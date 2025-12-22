import os
import re
import csv
import math
import json
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from datasets import load_dataset, DownloadConfig
from transformers import ViltProcessor, ViltForQuestionAnswering

# Optional (timeout control for HF datasets download)
try:
    import aiohttp
    HAS_AIOHTTP = True
except Exception:
    HAS_AIOHTTP = False


# -------------------------
# Metrics
# -------------------------
def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return float((p * (p.log() - q.log())).sum().item())

def js_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    # Jensen-Shannon divergence (symmetric, bounded)
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, eps=eps) + 0.5 * kl_div(q, m, eps=eps)


# -------------------------
# Image perturbations
# -------------------------
def blank_like(img: Image.Image) -> Image.Image:
    return Image.new("RGB", img.size, color=(128, 128, 128))

def lowres_like(img: Image.Image, side: int = 64) -> Image.Image:
    w, h = img.size
    s = min(side, w, h)
    small = img.resize((s, s), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)

def blur_like(img: Image.Image, radius: float = 2.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def center_crop_like(img: Image.Image, crop_ratio: float = 0.7) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - cw) // 2
    top = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.BILINEAR)


# -------------------------
# Visual-cue heuristic
# -------------------------
VISUAL_CUE_PATTERNS = [
    r"\b(in the picture|in this picture|in the image|in this image)\b",
    r"\b(shown|shows|showing|pictured|depicted)\b",
    r"\b(figure|diagram|graph|chart|map)\b",
    r"\b(which animal|what animal)\b",
    r"\b(what color|which color|colour)\b",
    r"\b(how many|number of)\b",
    r"\b(these|this)\b",
    r"\b(look at)\b",
    r"\b(photo|photograph|drawing)\b",
]

def is_visual_cued(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in VISUAL_CUE_PATTERNS)


# -------------------------
# Data record
# -------------------------
@dataclass
class Record:
    dataset_idx: int
    question: str
    visual_cued: int

    pred_orig: str
    p_orig: float

    pred_blank: str
    p_blank: float

    pred_blur: str
    p_blur: float

    pred_lowres: str
    p_lowres: float

    pred_crop: str
    p_crop: float

    # swap stats
    swap_k: int
    swap_same_rate: float
    swap_mean_kl: float
    swap_mean_js: float

    # blank/corrupt divergence vs orig
    kl_blank: float
    js_blank: float
    kl_blur: float
    js_blur: float
    kl_lowres: float
    js_lowres: float
    kl_crop: float
    js_crop: float

    # how much orig top answer probability changes under perturbations
    delta_p_blank: float
    delta_p_blur: float
    delta_p_lowres: float
    delta_p_crop: float

    # convenience flags
    same_orig_blank: int
    unexpected_dominance: int


# -------------------------
# Model inference
# -------------------------
def predict_probs(
    processor: ViltProcessor,
    model: ViltForQuestionAnswering,
    device: str,
    images: List[Image.Image],
    questions: List[str],
    max_length: int = 40,
) -> torch.Tensor:
    with torch.no_grad():
        inputs = processor(
            images=images,
            text=questions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).detach().cpu()
    return probs

def top_answer(model: ViltForQuestionAnswering, probs_1d: torch.Tensor) -> Tuple[str, int, float]:
    tid = int(torch.argmax(probs_1d).item())
    ans = model.config.id2label[tid]
    return ans, tid, float(probs_1d[tid].item())


# -------------------------
# HF cache setup
# -------------------------
def setup_hf_cache(base_dir: Path) -> Dict[str, Path]:
    base_dir = base_dir.resolve()
    hf_home = base_dir / "hf_home"
    hf_hub = base_dir / "hf_hub"
    hf_datasets = base_dir / "hf_datasets"
    hf_transformers = base_dir / "hf_transformers"
    for p in [hf_home, hf_hub, hf_datasets, hf_transformers]:
        p.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_hub)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_transformers)

    # compatibility
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)

    # timeouts (slow networks)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

    return {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_hub,
        "HF_DATASETS_CACHE": hf_datasets,
        "HF_TRANSFORMERS_CACHE": hf_transformers,
    }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    parser.add_argument("--model_name", type=str, default="dandelin/vilt-b32-finetuned-vqa")
    parser.add_argument("--dataset_name", type=str, default="derek-thomas/ScienceQA")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_samples", type=int, default=1000, help="How many rows to scan from the split")
    parser.add_argument("--max_length", type=int, default=40, help="Max text tokens (ViLT VQA uses 40)")
    parser.add_argument("--swap_k", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)

    # thresholds for unexpected dominance
    parser.add_argument("--th_same_rate", type=float, default=0.85)
    parser.add_argument("--th_swap_js", type=float, default=0.10)
    parser.add_argument("--require_visual_cue", action="store_true", help="Only consider visual-cued questions for unexpected dominance")
    parser.add_argument("--max_save_cases", type=int, default=200, help="How many unexpected cases to save into folder")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = Path(args.base_dir).resolve()
    paths = setup_hf_cache(base_dir)

    out_csv = base_dir / "dominance_results.csv"
    out_json = base_dir / "unexpected_cases.json"
    case_dir = base_dir / "unexpected_cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load model/processor
    processor = ViltProcessor.from_pretrained(args.model_name, cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]))
    model = ViltForQuestionAnswering.from_pretrained(args.model_name, cache_dir=str(paths["HF_TRANSFORMERS_CACHE"])).to(device)
    model.eval()

    # Load dataset slice
    download_config = DownloadConfig(resume_download=True, max_retries=50)
    storage_options = None
    if HAS_AIOHTTP:
        storage_options = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}}

    split_expr = f"{args.split}[:{args.max_samples}]"
    ds = load_dataset(
        args.dataset_name,
        split=split_expr,
        cache_dir=str(paths["HF_DATASETS_CACHE"]),
        download_config=download_config,
        storage_options=storage_options,
    )

    print(f"Loaded: {args.dataset_name} {split_expr} rows: {len(ds)}")
    print("Columns:", ds.column_names)

    # build image pool for swaps (only valid images)
    valid_indices: List[int] = []
    images_pool: List[Image.Image] = []
    for i in range(len(ds)):
        img = ds[i].get("image", None)
        if isinstance(img, Image.Image):
            images_pool.append(img.convert("RGB"))
            valid_indices.append(i)
    print("Usable samples with image:", len(images_pool))
    if len(images_pool) < max(10, args.swap_k):
        raise RuntimeError("Too few images in pool for swaps. Increase max_samples or use another dataset.")

    # CSV writer
    fieldnames = list(asdict(Record(
        dataset_idx=0, question="", visual_cued=0,
        pred_orig="", p_orig=0.0,
        pred_blank="", p_blank=0.0,
        pred_blur="", p_blur=0.0,
        pred_lowres="", p_lowres=0.0,
        pred_crop="", p_crop=0.0,
        swap_k=0, swap_same_rate=0.0, swap_mean_kl=0.0, swap_mean_js=0.0,
        kl_blank=0.0, js_blank=0.0,
        kl_blur=0.0, js_blur=0.0,
        kl_lowres=0.0, js_lowres=0.0,
        kl_crop=0.0, js_crop=0.0,
        delta_p_blank=0.0, delta_p_blur=0.0, delta_p_lowres=0.0, delta_p_crop=0.0,
        same_orig_blank=0, unexpected_dominance=0
    )).keys())

    records: List[Record] = []
    unexpected: List[Dict[str, Any]] = []
    saved_count = 0

    # iterate through valid samples
    for pool_i, ds_i in enumerate(valid_indices):
        ex = ds[ds_i]
        q = ex.get("question", "")
        if not isinstance(q, str) or len(q.strip()) == 0:
            continue

        img0 = images_pool[pool_i]
        vcue = 1 if is_visual_cued(q) else 0

        # Build perturbations
        img_blank = blank_like(img0)
        img_blur = blur_like(img0, radius=2.0)
        img_lowres = lowres_like(img0, side=64)
        img_crop = center_crop_like(img0, crop_ratio=0.7)

        # Batch predict orig + (blank/blur/lowres/crop)
        imgs_batch = [img0, img_blank, img_blur, img_lowres, img_crop]
        qs_batch = [q] * len(imgs_batch)
        probs = predict_probs(processor, model, device, imgs_batch, qs_batch, max_length=args.max_length)

        p0 = probs[0]
        ans0, tid0, prob0 = top_answer(model, p0)

        p_blank = probs[1]
        ansb, tidb, probb = top_answer(model, p_blank)

        p_blur = probs[2]
        ans_blur, tid_blur, prob_blur = top_answer(model, p_blur)

        p_lowres = probs[3]
        ans_lr, tid_lr, prob_lr = top_answer(model, p_lowres)

        p_crop = probs[4]
        ans_cr, tid_cr, prob_cr = top_answer(model, p_crop)

        # Divergences vs orig
        k_blank = kl_div(p0, p_blank)
        j_blank = js_div(p0, p_blank)

        k_blur = kl_div(p0, p_blur)
        j_blur = js_div(p0, p_blur)

        k_lr = kl_div(p0, p_lowres)
        j_lr = js_div(p0, p_lowres)

        k_cr = kl_div(p0, p_crop)
        j_cr = js_div(p0, p_crop)

        # Probability drop of orig top-answer under perturbations
        delta_blank = float(p_blank[tid0].item() - p0[tid0].item())
        delta_blur = float(p_blur[tid0].item() - p0[tid0].item())
        delta_lr = float(p_lowres[tid0].item() - p0[tid0].item())
        delta_cr = float(p_crop[tid0].item() - p0[tid0].item())

        # Swap stats
        swap_ids = random.sample(range(len(images_pool)), k=min(args.swap_k, len(images_pool)))
        swap_imgs = [images_pool[j] for j in swap_ids]

        same_cnt = 0
        kls: List[float] = []
        jss: List[float] = []

        # batched swaps
        for b in range(0, len(swap_imgs), args.batch_size):
            batch_imgs = swap_imgs[b:b + args.batch_size]
            batch_qs = [q] * len(batch_imgs)
            psw = predict_probs(processor, model, device, batch_imgs, batch_qs, max_length=args.max_length)

            top_ids = torch.argmax(psw, dim=-1).tolist()
            same_cnt += sum(1 for t in top_ids if t == tid0)

            for r in range(psw.shape[0]):
                kls.append(kl_div(p0, psw[r]))
                jss.append(js_div(p0, psw[r]))

        same_rate = same_cnt / len(swap_imgs)
        mean_kl = float(sum(kls) / len(kls)) if kls else 0.0
        mean_js = float(sum(jss) / len(jss)) if jss else 0.0

        same_orig_blank = 1 if (tid0 == tidb) else 0

        # Unexpected dominance rule (default: visual-cued recommended)
        consider = True
        if args.require_visual_cue and vcue == 0:
            consider = False

        unexpected_flag = 0
        if consider:
            if same_orig_blank == 1 and same_rate >= args.th_same_rate and mean_js <= args.th_swap_js:
                unexpected_flag = 1

        rec = Record(
            dataset_idx=int(ds_i),
            question=q,
            visual_cued=vcue,

            pred_orig=ans0, p_orig=prob0,
            pred_blank=ansb, p_blank=probb,
            pred_blur=ans_blur, p_blur=prob_blur,
            pred_lowres=ans_lr, p_lowres=prob_lr,
            pred_crop=ans_cr, p_crop=prob_cr,

            swap_k=len(swap_imgs),
            swap_same_rate=float(same_rate),
            swap_mean_kl=float(mean_kl),
            swap_mean_js=float(mean_js),

            kl_blank=float(k_blank), js_blank=float(j_blank),
            kl_blur=float(k_blur), js_blur=float(j_blur),
            kl_lowres=float(k_lr), js_lowres=float(j_lr),
            kl_crop=float(k_cr), js_crop=float(j_cr),

            delta_p_blank=float(delta_blank),
            delta_p_blur=float(delta_blur),
            delta_p_lowres=float(delta_lr),
            delta_p_crop=float(delta_cr),

            same_orig_blank=same_orig_blank,
            unexpected_dominance=unexpected_flag,
        )

        records.append(rec)

        # Save unexpected cases with artifacts
        if unexpected_flag == 1 and saved_count < args.max_save_cases:
            # Create folder
            cdir = case_dir / f"idx_{ds_i}"
            cdir.mkdir(parents=True, exist_ok=True)

            # Save images
            img0.save(cdir / "orig.png")
            img_blank.save(cdir / "blank.png")
            img_blur.save(cdir / "blur.png")
            img_lowres.save(cdir / "lowres.png")
            img_crop.save(cdir / "crop.png")

            # Save a few swaps
            for si, simg in enumerate(swap_imgs[:min(8, len(swap_imgs))]):
                simg.save(cdir / f"swap_{si+1:02d}.png")

            # Save report
            report = {
                "dataset_idx": int(ds_i),
                "question": q,
                "visual_cued": vcue,
                "pred_orig": ans0, "p_orig": prob0,
                "pred_blank": ansb, "p_blank": probb,
                "swap_same_rate": float(same_rate),
                "swap_mean_js": float(mean_js),
                "swap_mean_kl": float(mean_kl),
                "kl_blank": float(k_blank),
                "js_blank": float(j_blank),
                "delta_p_blank": float(delta_blank),
            }
            with open(cdir / "report.json", "w") as f:
                json.dump(report, f, indent=2)

            unexpected.append(report)
            saved_count += 1

        # simple progress
        if (len(records) % 50) == 0:
            n_unexp = sum(r.unexpected_dominance for r in records)
            print(f"Processed {len(records)} / {len(valid_indices)} | unexpected={n_unexp}")

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    # Write JSON list
    with open(out_json, "w") as f:
        json.dump(unexpected, f, indent=2)

    # Summary
    total = len(records)
    unexp = sum(r.unexpected_dominance for r in records)
    vcue_total = sum(r.visual_cued for r in records)
    vcue_unexp = sum(r.unexpected_dominance for r in records if r.visual_cued == 1)
    print("\nDone.")
    print("Total evaluated:", total)
    print("Visual-cued:", vcue_total)
    print("Unexpected dominance cases:", unexp)
    print("Unexpected among visual-cued:", vcue_unexp)
    print("Saved CSV:", str(out_csv))
    print("Saved unexpected JSON:", str(out_json))
    print("Saved artifacts dir:", str(case_dir))


if __name__ == "__main__":
    main()
