# eval_qwen25vl_mmbench_video_lora.py
import os
import json
import math
import re
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
from datasets import load_dataset, DownloadConfig
from transformers import AutoProcessor

from peft import PeftModel

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    QwenVLModelClass = Qwen2_5_VLForConditionalGeneration
except Exception:
    QwenVLModelClass = None

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False

# qwen-vl-utils (권장)
try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except Exception:
    HAS_QWEN_VL_UTILS = False


# -------------------------
# HF cache setup (training 코드와 동일)
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
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)

    os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

    return {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_hub,
        "HF_DATASETS_CACHE": hf_datasets,
        "HF_TRANSFORMERS_CACHE": hf_transformers,
    }


# -------------------------
# Download / extract videos.zip (training 코드와 동일)
# -------------------------
def ensure_mmbench_videos(
    repo_id: str,
    videos_dir: Path,
    hf_cache_dir: Path,
    filename: str = "videos.zip",
):
    if not HAS_HF_HUB:
        raise RuntimeError("huggingface_hub not installed.")

    videos_dir.mkdir(parents=True, exist_ok=True)
    marker = videos_dir / ".extracted_ok"
    if marker.exists():
        return

    zip_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        cache_dir=str(hf_cache_dir),
    )
    zip_path = Path(zip_path)

    import zipfile
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(videos_dir))

    if not (videos_dir / "video").exists():
        print(f"[WARN] Extracted but {videos_dir/'video'} not found. Check: {videos_dir}")

    marker.write_text("ok\n")


# -------------------------
# Text normalization + token F1
# -------------------------
_punct_re = re.compile(r"[^a-z0-9\s]+")

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = _punct_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_f1(pred: str, gt: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gt).split()
    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0
    from collections import Counter
    cp, cg = Counter(p), Counter(g)
    common = sum((cp & cg).values())
    if common == 0:
        return 0.0
    prec = common / max(1, len(p))
    rec = common / max(1, len(g))
    return (2 * prec * rec) / max(1e-12, (prec + rec))


# -------------------------
# Safe process_vision_info wrapper
#   - qwen-vl-utils 버전에 따라 return_video_kwargs 유무가 달라짐
# -------------------------
def safe_process_vision(messages):
    """
    Returns: (image_inputs, video_inputs, video_kwargs_dict)
    """
    if not HAS_QWEN_VL_UTILS:
        raise RuntimeError(
            "qwen-vl-utils is required for this evaluation path.\n"
            "Install: pip install qwen-vl-utils[decord] (or at least qwen-vl-utils)"
        )

    # 최신 버전: return_video_kwargs=True 지원
    try:
        out = process_vision_info(messages, return_video_kwargs=True)
        # 보통 (image_inputs, video_inputs, video_kwargs)
        if isinstance(out, tuple) and len(out) == 3:
            return out[0], out[1], out[2] if out[2] is not None else {}
    except TypeError:
        pass

    # 구버전 fallback: (image_inputs, video_inputs)
    out = process_vision_info(messages)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1], {}
    raise RuntimeError("Unexpected return type from process_vision_info().")


# -------------------------
# Build one sample inputs
# -------------------------
def build_one_input(
    ex: Dict[str, Any],
    processor,
    videos_root: Path,
    max_length: int,
    nframes: int,
    fps: float,
    max_pixels: int,
) -> Dict[str, Any]:
    q = ex.get("question", "")
    a = ex.get("answer", "")
    vp = ex.get("video_path", "")

    q = q if isinstance(q, str) else str(q)
    a = a if isinstance(a, str) else str(a)
    vp = vp if isinstance(vp, str) else str(vp)

    rel = vp.lstrip("./")
    video_path = (videos_root / rel).resolve()
    if not video_path.exists():
        return {"skip": True, "reason": f"missing_video: {video_path}", "question": q, "answer": a, "video_path": str(video_path)}

    # qwen-vl-utils 권장 포맷: {"type":"video","video":"file:///abs/path.mp4", ...}
    vid_ele = {
        "type": "video",
        "video": video_path.as_uri(),
        "max_pixels": int(max_pixels),
    }
    # 프레임 제한: nframes 우선(고정 프레임) / 아니면 fps
    if nframes > 0:
        vid_ele["nframes"] = int(nframes)
    else:
        vid_ele["fps"] = float(fps)

    messages = [
        {
            "role": "user",
            "content": [
                vid_ele,
                {"type": "text", "text": f"Question: {q}\nAnswer:"},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = safe_process_vision(messages)

    # fps 중복 전달 방지:
    # 어떤 버전에선 video_kwargs에 fps 관련 항목이 들어올 수 있어,
    # processor(..., fps=...)를 같이 주면 "multiple values for fps"가 날 수 있음.
    extra = dict(video_kwargs) if isinstance(video_kwargs, dict) else {}
    proc_kwargs = dict(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # fps를 별도 kwarg로 주는 건 nframes 미사용 + extra에 fps가 없을 때만
    if (nframes <= 0) and ("fps" not in extra):
        proc_kwargs["fps"] = float(fps)
    # extra는 마지막에 병합
    proc_kwargs.update(extra)

    enc = processor(**proc_kwargs)

    input_ids = enc["input_ids"]
    if input_ids.shape[1] > max_length:
        return {
            "skip": True,
            "reason": f"too_long: L={input_ids.shape[1]} > max_length={max_length}",
            "question": q,
            "answer": a,
            "video_path": str(video_path),
            "seq_len": int(input_ids.shape[1]),
        }

    # tensor들만 따로
    tensors = {k: v for k, v in enc.items() if isinstance(v, torch.Tensor)}
    return {
        "skip": False,
        "question": q,
        "answer": a,
        "video_path": str(video_path),
        "tensors": tensors,
        "seq_len": int(input_ids.shape[1]),
    }


# -------------------------
# Main eval
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")

    # LoRA adapter (training output/final)
    ap.add_argument("--adapter_dir", type=str, required=True, help="e.g., .../qwen25vl_mmbench_video/baseline/llm_attn/final")

    ap.add_argument("--dataset_name", type=str, default="lscpku/MMBench-Video")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_eval_samples", type=int, default=200)

    ap.add_argument("--videos_dir", type=str, default="")
    ap.add_argument("--auto_download_videos", action="store_true")

    # video sampling controls (중요)
    ap.add_argument("--nframes", type=int, default=4, help="fixed number of frames. If >0, overrides fps.")
    ap.add_argument("--fps", type=float, default=0.25, help="used only when nframes<=0")
    ap.add_argument("--video_max_pixels", type=int, default=128 * 28 * 28)

    # processor pixels (이미지/비디오 토큰 수에 영향)
    ap.add_argument("--min_pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=128 * 28 * 28)

    # backend control: env var (kwarg backend는 경고/무시됨)
    ap.add_argument("--force_video_reader", type=str, default="", choices=["", "decord", "torchvision"])

    ap.add_argument("--device_map", type=str, default="cuda:0")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--load_in_4bit", action="store_true")

    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)

    ap.add_argument("--save_jsonl", type=str, default="", help="predictions jsonl path")
    ap.add_argument("--save_summary", type=str, default="", help="summary json path")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if QwenVLModelClass is None:
        raise RuntimeError("Could not import Qwen2_5_VLForConditionalGeneration. Update transformers.")
    if not HAS_QWEN_VL_UTILS:
        raise RuntimeError(
            "qwen-vl-utils가 필요합니다. (권장) \n"
            "pip install qwen-vl-utils[decord] (또는 최소 qwen-vl-utils)\n"
            "비디오 예시는 Qwen2.5-VL 모델 카드가 이 방식을 권장합니다."
        )

    # backend는 env var로 강제 (공식 가이드)
    if args.force_video_reader:
        os.environ["FORCE_QWENVL_VIDEO_READER"] = args.force_video_reader

    base_dir = Path(args.base_dir).resolve()
    paths = setup_hf_cache(base_dir)

    adapter_dir = Path(args.adapter_dir).resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter_dir not found: {adapter_dir}")

    # save paths
    save_jsonl = Path(args.save_jsonl).resolve() if args.save_jsonl else (adapter_dir / "eval_preds.jsonl")
    save_summary = Path(args.save_summary).resolve() if args.save_summary else (adapter_dir / "eval_summary.json")

    # Processor: training에서 저장한 processor를 쓰는 게 가장 안전
    # (없으면 base model에서 로드)
    proc_src = str(adapter_dir) if (adapter_dir / "preprocessor_config.json").exists() else args.model_name
    processor = AutoProcessor.from_pretrained(
        proc_src,
        cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]),
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
    )

    # Model load
    model_kwargs = dict(
        cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]),
        device_map=args.device_map,
    )

    if args.load_in_4bit:
        if not HAS_BNB:
            raise RuntimeError("BitsAndBytesConfig not available. Install bitsandbytes.")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = bnb
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    base_model = QwenVLModelClass.from_pretrained(args.model_name, **model_kwargs)
    base_model.config.use_cache = True
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    # Videos
    videos_root = Path(args.videos_dir).resolve() if args.videos_dir else (base_dir / "mmbench_video_assets")
    if args.auto_download_videos:
        ensure_mmbench_videos(
            repo_id=args.dataset_name,
            videos_dir=videos_root,
            hf_cache_dir=paths["HF_HUB_CACHE"],
            filename="videos.zip",
        )

    # Dataset
    dl_cfg = DownloadConfig(resume_download=True, max_retries=50)
    ds = load_dataset(
        args.dataset_name,
        split=f"{args.split}[:{args.max_eval_samples}]",
        cache_dir=str(paths["HF_DATASETS_CACHE"]),
        download_config=dl_cfg,
    )
    ds = ds.filter(lambda ex: isinstance(ex.get("question", None), str)
                          and isinstance(ex.get("answer", None), str)
                          and isinstance(ex.get("video_path", None), str))
    print("Rows:", len(ds))

    # Eval loop
    total = len(ds)
    n_ok = 0
    n_skip = 0
    sum_em = 0.0
    sum_f1 = 0.0

    save_jsonl.parent.mkdir(parents=True, exist_ok=True)
    f = open(save_jsonl, "w", encoding="utf-8")

    with torch.no_grad():
        for i in range(total):
            ex = ds[i]
            pack = build_one_input(
                ex=ex,
                processor=processor,
                videos_root=videos_root,
                max_length=args.max_length,
                nframes=args.nframes,
                fps=args.fps,
                max_pixels=args.video_max_pixels,
            )

            rec: Dict[str, Any] = {
                "idx": i,
                "video_path": pack.get("video_path", ex.get("video_path", "")),
                "question": pack.get("question", ex.get("question", "")),
                "gt_answer": pack.get("answer", ex.get("answer", "")),
            }

            if pack.get("skip", False):
                n_skip += 1
                rec["skip"] = True
                rec["reason"] = pack.get("reason", "unknown")
                rec["seq_len"] = pack.get("seq_len", None)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (i + 1) % args.log_every == 0:
                    print(f"[{i+1}/{total}] ok={n_ok} skip={n_skip} (last skip: {rec['reason']})")
                continue

            tensors = pack["tensors"]
            # move to device
            inputs = {k: v.to(args.device) for k, v in tensors.items()}

            gen_kwargs = dict(
                max_new_tokens=int(args.max_new_tokens),
                do_sample=bool(args.do_sample),
            )
            if args.do_sample:
                gen_kwargs.update(dict(temperature=float(args.temperature), top_p=float(args.top_p)))

            # generated_ids = model.generate(**inputs, **gen_kwargs)

            # in_len = int(inputs["input_ids"].shape[1])
            # out_ids = generated_ids[0, in_len:]
            # pred = processor.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            generated_ids = model.generate(**inputs, **gen_kwargs)

            in_len = int(inputs["input_ids"].shape[1])
            out_len = int(generated_ids.shape[1])

            # 어떤 모델/버전에선 generate가 input을 포함하지 않고 "new tokens only"를 줄 수도 있어서 방어
            if out_len <= in_len:
                gen_part = generated_ids[0]
            else:
                gen_part = generated_ids[0, in_len:]

            # 디버그: 처음 5개만 출력해보면 원인 바로 보입니다
            if i < 5:
                print(f"[DBG] in_len={in_len} out_len={out_len} gen_len={int(gen_part.numel())}")

            pred = processor.batch_decode(
                gen_part.unsqueeze(0),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0].strip()

            # 혹시라도 gen_part가 비어서 pred가 빈 문자열이면(=gen_len==0),
            # 전체를 디코드해서 "Answer:" 뒤만 뽑는 fallback
            if 1:#pred == "":
                full = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]
                pred = full.split("Answer:")[-1].strip()  # 프롬프트에 Answer:를 넣었으니 이게 꽤 잘 먹습니다.
                print("full, pred: ", full, pred)


            gt = rec["gt_answer"]
            em = 1.0 if normalize_text(pred) == normalize_text(gt) else 0.0
            f1 = token_f1(pred, gt)

            rec.update({
                "skip": False,
                "pred_answer": pred,
                "em": em,
                "f1": f1,
                "seq_len": pack.get("seq_len", None),
            })

            n_ok += 1
            sum_em += float(em)
            sum_f1 += float(f1)

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (i + 1) % args.log_every == 0:
                avg_em = (sum_em / max(1, n_ok))
                avg_f1 = (sum_f1 / max(1, n_ok))
                print(f"[{i+1}/{total}] ok={n_ok} skip={n_skip} | EM={avg_em:.3f} F1={avg_f1:.3f}")

            # (선택) 주기적으로 캐시 비우기
            if torch.cuda.is_available() and ((i + 1) % 50 == 0):
                torch.cuda.empty_cache()

    f.close()

    summary = {
        "total": total,
        "ok": n_ok,
        "skip": n_skip,
        "avg_em": (sum_em / max(1, n_ok)),
        "avg_f1": (sum_f1 / max(1, n_ok)),
        "args": vars(args),
    }
    save_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(save_summary, "w", encoding="utf-8") as wf:
        json.dump(summary, wf, indent=2, ensure_ascii=False)

    print("Saved preds:", str(save_jsonl))
    print("Saved summary:", str(save_summary))


if __name__ == "__main__":
    main()
