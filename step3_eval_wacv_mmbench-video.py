import os
import re
import json
import csv
import math
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
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

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

class AttnStatsCollector:
    def __init__(self, prompt_len, gen_start, prompt_text_mask, prompt_vis_mask, layerid_to_group):
        self.prompt_len = int(prompt_len)
        self.gen_start = int(gen_start)
        self.prompt_text_mask = prompt_text_mask  # [prompt_len] bool (on same device as attn)
        self.prompt_vis_mask  = prompt_vis_mask   # [prompt_len] bool
        self.layerid_to_group = layerid_to_group  # dict: layer_id -> "early"/"middle"/"late"
        self.at_list = {"early": [], "middle": [], "late": []}
        self.ao_list = {"early": [], "middle": [], "late": []}

    @torch.no_grad()
    def hook(self, layer_id):
        def _fn(module, inp, out):
            # out is usually tuple: (hidden_states, attn_weights, ...)
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                return
            attn = out[1]
            if isinstance(attn, (tuple, list)):
                # sometimes returns (attn_weights, ...)
                attn = attn[0] if len(attn) > 0 else None
            if not torch.is_tensor(attn):
                return
            # expected: [B, H, L, L]
            if attn.dim() != 4:
                return

            gname = self.layerid_to_group.get(layer_id, None)
            if gname is None:
                return

            # slice only what we need: generated queries -> prompt keys
            # attn: [1,H,L,L]
            if self.gen_start >= attn.shape[2]:
                return
            A = attn[0, :, self.gen_start:, :self.prompt_len]  # [H, Ngen, prompt_len]

            # ensure masks on same device
            txtm = self.prompt_text_mask.to(A.device)
            vism = self.prompt_vis_mask.to(A.device)

            t_sum = A[:, :, txtm].sum()
            o_sum = A[:, :, vism].sum()
            denom = (t_sum + o_sum).clamp(min=1e-12)

            AT = (t_sum / denom).item()
            AO = (o_sum / denom).item()

            self.at_list[gname].append(AT)
            self.ao_list[gname].append(AO)
        return _fn


class TempForceAttnOnLayers:
    """
    선택 레이어들만 layer.forward를 감싸서 output_attentions=True를 강제.
    (global output_attentions=False로 model forward를 호출해야 함)
    """
    def __init__(self, layers, target_layer_ids):
        self.layers = layers
        self.target = set(target_layer_ids)
        self.orig = {}

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            if i not in self.target:
                continue
            if hasattr(layer, "_orig_forward_eval"):
                continue
            self.orig[i] = layer.forward
            orig = layer.forward

            def wrapped(*args, __orig=orig, **kwargs):
                # 중요한 포인트: LM 전체는 output_attentions=False로 호출하지만,
                # 여기서만 True로 강제해 attn_weights를 out[1]로 받는다.
                kwargs["output_attentions"] = True
                # 평가용 forward는 use_cache=False로 돌리는 게 안전 (tuple 구조 단순화)
                kwargs["use_cache"] = False
                return __orig(*args, **kwargs)

            layer._orig_forward_eval = orig
            layer.forward = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "_orig_forward_eval"):
                layer.forward = layer._orig_forward_eval
                delattr(layer, "_orig_forward_eval")
        return False


# -------------------------
# HF cache setup (same as train)
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
# Download / extract videos.zip (same as train)
# -------------------------
def ensure_mmbench_videos(
    repo_id: str,
    videos_dir: Path,
    hf_cache_dir: Path,
    filename: str = "videos.zip",
):
    if not HAS_HF_HUB:
        raise RuntimeError("huggingface_hub not installed. pip install huggingface_hub (and optionally hf-xet).")

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
        print(f"[WARN] Extracted but {videos_dir/'video'} not found. Check structure under: {videos_dir}")
    marker.write_text("ok\n")


# -------------------------
# Qwen vision token ids helper
# -------------------------
def _get_qwen_vision_token_ids(model, tokenizer) -> Tuple[int, int]:
    image_tid = getattr(getattr(model, "config", None), "image_token_id", None)
    video_tid = getattr(getattr(model, "config", None), "video_token_id", None)

    if isinstance(image_tid, int) and isinstance(video_tid, int):
        return int(image_tid), int(video_tid)

    # fallback: tokenizer에서 찾기
    image_tid2 = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_tid2 = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    if image_tid2 is None or video_tid2 is None:
        raise RuntimeError("Could not resolve <|image_pad|>/<|video_pad|> token ids.")
    return int(image_tid2), int(video_tid2)


# -------------------------
# Answer normalization / matching
# -------------------------
def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # light punctuation normalization
    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"[ \t]+", " ", s)
    return s

def is_exact_match(pred: str, gt: str) -> bool:
    return normalize_text(pred) == normalize_text(gt)


# -------------------------
# Effective rank
# -------------------------
@torch.no_grad()
def effective_rank(X: torch.Tensor, eps: float = 1e-12) -> float:
    """
    X: [N, D]
    Effective rank = exp( H(p) ), p_i = s_i / sum(s)
    """
    if X is None or X.numel() == 0:
        return float("nan")
    X = X.float()
    if X.shape[0] < 2 or X.shape[1] < 2:
        return float("nan")
    X = X - X.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(X)  # [min(N,D)]
        s = torch.clamp(s, min=0.0)
        if not torch.isfinite(s).all():
            return float("nan")
        z = s.sum().clamp(min=eps)
        p = s / z
        H = -(p * (p + eps).log()).sum()
        er = torch.exp(H)
        return float(er.item())
    except Exception:
        return float("nan")


# -------------------------
# Linear CKA (token-sample CKA)
# -------------------------
@torch.no_grad()
def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    X, Y: [N, D] with same N
    CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    """
    if X is None or Y is None:
        return float("nan")
    if X.numel() == 0 or Y.numel() == 0:
        return float("nan")
    if X.shape[0] != Y.shape[0]:
        return float("nan")
    if X.shape[0] < 2:
        return float("nan")

    X = X.float() - X.float().mean(dim=0, keepdim=True)
    Y = Y.float() - Y.float().mean(dim=0, keepdim=True)

    XT_Y = X.T @ Y
    XT_X = X.T @ X
    YT_Y = Y.T @ Y

    hsic = (XT_Y * XT_Y).sum()
    norm_x = torch.sqrt((XT_X * XT_X).sum().clamp(min=eps))
    norm_y = torch.sqrt((YT_Y * YT_Y).sum().clamp(min=eps))
    cka = hsic / (norm_x * norm_y + eps)
    return float(cka.item())


# -------------------------
# Layer access + patching (attention only for selected layers)
# -------------------------
def getattr_nested(obj, path: str) -> Optional[Any]:
    cur = obj
    for p in path.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur

def find_language_model(model) -> nn.Module:
    # train 코드와 동일한 우선순위로 찾기
    lm = getattr(model, "language_model", None)
    if lm is None:
        lm = getattr_nested(model, "model.language_model")
    if lm is None:
        lm = getattr(model, "model", None)
    if lm is None:
        raise RuntimeError("Could not locate language model module.")
    return lm

def find_transformer_layers(lm: nn.Module) -> List[nn.Module]:
    # 흔한 경로들
    for path in ["model.layers", "layers", "decoder.layers"]:
        layers = getattr_nested(lm, path)
        if isinstance(layers, (list, nn.ModuleList)):
            return list(layers)
    raise RuntimeError("Could not locate transformer layers list inside language model.")

def select_layer_groups(num_layers: int) -> Dict[str, List[int]]:
    """
    paper style: early=first 2, middle=middle 2, late=last 2
    """
    if num_layers < 4:
        return {"early": [0], "middle": [], "late": [num_layers-1]}
    early = [0, 1]
    mid0 = max(0, num_layers // 2 - 1)
    middle = [mid0, min(num_layers - 1, mid0 + 1)]
    late = [num_layers - 2, num_layers - 1]
    return {"early": early, "middle": middle, "late": late}

# -------------------------
# Build prompt inputs (video + question)
# -------------------------
def build_prompt_inputs(
    ex: Dict[str, Any],
    processor,
    videos_root: Path,
    num_frames: int,
    video_backend: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    q = ex.get("question", "")
    a = ex.get("answer", "")
    vp = ex.get("video_path", "")
    q = q if isinstance(q, str) else str(q)
    a = a if isinstance(a, str) else str(a)
    vp = vp if isinstance(vp, str) else str(vp)

    rel = vp.lstrip("./")
    video_path = (videos_root / rel).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    user_conv = [{
        "role": "user",
        "content": [
            {"type": "video", "path": str(video_path)},
            {"type": "text", "text": f"Question: {q}\nAnswer:"},
        ],
    }]

    enc = processor.apply_chat_template(
        user_conv,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        num_frames=num_frames,
        # videos_kwargs={"backend": video_backend, "do_sample_frames": True, "num_frames": num_frames},
    )

    if "pixel_values_videos" in enc and isinstance(enc["pixel_values_videos"], torch.Tensor):
        enc["pixel_values_videos"] = enc["pixel_values_videos"].to(torch.float16)

    meta = {
        "question": q,
        "gt_answer": a,
        "video_path": vp,
        "video_abs_path": str(video_path),
        "video_id": Path(video_path).stem,
    }
    return enc, meta


class InputEmbedCatcher:
    def __init__(self, prompt_len: int):
        self.prompt_len = int(prompt_len)
        self.inputs_embeds_prompt = None  # [1, prompt_len, D]
        self.done = False

    def hook(self, module, args, kwargs):
        if self.done:
            return
        inp_emb = kwargs.get("inputs_embeds", None)
        if inp_emb is None:
            return
        if not torch.isfinite(inp_emb).all():
            return
        try:
            self.inputs_embeds_prompt = inp_emb[:, : self.prompt_len, :].detach().float().cpu()
            self.done = True
        except Exception:
            return


@torch.no_grad()
def finalize_mdi_aei(at_list, ao_list, nT, nO):
    # 평균 AT/AO
    if len(at_list) == 0:
        return dict(AT=float("nan"), AO=float("nan"), MDI=float("nan"), AEI_T=float("nan"), AEI_O=float("nan"))
    AT = float(sum(at_list) / len(at_list))
    AO = float(sum(ao_list) / len(ao_list))

    eps = 1e-12
    mdi = (AT / max(nT, 1)) / (AO / max(nO, 1) + eps)

    QT = nT / (nT + nO + eps)
    QO = nO / (nT + nO + eps)
    aei_t = AT / (QT + eps)
    aei_o = AO / (QO + eps)
    return dict(AT=AT, AO=AO, MDI=mdi, AEI_T=aei_t, AEI_O=aei_o)

# -------------------------
# Main eval
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")

    ap.add_argument("--ckpt_dir", type=str, required=True, help="LoRA checkpoint dir (e.g., .../final or checkpoint_step_XXX)")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--device_map", type=str, default="cuda:0")
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--dataset_name", type=str, default="lscpku/MMBench-Video")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_eval_samples", type=int, default=400)

    ap.add_argument("--videos_dir", type=str, default="")
    ap.add_argument("--auto_download_videos", action="store_true")
    ap.add_argument("--num_frames", type=int, default=4)
    ap.add_argument("--video_backend", type=str, default="decord")

    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--save_mismatch_jsonl", action="store_true")

    # CKA token subsample
    ap.add_argument("--cka_tokens", type=int, default=256)

    args = ap.parse_args()

    if QwenVLModelClass is None:
        raise RuntimeError("Could not import Qwen2_5_VLForConditionalGeneration. Update transformers.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = Path(args.base_dir).resolve()
    paths = setup_hf_cache(base_dir)

    ckpt_dir = Path(args.ckpt_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (ckpt_dir / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # processor: prefer ckpt_dir (saved by train), fallback model_name
    try:
        processor = AutoProcessor.from_pretrained(str(ckpt_dir))
    except Exception:
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]),
        )
    tokenizer = processor.tokenizer

    # Disable flash/sdpa to get attentions reliably (best-effort)
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    # model load
    model_kwargs = dict(
        cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]),
        device_map=args.device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",  # best effort
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

    base_model = QwenVLModelClass.from_pretrained(args.model_name, **model_kwargs)
    base_model.config.use_cache = True  # generation uses cache; forward metrics will set use_cache=False
    model = PeftModel.from_pretrained(base_model, str(ckpt_dir))
    model.eval()

    image_tid, video_tid = _get_qwen_vision_token_ids(base_model, tokenizer)

    lm = find_language_model(model)
    layers = find_transformer_layers(lm)
    groups = select_layer_groups(len(layers))

    keep = sorted(set(groups["early"] + groups["middle"] + groups["late"]))
    layerid_to_group = {}
    for gname, ids in groups.items():
        for lid in ids:
            layerid_to_group[lid] = gname

    # videos root
    videos_root = Path(args.videos_dir).resolve() if args.videos_dir else (base_dir / "mmbench_video_assets")
    if args.auto_download_videos:
        ensure_mmbench_videos(
            repo_id=args.dataset_name,
            videos_dir=videos_root,
            hf_cache_dir=paths["HF_HUB_CACHE"],
            filename="videos.zip",
        )

    # dataset
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
    print("Eval rows:", len(ds))

    # outputs
    csv_path = out_dir / "eval_all.csv"
    mismatch_path = out_dir / "mismatch.jsonl"

    fieldnames = [
        "idx", "video_path", "video_abs_path", "video_id",
        "question", "gt_answer", "pred_answer",
        "is_correct",
        "prompt_len", "gen_len",
        "n_text_tokens", "n_video_tokens",
        # MDI/AEI per group
        "MDI_early", "AEI_T_early", "AEI_O_early",
        "MDI_middle", "AEI_T_middle", "AEI_O_middle",
        "MDI_late", "AEI_T_late", "AEI_O_late",
        # effective rank + CKA
        "effrank_text", "effrank_video",
        "cka_text_video",
    ]

    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    writer.writeheader()

    fj = open(mismatch_path, "w", encoding="utf-8") if args.save_mismatch_jsonl else None

    it = range(len(ds))
    if tqdm is not None:
        it = tqdm(it, desc="eval")

    for i in it:
        ex = ds[i]

        # 1) build prompt inputs
        try:
            enc, meta = build_prompt_inputs(
                ex=ex,
                processor=processor,
                videos_root=videos_root,
                num_frames=args.num_frames,
                video_backend=args.video_backend,
            )
        except Exception as e:
            print(f"[SKIP] idx={i} prompt build failed: {e}")
            continue

        # move to device
        batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
        prompt_len = int(batch["input_ids"].shape[1])

        # 2) generate
        with torch.no_grad():
            out_ids = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        # out_ids: [1, prompt_len + gen_len]
        gen_ids = out_ids[0, prompt_len:]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        gt = meta["gt_answer"]
        correct = is_exact_match(pred, gt)

        # 3) forward once on full sequence:
        #    - MDI/AEI: hook로 selected layers에서만 attention 통계 수집
        #    - effective rank/CKA: LM entry inputs_embeds(prompt) 캡쳐
        full_ids = out_ids
        full_attn_mask = torch.ones_like(full_ids, device=full_ids.device)

        fwd_inputs = dict(batch)
        fwd_inputs["input_ids"] = full_ids
        fwd_inputs["attention_mask"] = full_attn_mask

        # prompt masks (prompt ids 기준)
        prompt_ids = batch["input_ids"][0]  # prompt only
        prompt_vis_mask = (prompt_ids == image_tid) | (prompt_ids == video_tid)
        prompt_txt_mask = ~prompt_vis_mask
        n_text = int(prompt_txt_mask.sum().item())
        n_vis  = int(prompt_vis_mask.sum().item())

        # inputs_embeds catcher
        catcher = InputEmbedCatcher(prompt_len=prompt_len)
        h_catch = lm.register_forward_pre_hook(catcher.hook, with_kwargs=True)

        # attention stats collector
        collector = AttnStatsCollector(
            prompt_len=prompt_len,
            gen_start=prompt_len,
            prompt_text_mask=prompt_txt_mask,
            prompt_vis_mask=prompt_vis_mask,
            layerid_to_group=layerid_to_group,
        )

        attn_handles = []
        try:
            with TempForceAttnOnLayers(layers, keep):
                for lid in keep:
                    attn_handles.append(layers[lid].register_forward_hook(collector.hook(lid)))

                # 중요: global output_attentions=False
                _ = model(
                    **fwd_inputs,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
        finally:
            for hh in attn_handles:
                try:
                    hh.remove()
                except Exception:
                    pass
            try:
                h_catch.remove()
            except Exception:
                pass

        # finalize MDI/AEI (collector 결과만 사용)
        m_early  = finalize_mdi_aei(collector.at_list["early"],  collector.ao_list["early"],  n_text, n_vis)
        m_middle = finalize_mdi_aei(collector.at_list["middle"], collector.ao_list["middle"], n_text, n_vis)
        m_late   = finalize_mdi_aei(collector.at_list["late"],   collector.ao_list["late"],   n_text, n_vis)

        mdi_early, aei_t_early, aei_o_early = m_early["MDI"],  m_early["AEI_T"],  m_early["AEI_O"]
        mdi_mid,   aei_t_mid,   aei_o_mid   = m_middle["MDI"], m_middle["AEI_T"], m_middle["AEI_O"]
        mdi_late,  aei_t_late,  aei_o_late  = m_late["MDI"],   m_late["AEI_T"],   m_late["AEI_O"]

        # effective rank + CKA on LM-entry embeds (prompt only)
        eff_txt = eff_vis = cka_tv = float("nan")
        if catcher.inputs_embeds_prompt is not None:
            emb = catcher.inputs_embeds_prompt[0]  # [prompt_len, D] on CPU

            txt_mask_cpu = prompt_txt_mask.detach().cpu()
            vis_mask_cpu = prompt_vis_mask.detach().cpu()

            Xtxt = emb[txt_mask_cpu]
            Xvis = emb[vis_mask_cpu]

            eff_txt = effective_rank(Xtxt)
            eff_vis = effective_rank(Xvis)

            nt = Xtxt.shape[0]
            nv = Xvis.shape[0]
            n = min(nt, nv, args.cka_tokens)
            if n >= 2:
                g = torch.Generator().manual_seed(args.seed + i)
                tidx = torch.randperm(nt, generator=g)[:n]
                vidx = torch.randperm(nv, generator=g)[:n]
                cka_tv = linear_cka(Xtxt[tidx], Xvis[vidx])

        row = {
            "idx": i,
            "video_path": meta["video_path"],
            "video_abs_path": meta["video_abs_path"],
            "video_id": meta["video_id"],
            "question": meta["question"],
            "gt_answer": gt,
            "pred_answer": pred,
            "is_correct": int(correct),
            "prompt_len": prompt_len,
            "gen_len": int(gen_ids.numel()),
            "n_text_tokens": n_text,
            "n_video_tokens": n_vis,
            "MDI_early": mdi_early,
            "AEI_T_early": aei_t_early,
            "AEI_O_early": aei_o_early,
            "MDI_middle": mdi_mid,
            "AEI_T_middle": aei_t_mid,
            "AEI_O_middle": aei_o_mid,
            "MDI_late": mdi_late,
            "AEI_T_late": aei_t_late,
            "AEI_O_late": aei_o_late,
            "effrank_text": eff_txt,
            "effrank_video": eff_vis,
            "cka_text_video": cka_tv,
        }
        writer.writerow(row)

        if (not correct) and fj is not None:
            record = dict(row)
            # JSONL에는 사람이 보기 쉽게 원문도 그대로 보관
            record["question_raw"] = meta["question"]
            record["gt_answer_raw"] = gt
            record["pred_answer_raw"] = pred
            fj.write(json.dumps(record, ensure_ascii=False) + "\n")

    fcsv.close()
    if fj is not None:
        fj.close()

    print("Saved CSV:", str(csv_path))
    if args.save_mismatch_jsonl:
        print("Saved mismatch JSONL:", str(mismatch_path))


if __name__ == "__main__":
    main()
