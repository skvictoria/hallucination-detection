# eval_qwen25vl_mmbench_video_lora_with_stats.py
import os
import json
import math
import random
import argparse
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import string
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)

    os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

    return {
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": hf_hub,
        "HF_DATASETS_CACHE": hf_datasets,
        "HF_TRANSFORMERS_CACHE": hf_transformers,
    }

def downsample_tokens_meanpool(x_2d: torch.Tensor, target_n: int) -> torch.Tensor:
    """
    x_2d: [N, D] -> [target_n, D] by mean-pooling over contiguous chunks.
    """
    assert x_2d.dim() == 2
    N, D = x_2d.shape
    target_n = int(target_n)
    if target_n <= 0:
        return x_2d[:0]
    if N <= target_n:
        return x_2d

    # chunk boundaries (approximately equal-sized)
    # indices in [0, N]
    bounds = torch.linspace(0, N, steps=target_n + 1, device=x_2d.device)
    bounds = bounds.floor().long().clamp(0, N)

    out = []
    for i in range(target_n):
        s = int(bounds[i].item())
        e = int(bounds[i + 1].item())
        if e <= s:
            # if empty chunk occurs due to flooring, take one element safely
            s = min(s, N - 1)
            e = min(s + 1, N)
        out.append(x_2d[s:e].mean(dim=0, keepdim=True))  # [1, D]
    return torch.cat(out, dim=0)  # [target_n, D]



# -------------------------
# videos.zip download/extract
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

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(videos_dir))

    if not (videos_dir / "video").exists():
        print(f"[WARN] Extracted but {videos_dir/'video'} not found. Check extracted structure under: {videos_dir}")

    marker.write_text("ok\n")


# -------------------------
# prompt_len inference (marker-based)
# -------------------------
def _find_last_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> Optional[int]:
    L = int(haystack.numel())
    M = int(needle.numel())
    if M <= 0 or M > L:
        return None
    for s in range(L - M, -1, -1):
        if torch.equal(haystack[s:s+M], needle):
            return int(s)
    return None

def infer_prompt_len_from_marker(input_ids_1d: torch.Tensor, tokenizer) -> int:
    for mt in ["<|im_start|>assistant\n", "<|im_start|>assistant"]:
        m = tokenizer(mt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(input_ids_1d.device)
        pos = _find_last_subsequence(input_ids_1d, m)
        if pos is not None:
            return int(pos + m.numel())
    return -1


# -------------------------
# slice seq-aligned tensors
# -------------------------
def _slice_seq_aligned_tensors(batch: Dict[str, torch.Tensor], new_L: int) -> Dict[str, torch.Tensor]:
    assert "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor)
    old_L = int(batch["input_ids"].shape[-1])

    keep_keys = {"pixel_values_videos", "video_grid_thw", "second_per_grid_ts"}
    out = {}

    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k in keep_keys:
            out[k] = v
            continue

        # [B, L]
        if v.dim() == 2 and v.shape[-1] == old_L:
            out[k] = v[..., :new_L]
            continue

        # [B, ?, L]
        if v.dim() == 3 and v.shape[-1] == old_L:
            out[k] = v[..., :new_L]
            continue

        # [3, L]
        if v.dim() == 2 and v.shape[0] == 3 and v.shape[1] == old_L:
            out[k] = v[:, :new_L]
            continue

        # [L]
        if v.dim() == 1 and v.shape[0] == old_L:
            out[k] = v[:new_L]
            continue

        out[k] = v

    return out


# -------------------------
# cache dataset
# -------------------------
class MMBenchVideoCachedDataset(Dataset):
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.files = sorted([p for p in cache_dir.glob("*.pt")])
        if len(self.files) == 0:
            raise RuntimeError(f"No cache files found in {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        payload = torch.load(self.files[idx], map_location="cpu")
        payload["_cache_path"] = str(self.files[idx])
        return payload


def collate_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise RuntimeError("This script supports batch_size=1 (video). Use iteration / grad_accum-like logic if needed.")
    return batch[0]


# -------------------------
# build eval cache
# -------------------------
def build_mmbench_video_eval_cache(
    ds,
    processor,
    tokenizer,
    videos_root: Path,
    cache_dir: Path,
    max_length: int,
    num_frames: int,
    video_backend: str,
):
    cache_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(len(ds)):
        ex = ds[idx]
        q = ex.get("question", "")
        a = ex.get("answer", "")
        vp = ex.get("video_path", "")
        q = q if isinstance(q, str) else str(q)
        a = a if isinstance(a, str) else str(a)

        if not isinstance(vp, str):
            print(f"[SKIP] bad video_path idx={idx}")
            continue

        # resolve path
        if vp.startswith("./"):
            rel = vp[2:]
        elif vp.startswith("/"):
            rel = vp[1:]
        else:
            rel = vp
        video_path = (videos_root / rel).resolve()
        if not video_path.exists():
            print(f"[SKIP] missing video idx={idx}: {video_path}")
            continue

        user_conv = [{
            "role": "user",
            "content": [
                {"type": "video", "path": str(video_path)},
                {"type": "text", "text": f"Question: {q}\nAnswer:"},
            ],
        }]
        full_conv = user_conv + [{
            "role": "assistant",
            "content": [{"type": "text", "text": a}],
        }]

        enc = processor.apply_chat_template(
            full_conv,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            videos_kwargs={
                "backend": video_backend,
                "do_sample_frames": True,
                "num_frames": num_frames,
            },
        )

        if "pixel_values_videos" in enc and isinstance(enc["pixel_values_videos"], torch.Tensor):
            enc["pixel_values_videos"] = enc["pixel_values_videos"].to(torch.float16)

        input_ids = enc["input_ids"]  # [1, L]
        L = int(input_ids.shape[1])
        if L > max_length:
            print(f"[SKIP] too long seq (L={L}) idx={idx}. Reduce frames/pixels or increase max_length.")
            continue

        prompt_len = infer_prompt_len_from_marker(input_ids[0], tokenizer)
        if prompt_len < 0:
            print(f"[SKIP] cannot find assistant marker idx={idx}")
            continue

        attention_mask = enc.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        if int((labels != -100).sum().item()) == 0:
            labels[:, -1] = input_ids[:, -1]

        payload: Dict[str, Any] = {
            "idx": int(idx),
            "question": q,
            "answer": a,
            "video_path": vp,
            "prompt_len": int(prompt_len),

            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "labels": labels.cpu(),
        }

        for k in ["pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]:
            if k in enc and isinstance(enc[k], torch.Tensor):
                payload[k] = enc[k].cpu()

        torch.save(payload, cache_dir / f"{idx:06d}.pt")
        if (idx + 1) % 50 == 0:
            print(f"[CACHE] saved up to idx={idx}")

        del enc
        gc.collect()
        torch.cuda.empty_cache()

    print("Eval cache build done:", str(cache_dir))


# -------------------------
# metrics: EM / F1 (SQuAD-style)
# -------------------------
_ARTICLES = {"a", "an", "the"}

def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)

def exact_match(pred: str, gt: str) -> float:
    return float(_normalize_text(pred) == _normalize_text(gt))

def f1_score(pred: str, gt: str) -> float:
    pred_toks = _normalize_text(pred).split()
    gt_toks = _normalize_text(gt).split()
    if len(pred_toks) == 0 and len(gt_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gt_toks) == 0:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gt_toks:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / max(1, len(pred_toks))
    recall = num_same / max(1, len(gt_toks))
    return (2 * precision * recall) / max(1e-12, (precision + recall))


# -------------------------
# decode GT from full input_ids using prompt_len (+ optional <|im_end|>)
# -------------------------
def decode_gt_from_full_ids(input_ids_1d: torch.Tensor, prompt_len: int, tokenizer) -> str:
    ans_ids = input_ids_1d[prompt_len:]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        end_pos = (ans_ids == im_end_id).nonzero(as_tuple=False)
        if end_pos.numel() > 0:
            ans_ids = ans_ids[: int(end_pos[0].item())]
    return tokenizer.decode(ans_ids, skip_special_tokens=True).strip()


# =====================================================================
# NEW: modality masks + effective rank + MDI/AEI
# =====================================================================

def _guess_video_placeholder_id(tokenizer, input_ids_1d: torch.Tensor) -> Optional[int]:
    """
    Qwen2.5-VL 계열에서 비디오 placeholder 토큰 id를 유추.
    환경/버전에 따라 token string이 다를 수 있어 후보를 여러 개 두고,
    실제 input_ids에서 가장 많이 등장하는 후보를 선택.
    """
    candidates = [
        "<|video_pad|>",
        "<|video_placeholder|>",
        "<|video|>",
        "<video>",
        "<|vision_pad|>",
        "<|image_pad|>",  # 혹시 video가 image_pad로 들어오는 케이스 방어
    ]
    best = None
    best_cnt = 0
    for tok in candidates:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            cnt = int((input_ids_1d == tid).sum().item())
            if cnt > best_cnt:
                best_cnt = cnt
                best = tid
    if best is not None and best_cnt > 0:
        return best

    # fallback: additional_special_tokens 중 'video' 포함 토큰 탐색
    try:
        for tok in getattr(tokenizer, "additional_special_tokens", []) or []:
            if "video" in tok.lower():
                tid = tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    cnt = int((input_ids_1d == tid).sum().item())
                    if cnt > 0:
                        return tid
    except Exception:
        pass

    return None


def build_modality_masks(prompt_input_ids_1d: torch.Tensor, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    prompt 구간(길이 = prompt_len)에서
    - video_mask: video placeholder 위치
    - text_mask: 나머지(간단히 ~video_mask)
    반환.
    """
    vid_id = _guess_video_placeholder_id(tokenizer, prompt_input_ids_1d)
    if vid_id is None:
        video_mask = torch.zeros_like(prompt_input_ids_1d, dtype=torch.bool)
        text_mask = torch.ones_like(prompt_input_ids_1d, dtype=torch.bool)
        return video_mask, text_mask, -1

    video_mask = (prompt_input_ids_1d == int(vid_id))
    text_mask = ~video_mask
    return video_mask, text_mask, int(vid_id)


def effective_rank(x_2d: torch.Tensor, eps: float = 1e-12) -> float:
    """
    x: [N, D]
    effective rank = exp( -sum_i p_i log p_i ), p_i = (s_i^2) / sum_j (s_j^2)
    """
    if x_2d.numel() == 0:
        return 0.0
    x = x_2d.float()
    x = x - x.mean(dim=0, keepdim=True)
    # SVD vals
    s = torch.linalg.svdvals(x)  # [min(N,D)]
    lam = (s ** 2)
    denom = lam.sum().clamp_min(eps)
    p = (lam / denom).clamp_min(eps)
    h = -(p * p.log()).sum()
    r = torch.exp(h)
    return float(r.item())


def sample_rows(x_2d: torch.Tensor, max_rows: int, seed: int = 0) -> torch.Tensor:
    n = int(x_2d.shape[0])
    if max_rows <= 0 or n <= max_rows:
        return x_2d
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:max_rows]
    return x_2d[idx.to(x_2d.device)]


def get_lm_layers(model) -> Optional[nn.ModuleList]:
    """
    Qwen2.5-VL / PEFT wrapper에서 transformer block list를 최대한 robust하게 찾기.
    보통:
      - base_model.model.layers
      - model.model.layers
      - model.base_model.model.model.layers (PEFT)
    """
    # unwrap peft
    m = model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        # PeftModel
        m0 = m.base_model.model
    else:
        m0 = m

    # try common paths
    for cand in [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("transformer", "layers"),
        ("model", "decoder", "layers"),
    ]:
        obj = m0
        ok = True
        for attr in cand:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, (list, nn.ModuleList)):
            return obj

    return None


class BlockOutputCollector:
    """
    특정 transformer block들의 output hidden을 hook으로 받아 저장.
    (output_hidden_states=True 없이도 last/중간 hidden을 얻기 위함)
    """
    def __init__(self, layers: nn.ModuleList, pick_indices: List[int]):
        self.layers = layers
        self.pick = sorted(list(set(int(i) for i in pick_indices)))
        self.handles = []
        self.outputs: Dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inp, out):
            # out이 tuple이면 첫 원소가 hidden인 경우가 많음
            h = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(h, torch.Tensor):
                self.outputs[layer_idx] = h
        return hook

    def register(self):
        for i in self.pick:
            if 0 <= i < len(self.layers):
                self.handles.append(self.layers[i].register_forward_hook(self._make_hook(i)))

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []


def layer_groups(num_layers: int) -> Dict[str, List[int]]:
    if num_layers <= 0:
        return {"early": [], "middle": [], "late": [], "all": []}
    early = [0, 1] if num_layers >= 2 else [0]
    mid0 = max(0, num_layers // 2 - 1)
    middle = [mid0, min(num_layers - 1, mid0 + 1)] if num_layers >= 2 else [0]
    late = [max(0, num_layers - 2), num_layers - 1] if num_layers >= 2 else [0]
    all_layers = list(range(num_layers))
    return {"early": early, "middle": middle, "late": late, "all": all_layers}


@torch.no_grad()
def compute_stats_for_example(
    model,
    tokenizer,
    prompt_inputs: Dict[str, torch.Tensor],   # prompt only (sliced to prompt_len)
    out_ids: torch.Tensor,                    # [1, prompt_len + gen_len]
    prompt_len: int,
    rank_max_tokens: int,
    stats_max_steps: int,
    stats_stride: int,
    amp_dtype=torch.float16,
) -> Dict[str, Any]:
    """
    - effective rank (text/video) from selected block outputs (early/middle/late/last)
    - MDI/AEI from step-wise attentions over generated tokens (teacher forcing)
    """
    device = out_ids.device
    prompt_ids_1d = prompt_inputs["input_ids"][0]  # [prompt_len]
    vmask, tmask, vid_id = build_modality_masks(prompt_ids_1d, tokenizer)
    vmask = vmask.to(device)
    tmask = tmask.to(device)
    n_video = int(vmask.sum().item())
    n_text = int(tmask.sum().item())
    n_total = n_video + n_text

    # identify LM layers
    layers = get_lm_layers(model)
    num_layers = len(layers) if layers is not None else 0
    groups = layer_groups(num_layers)

    # -------- effective rank via hooks (no output_hidden_states=True) --------
    rank_res: Dict[str, Any] = {}
    if layers is not None and num_layers > 0:
        # 대표 레이어 인덱스: early(1), middle(mid last), late(last), last(last)
        pick = [
            groups["early"][-1],
            groups["middle"][-1],
            groups["late"][-1],
        ]
        pick = sorted(list(set(pick)))
        collector = BlockOutputCollector(layers, pick)
        collector.register()

        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=amp_dtype):
                # prefill forward: cache 생성 + hook으로 hidden 획득
                #pre = model(**prompt_inputs, use_cache=True, output_attentions=False, return_dict=True)
                prefill_inputs = dict(prompt_inputs)
                prefill_inputs.setdefault("use_cache", True)
                pre = model(**prefill_inputs, output_attentions=False, return_dict=True)
        finally:
            collector.remove()

        # collector.outputs[layer_idx] : [1, prompt_len, D]
        for li in pick:
            h = collector.outputs.get(li, None)
            if h is None:
                continue
            h_prompt = h[0, :prompt_len]  # [prompt_len, D]
            h_text = h_prompt[tmask]
            h_video = h_prompt[vmask]
            Nt = int(h_text.shape[0])
            Nv = int(h_video.shape[0])

            # (옵션) 너무 긴 텍스트를 대비해 상한 적용: 둘 다 같은 N으로 맞추기 위해 target_n을 정의
            target_n = Nt
            if rank_max_tokens > 0:
                target_n = min(target_n, int(rank_max_tokens))

            if target_n <= 0:
                er_text = 0.0
                er_video = 0.0
            else:
                # text도 target_n으로 맞춤 (Nt가 target_n보다 크면 subsample)
                h_text2 = sample_rows(h_text, target_n, seed=0) if Nt > target_n else h_text
                # video는 mean-pool로 target_n으로 축소
                h_video2 = downsample_tokens_meanpool(h_video, target_n) if Nv > target_n else h_video

                er_text = effective_rank(h_text2) if h_text2.numel() > 0 else 0.0
                er_video = effective_rank(h_video2) if h_video2.numel() > 0 else 0.0

                print("after downsampling: ", Nt, Nv, er_text, er_video)

            rank_res[f"effrank_text_L{li}"] = er_text
            rank_res[f"effrank_video_to_text_L{li}"] = er_video
            rank_res[f"rank_target_n_L{li}"] = int(target_n)
            rank_res[f"Nt_L{li}"] = int(Nt)
            rank_res[f"Nv_L{li}"] = int(Nv)

        past = getattr(pre, "past_key_values", None)
        attn_mask = prompt_inputs.get("attention_mask", torch.ones_like(prompt_inputs["input_ids"]))
    else:
        past = None
        attn_mask = prompt_inputs.get("attention_mask", torch.ones_like(prompt_inputs["input_ids"]))

    # -------- MDI/AEI via step-wise attentions --------
    # init accumulators
    acc = {}
    for gname in ["early", "middle", "late", "all"]:
        acc[gname] = {"t": 0.0, "v": 0.0, "cnt": 0}

    if past is None or num_layers == 0 or n_total <= 0 or n_video <= 0:
        # video 토큰이 없으면 MDI는 정의가 애매(division by 0)
        # 그래도 AEI/MDI를 스킵하고 rank만 반환하도록 처리
        return {
            "n_text_tokens": n_text,
            "n_video_tokens": n_video,
            "video_token_id": vid_id,
            "num_layers": num_layers,
            **rank_res,
            "mdi_early": None, "mdi_middle": None, "mdi_late": None, "mdi_all": None,
            "aei_text_early": None, "aei_text_middle": None, "aei_text_late": None, "aei_text_all": None,
            "aei_video_early": None, "aei_video_middle": None, "aei_video_late": None, "aei_video_all": None,
            "stats_steps_used": 0,
        }

    # generated token ids (query tokens)
    gen_ids = out_ids[0, prompt_len:]
    if stats_max_steps > 0:
        gen_ids = gen_ids[:stats_max_steps]
    if stats_stride > 1:
        gen_ids = gen_ids[::stats_stride]
    steps_used = int(gen_ids.numel())

    # attention mask grows by 1 each step
    cur_mask = attn_mask
    cur_past = past

    eps = 1e-12
    for si in range(steps_used):
        tok_id = gen_ids[si].view(1, 1)  # [1,1]
        one = torch.ones((1, 1), dtype=cur_mask.dtype, device=device)
        cur_mask = torch.cat([cur_mask, one], dim=1)

        # some models want prepared inputs (position_ids etc.)
        try:
            model_inputs = model.prepare_inputs_for_generation(
                input_ids=tok_id,
                past_key_values=cur_past,
                attention_mask=cur_mask,
                #use_cache=True,
            )
        except Exception:
            model_inputs = {
                "input_ids": tok_id,
                "past_key_values": cur_past,
                "attention_mask": cur_mask,
                #"use_cache": True,
            }
        model_inputs.setdefault("use_cache", True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=amp_dtype):
            out = model(**model_inputs, output_attentions=True, return_dict=True)#, use_cache=True)

        cur_past = getattr(out, "past_key_values", cur_past)
        atts = getattr(out, "attentions", None)
        if atts is None:
            # output_attentions를 지원 못하거나 eager로 안 떨어진 경우
            break

        # atts: tuple(len=num_layers), each [1, H, 1, kv_len]
        for gname, layer_ids in groups.items():
            if len(layer_ids) == 0:
                continue
            for li in layer_ids:
                if li < 0 or li >= len(atts):
                    continue
                a = atts[li]
                if not isinstance(a, torch.Tensor) or a.dim() != 4:
                    continue
                # prompt keys only: [:prompt_len]
                a_prompt = a[..., :prompt_len]  # [1,H,1,prompt_len]
                # modality masses
                t_mass = a_prompt[..., tmask].sum(dim=-1)  # [1,H,1]
                v_mass = a_prompt[..., vmask].sum(dim=-1)  # [1,H,1]
                denom = (t_mass + v_mass).clamp_min(eps)

                # per-step normalized share over (text+video prompt tokens)
                t_share = (t_mass / denom).mean(dim=1).squeeze().item()
                v_share = (v_mass / denom).mean(dim=1).squeeze().item()

                acc[gname]["t"] += float(t_share)
                acc[gname]["v"] += float(v_share)
                acc[gname]["cnt"] += 1

        del out, atts
        gc.collect()
        torch.cuda.empty_cache()

    def finalize_group(gname: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        cnt = acc[gname]["cnt"]
        if cnt <= 0 or n_total <= 0 or n_video <= 0 or n_text <= 0:
            return None, None, None
        A_text = acc[gname]["t"] / cnt
        A_video = acc[gname]["v"] / cnt
        # normalize to be safe
        s = max(eps, A_text + A_video)
        A_text /= s
        A_video /= s

        mdi = (A_text / n_text) / max(eps, (A_video / n_video))

        token_share_text = n_text / n_total
        token_share_video = n_video / n_total
        aei_text = A_text / max(eps, token_share_text)
        aei_video = A_video / max(eps, token_share_video)
        return float(mdi), float(aei_text), float(aei_video)

    out_stats: Dict[str, Any] = {
        "n_text_tokens": n_text,
        "n_video_tokens": n_video,
        "video_token_id": vid_id,
        "num_layers": num_layers,
        "stats_steps_used": steps_used,
        **rank_res,
    }

    for gname in ["early", "middle", "late", "all"]:
        mdi, aei_t, aei_v = finalize_group(gname)
        out_stats[f"mdi_{gname}"] = mdi
        out_stats[f"aei_text_{gname}"] = aei_t
        out_stats[f"aei_video_{gname}"] = aei_v

    return out_stats


# -------------------------
# evaluation loop
# -------------------------
@torch.no_grad()
def run_eval(
    model,
    processor,
    loader,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    print_every: int,
    out_jsonl: Optional[Path],
    # NEW
    collect_stats: bool,
    rank_max_tokens: int,
    stats_max_steps: int,
    stats_stride: int,
) -> Dict[str, Any]:
    tok = processor.tokenizer
    model.eval()

    old_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = True

    n = 0
    sum_em = 0.0
    sum_f1 = 0.0
    n_fail = 0

    # NEW: stats aggregation
    stat_n = 0
    stat_sums: Dict[str, float] = {}

    jf = None
    if out_jsonl is not None:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jf = out_jsonl.open("w", encoding="utf-8")

    try:
        for it, batch in enumerate(loader):
            bt: Dict[str, Any] = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    bt[k] = v.to(device)
                else:
                    bt[k] = v

            if "input_ids" not in bt:
                n_fail += 1
                continue

            input_ids = bt["input_ids"]  # [1, L]
            ids_1d = input_ids[0]
            L = int(ids_1d.numel())

            prompt_len = int(bt.get("prompt_len", -1))
            if prompt_len <= 0 or prompt_len >= L:
                prompt_len = infer_prompt_len_from_marker(ids_1d, tok)
            if prompt_len <= 0 or prompt_len >= L:
                n_fail += 1
                continue

            gt = decode_gt_from_full_ids(ids_1d, prompt_len, tok)

            # generation inputs: slice seq-aligned tensors to prompt_len
            gen_batch = _slice_seq_aligned_tensors(bt, prompt_len)
            gen_inputs = {k: v for k, v in gen_batch.items() if k != "labels"}

            try:
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                    out_ids = model.generate(
                        **gen_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=bool(do_sample),
                        temperature=float(temperature) if do_sample else None,
                        top_p=float(top_p) if do_sample else None,
                    )
            except Exception as e:
                n_fail += 1
                if (it + 1) % max(1, print_every) == 0:
                    print(f"[EVAL] generate failed at iter={it}: {repr(e)}")
                continue

            if out_ids.dim() == 2 and out_ids.shape[1] > prompt_len:
                gen_part = out_ids[0, prompt_len:]
            else:
                gen_part = out_ids[0] if out_ids.dim() == 2 else out_ids

            pred = tok.decode(gen_part, skip_special_tokens=True).strip()

            em = exact_match(pred, gt)
            f1 = f1_score(pred, gt)

            n += 1
            sum_em += em
            sum_f1 += f1

            # NEW: compute stats
            stats = None
            if collect_stats:
                try:
                    # prompt-only inputs already (gen_inputs) + out_ids from generate
                    stats = compute_stats_for_example(
                        model=model,
                        tokenizer=tok,
                        prompt_inputs=gen_inputs,
                        out_ids=out_ids if out_ids.dim() == 2 else out_ids.unsqueeze(0),
                        prompt_len=prompt_len,
                        rank_max_tokens=rank_max_tokens,
                        stats_max_steps=stats_max_steps,
                        stats_stride=stats_stride,
                        amp_dtype=torch.float16,
                    )
                    # aggregate means for numeric entries
                    for k, v in stats.items():
                        if isinstance(v, (int, float)) and v is not None and math.isfinite(float(v)):
                            stat_sums[k] = stat_sums.get(k, 0.0) + float(v)
                    stat_n += 1
                except Exception as e:
                    if (it + 1) % max(1, print_every) == 0:
                        print(f"[STATS] failed at iter={it}: {repr(e)}")
                    stats = None

            if (print_every > 0) and (n % print_every == 0):
                q = bt.get("question", "")
                print("=" * 80)
                print(f"[EVAL] n={n} EM={sum_em/n:.4f} F1={sum_f1/n:.4f}  (fails={n_fail})")
                if isinstance(q, str) and len(q) > 0:
                    print("- Q --------------------------------")
                    print(q)
                print("- GT -------------------------------")
                print(gt)
                print("- PRED -----------------------------")
                print(pred)
                if stats is not None:
                    print("- STATS (late/all) ------------------")
                    print({
                        "mdi_late": stats.get("mdi_late", None),
                        "aei_text_late": stats.get("aei_text_late", None),
                        "effrank_text_last": stats.get("effrank_text_L{}".format(stats.get("num_layers", 0)-1), None),
                        "effrank_video_last": stats.get("effrank_video_L{}".format(stats.get("num_layers", 0)-1), None),
                        "n_text_tokens": stats.get("n_text_tokens", None),
                        "n_video_tokens": stats.get("n_video_tokens", None),
                        "steps_used": stats.get("stats_steps_used", None),
                    })
                print("=" * 80)

            if jf is not None:
                rec = {
                    "idx": int(bt.get("idx", -1)) if isinstance(bt.get("idx", -1), int) else bt.get("idx", None),
                    "question": bt.get("question", None),
                    "video_path": bt.get("video_path", None),
                    "gt": gt,
                    "pred": pred,
                    "em": em,
                    "f1": f1,
                }
                if stats is not None:
                    rec.update(stats)
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    finally:
        if jf is not None:
            jf.close()
        if old_cache is not None:
            model.config.use_cache = old_cache

    out = {
        "n": n,
        "fails": n_fail,
        "EM": (sum_em / n) if n > 0 else 0.0,
        "F1": (sum_f1 / n) if n > 0 else 0.0,
    }

    # NEW: mean stats
    if collect_stats and stat_n > 0:
        means = {}
        for k, s in stat_sums.items():
            means[f"mean_{k}"] = s / stat_n
        out["stats_n"] = stat_n
        out.update(means)

    return out


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")

    ap.add_argument("--ckpt_dir", type=str, required=True, help="e.g., .../final or checkpoint_step_xxx")

    ap.add_argument("--dataset_name", type=str, default="lscpku/MMBench-Video")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_eval_samples", type=int, default=400)

    ap.add_argument("--videos_dir", type=str, default="")
    ap.add_argument("--auto_download_videos", action="store_true")

    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--device_map", type=str, default="cuda:0")
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--max_length", type=int, default=1400)
    ap.add_argument("--num_frames", type=int, default=4)
    ap.add_argument("--video_backend", type=str, default="decord")

    ap.add_argument("--cache_dir", type=str, default="")
    ap.add_argument("--build_cache", action="store_true")
    ap.add_argument("--use_cache", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--out_jsonl", type=str, default="")

    # NEW: collect stats (effective rank / MDI / AEI)
    ap.add_argument("--collect_stats", action="store_true",
                    help="Compute effective-rank(video/text) and MDI/AEI during evaluation.")
    ap.add_argument("--rank_max_tokens", type=int, default=512,
                    help="Subsample max tokens per modality for effective-rank computation.")
    ap.add_argument("--stats_max_steps", type=int, default=64,
                    help="How many generated tokens to use for MDI/AEI (<= max_new_tokens).")
    ap.add_argument("--stats_stride", type=int, default=1,
                    help="Use every k-th generated token for stats (speed/accuracy tradeoff).")

    args = ap.parse_args()

    if QwenVLModelClass is None:
        raise RuntimeError("Could not import Qwen2_5_VLForConditionalGeneration. Update transformers.")

    base_dir = Path(args.base_dir).resolve()
    paths = setup_hf_cache(base_dir)

    ckpt_dir = Path(args.ckpt_dir).resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

    # processor
    try:
        processor = AutoProcessor.from_pretrained(str(ckpt_dir))
    except Exception:
        min_pixels = 128 * 28 * 28
        max_pixels = 256 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    tokenizer = processor.tokenizer

    # model load
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

    # force eager attention if possible (output_attentions 안정성 ↑)
    try:
        if hasattr(base_model.config, "attn_implementation"):
            base_model.config.attn_implementation = "eager"
    except Exception:
        pass

    try:
        model = PeftModel.from_pretrained(base_model, str(ckpt_dir))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load LoRA adapter from ckpt_dir={ckpt_dir}. "
            f"Make sure ckpt_dir is the folder saved by model.save_pretrained().\n"
            f"Error: {repr(e)}"
        )

    model.eval()
    try:
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
    except Exception:
        pass

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
    print("Rows:", len(ds))

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (base_dir / "mmbench_video_eval_cache_fixed")
    if args.build_cache:
        build_mmbench_video_eval_cache(
            ds=ds,
            processor=processor,
            tokenizer=tokenizer,
            videos_root=videos_root,
            cache_dir=cache_dir,
            max_length=args.max_length,
            num_frames=args.num_frames,
            video_backend=args.video_backend,
        )
        return

    if args.use_cache:
        eval_set = MMBenchVideoCachedDataset(cache_dir)
    else:
        raise RuntimeError("For stability, use --use_cache. (Online eval can be added if needed.)")

    loader = DataLoader(
        eval_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_single,
        num_workers=2,
        pin_memory=True,
    )

    out_jsonl = Path(args.out_jsonl).resolve() if args.out_jsonl else None

    results = run_eval(
        model=model,
        processor=processor,
        loader=loader,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        print_every=args.print_every,
        out_jsonl=out_jsonl,
        collect_stats=args.collect_stats,
        rank_max_tokens=args.rank_max_tokens,
        stats_max_steps=args.stats_max_steps,
        stats_stride=args.stats_stride,
    )

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))

    # save summary
    summary_path = (out_jsonl.parent / "eval_summary.json") if out_jsonl is not None else (cache_dir / "eval_summary.json")
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved summary:", str(summary_path))
    if out_jsonl is not None:
        print("Saved predictions+stats:", str(out_jsonl))


if __name__ == "__main__":
    main()
