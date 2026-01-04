# train_qwen25vl_mmbench_video_lora.py
import os
import json
import math
import random
import argparse
import inspect
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, DownloadConfig
from transformers import AutoProcessor

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False

try:
    # Transformers model class
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


# -------------------------
# LoRA targets
# -------------------------
def find_projector_linear_module_names(model) -> List[str]:
    """
    Qwen2.5-VL 쪽 projector/merger류 Linear 모듈을 최대한 넓게 탐색.
    """
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            n = name.lower()
            if any(k in n for k in [
                "mm_projector", "multi_modal_projector",
                "vision_proj", "vision_projector",
                "visual", "merger", "connector", "projector"
            ]):
                names.append(name)
    # 중복 제거(순서 유지)
    seen = set()
    out = []
    for x in names:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_lora_targets(model, placement: str) -> List[str]:
    placement = placement.lower()
    attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp = ["gate_proj", "up_proj", "down_proj"]
    proj_fullnames = find_projector_linear_module_names(model)

    if placement == "llm_attn":
        return attn
    if placement == "llm_mlp":
        return mlp
    if placement == "projector":
        if len(proj_fullnames) == 0:
            raise RuntimeError("Could not find projector/visual Linear modules. Inspect model.named_modules().")
        return proj_fullnames
    if placement == "all":
        if len(proj_fullnames) == 0:
            raise RuntimeError("Could not find projector/visual Linear modules. Inspect model.named_modules().")
        return attn + mlp
    raise ValueError(f"Unknown placement: {placement}. Use llm_attn, llm_mlp, projector, all")


# =========================================================
# Rank-Enhancing Token Fuser (RTF)
#   - Qwen2.5-VL에선 language_model pre-hook로 inputs_embeds를 수정
# =========================================================
class RankEnhancingTokenFuser(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        top_q: int = 64,
        exchange_ratio: float = 0.10,
        token_subsample: int = 256,
        symmetric: bool = True,
        init_alpha: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.top_q = top_q
        self.exchange_ratio = exchange_ratio
        self.token_subsample = token_subsample
        self.symmetric = symmetric

        init_logit = math.log(init_alpha / (1.0 - init_alpha + 1e-12) + 1e-12)
        self.alpha_img_logits = nn.Parameter(torch.full((hidden_size,), float(init_logit)))
        self.alpha_txt_logits = nn.Parameter(torch.full((hidden_size,), float(init_logit)))

    @torch.no_grad()
    def _channel_importance(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        if X is None or X.numel() == 0:
            return None
        if not torch.isfinite(X).all():
            return None

        T, D = X.shape
        if T < 2 or D < 2:
            return None

        if self.token_subsample > 0 and T > self.token_subsample:
            X = X[: self.token_subsample]

        X = X.float()
        X = X - X.mean(dim=0, keepdim=True)

        def var_importance(Z):
            I = (Z * Z).mean(dim=0)
            return torch.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)

        q = min(self.top_q, X.shape[0] - 1, X.shape[1])
        if q < 1:
            return var_importance(X)

        try:
            _, S, V = torch.svd_lowrank(X, q=q, niter=2)
            I = (V.pow(2) @ S.pow(2))
            I = torch.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)
            if not torch.isfinite(I).all():
                return var_importance(X)
            return I
        except Exception:
            return var_importance(X)

    def forward(
        self,
        text_embeds: torch.Tensor,   # [B, Lt, D]
        image_embeds: torch.Tensor,  # [B, Li, D] (여기선 video tokens도 포함해서 "vision tokens"로 사용)
        text_valid_mask: Optional[torch.Tensor] = None,  # [B, Lt] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if (not torch.isfinite(text_embeds).all()) or (not torch.isfinite(image_embeds).all()):
            return text_embeds, image_embeds

        dev = text_embeds.device
        orig_dtype = text_embeds.dtype
        B, Lt, D = text_embeds.shape
        _, Li, _ = image_embeds.shape

        if text_valid_mask is None:
            text_valid_mask = torch.ones((B, Lt), device=dev, dtype=torch.bool)
        else:
            text_valid_mask = text_valid_mask.to(dev)

        text_f = text_embeds.float()
        img_f  = image_embeds.float()

        m = text_valid_mask.unsqueeze(-1)
        denom = m.sum(dim=1, keepdim=True).clamp(min=1).float()
        text_pool = (text_f * m.float()).sum(dim=1, keepdim=True) / denom
        img_pool  = img_f.mean(dim=1, keepdim=True)

        text_out = text_f.clone()
        img_out  = img_f.clone()

        k_low = max(1, int(round(self.exchange_ratio * D)))

        alpha_img = torch.sigmoid(self.alpha_img_logits.to(dev).float()).view(1, 1, D)
        alpha_txt = torch.sigmoid(self.alpha_txt_logits.to(dev).float()).view(1, 1, D)

        eps = 1e-6

        for b in range(B):
            # vision side
            with torch.no_grad():
                I_img = self._channel_importance(img_out[b])
                sel_img = None if (I_img is None or (not torch.isfinite(I_img).all())) else torch.topk(I_img, k=k_low, largest=False).indices
            if sel_img is not None and sel_img.numel() > 0:
                mask = torch.zeros((D,), device=dev, dtype=torch.float32)
                mask[sel_img] = 1.0
                mask = mask.view(1, 1, D)

                img_b = img_out[b:b+1]
                src   = text_pool[b:b+1].expand(-1, Li, -1)

                dst_sel = img_b[:, :, sel_img]
                src_sel = src[:, :, sel_img]
                rms_dst = torch.sqrt((dst_sel**2).mean() + eps)
                rms_src = torch.sqrt((src_sel**2).mean() + eps)
                scale = (rms_dst / (rms_src + eps)).clamp(0.1, 10.0)
                src = src * scale

                blended = (alpha_img * src + (1.0 - alpha_img) * img_b)
                img_out[b:b+1] = img_b * (1.0 - mask) + blended * mask

            # text side
            if self.symmetric:
                with torch.no_grad():
                    valid = text_valid_mask[b]
                    Xtxt = text_out[b][valid]
                    I_txt = self._channel_importance(Xtxt) if Xtxt.numel() > 0 else None
                    sel_txt = None if (I_txt is None or (not torch.isfinite(I_txt).all())) else torch.topk(I_txt, k=k_low, largest=False).indices
                if sel_txt is not None and sel_txt.numel() > 0:
                    mask = torch.zeros((D,), device=dev, dtype=torch.float32)
                    mask[sel_txt] = 1.0
                    mask = mask.view(1, 1, D)

                    txt_b = text_out[b:b+1]
                    src   = img_pool[b:b+1].expand(-1, Lt, -1)

                    dst_sel = txt_b[:, :, sel_txt]
                    src_sel = src[:, :, sel_txt]
                    rms_dst = torch.sqrt((dst_sel**2).mean() + eps)
                    rms_src = torch.sqrt((src_sel**2).mean() + eps)
                    scale = (rms_dst / (rms_src + eps)).clamp(0.1, 10.0)
                    src = src * scale

                    blended = (alpha_txt * src + (1.0 - alpha_txt) * txt_b)
                    text_out[b:b+1] = txt_b * (1.0 - mask) + blended * mask

        if (not torch.isfinite(text_out).all()) or (not torch.isfinite(img_out).all()):
            return text_embeds, image_embeds

        return text_out.to(orig_dtype), img_out.to(orig_dtype)


def _get_qwen_vision_token_ids(model, tokenizer) -> Tuple[int, int]:
    """
    Qwen2.5-VL config에 image_token_id / video_token_id가 명시됨. (문서에도 나옴) :contentReference[oaicite:1]{index=1}
    """
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


def inject_rtf_into_qwen(
    model: nn.Module,
    tokenizer,
    exchange_ratio: float,
    top_q: int,
    token_subsample: int,
    symmetric: bool,
):
    # hidden size
    hidden = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden is None and hasattr(model, "model") and hasattr(model.model, "config"):
        hidden = getattr(model.model.config, "hidden_size", None)
    if hidden is None:
        raise RuntimeError("[RTF] Could not infer hidden_size from model config.")

    image_tid, video_tid = _get_qwen_vision_token_ids(model, tokenizer)

    model.rtf = RankEnhancingTokenFuser(
        hidden_size=int(hidden),
        top_q=top_q,
        exchange_ratio=exchange_ratio,
        token_subsample=token_subsample,
        symmetric=symmetric,
        init_alpha=0.5,
    )

    # hook target: language_model가 있으면 거기에, 없으면 model.model에
    lm = getattr(model, "language_model", None)
    if lm is None and hasattr(model, "model"):
        lm = getattr(model.model, "language_model", None)
    if lm is None:
        lm = getattr(model, "model", None)
    if lm is None:
        print("[RTF] Could not find language model module. RTF disabled.")
        return

    if getattr(model, "_rtf_hook_handle", None) is not None:
        # already injected
        return

    def pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        attention_mask = kwargs.get("attention_mask", None)

        if input_ids is None or inputs_embeds is None:
            return

        if (not torch.isfinite(inputs_embeds).all()):
            return

        dev = inputs_embeds.device
        if getattr(model, "_rtf_device", None) != dev:
            model.rtf.to(dev)
            model._rtf_device = dev

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=dev, dtype=torch.long)

        # vision positions: <|image_pad|> or <|video_pad|>
        vision_mask = (input_ids == image_tid) | (input_ids == video_tid)
        # text valid: not vision & not pad
        text_mask = (~vision_mask) & attention_mask.bool()

        B, L, D = inputs_embeds.shape
        new_embeds = inputs_embeds

        # batch-wise apply
        for b in range(B):
            vid_idx = torch.nonzero(vision_mask[b], as_tuple=False).squeeze(-1)
            txt_idx = torch.nonzero(text_mask[b], as_tuple=False).squeeze(-1)
            if vid_idx.numel() < 2 or txt_idx.numel() < 2:
                continue

            txt = new_embeds[b, txt_idx, :].unsqueeze(0)  # [1,Lt,D]
            vis = new_embeds[b, vid_idx, :].unsqueeze(0)  # [1,Lv,D]
            txt_valid = torch.ones((1, txt.shape[1]), device=dev, dtype=torch.bool)

            txt_new, vis_new = model.rtf(txt, vis, text_valid_mask=txt_valid)
            if torch.isfinite(txt_new).all() and torch.isfinite(vis_new).all():
                new_embeds[b, txt_idx, :] = txt_new[0]
                new_embeds[b, vid_idx, :] = vis_new[0]

        kwargs["inputs_embeds"] = new_embeds

    model._rtf_hook_handle = lm.register_forward_pre_hook(pre_hook, with_kwargs=True)
    print(f"[RTF] Injected via pre-hook on {lm.__class__.__name__} (exchange_ratio={exchange_ratio}, top_q={top_q}, symmetric={symmetric})")


# -------------------------
# Download / extract videos.zip
# -------------------------
def ensure_mmbench_videos(
    repo_id: str,
    videos_dir: Path,
    hf_cache_dir: Path,
    filename: str = "videos.zip",
):
    """
    lscpku/MMBench-Video의 Files 탭에 videos.zip가 따로 존재(약 13.3GB). :contentReference[oaicite:2]{index=2}
    """
    if not HAS_HF_HUB:
        raise RuntimeError("huggingface_hub not installed. pip install huggingface_hub (and optionally hf-xet).")

    videos_dir.mkdir(parents=True, exist_ok=True)
    marker = videos_dir / ".extracted_ok"

    # 이미 video 폴더가 있거나 marker가 있으면 패스
    if marker.exists():
        return

    # 다운로드
    zip_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        cache_dir=str(hf_cache_dir),
    )
    zip_path = Path(zip_path)

    # 압축 해제
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(videos_dir))

    # 간단 검증: video/ 폴더가 생겼는지
    # (데이터셋의 video_path가 "./video/xxx.mp4" 형태) :contentReference[oaicite:3]{index=3}
    if not (videos_dir / "video").exists():
        # 그래도 marker는 찍되, 사용자가 경로를 확인할 수 있게 안내
        print(f"[WARN] Extracted but {videos_dir/'video'} not found. Check extracted structure under: {videos_dir}")

    marker.write_text("ok\n")

class MMBenchVideoCachedDataset(Dataset):
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.files = sorted([p for p in cache_dir.glob("*.pt")])
        if len(self.files) == 0:
            raise RuntimeError(f"No cache files found in {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return torch.load(self.files[idx], map_location="cpu")


# -------------------------
# Dataset (MMBench-Video)
# -------------------------
class MMBenchVideoLoRADataset(Dataset):
    def __init__(
        self,
        hf_ds,
        processor,
        tokenizer,
        videos_root: Path,
        max_length: int = 1024,
        video_fps: int = 1,
        num_frames: int = 8, video_backend: str = ""
    ):
        self.ds = hf_ds
        self.processor = processor
        self.tokenizer = tokenizer
        self.videos_root = videos_root
        self.max_length = max_length
        self.video_fps = video_fps
        self.num_frames = num_frames
        self.video_backend = video_backend

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.ds[idx]
        q = ex.get("question", "")
        a = ex.get("answer", "")
        vp = ex.get("video_path", "")
        q = q if isinstance(q, str) else str(q)
        a = a if isinstance(a, str) else str(a)

        if not isinstance(q, str) or not isinstance(a, str) or not isinstance(vp, str):
            raise RuntimeError("Bad example schema.")

        # dataset viewer 상 video_path는 "./video/xxx.mp4" 형태 :contentReference[oaicite:4]{index=4}
        rel = vp.lstrip("./")
        video_path = (self.videos_root / rel).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}\nDid you download/extract videos.zip?")

        # Qwen2.5-VL video usage: {"type":"video","path":...} + processor.apply_chat_template(fps=...) :contentReference[oaicite:5]{index=5}
        user_conv = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(video_path)},
                    {"type": "text", "text": f"Question: {q}\nAnswer:"},
                ],
            }
        ]
        #full_conv = user_conv + [{"role": "assistant", "content": a}]
        full_conv = user_conv + [{"role": "assistant", "content": [{"type": "text", "text": a}]}]


        # prompt length (text-only tokenize; video decode는 1번만 하고 싶어서)
        prompt_text = self.processor.apply_chat_template(
            user_conv,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        prompt_len = int(prompt_ids.shape[1])

        videos_kwargs = {
            "backend":"decord",
            "do_sample_frames": True,
            "num_frames": self.num_frames,  # 길이와 무관하게 고정 프레임만
        }
        if self.video_backend:
            videos_kwargs["backend"] = self.video_backend
        enc = self.processor.apply_chat_template(
            full_conv,
            #fps=self.video_fps,
            num_frames=self.num_frames,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            #videos_kwargs=videos_kwargs,
        )

        if "pixel_values_videos" in enc and isinstance(enc["pixel_values_videos"], torch.Tensor):
            enc["pixel_values_videos"] = enc["pixel_values_videos"].to(torch.float16)

        input_ids = enc["input_ids"]               # [1, L]
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        # supervised token 최소 1개 보장
        if int((labels != -100).sum().item()) == 0:
            labels[:, -1] = input_ids[:, -1]

        # return everything (processor output keys may include pixel_values_videos, video_grid_thw, etc.)
        out = dict(enc)
        out["input_ids"] = input_ids
        out["attention_mask"] = attention_mask
        out["labels"] = labels

        # truncate other seq-aligned keys if any
        for k in ["position_ids"]:
            if k in out and isinstance(out[k], torch.Tensor) and out[k].shape[:2] == enc["input_ids"].shape[:2]:
                out[k] = out[k][:, : self.max_length]

        return out


def collate_single(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # 비디오 프레임 수가 샘플마다 달라질 수 있어 안전하게 batch_size=1만 지원
    if len(batch) != 1:
        raise RuntimeError("This script currently supports batch_size=1 for video (variable frame counts). Use grad_accum.")
    return batch[0]

def build_mmbench_video_cache(
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

    # (중요) truncation으로 input_ids만 자르면 token/features mismatch가 나므로,
    # 캐시 생성 시에는 길이가 max_length를 넘으면 "스킵/에러"로 처리하고
    # num_frames/max_pixels를 줄여서 해결하는 게 안전합니다.
    for idx in range(len(ds)):
        ex = ds[idx]
        q = ex.get("question", "")
        a = ex.get("answer", "")
        vp = ex.get("video_path", "")
        q = q if isinstance(q, str) else str(q)
        a = a if isinstance(a, str) else str(a)

        # video path resolve
        if vp.startswith("./"):
            rel = vp[2:]
        elif vp.startswith("/"):
            rel = vp[1:]
        else:
            rel = vp
        video_path = (videos_root / rel).resolve()
        if not video_path.exists():
            print(f"[SKIP] missing video: {video_path}")
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

        # prompt_len (텍스트만)
        prompt_text = processor.apply_chat_template(
            user_conv,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        prompt_len = int(prompt_ids.shape[1])

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

        # fp16로 내려서 캐시 크기 절감 (디코딩 이후 텐서 크기만 줄임)
        if "pixel_values_videos" in enc and isinstance(enc["pixel_values_videos"], torch.Tensor):
            enc["pixel_values_videos"] = enc["pixel_values_videos"].to(torch.float16)

        input_ids = enc["input_ids"]
        if input_ids.shape[1] > max_length:
            # 여기서 자르면 mismatch가 다시 터질 수 있으므로 "스킵" 권장
            print(f"[SKIP] too long seq (L={input_ids.shape[1]}) idx={idx}. "
                  f"Reduce num_frames/max_pixels or increase max_length.")
            continue

        attention_mask = enc.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        if int((labels != -100).sum().item()) == 0:
            labels[:, -1] = input_ids[:, -1]

        # 저장할 payload 구성
        payload = {
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "labels": labels.cpu(),
        }

        # 모델 forward에 필요한 비디오 키들 포함
        for k in ["pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]:
            if k in enc and isinstance(enc[k], torch.Tensor):
                payload[k] = enc[k].cpu()

        torch.save(payload, cache_dir / f"{idx:06d}.pt")

        print(f"[CACHE] saved {idx:06d}.pt")
        for var_name in ['inputs', 'enc', 'video_tensor', 'pixel_values']:
                if var_name in locals():
                    del locals()[var_name]
        
        gc.collect()
        torch.cuda.empty_cache()


# -------------------------
# Training
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")

    ap.add_argument("--dataset_name", type=str, default="lscpku/MMBench-Video")
    ap.add_argument("--train_split", type=str, default="test")  # 이 데이터셋은 test split 2k rows로 보임 :contentReference[oaicite:6]{index=6}
    ap.add_argument("--max_train_samples", type=int, default=400)

    ap.add_argument("--videos_dir", type=str, default="")  # default: base_dir/mmbench_video_assets
    ap.add_argument("--auto_download_videos", action="store_true")

    ap.add_argument("--video_fps", type=int, default=1)

    ap.add_argument("--placement", type=str, required=True,
                    choices=["llm_attn", "llm_mlp", "projector", "all"])

    ap.add_argument("--output_dir", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--device_map", type=str, default="cuda:0")  # model load placement
    ap.add_argument("--device", type=str, default="cuda:0")      # batch .to() device
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=20)

    # RTF
    ap.add_argument("--rtf_exchange_ratio", type=float, default=0.10)
    ap.add_argument("--rtf_top_q", type=int, default=64)
    ap.add_argument("--rtf_token_subsample", type=int, default=256)
    ap.add_argument("--rtf_symmetric", action="store_true")
    ap.add_argument("--rtf_disable", action="store_true")
    ap.add_argument("--cache_dir", type=str, default="")
    ap.add_argument("--build_cache", action="store_true")   # 캐시 생성만 하고 종료
    ap.add_argument("--use_cache", action="store_true")
    ap.add_argument("--num_frames", type=int, default=4)   # 8~16 추천
    ap.add_argument("--video_backend", type=str, default="decord")  # "decord" 등(설치돼 있으면)


    ap.add_argument("--dry_run", action="store_true", help="LoRA target 확인만 하고 종료")
    args = ap.parse_args()

    if args.batch_size != 1:
        raise RuntimeError("For MMbench-video, set --batch_size 1 (use --grad_accum for effective batch size).")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if QwenVLModelClass is None:
        raise RuntimeError("Could not import Qwen2_5_VLForConditionalGeneration. Update transformers.")

    base_dir = Path(args.base_dir).resolve()
    paths = setup_hf_cache(base_dir)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (base_dir / "qwen25vl_mmbench_video/baseline" / args.placement)
    out_dir.mkdir(parents=True, exist_ok=True)

    min_pixels = 128 * 28 * 28
    max_pixels = 128 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    tokenizer = processor.tokenizer

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

    model = QwenVLModelClass.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Optional RTF
    if not args.rtf_disable:
        print("rtf is not disable!")
        inject_rtf_into_qwen(
            model=model,
            tokenizer=tokenizer,
            exchange_ratio=args.rtf_exchange_ratio,
            top_q=args.rtf_top_q,
            token_subsample=args.rtf_token_subsample,
            symmetric=args.rtf_symmetric,
        )

    # LoRA
    target_modules = build_lora_targets(model, args.placement)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"[{args.placement}] trainable params: {trainable} / {total} ({100*trainable/total:.4f}%)")

    if args.dry_run:
        hit = [n for n, _ in model.named_modules() if "lora_" in n]
        print("LoRA modules (first 80):")
        for x in hit[:80]:
            print("  ", x)
        rtf_params = [n for n, p in model.named_parameters() if "rtf." in n and p.requires_grad]
        if len(rtf_params) > 0:
            print("RTF trainable params:")
            for n in rtf_params:
                print("  ", n)
        print("Dry run done.")
        return

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
        split=f"{args.train_split}[:{args.max_train_samples}]",
        cache_dir=str(paths["HF_DATASETS_CACHE"]),
        download_config=dl_cfg,
    )

    # 필요한 필드 확인
    def _ok(ex):
        return (
            isinstance(ex.get("question", None), str)
            and isinstance(ex.get("answer", None), str)
            and isinstance(ex.get("video_path", None), str)
        )
    # ds = ds.filter(_ok)
    # print("Train rows:", len(ds))
    ds = ds.filter(lambda ex: isinstance(ex.get("question", None), str)
                        and isinstance(ex.get("answer", None), str)
                        and isinstance(ex.get("video_path", None), str))
    print("Rows:", len(ds))
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (base_dir / "mmbench_video_cache")
    # (1) 캐시 생성만 하고 종료
    if args.build_cache:
        build_mmbench_video_cache(
            ds=ds,
            processor=processor,
            tokenizer=tokenizer,
            videos_root=videos_root,
            cache_dir=cache_dir,
            max_length=args.max_length,
            num_frames=args.num_frames,
            video_backend=args.video_backend,
        )
        print("Cache build done:", cache_dir)
        return

    # (2) 학습에서 캐시 사용
    if args.use_cache:
        train_set = MMBenchVideoCachedDataset(cache_dir)
    else:
        # 기존 online 방식(권장 X)
        train_set = MMBenchVideoLoRADataset(
            ds,
            processor=processor,
            tokenizer=tokenizer,
            videos_root=videos_root,
            max_length=args.max_length,
            video_fps=args.video_fps,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_single,
        num_workers=2,      # 캐시 사용 시에는 workers 올려도 비교적 안전
        pin_memory=True,
    )

    # Optim
    if args.load_in_4bit:
        from bitsandbytes.optim import PagedAdamW8bit
        optim = PagedAdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return max(0.0, float(args.max_steps - step) / float(max(1, args.max_steps - args.warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    model.train()
    step = 0
    optim.zero_grad(set_to_none=True)
    running_loss = 0.0


    while step < args.max_steps:
        for batch in train_loader:
            batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(**batch)
            loss_task = out.loss
            if not torch.isfinite(loss_task):
                raise RuntimeError("NaN/Inf loss detected")

            loss = loss_task / args.grad_accum
            loss.backward()
            running_loss += float(loss_task.item())

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            if (step + 1) % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"step {step+1}/{args.max_steps} | loss={running_loss/args.log_every:.4f} | lr={lr:.2e}")
                running_loss = 0.0

            if (step + 1) % args.save_every == 0:
                ckpt = out_dir / f"checkpoint_step_{step+1}"
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt)
                processor.save_pretrained(ckpt)
                with open(ckpt / "train_args.json", "w") as f:
                    json.dump(vars(args), f, indent=2)
                print("Saved:", str(ckpt))

            step += 1
            if step >= args.max_steps:
                break

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    with open(final_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Training done. Final saved:", str(final_dir))


if __name__ == "__main__":
    main()
