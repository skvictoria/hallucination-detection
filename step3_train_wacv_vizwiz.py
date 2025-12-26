import os
import json
import math
import random
import argparse
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

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

# LLaVA import (transformers 버전에 따라 다름)
try:
    from transformers import LlavaForConditionalGeneration
    LlavaModelClass = LlavaForConditionalGeneration
except Exception:
    try:
        from transformers import AutoModelForVision2Seq
        LlavaModelClass = AutoModelForVision2Seq
    except Exception:
        LlavaModelClass = None

def first_nonfinite_trainable_param(model):
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.data is not None and (not torch.isfinite(p.data).all()):
                return n
            if p.grad is not None and (not torch.isfinite(p.grad).all()):
                return n + " (grad)"
    return None

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
# VizWiz prompt / label
# -------------------------
def build_train_prompt(question: str) -> str:
    # open-ended VQA: 짧게 답하도록 유도 + unanswerable 허용
    return (
        "USER: <image>\n"
        f"Question: {question}\n"
        "Answer the question using a short phrase. "
        "If the image does not contain enough information, answer exactly: unanswerable.\n"
        "ASSISTANT:"
    )

def normalize_ans(s: str) -> str:
    s = (s or "").strip()
    s = " ".join(s.split())
    return s

def choose_vizwiz_target(ex: Dict[str, Any]) -> str:
    """
    HuggingFaceM4/VizWiz schema (train/val):
      - answers: list[str] (10 crowdsourced)
      - answerable: int32 (0/1)
    """
    answerable = ex.get("answerable", 1)
    if isinstance(answerable, (int, float)) and int(answerable) == 0:
        return "unanswerable"

    answers = ex.get("answers", None)
    if not isinstance(answers, list) or len(answers) == 0:
        return "unanswerable"

    # majority vote (가장 흔한 답)
    norm = [normalize_ans(a).lower() for a in answers if isinstance(a, str) and a.strip()]
    if len(norm) == 0:
        return "unanswerable"
    most_common = Counter(norm).most_common(1)[0][0]
    return most_common


# -------------------------
# image token id helper
# -------------------------
def get_image_token_id(model, tokenizer) -> int:
    tid = getattr(getattr(model, "config", None), "image_token_index", None)
    if isinstance(tid, int) and tid >= 0:
        return tid

    candidates = ["<image>", "<im_start>", "<image_token>", "<IMG>", "<Image>"]
    for tok in candidates:
        tid2 = tokenizer.convert_tokens_to_ids(tok)
        if tid2 is not None and tid2 != tokenizer.unk_token_id:
            return tid2

    for tok in getattr(tokenizer, "additional_special_tokens", []) or []:
        tid2 = tokenizer.convert_tokens_to_ids(tok)
        if tid2 is not None and tid2 != tokenizer.unk_token_id and "image" in tok.lower():
            return tid2

    raise RuntimeError("Could not find image token id. Check your LLaVA processor/tokenizer.")


# -------------------------
# Safety: pad_token vs image_token 충돌 방지
# -------------------------
def ensure_safe_pad(tokenizer, model):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    img_tid = getattr(model.config, "image_token_index", None)
    if img_tid is None:
        img_tid = tokenizer.convert_tokens_to_ids("<image>")
        if img_tid is None or img_tid == tokenizer.unk_token_id:
            raise RuntimeError("model.config.image_token_index is None and tokenizer has no <image> token.")
        model.config.image_token_index = img_tid

    if tokenizer.pad_token_id == model.config.image_token_index:
        tokenizer.add_special_tokens({"pad_token": "<pad2>"})
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id


# -------------------------
# LoRA targets
# -------------------------
def find_projector_linear_module_names(model) -> List[str]:
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if ("mm_projector" in name) or ("multi_modal_projector" in name) or ("vision_proj" in name):
                names.append(name)
    return names


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
            raise RuntimeError("Could not find projector Linear modules. Inspect model.named_modules().")
        return proj_fullnames
    if placement == "all":
        if len(proj_fullnames) == 0:
            raise RuntimeError("Could not find projector Linear modules. Inspect model.named_modules().")
        return attn + mlp + proj_fullnames

    raise ValueError(f"Unknown placement: {placement}. Use llm_attn, llm_mlp, projector, all")


# =========================================================
# Rank-Enhancing Token Fuser (RTF) for LLaVA (Text/Image)
# (원 코드 그대로)
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

        # 안전한 폴백: 채널 분산(항상 안정적)
        def var_importance(Z):
            I = (Z * Z).mean(dim=0)  # [D]
            return torch.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)

        q = min(self.top_q, X.shape[0] - 1, X.shape[1])
        if q < 1:
            return var_importance(X)

        try:
            U, S, V = torch.svd_lowrank(X, q=q, niter=2)
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
        image_embeds: torch.Tensor,  # [B, Li, D]
        text_valid_mask: Optional[torch.Tensor] = None,  # [B, Lt] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 0) 입력이 이미 비정상이면 skip
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

        # 1) 수치 안정성: pooling/blending은 fp32로
        text_f = text_embeds.float()
        img_f  = image_embeds.float()

        # masked mean (fp32 누적)
        m = text_valid_mask.unsqueeze(-1)  # [B,Lt,1] bool
        denom = m.sum(dim=1, keepdim=True).clamp(min=1).float()  # [B,1,1]
        text_pool = (text_f * m.float()).sum(dim=1, keepdim=True) / denom  # [B,1,D]
        img_pool  = img_f.mean(dim=1, keepdim=True)  # [B,1,D]

        # 2) output buffers (fp32)
        text_out = text_f.clone()
        img_out  = img_f.clone()

        k_low = max(1, int(round(self.exchange_ratio * D)))

        # alpha도 fp32
        alpha_img = torch.sigmoid(self.alpha_img_logits.to(dev).float()).view(1, 1, D)  # [1,1,D]
        alpha_txt = torch.sigmoid(self.alpha_txt_logits.to(dev).float()).view(1, 1, D)

        eps = 1e-6

        for b in range(B):
            # --------- IMAGE SIDE ----------
            with torch.no_grad():
                I_img = self._channel_importance(img_out[b])  # [D] or None
                if I_img is None or (not torch.isfinite(I_img).all()):
                    sel_img = None
                else:
                    sel_img = torch.topk(I_img, k=k_low, largest=False).indices  # [k_low]

            if sel_img is not None and sel_img.numel() > 0:
                # mask: [1,1,D] float32
                mask = torch.zeros((D,), device=dev, dtype=torch.float32)
                mask[sel_img] = 1.0
                mask = mask.view(1, 1, D)

                img_b = img_out[b:b+1]  # [1,Li,D]
                src   = text_pool[b:b+1].expand(-1, Li, -1)  # [1,Li,D]

                # (중요) 스케일 매칭: src가 dst보다 너무 크면 overflow 유발
                # selected channels에서 RMS 맞추기 (scalar)
                dst_sel = img_b[:, :, sel_img]     # [1,Li,k]
                src_sel = src[:, :, sel_img]
                rms_dst = torch.sqrt((dst_sel**2).mean() + eps)
                rms_src = torch.sqrt((src_sel**2).mean() + eps)
                scale = (rms_dst / (rms_src + eps)).clamp(0.1, 10.0)  # 과도한 확대/축소 방지
                src = src * scale

                blended = (alpha_img * src + (1.0 - alpha_img) * img_b)  # [1,Li,D]
                img_out[b:b+1] = img_b * (1.0 - mask) + blended * mask

            # --------- TEXT SIDE (symmetric) ----------
            if self.symmetric:
                with torch.no_grad():
                    valid = text_valid_mask[b]  # [Lt]
                    Xtxt = text_out[b][valid]   # [Tvalid,D]
                    I_txt = self._channel_importance(Xtxt) if Xtxt.numel() > 0 else None
                    if I_txt is None or (not torch.isfinite(I_txt).all()):
                        sel_txt = None
                    else:
                        sel_txt = torch.topk(I_txt, k=k_low, largest=False).indices

                if sel_txt is not None and sel_txt.numel() > 0:
                    mask = torch.zeros((D,), device=dev, dtype=torch.float32)
                    mask[sel_txt] = 1.0
                    mask = mask.view(1, 1, D)

                    txt_b = text_out[b:b+1]  # [1,Lt,D]
                    src   = img_pool[b:b+1].expand(-1, Lt, -1)  # [1,Lt,D]

                    dst_sel = txt_b[:, :, sel_txt]
                    src_sel = src[:, :, sel_txt]
                    rms_dst = torch.sqrt((dst_sel**2).mean() + eps)
                    rms_src = torch.sqrt((src_sel**2).mean() + eps)
                    scale = (rms_dst / (rms_src + eps)).clamp(0.1, 10.0)
                    src = src * scale

                    blended = (alpha_txt * src + (1.0 - alpha_txt) * txt_b)
                    text_out[b:b+1] = txt_b * (1.0 - mask) + blended * mask

        # 3) 결과 sanity 체크: non-finite면 원본 반환 (아예 RTF 효과 skip)
        if (not torch.isfinite(text_out).all()) or (not torch.isfinite(img_out).all()):
            return text_embeds, image_embeds

        # 4) dtype 복원
        return text_out.to(orig_dtype), img_out.to(orig_dtype)



def inject_rtf_into_llava(
    model: nn.Module,
    image_token_id: int,
    exchange_ratio: float,
    top_q: int,
    token_subsample: int,
    symmetric: bool,
):
    merge_owner = None
    merge_name = None
    for owner in [model, getattr(model, "model", None)]:
        if owner is None:
            continue
        for name in ["_merge_input_ids_with_image_features", "merge_input_ids_with_image_features"]:
            if hasattr(owner, name):
                merge_owner = owner
                merge_name = name
                break
        if merge_owner is not None:
            break

    if merge_owner is None or merge_name is None:
        print("[RTF] Could not find merge function. RTF disabled.")
        return

    hidden = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden is None:
        hidden = getattr(getattr(model, "language_model", None).config, "hidden_size", None)
    if hidden is None:
        raise RuntimeError("[RTF] Could not infer hidden_size from model config.")

    model.rtf = RankEnhancingTokenFuser(
        hidden_size=int(hidden),
        top_q=top_q,
        exchange_ratio=exchange_ratio,
        token_subsample=token_subsample,
        symmetric=symmetric,
        init_alpha=0.5,
    )

    orig_merge = getattr(merge_owner, merge_name)
    sig = inspect.signature(orig_merge)

    def wrapped_merge(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        ba = bound.arguments

        input_ids = ba.get("input_ids", None)
        inputs_embeds = ba.get("inputs_embeds", None)
        image_features = ba.get("image_features", None)
        attention_mask = ba.get("attention_mask", None)

        if image_features is None and "vision_features" in ba:
            image_features = ba["vision_features"]
            image_key = "vision_features"
        else:
            image_key = "image_features"

        if input_ids is None or inputs_embeds is None or image_features is None:
            return orig_merge(*args, **kwargs)

        # ---- 백업 (RTF 실패 시 원복용)
        orig_inputs_embeds = inputs_embeds
        orig_image_features = image_features

        dev = inputs_embeds.device
        if attention_mask is None:
            attention_mask = (input_ids != 0).to(dtype=torch.long, device=dev)
        else:
            attention_mask = attention_mask.to(dev)
        input_ids = input_ids.to(dev)

        if getattr(model, "_rtf_device", None) != dev:
            model.rtf.to(dev)
            model._rtf_device = dev

        text_valid_mask = attention_mask.bool() & (input_ids != int(image_token_id))

        # ---- 입력 자체가 비정상이면 RTF 스킵
        if (not torch.isfinite(inputs_embeds).all()) or (not torch.isfinite(image_features).all()):
            ba["attention_mask"] = attention_mask
            return orig_merge(*bound.args, **bound.kwargs)

        text_new, img_new = model.rtf(inputs_embeds, image_features, text_valid_mask=text_valid_mask)

        # ---- RTF 출력이 비정상이면 "무조건 원본으로 폴백"
        if (not torch.isfinite(text_new).all()) or (not torch.isfinite(img_new).all()):
            ba["inputs_embeds"] = orig_inputs_embeds
            ba[image_key] = orig_image_features
            ba["attention_mask"] = attention_mask
            return orig_merge(*bound.args, **bound.kwargs)

        # 정상일 때만 적용
        ba["inputs_embeds"] = text_new
        ba[image_key] = img_new
        ba["attention_mask"] = attention_mask

        return orig_merge(*bound.args, **bound.kwargs)


    setattr(merge_owner, merge_name, wrapped_merge)
    print(f"[RTF] Injected into {merge_owner.__class__.__name__}.{merge_name} (exchange_ratio={exchange_ratio}, top_q={top_q}, symmetric={symmetric})")


# -------------------------
# Dataset (VizWiz)
# -------------------------
class VizWizLoRADataset(Dataset):
    def __init__(self, hf_ds, tokenizer, image_processor, model, max_length: int = 512):
        self.ds = hf_ds
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model = model
        self.max_length = max_length

        self.tokenizer.padding_side = "right"
        self.image_token_id = get_image_token_id(self.model, self.tokenizer)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.ds[idx]
        img = ex.get("image", None)
        if img is None:
            raise RuntimeError("Missing image.")
        img = img.convert("RGB")

        # q = ex.get("question", "")
        # prompt = build_train_prompt(q)

        # ans = choose_vizwiz_target(ex)
        # full_text = prompt + " " + ans

        # pv = self.image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        # enc = self.tokenizer(
        #     full_text,
        #     return_tensors="pt",
        #     truncation=True,
        #     max_length=self.max_length,
        #     add_special_tokens=True,
        #     padding=False,
        # )
        # input_ids = enc["input_ids"].squeeze(0)
        # attention_mask = enc["attention_mask"].squeeze(0)

        # enc_p = self.tokenizer(
        #     prompt,
        #     return_tensors="pt",
        #     truncation=True,
        #     max_length=self.max_length,
        #     add_special_tokens=True,
        #     padding=False,
        # )
        # prompt_len = enc_p["input_ids"].shape[1]

        # labels = input_ids.clone()
        # labels[:prompt_len] = -100

        # img_count = int((input_ids == self.image_token_id).sum().item())
        # if img_count != 1:
        #     raise ValueError(
        #         f"Bad image token count (pre-pad): {img_count} (expected 1). "
        #         f"image_token_id={self.image_token_id}\n"
        #         f"Prompt head: {prompt[:200]}"
        #     )

        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "pixel_values": pv,
        #     "labels": labels,
        # }
        ANSWER_BUDGET = 16  # 16~32 추천

        # --- inside __getitem__ ---
        q = ex.get("question", "")
        prompt = build_train_prompt(q)

        ans = choose_vizwiz_target(ex)
        ans = normalize_ans(ans)  # 공백/빈문자 방지
        if len(ans) == 0:
            ans = "unanswerable"

        # 1) prompt 먼저: answer budget만큼 빼고 자르기
        enc_p = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - ANSWER_BUDGET,
            add_special_tokens=True,
            padding=False,
        )
        prompt_ids = enc_p["input_ids"]          # [1, Lp]
        prompt_mask = enc_p["attention_mask"]    # [1, Lp]
        prompt_len = prompt_ids.shape[1]

        # 2) answer는 별도로: special token 없이 짧게
        enc_a = self.tokenizer(
            " " + ans,
            return_tensors="pt",
            truncation=True,
            max_length=ANSWER_BUDGET,
            add_special_tokens=False,
            padding=False,
        )
        ans_ids = enc_a["input_ids"]             # [1, La]
        ans_mask = enc_a["attention_mask"]       # [1, La]

        # 3) concat
        input_ids = torch.cat([prompt_ids, ans_ids], dim=1).squeeze(0)            # [Lp+La]
        attention_mask = torch.cat([prompt_mask, ans_mask], dim=1).squeeze(0)

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # ---- 여기서 진짜로 supervised token이 있는지 강제 체크 (중요) ----
        if int((labels != -100).sum().item()) == 0:
            # 이 경우는 거의 없어야 함. 그래도 혹시 생기면 안전하게 강제 1토큰이라도 남기기.
            labels[-1] = input_ids[-1]

        # <image> placeholder exactly 1
        img_count = int((input_ids == self.image_token_id).sum().item())
        if img_count != 1:
            raise ValueError(f"Bad image token count: {img_count} (expected 1)")

        # pixel_values
        pv = self.image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pv,
            "labels": labels,
        }



def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int,
    image_token_id: int,
    cast_pixel_values_fp16: bool,
) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].shape[0] for x in batch)

    def pad_1d(x: torch.Tensor, pad_value: int):
        n = x.shape[0]
        if n == max_len:
            return x
        pad = torch.full((max_len - n,), pad_value, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    input_ids = torch.stack([pad_1d(x["input_ids"], pad_id) for x in batch], dim=0)
    attention_mask = torch.stack([pad_1d(x["attention_mask"], 0) for x in batch], dim=0)
    labels = torch.stack([pad_1d(x["labels"], -100) for x in batch], dim=0)
    pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)

    if pad_id == image_token_id:
        raise RuntimeError(f"pad_id == image_token_id ({pad_id}). ensure_safe_pad() failed.")

    for b in range(input_ids.shape[0]):
        cnt = int((input_ids[b] == image_token_id).sum().item())
        if cnt != 1:
            raise ValueError(
                f"Bad image token count (post-pad): {cnt} (expected 1). "
                f"pad_id={pad_id}, image_token_id={image_token_id}, seq_len={input_ids.shape[1]}"
            )

    if cast_pixel_values_fp16:
        pixel_values = pixel_values.to(dtype=torch.float16)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }


# -------------------------
# Training
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")

    # VizWiz dataset
    ap.add_argument("--dataset_name", type=str, default="HuggingFaceM4/VizWiz")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--max_train_samples", type=int, default=400)

    ap.add_argument("--placement", type=str, required=True,
                    choices=["llm_attn", "llm_mlp", "projector", "all"])

    ap.add_argument("--output_dir", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--device_map", type=str, default="cuda:0")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--max_length", type=int, default=512)
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

    ap.add_argument("--dry_run", action="store_true", help="LoRA target 확인만 하고 종료")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if LlavaModelClass is None:
        raise RuntimeError("Could not import LLaVA model class. Update transformers.")

    base_dir = Path(args.base_dir).resolve()
    paths = setup_hf_cache(base_dir)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (base_dir / "step3_wacv_vizwiz" / args.placement)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]),
        patch_size=32,
        vision_feature_select_strategy="default",
    )
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    tokenizer.padding_side = "right"

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

    model = LlavaModelClass.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ensure_safe_pad(tokenizer, model)

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    image_token_id = get_image_token_id(model, tokenizer)
    print("pad_token_id:", tokenizer.pad_token_id, "| image_token_id:", image_token_id)

    if not args.rtf_disable:
        inject_rtf_into_llava(
            model=model,
            image_token_id=image_token_id,
            exchange_ratio=args.rtf_exchange_ratio,
            top_q=args.rtf_top_q,
            token_subsample=args.rtf_token_subsample,
            symmetric=args.rtf_symmetric,
        )

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
        print("LoRA modules (first 60):")
        for x in hit[:60]:
            print("  ", x)
        rtf_params = [n for n, p in model.named_parameters() if "rtf." in n and p.requires_grad]
        if len(rtf_params) > 0:
            print("RTF trainable params:")
            for n in rtf_params:
                print("  ", n)
        print("Dry run done.")
        return

    dl_cfg = DownloadConfig(resume_download=True, max_retries=50)
    ds = load_dataset(
        args.dataset_name,
        split=f"{args.train_split}[:{args.max_train_samples}]",
        cache_dir=str(paths["HF_DATASETS_CACHE"]),
        download_config=dl_cfg,
    )
    ds = ds.filter(lambda ex: ex.get("image", None) is not None and isinstance(ex.get("question", None), str))
    print("Train rows:", len(ds))

    train_set = VizWizLoRADataset(ds, tokenizer, image_processor, model, max_length=args.max_length)

    cast_pv_fp16 = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(
            b,
            pad_id=tokenizer.pad_token_id,
            image_token_id=image_token_id,
            cast_pixel_values_fp16=cast_pv_fp16,
        ),
    )

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
            n_supervised = int((batch["labels"] != -100).sum().item())
            if n_supervised == 0:
                print("WARNING: no supervised tokens in this batch -> loss may become NaN")

            batch = {k: v.to(args.device_map) for k, v in batch.items()}
            out = model(**batch)
            labels = batch["labels"]
            valid_n = int((labels != -100).sum().item())
            #print("valid_label_tokens:", valid_n)

            # logits NaN 여부
            logits = out.logits
            #print("logits finite:", bool(torch.isfinite(logits).all().item()))

            if valid_n == 0:
                print(">>> ALL LABELS ARE -100 (no target tokens) -> loss becomes NaN")
                raise RuntimeError("No valid label tokens in this batch")
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite logits (NaN/Inf) detected")
            loss_task = out.loss

            if not torch.isfinite(loss_task):
                bad = first_nonfinite_trainable_param(model)
                print("LOSS IS NON-FINITE. first bad param:", bad)
                print("loss:", loss_task.item())
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
