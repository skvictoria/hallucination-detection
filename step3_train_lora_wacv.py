import os
import json
import math
import random
import argparse
import inspect
import types
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
# Prompt / label
# -------------------------
def build_train_prompt(question: str, choices: Optional[List[str]]) -> str:
    if choices and isinstance(choices, list) and len(choices) > 0:
        opts = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices[:8])])
        body = f"Question: {question}\nChoices: {opts}\n"
    else:
        body = f"Question: {question}\n"

    return (
        "USER: <image>\n"
        f"{body}"
        "Answer with the letter only (A, B, C, D, ...). Do not explain.\n"
        "ASSISTANT:"
    )


def get_answer_letter(ex: Dict[str, Any]) -> str:
    a = ex.get("answer", None)
    if isinstance(a, int):
        return chr(65 + int(a))
    if isinstance(a, str):
        a = a.strip()
        if len(a) == 1 and a.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return a.upper()
    return "A"


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
# - Paper-style: channel importance via (truncated) SVD
# - bottom-k' channels only, adaptive blending with learnable alpha
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

        # alpha in [0,1] via sigmoid(logits). We keep per-channel alphas (size D).
        # init_alpha=0.5 -> logits=0
        init_logit = math.log(init_alpha / (1.0 - init_alpha + 1e-12) + 1e-12)
        self.alpha_img_logits = nn.Parameter(torch.full((hidden_size,), float(init_logit)))
        self.alpha_txt_logits = nn.Parameter(torch.full((hidden_size,), float(init_logit)))

    @torch.no_grad()
    def _channel_importance(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        """
        X: [T, D] (float/half ok). We compute truncated right-singular vectors via svd_lowrank.
        Returns I: [D] where I_c = sum_{i=1..q} (S_i^2) * (v_{c,i}^2)
        """
        if X is None or X.numel() == 0:
            return None
        T, D = X.shape
        if T < 2 or D < 2:
            return None

        # subsample tokens for speed
        if self.token_subsample > 0 and T > self.token_subsample:
            X = X[: self.token_subsample]

        X = X.float()
        X = X - X.mean(dim=0, keepdim=True)

        q = min(self.top_q, X.shape[0] - 1, X.shape[1])
        if q < 1:
            return None

        try:
            # U: [T, q], S: [q], V: [D, q]
            U, S, V = torch.svd_lowrank(X, q=q, niter=2)
        except Exception:
            # fallback (heavier)
            # full_matrices=False -> Vh: [min(T,D), D]
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            q2 = min(q, Vh.shape[0])
            V = Vh[:q2].transpose(0, 1)  # [D, q2]
            S = S[:q2]

        I = (V.pow(2) @ S.pow(2))  # [D]
        return I

    def forward(
        self,
        text_embeds: torch.Tensor,   # [B, Lt, D]
        image_embeds: torch.Tensor,  # [B, Li, D]
        text_valid_mask: Optional[torch.Tensor] = None,  # [B, Lt] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (text_embeds_new, image_embeds_new)
        """
        dev = text_embeds.device
        dtype = text_embeds.dtype
        B, Lt, D = text_embeds.shape
        Bi, Li, Di = image_embeds.shape
        assert B == Bi and D == Di, "Shape mismatch between text and image embeddings"

        # pool other-modality features (broadcast targets y_c)
        # text pool: masked mean over valid text tokens (exclude pad and <image> placeholder)
        if text_valid_mask is None:
            text_valid_mask = torch.ones((B, Lt), device=text_embeds.device, dtype=torch.bool)

        # avoid div by zero
        denom = text_valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(text_embeds.dtype)  # [B,1]
        text_pool = (text_embeds * text_valid_mask.unsqueeze(-1).to(text_embeds.dtype)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)  # [B,1,D]
        img_pool = image_embeds.mean(dim=1, keepdim=True)  # [B,1,D]

        # prepare outputs
        text_out = text_embeds
        img_out = image_embeds

        # compute per-sample bottom channels and apply blending via masks
        k_low = max(1, int(round(self.exchange_ratio * D)))

        alpha_img = torch.sigmoid(self.alpha_img_logits.to(dev)).to(dtype).view(1, 1, D)
        alpha_txt = torch.sigmoid(self.alpha_txt_logits.to(dev)).to(dtype).view(1, 1, D)

        for b in range(B):
            # ---- image: find low-informative channels in image_embeds[b]
            with torch.no_grad():
                I_img = self._channel_importance(img_out[b])  # [D] or None
                if I_img is None:
                    sel_img = None
                else:
                    sel_img = torch.topk(I_img, k=k_low, largest=False).indices  # [k_low]

            if sel_img is not None and sel_img.numel() > 0:
                mask = torch.zeros((D,), device=dev, dtype=dtype)
                #mask = torch.zeros((D,), device=img_out.device, dtype=img_out.dtype)
                mask[sel_img] = 1.0
                mask = mask.view(1, 1, D)

                # Eq-style: new_low = alpha * (other) + (1-alpha) * (self)
                img_out_b = img_out[b:b+1]  # [1,Li,D]
                txt_pool_b = text_pool[b:b+1].expand(-1, Li, -1)  # [1,Li,D]
                img_out = torch.cat([
                    img_out[:b],
                    img_out_b * (1 - mask) + (alpha_img * txt_pool_b + (1 - alpha_img) * img_out_b) * mask,
                    img_out[b+1:]
                ], dim=0)

            # ---- text (optional symmetric): low-informative channels in text_embeds[b]
            if self.symmetric:
                with torch.no_grad():
                    # only valid tokens for importance estimation
                    valid = text_valid_mask[b]
                    Xtxt = text_out[b][valid]  # [Tvalid, D]
                    I_txt = self._channel_importance(Xtxt) if Xtxt.numel() > 0 else None
                    sel_txt = torch.topk(I_txt, k=k_low, largest=False).indices if I_txt is not None else None

                if sel_txt is not None and sel_txt.numel() > 0:
                    mask = torch.zeros((D,), device=dev, dtype=dtype)
                    #mask = torch.zeros((D,), device=text_out.device, dtype=text_out.dtype)
                    mask[sel_txt] = 1.0
                    mask = mask.view(1, 1, D)

                    text_out_b = text_out[b:b+1]  # [1,Lt,D]
                    img_pool_b = img_pool[b:b+1].expand(-1, Lt, -1)  # [1,Lt,D]
                    text_out = torch.cat([
                        text_out[:b],
                        text_out_b * (1 - mask) + (alpha_txt * img_pool_b + (1 - alpha_txt) * text_out_b) * mask,
                        text_out[b+1:]
                    ], dim=0)

        return text_out, img_out


def inject_rtf_into_llava(
    model: nn.Module,
    image_token_id: int,
    exchange_ratio: float,
    top_q: int,
    token_subsample: int,
    symmetric: bool,
):
    """
    Monkey-patch LLaVA's merge function so we can access BOTH:
      - text inputs_embeds (from input_ids)
      - image_features (after projector)
    and apply paper-style RTF before merging.
    """
    # find merge function on model or model.model
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
        # some configs store it differently
        hidden = getattr(getattr(model, "language_model", None).config, "hidden_size", None)
    if hidden is None:
        raise RuntimeError("[RTF] Could not infer hidden_size from model config.")

    # attach fuser (trainable params will be optimized together with LoRA)
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

        # >>> 추가: 모두 같은 device로
        dev = inputs_embeds.device
        if attention_mask is None:
            attention_mask = (input_ids != 0).to(dtype=torch.long, device=dev)
        else:
            attention_mask = attention_mask.to(dev)
        input_ids = input_ids.to(dev)

        # >>> 추가: rtf를 merge가 도는 디바이스로 이동 (한번만)
        if getattr(model, "_rtf_device", None) != dev:
            model.rtf.to(dev)
            model._rtf_device = dev

        text_valid_mask = attention_mask.bool() & (input_ids != int(image_token_id))

        text_new, img_new = model.rtf(inputs_embeds, image_features, text_valid_mask=text_valid_mask)

        ba["inputs_embeds"] = text_new
        ba[image_key] = img_new
        ba["attention_mask"] = attention_mask

        return orig_merge(*bound.args, **bound.kwargs)


    setattr(merge_owner, merge_name, wrapped_merge)
    print(f"[RTF] Injected into {merge_owner.__class__.__name__}.{merge_name} (exchange_ratio={exchange_ratio}, top_q={top_q}, symmetric={symmetric})")


# -------------------------
# Dataset
# -------------------------
class ScienceQALoRADataset(Dataset):
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

        question = ex.get("question", "")
        choices = ex.get("choices", None)
        prompt = build_train_prompt(question, choices)
        ans = get_answer_letter(ex)
        full_text = prompt + " " + ans

        pv = self.image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=False,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        enc_p = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=False,
        )
        prompt_len = enc_p["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # <image> placeholder exactly 1
        img_count = int((input_ids == self.image_token_id).sum().item())
        if img_count != 1:
            raise ValueError(
                f"Bad image token count (pre-pad): {img_count} (expected 1). "
                f"image_token_id={self.image_token_id}\n"
                f"Prompt head: {prompt[:200]}"
            )

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
    ap.add_argument("--dataset_name", type=str, default="derek-thomas/ScienceQA")

    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--max_train_samples", type=int, default=400)

    ap.add_argument("--placement", type=str, required=True,
                    choices=["llm_attn", "llm_mlp", "projector", "all"])

    ap.add_argument("--output_dir", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--device_map", type=str, default="auto")
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

    # New: Rank-Enhancing Token Fuser (paper-style)
    ap.add_argument("--rtf_exchange_ratio", type=float, default=0.10,
                    help="bottom-k' 채널 비율 (예: 0.1=10%)")
    ap.add_argument("--rtf_top_q", type=int, default=64,
                    help="채널 중요도 계산에 사용할 truncated SVD rank q")
    ap.add_argument("--rtf_token_subsample", type=int, default=256,
                    help="SVD 계산 시 토큰 subsample (속도용)")
    ap.add_argument("--rtf_symmetric", action="store_true",
                    help="True면 text<->image 양방향 blending, False면 image만 blending")
    ap.add_argument("--rtf_disable", action="store_true",
                    help="RTF 비활성화 (디버그용)")

    ap.add_argument("--dry_run", action="store_true", help="LoRA target 확인만 하고 종료")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if LlavaModelClass is None:
        raise RuntimeError("Could not import LLaVA model class. Update transformers.")

    base_dir = Path(args.base_dir).resolve()
    paths = setup_hf_cache(base_dir)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (base_dir / "step3" / args.placement)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Processor
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]),
        patch_size=32,
        vision_feature_select_strategy="default",
    )
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    tokenizer.padding_side = "right"

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

    model = LlavaModelClass.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ensure_safe_pad(tokenizer, model)

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # image token id
    image_token_id = get_image_token_id(model, tokenizer)
    print("pad_token_id:", tokenizer.pad_token_id, "| image_token_id:", image_token_id)

    # Inject RTF (before LoRA ok, after LoRA ok — 여기서는 LoRA 전에 주입)
    if not args.rtf_disable:
        inject_rtf_into_llava(
            model=model,
            image_token_id=image_token_id,
            exchange_ratio=args.rtf_exchange_ratio,
            top_q=args.rtf_top_q,
            token_subsample=args.rtf_token_subsample,
            symmetric=args.rtf_symmetric,
        )

    # LoRA 설정
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

    # trainable params info
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"[{args.placement}] trainable params: {trainable} / {total} ({100*trainable/total:.4f}%)")

    if args.dry_run:
        hit = []
        for n, _ in model.named_modules():
            if "lora_" in n:
                hit.append(n)
        print("LoRA modules (first 60):")
        for x in hit[:60]:
            print("  ", x)
        # RTF params 확인
        rtf_params = [n for n, p in model.named_parameters() if "rtf." in n and p.requires_grad]
        if len(rtf_params) > 0:
            print("RTF trainable params:")
            for n in rtf_params:
                print("  ", n)
        print("Dry run done.")
        return

    # Dataset
    dl_cfg = DownloadConfig(resume_download=True, max_retries=50)
    ds = load_dataset(
        args.dataset_name,
        split=f"{args.train_split}[:{args.max_train_samples}]",
        cache_dir=str(paths["HF_DATASETS_CACHE"]),
        download_config=dl_cfg,
    )
    ds = ds.filter(lambda ex: ex.get("image", None) is not None)
    print("Train rows:", len(ds))

    train_set = ScienceQALoRADataset(ds, tokenizer, image_processor, model, max_length=args.max_length)

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

    # Optimizer
    if args.load_in_4bit:
        from bitsandbytes.optim import PagedAdamW8bit
        optim = PagedAdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
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
            out = model(**batch)
            loss_task = out.loss

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
