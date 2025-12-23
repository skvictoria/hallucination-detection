import os
import re
import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
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
    # 출력 안정화를 위해 letter-only 학습
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

    # fallback search
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
    # pad token 없으면 추가
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    img_tid = getattr(model.config, "image_token_index", None)
    if img_tid is None:
        # set from tokenizer if possible
        img_tid = tokenizer.convert_tokens_to_ids("<image>")
        if img_tid is None or img_tid == tokenizer.unk_token_id:
            raise RuntimeError("model.config.image_token_index is None and tokenizer has no <image> token.")
        model.config.image_token_index = img_tid

    # pad_token_id가 image_token_index와 같으면 새 pad 토큰 추가
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


# -------------------------
# Effective rank helper (token matrix -> effective rank)
# -------------------------
def effective_rank_from_tokens(H: torch.Tensor, eps: float = 1e-12) -> float:
    """
    H: [n_tokens, d]
    effective rank = exp(entropy(eigvals(H H^T)))
    """
    if H is None or H.numel() == 0:
        return float("nan")
    n = H.shape[0]
    if n < 2:
        return 1.0
    X = H.float()
    X = X - X.mean(dim=0, keepdim=True)
    G = X @ X.t()
    G = (G + G.t()) * 0.5
    try:
        eig = torch.linalg.eigvalsh(G)
    except Exception:
        eig = torch.linalg.eigvals(G).real
    eig = torch.clamp(eig, min=0.0)
    s = float(eig.sum().item())
    if s <= eps:
        return float("nan")
    p = eig / (s + eps)
    ent = -torch.sum(p * torch.log(p + eps)).item()
    return float(math.exp(ent))


def build_masks_from_expanded_len(
    prompt_ids_no_pad: List[int],
    image_token_id: int,
    expanded_len_no_pad: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    prompt_ids_no_pad: padding 제거된 input_ids (길이 L_real)
    expanded_len_no_pad: merge 후 padding 제외된 길이
    """
    L = len(prompt_ids_no_pad)
    pos = [i for i, tid in enumerate(prompt_ids_no_pad) if tid == image_token_id]
    n = len(pos)
    if n == 0:
        raise RuntimeError("No <image> token found in prompt_ids_no_pad.")
    numerator = expanded_len_no_pad - L + n
    m = int(round(numerator / n)) if (numerator % n != 0) else (numerator // n)
    if m <= 0:
        raise RuntimeError(f"Invalid inferred m={m}, expanded_len={expanded_len_no_pad}, L={L}, n={n}")

    text_mask = torch.zeros(expanded_len_no_pad, dtype=torch.bool, device="cpu")
    img_mask = torch.zeros(expanded_len_no_pad, dtype=torch.bool, device="cpu")
    cur = 0
    for i in range(L):
        if prompt_ids_no_pad[i] == image_token_id:
            img_mask[cur:cur + m] = True
            cur += m
        else:
            if cur < expanded_len_no_pad:
                text_mask[cur] = True
            cur += 1

    if cur != expanded_len_no_pad and cur < expanded_len_no_pad:
        text_mask[cur:expanded_len_no_pad] = True

    return text_mask, img_mask


# -------------------------
# Dataset: 이미지/텍스트 분리 전처리
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

        # 1) image -> pixel_values
        pv = self.image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        # 2) text -> token ids (no padding here; dynamic padding in collate)
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

        # 핵심: <image> placeholder가 정확히 1개인지 강제 (pad 전)
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

    # 매우 중요: pad 토큰이 image 토큰이면 catastrophic
    if pad_id == image_token_id:
        raise RuntimeError(f"pad_id == image_token_id ({pad_id}). ensure_safe_pad() failed.")

    # 매우 중요: padding 후에도 <image> placeholder가 배치 각 샘플마다 정확히 1개인지 체크
    for b in range(input_ids.shape[0]):
        cnt = int((input_ids[b] == image_token_id).sum().item())
        if cnt != 1:
            # 이 경우가 발생하면 LLaVA가 'image tokens 507' 같은 에러를 내기 쉬움
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

    # New: effective-rank regularization (weak-modality rescue)
    ap.add_argument("--rank_reg_weight", type=float, default=0.2,
                    help=">0이면 image/token effective-rank penalty를 추가. (권장: 0.01~0.2)")
    ap.add_argument("--rank_target_img", type=float, default=32.0,
                    help="image-token effective rank 목표 하한 (subsample 후 기준).")
    ap.add_argument("--rank_balance_margin", type=float, default=0.0,
                    help=">0이면 (rank_text - rank_img - margin) 양수일 때 추가 페널티.")
    ap.add_argument("--rank_tokens", type=int, default=64,
                    help="effective rank 계산 시 modality별로 사용할 토큰 수(속도/안정성).")
    ap.add_argument("--rank_reg_every", type=int, default=1,
                    help="몇 step마다 rank penalty를 적용할지(1=매 step).")

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

    # pad_token 안전화(중요)
    ensure_safe_pad(tokenizer, model)

    # QLoRA 준비
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

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

    # image token id (after pad safe)
    image_token_id = get_image_token_id(model, tokenizer)
    print("pad_token_id:", tokenizer.pad_token_id, "| image_token_id:", image_token_id)

    if args.dry_run:
        hit = []
        for n, _ in model.named_modules():
            if "lora_" in n:
                hit.append(n)
        print("LoRA modules (first 60):")
        for x in hit[:60]:
            print("  ", x)
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

    # running logs
    running_loss = 0.0
    running_rank_img = 0.0
    running_rank_txt = 0.0
    running_rank_n = 0

    while step < args.max_steps:
        for batch in train_loader:
            # device_map="auto"인 경우, 강제 .to()는 피함(Accelerate가 처리)
            need_rank = (args.rank_reg_weight > 0.0) and ((step + 1) % args.rank_reg_every == 0)

            if need_rank:
                out = model(**batch, output_hidden_states=True, return_dict=True)
            else:
                out = model(**batch)

            loss_task = out.loss

            # -------------------------
            # Rank regularization (optional)
            # -------------------------
            loss_rank = 0.0
            rank_img_val = float("nan")
            rank_txt_val = float("nan")

            if need_rank:
                # hidden_states[-1]: [B, seq_len_merged, D]
                last = out.hidden_states[-1]  # type: ignore
                # NOTE: batch_size>1 지원: 샘플별로 계산 후 평균
                B = last.shape[0]
                seq_merged = last.shape[1]
                for b in range(B):
                    # 원래(merge 전) real token length (pad 제외)
                    attn = batch["attention_mask"][b]
                    L_real = int(attn.sum().item())
                    P = int(attn.shape[0] - L_real)  # pad count (merge 전)

                    # merge 후 pad count는 동일하게 유지된다고 가정(오른쪽 padding)
                    expanded_real_len = int(seq_merged - P)
                    if expanded_real_len <= 0:
                        continue

                    prompt_ids_no_pad = batch["input_ids"][b, :L_real].tolist()
                    # mask build on CPU for simplicity
                    tmask, imask = build_masks_from_expanded_len(
                        prompt_ids_no_pad=prompt_ids_no_pad,
                        image_token_id=image_token_id,
                        expanded_len_no_pad=expanded_real_len,
                    )

                    H = last[b, :expanded_real_len, :]  # [expanded_real_len, D]
                    Ht = H[tmask.to(H.device)]
                    Hv = H[imask.to(H.device)]

                    # subsample tokens for speed (deterministic slice)
                    if args.rank_tokens > 0:
                        if Ht.shape[0] > args.rank_tokens:
                            Ht = Ht[:args.rank_tokens]
                        if Hv.shape[0] > args.rank_tokens:
                            Hv = Hv[:args.rank_tokens]

                    r_txt = effective_rank_from_tokens(Ht)
                    r_img = effective_rank_from_tokens(Hv)

                    # penalty: push image rank up
                    # relu(target - r_img)
                    if not math.isnan(r_img):
                        loss_rank = loss_rank + max(0.0, args.rank_target_img - r_img)

                    # optional balance: prevent text >> image
                    if args.rank_balance_margin > 0 and (not math.isnan(r_txt)) and (not math.isnan(r_img)):
                        loss_rank = loss_rank + max(0.0, (r_txt - r_img - args.rank_balance_margin))

                    # logging (batch 평균)
                    if not math.isnan(r_img):
                        running_rank_img += r_img
                    if not math.isnan(r_txt):
                        running_rank_txt += r_txt
                    running_rank_n += 1

                if isinstance(loss_rank, float):
                    loss_rank = torch.tensor(loss_rank, dtype=loss_task.dtype, device=loss_task.device)

            # total loss
            loss = loss_task + (args.rank_reg_weight * loss_rank if need_rank else 0.0)
            loss = loss / args.grad_accum
            loss.backward()

            running_loss += float(loss_task.item())

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            if (step + 1) % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                msg = f"step {step+1}/{args.max_steps} | loss={running_loss/args.log_every:.4f} | lr={lr:.2e}"
                if args.rank_reg_weight > 0 and running_rank_n > 0:
                    msg += f" | effrank_img={running_rank_img/running_rank_n:.2f} | effrank_txt={running_rank_txt/running_rank_n:.2f}"
                print(msg)
                running_loss = 0.0
                running_rank_img = 0.0
                running_rank_txt = 0.0
                running_rank_n = 0

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
