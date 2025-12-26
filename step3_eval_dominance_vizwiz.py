# step3_eval_wacv_vizwiz.py
# - VizWiz용 evaluation 스크립트 (ScienceQA -> VizWiz 변경)
# - orig / blank / swap 답변, MDI/AEI(attn), effrank/CKA(hidden) 기록
# - NO-PADDING 입력 빌더 유지 (pad/image 충돌 방지)
#
# 실행 예:
# python step3_eval_wacv_vizwiz.py \
#   --adapter_dir /path/to/step3/llm_mlp/final \
#   --split validation --max_samples 200 \
#   --out_dir /path/to/out
#
import os
import re
import csv
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

import torch
from PIL import Image

from datasets import load_dataset, DownloadConfig
from transformers import AutoProcessor

from peft import PeftModel

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

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False


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

def is_visual_cued(question: str) -> int:
    q = (question or "").lower()
    return 1 if any(re.search(p, q) for p in VISUAL_CUE_PATTERNS) else 0


# -------------------------
# perturbations / normalize
# -------------------------
def blank_like(img: Image.Image) -> Image.Image:
    return Image.new("RGB", img.size, color=(128, 128, 128))

def normalize_answer(s: str) -> str:
    # VizWiz는 open-ended라 너무 과하게 정규화하면 정보가 날아갈 수 있음
    # 여기서는 "비교"를 위해 최소한으로만 정리 (lower + 공백 정리)
    s = (s or "").strip().lower()
    s = re.sub(r"^(assistant:|answer:)\s*", "", s)
    s = " ".join(s.split())
    return s


# -------------------------
# VizWiz GT helper (optional)
# -------------------------
def normalize_vizwiz_gt(s: str) -> str:
    s = (s or "").strip().lower()
    s = " ".join(s.split())
    return s

def choose_vizwiz_target(ex: Dict[str, Any]) -> str:
    """
    HuggingFaceM4/VizWiz schema (train/val/test)
      - question: str
      - image: PIL
      - answers: list[str] (train/val)
      - answerable: 0/1 (train/val)
    """
    answerable = ex.get("answerable", 1)
    if isinstance(answerable, (int, float)) and int(answerable) == 0:
        return "unanswerable"

    answers = ex.get("answers", None)
    if not isinstance(answers, list) or len(answers) == 0:
        return "unanswerable"

    norm = [normalize_vizwiz_gt(a) for a in answers if isinstance(a, str) and a.strip()]
    if len(norm) == 0:
        return "unanswerable"
    most_common = Counter(norm).most_common(1)[0][0]
    return most_common


# -------------------------
# token helpers
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

    raise RuntimeError("Could not find image token id in tokenizer/model config.")

def sync_image_token_index(model, tokenizer) -> int:
    tok_tid = tokenizer.convert_tokens_to_ids("<image>")
    if tok_tid is not None and tok_tid != tokenizer.unk_token_id:
        cfg_tid = getattr(model.config, "image_token_index", None)
        if cfg_tid is None or int(cfg_tid) != int(tok_tid):
            model.config.image_token_index = int(tok_tid)
    return int(model.config.image_token_index)

def ensure_safe_pad(tokenizer, model):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    sync_image_token_index(model, tokenizer)

    if tokenizer.pad_token_id == model.config.image_token_index:
        tokenizer.add_special_tokens({"pad_token": "<pad2>"})
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

def infer_input_device(model) -> torch.device:
    for _, p in model.named_parameters():
        if isinstance(p, torch.Tensor) and p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# prompt builder (VizWiz open-ended)
# -------------------------
def build_prompt_vizwiz(question: str) -> str:
    q = (question or "").strip()
    return (
        "USER: <image>\n"
        f"Question: {q}\n"
        "Answer the question using a short phrase. "
        "If the image does not contain enough information, answer exactly: unanswerable.\n"
        "ASSISTANT:"
    )


# -------------------------
# MDI/AEI from generate attentions
# -------------------------
def build_context_masks_from_generate_attn(
    prompt_input_ids: List[int],
    image_token_id: int,
    context_len_expanded: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    L = len(prompt_input_ids)
    positions = [i for i, tid in enumerate(prompt_input_ids) if tid == image_token_id]
    n = len(positions)
    if n == 0:
        raise RuntimeError("No <image> token found in prompt_input_ids.")

    numerator = context_len_expanded - L + n
    m = int(round(numerator / n)) if (numerator % n != 0) else (numerator // n)
    if m <= 0:
        raise RuntimeError(f"Invalid inferred image token count m={m}.")

    text_mask = torch.zeros(context_len_expanded, dtype=torch.bool)
    image_mask = torch.zeros(context_len_expanded, dtype=torch.bool)

    cur = 0
    for i in range(L):
        if prompt_input_ids[i] == image_token_id:
            image_mask[cur:cur + m] = True
            cur += m
        else:
            if cur < context_len_expanded:
                text_mask[cur] = True
            cur += 1

    if cur != context_len_expanded and cur < context_len_expanded:
        text_mask[cur:context_len_expanded] = True

    num_img = int(image_mask.sum().item())
    num_txt = int(text_mask.sum().item())
    return text_mask, image_mask, num_img, num_txt


def compute_mdi_aei(
    gen_attentions,
    prompt_input_ids: List[int],
    image_token_id: int,
) -> Dict[str, float]:
    if gen_attentions is None or len(gen_attentions) == 0:
        return {
            "AT": float("nan"), "AO": float("nan"), "MDI": float("nan"),
            "AEI_T": float("nan"), "AEI_O": float("nan"),
            "context_len": float("nan"), "num_text_tokens": float("nan"), "num_image_tokens": float("nan"),
        }

    first_step = gen_attentions[0]
    first_layer = first_step[0]
    context_len = int(first_layer.shape[-1])

    text_mask, image_mask, num_img, num_text = build_context_masks_from_generate_attn(
        prompt_input_ids, image_token_id, context_len
    )

    A_text, A_img = 0.0, 0.0
    for step_attn in gen_attentions:
        for layer_attn in step_attn:
            a = layer_attn[0, :, 0, :context_len]  # [heads, key_len]
            vec = a.sum(dim=0)
            A_text += float(vec[text_mask].sum().item())
            A_img += float(vec[image_mask].sum().item())

    total = A_text + A_img
    if total <= 0 or num_img == 0 or num_text == 0:
        return {
            "AT": float("nan"), "AO": float("nan"), "MDI": float("nan"),
            "AEI_T": float("nan"), "AEI_O": float("nan"),
            "context_len": float(context_len), "num_text_tokens": float(num_text), "num_image_tokens": float(num_img),
        }

    AT = A_text / total
    AO = A_img / total

    mdi = (AT / num_text) / (AO / num_img)
    QT = num_text / (num_text + num_img)
    QO = num_img / (num_text + num_img)
    aei_t = AT / QT
    aei_o = AO / QO

    return {
        "AT": float(AT), "AO": float(AO), "MDI": float(mdi),
        "AEI_T": float(aei_t), "AEI_O": float(aei_o),
        "context_len": float(context_len), "num_text_tokens": float(num_text), "num_image_tokens": float(num_img),
    }


# -------------------------
# prompt hidden-state diagnostics: effrank / CKA
# -------------------------
def effective_rank_from_tokens(H: torch.Tensor, eps: float = 1e-12) -> float:
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

def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    if X is None or Y is None:
        return float("nan")
    if X.shape[0] < 2 or Y.shape[0] < 2:
        return float("nan")
    if X.shape[0] != Y.shape[0]:
        return float("nan")

    Xc = X.float() - X.float().mean(dim=0, keepdim=True)
    Yc = Y.float() - Y.float().mean(dim=0, keepdim=True)

    K = Xc @ Xc.t()
    L = Yc @ Yc.t()

    n = K.shape[0]
    Hm = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
    Kc = Hm @ K @ Hm
    Lc = Hm @ L @ Hm

    hsic = (Kc * Lc).sum()
    normx = torch.sqrt((Kc * Kc).sum() + eps)
    normy = torch.sqrt((Lc * Lc).sum() + eps)
    return float((hsic / (normx * normy + eps)).item())

def build_masks_from_expanded_len(
    prompt_ids_no_pad: List[int],
    image_token_id: int,
    expanded_len_no_pad: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    L = len(prompt_ids_no_pad)
    pos = [i for i, tid in enumerate(prompt_ids_no_pad) if tid == image_token_id]
    n = len(pos)
    if n == 0:
        raise RuntimeError("No <image> token found in prompt_ids_no_pad.")

    numerator = expanded_len_no_pad - L + n
    m = int(round(numerator / n)) if (numerator % n != 0) else (numerator // n)
    if m <= 0:
        raise RuntimeError(f"Invalid inferred m={m}.")

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
# NO-PADDING input builder (critical fix)
# -------------------------
def prepare_llava_inputs_no_padding(
    tokenizer,
    image_processor,
    model,
    image: Image.Image,
    prompt: str,
    max_length: int,
    debug: bool = False,
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    image = image.convert("RGB")

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        padding=False,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]

    image_token_id = int(model.config.image_token_index)
    cnt = int((input_ids == image_token_id).sum().item())
    if cnt != 1:
        if debug:
            print("---- DEBUG INPUT ----")
            print("prompt head:", prompt[:200].replace("\n", "\\n"))
            print("seq_len:", input_ids.shape[1])
            print("image_token_id:", image_token_id, "pad_token_id:", tokenizer.pad_token_id)
            print("image_token_count:", cnt)
            if attention_mask is not None:
                zeros = int((attention_mask == 0).sum().item())
                print("attention_mask zeros:", zeros)
        raise ValueError(
            f"Bad image token count in input_ids: {cnt} (expected 1). "
            f"This typically indicates pad_token_id/image_token_index mismatch or unwanted padding."
        )

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }
    prompt_ids = input_ids[0].tolist()
    return inputs, prompt_ids


@torch.no_grad()
def compute_prompt_stats(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    question: str,
    image_token_id: int,
    max_length: int,
    rank_tokens: int,
    cka_tokens: int,
    debug: bool = False,
) -> Dict[str, float]:
    prompt = build_prompt_vizwiz(question)

    inputs, _ = prepare_llava_inputs_no_padding(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
        image=image,
        prompt=prompt,
        max_length=max_length,
        debug=debug,
    )

    dev = infer_input_device(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    if dev.type == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    out = model(**inputs, output_hidden_states=True, return_dict=True)
    H = out.hidden_states[-1][0]  # [seq_len_expanded, d]
    seq_expanded = int(H.shape[0])

    L_real = int(inputs["input_ids"].shape[1])
    prompt_ids_no_pad = inputs["input_ids"][0, :L_real].tolist()

    expanded_real_len = seq_expanded
    H = H[:expanded_real_len, :]

    tmask, imask = build_masks_from_expanded_len(prompt_ids_no_pad, image_token_id, expanded_real_len)
    Ht = H[tmask.to(H.device)]
    Hi = H[imask.to(H.device)]

    if rank_tokens > 0:
        Ht_rank = Ht[:rank_tokens] if Ht.shape[0] > rank_tokens else Ht
        Hi_rank = Hi[:rank_tokens] if Hi.shape[0] > rank_tokens else Hi
    else:
        Ht_rank, Hi_rank = Ht, Hi

    eff_t = effective_rank_from_tokens(Ht_rank)
    eff_i = effective_rank_from_tokens(Hi_rank)

    n = min(Ht.shape[0], Hi.shape[0], cka_tokens if cka_tokens > 0 else min(Ht.shape[0], Hi.shape[0]))
    if n >= 2:
        cka = linear_cka(Ht[:n], Hi[:n])
    else:
        cka = float("nan")

    return {
        "effrank_text": float(eff_t),
        "effrank_image": float(eff_i),
        "cka_text_image": float(cka),
        "num_text_tokens": float(Ht.shape[0]),
        "num_image_tokens": float(Hi.shape[0]),
        "expanded_len": float(expanded_real_len),
    }


# -------------------------
# Generation wrapper (NO-PADDING)
# -------------------------
@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    question: str,
    max_new_tokens: int,
    want_attn: bool,
    max_length: int,
    debug: bool = False,
) -> Tuple[str, Optional[Any], List[int]]:
    prompt = build_prompt_vizwiz(question)

    inputs, prompt_ids = prepare_llava_inputs_no_padding(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
        image=image,
        prompt=prompt,
        max_length=max_length,
        debug=debug,
    )

    dev = infer_input_device(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    if dev.type == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        return_dict_in_generate=True,
        output_attentions=want_attn,
        output_scores=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    seq = out.sequences[0]
    in_len = inputs["input_ids"].shape[1]
    out_ids = seq[in_len:]

    text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    text = normalize_answer(text)

    attn = None
    if want_attn:
        if hasattr(out, "attentions") and out.attentions is not None:
            attn = out.attentions
        elif hasattr(out, "decoder_attentions") and out.decoder_attentions is not None:
            attn = out.decoder_attentions

    return text, attn, prompt_ids


# -------------------------
# Output record
# -------------------------
@dataclass
class Record:
    dataset_idx: int
    visual_cued: int
    question: str

    # (optional) gt majority / answerable
    gt_answer: str
    gt_answerable: int

    ans_orig: str
    ans_blank: str
    same_orig_blank: int

    swap_k: int
    swap_same_rate: float

    # attention-based
    mdi_orig: float
    aei_t_orig: float
    aei_o_orig: float
    at_orig: float
    ao_orig: float

    mdi_blank: float
    aei_t_blank: float
    aei_o_blank: float
    at_blank: float
    ao_blank: float

    # prompt-hidden-based (diagnostics)
    effrank_text_orig: float
    effrank_image_orig: float
    cka_ti_orig: float

    effrank_text_blank: float
    effrank_image_blank: float
    cka_ti_blank: float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--adapter_dir", type=str, required=True)

    # VizWiz dataset
    ap.add_argument("--dataset_name", type=str, default="HuggingFaceM4/VizWiz")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--max_samples", type=int, default=200)

    ap.add_argument("--swap_k", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--device_map", type=str, default="auto")

    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--out_summary", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)

    # prompt diagnostic options
    ap.add_argument("--no_prompt_diag", action="store_true")
    ap.add_argument("--rank_tokens", type=int, default=64)
    ap.add_argument("--cka_tokens", type=int, default=64)

    # debug
    ap.add_argument("--debug_bad_inputs", action="store_true", help="image token count 이상 시 입력 디버그 출력")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if LlavaModelClass is None:
        raise RuntimeError("Could not import LLaVA model class. Update transformers.")

    base_dir = Path(args.base_dir).resolve()
    paths = setup_hf_cache(base_dir)

    adapter_dir = Path(args.adapter_dir).resolve()
    placement_name = adapter_dir.parts[-2] if adapter_dir.parts[-1] == "final" else adapter_dir.parts[-1]

    out_csv = Path(args.out_csv).resolve() if args.out_csv else (adapter_dir / f"eval_{placement_name}.csv")
    out_summary = Path(args.out_summary).resolve() if args.out_summary else (adapter_dir / f"summary_{placement_name}.json")

    want_prompt_diag = (not args.no_prompt_diag)

    # processor는 adapter_dir 우선 (학습과 일치)
    try:
        processor = AutoProcessor.from_pretrained(str(adapter_dir), cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]))
    except Exception:
        processor = AutoProcessor.from_pretrained(args.model_name, cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]))

    tokenizer = processor.tokenizer
    image_processor = processor.image_processor

    # Base model load
    model_kwargs = dict(
        cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]),
        device_map=args.device_map,
    )
    if args.load_in_4bit:
        if not HAS_BNB:
            raise RuntimeError("BitsAndBytesConfig not available.")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = bnb
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    base_model = LlavaModelClass.from_pretrained(args.model_name, **model_kwargs)
    base_model.eval()

    # pad/image 안전화 + image_token_index 동기화
    ensure_safe_pad(tokenizer, base_model)
    sync_image_token_index(base_model, tokenizer)

    # load adapter
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    image_token_id = int(base_model.config.image_token_index)
    if tokenizer.pad_token_id == image_token_id:
        raise RuntimeError(
            f"pad_token_id == image_token_id ({image_token_id}). "
            f"Tokenizer/model mismatch still present; check ensure_safe_pad()."
        )

    print("placement:", placement_name)
    print("pad_token_id:", tokenizer.pad_token_id, "| image_token_id:", image_token_id)
    print("prompt_diag:", want_prompt_diag)

    # Dataset load
    dl_cfg = DownloadConfig(resume_download=True, max_retries=50)
    ds = load_dataset(
        args.dataset_name,
        split=f"{args.split}[:{args.max_samples}]",
        cache_dir=str(paths["HF_DATASETS_CACHE"]),
        download_config=dl_cfg,
    )
    ds = ds.filter(lambda ex: ex.get("image", None) is not None and isinstance(ex.get("question", None), str))
    print("Eval rows:", len(ds))

    # pool for swaps
    images_pool = [ex["image"].convert("RGB") for ex in ds]
    if len(images_pool) < max(10, args.swap_k):
        raise RuntimeError("Too few images for swaps. Increase max_samples.")

    records: List[Record] = []

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fcsv:
        # header
        dummy = Record(
            dataset_idx=0, visual_cued=0, question="",
            gt_answer="",
            gt_answerable=1,
            ans_orig="", ans_blank="", same_orig_blank=0,
            swap_k=0, swap_same_rate=0.0,
            mdi_orig=0.0, aei_t_orig=0.0, aei_o_orig=0.0, at_orig=0.0, ao_orig=0.0,
            mdi_blank=0.0, aei_t_blank=0.0, aei_o_blank=0.0, at_blank=0.0, ao_blank=0.0,
            effrank_text_orig=0.0, effrank_image_orig=0.0, cka_ti_orig=0.0,
            effrank_text_blank=0.0, effrank_image_blank=0.0, cka_ti_blank=0.0,
        )
        fieldnames = list(asdict(dummy).keys())
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(ds)):
            ex = ds[i]
            q = ex.get("question", "")
            vcue = is_visual_cued(q)

            # GT (있으면)
            gt_ans = choose_vizwiz_target(ex)
            gt_answerable = int(ex.get("answerable", 1)) if ex.get("answerable", None) is not None else 1

            img0 = ex["image"].convert("RGB")
            img_blank = blank_like(img0)

            # orig generation + attn
            ans0, attn0, prompt_ids0 = generate_answer(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image=img0,
                question=q,
                max_new_tokens=args.max_new_tokens,
                want_attn=True,
                max_length=args.max_length,
                debug=args.debug_bad_inputs,
            )
            m0 = compute_mdi_aei(attn0, prompt_ids0, image_token_id)

            # blank generation + attn
            ansb, attnb, prompt_idsb = generate_answer(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image=img_blank,
                question=q,
                max_new_tokens=args.max_new_tokens,
                want_attn=True,
                max_length=args.max_length,
                debug=args.debug_bad_inputs,
            )
            mb = compute_mdi_aei(attnb, prompt_idsb, image_token_id)

            same_ob = 1 if (ans0 == ansb and len(ans0) > 0) else 0

            # swaps (answer only)
            swap_ids = random.sample(range(len(images_pool)), k=min(args.swap_k, len(images_pool)))
            same_cnt = 0
            for sid in swap_ids:
                img_sw = images_pool[sid]
                ans_sw, _, _ = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    image=img_sw,
                    question=q,
                    max_new_tokens=args.max_new_tokens,
                    want_attn=False,
                    max_length=args.max_length,
                    debug=args.debug_bad_inputs,
                )
                if ans_sw == ans0 and len(ans0) > 0:
                    same_cnt += 1
            same_rate = same_cnt / len(swap_ids)

            # prompt diagnostics (hidden-state 기반)
            if want_prompt_diag:
                s0 = compute_prompt_stats(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    image=img0,
                    question=q,
                    image_token_id=image_token_id,
                    max_length=args.max_length,
                    rank_tokens=args.rank_tokens,
                    cka_tokens=args.cka_tokens,
                    debug=args.debug_bad_inputs,
                )
                sb = compute_prompt_stats(
                    model=model,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    image=img_blank,
                    question=q,
                    image_token_id=image_token_id,
                    max_length=args.max_length,
                    rank_tokens=args.rank_tokens,
                    cka_tokens=args.cka_tokens,
                    debug=args.debug_bad_inputs,
                )
                eff_t0, eff_i0, cka0 = s0["effrank_text"], s0["effrank_image"], s0["cka_text_image"]
                eff_tb, eff_ib, ckab = sb["effrank_text"], sb["effrank_image"], sb["cka_text_image"]
            else:
                eff_t0 = eff_i0 = cka0 = float("nan")
                eff_tb = eff_ib = ckab = float("nan")

            rec = Record(
                dataset_idx=i,
                visual_cued=vcue,
                question=q,

                gt_answer=gt_ans,
                gt_answerable=gt_answerable,

                ans_orig=ans0,
                ans_blank=ansb,
                same_orig_blank=same_ob,

                swap_k=len(swap_ids),
                swap_same_rate=float(same_rate),

                mdi_orig=float(m0["MDI"]),
                aei_t_orig=float(m0["AEI_T"]),
                aei_o_orig=float(m0["AEI_O"]),
                at_orig=float(m0["AT"]),
                ao_orig=float(m0["AO"]),

                mdi_blank=float(mb["MDI"]),
                aei_t_blank=float(mb["AEI_T"]),
                aei_o_blank=float(mb["AEI_O"]),
                at_blank=float(mb["AT"]),
                ao_blank=float(mb["AO"]),

                effrank_text_orig=float(eff_t0),
                effrank_image_orig=float(eff_i0),
                cka_ti_orig=float(cka0),

                effrank_text_blank=float(eff_tb),
                effrank_image_blank=float(eff_ib),
                cka_ti_blank=float(ckab),
            )

            records.append(rec)
            writer.writerow(asdict(rec))
            fcsv.flush()

            if (i + 1) % 10 == 0:
                print(f"processed {i+1}/{len(ds)}")

    # summary
    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float)) and not (x != x)]
        return float(sum(xs) / len(xs)) if len(xs) > 0 else float("nan")

    summary = {
        "placement": placement_name,
        "n": len(records),
        "n_visual": sum(r.visual_cued for r in records),
        "n_non_visual": sum(1 for r in records if r.visual_cued == 0),

        "swap_same_rate_mean": mean([r.swap_same_rate for r in records]),
        "swap_same_rate_mean_visual": mean([r.swap_same_rate for r in records if r.visual_cued == 1]),
        "swap_same_rate_mean_non_visual": mean([r.swap_same_rate for r in records if r.visual_cued == 0]),

        "same_orig_blank_rate": mean([r.same_orig_blank for r in records]),
        "same_orig_blank_rate_visual": mean([r.same_orig_blank for r in records if r.visual_cued == 1]),
        "same_orig_blank_rate_non_visual": mean([r.same_orig_blank for r in records if r.visual_cued == 0]),

        "MDI_orig_mean": mean([r.mdi_orig for r in records]),
        "MDI_orig_mean_visual": mean([r.mdi_orig for r in records if r.visual_cued == 1]),
        "MDI_orig_mean_non_visual": mean([r.mdi_orig for r in records if r.visual_cued == 0]),

        "AEI_T_orig_mean": mean([r.aei_t_orig for r in records]),
        "AEI_T_orig_mean_visual": mean([r.aei_t_orig for r in records if r.visual_cued == 1]),
        "AEI_T_orig_mean_non_visual": mean([r.aei_t_orig for r in records if r.visual_cued == 0]),
    }

    if want_prompt_diag:
        summary.update({
            "effrank_text_orig_mean": mean([r.effrank_text_orig for r in records]),
            "effrank_image_orig_mean": mean([r.effrank_image_orig for r in records]),
            "cka_ti_orig_mean": mean([r.cka_ti_orig for r in records]),
        })

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved CSV:", str(out_csv))
    print("Saved summary:", str(out_summary))


if __name__ == "__main__":
    main()
