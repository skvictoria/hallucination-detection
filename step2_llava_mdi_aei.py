import os
import re
import csv
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
from PIL import Image

from datasets import load_dataset, DownloadConfig
from transformers import AutoProcessor

# LLaVA class import (transformers 버전에 따라 다름)
LlavaModelClass = None
try:
    from transformers import LlavaForConditionalGeneration  # 최신 transformers
    LlavaModelClass = LlavaForConditionalGeneration
except Exception:
    try:
        from transformers import AutoModelForVision2Seq  # 일부 버전
        LlavaModelClass = AutoModelForVision2Seq
    except Exception:
        LlavaModelClass = None

# bitsandbytes(4bit) optional
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
    hf_transformers = base_dir / "hf_hub"  # keep transformers under hub cache too (optional)
    for p in [hf_home, hf_hub, hf_datasets, hf_transformers]:
        p.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_hub)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_transformers)

    # compatibility
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)

    # slow network timeouts
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
    q = question.lower()
    return 1 if any(re.search(p, q) for p in VISUAL_CUE_PATTERNS) else 0


# -------------------------
# Image perturbations
# -------------------------
def blank_like(img: Image.Image) -> Image.Image:
    return Image.new("RGB", img.size, color=(128, 128, 128))


# -------------------------
# Text normalization (for answer equality)
# -------------------------
def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^(assistant:|answer:)\s*", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------------
# image token id (prefer model.config.image_token_index)
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
# Prompt builder (LLaVA)
# -------------------------
def build_prompt(question: str) -> str:
    return (
        "USER: <image>\n"
        f"{question}\n"
        "Answer with a short phrase or a single word. Do not explain.\n"
        "ASSISTANT:"
    )


# -------------------------
# Token mask builder for text vs image tokens in expanded context
# -------------------------
def build_context_masks_from_expanded_len(
    prompt_input_ids: List[int],
    image_token_id: int,
    context_len_expanded: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    LLaVA는 prompt_input_ids 안의 <image> placeholder(보통 1개)를
    내부적으로 여러 개의 image tokens로 확장한 뒤 LLM input에 merge한다.

    여기서는 expanded context length(=LLM hidden length 또는 generate attn key_len)를 보고
    placeholder 확장 개수 m을 역으로 추정해서 text/image mask를 만든다.
    """
    L = len(prompt_input_ids)
    positions = [i for i, tid in enumerate(prompt_input_ids) if tid == image_token_id]
    n = len(positions)
    if n == 0:
        raise RuntimeError("No <image> token found in prompt_input_ids. Ensure prompt includes <image>.")

    numerator = context_len_expanded - L + n
    if numerator % n != 0:
        m = int(round(numerator / n))
    else:
        m = numerator // n

    if m <= 0:
        raise RuntimeError(f"Invalid inferred image token count m={m}. context_len={context_len_expanded}, L={L}, n={n}")

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

    if cur != context_len_expanded:
        # 내부 구현 차이로 추가 토큰이 들어갈 수 있어 남는 부분은 text로 처리
        if cur < context_len_expanded:
            text_mask[cur:context_len_expanded] = True
        text_mask = text_mask[:context_len_expanded]
        image_mask = image_mask[:context_len_expanded]

    num_text = int(text_mask.sum().item())
    num_img = int(image_mask.sum().item())
    return text_mask, image_mask, num_text, num_img


# -------------------------
# MDI / AEI computation from generate attentions
# -------------------------
def compute_mdi_aei_from_generate_attentions(
    gen_attentions,               # tuple[gen_step] -> tuple[layer] -> (bs, heads, 1, key_len)
    prompt_input_ids: List[int],
    image_token_id: int,
) -> Dict[str, float]:
    if gen_attentions is None or len(gen_attentions) == 0:
        return {
            "AT": float("nan"),
            "AO": float("nan"),
            "MDI": float("nan"),
            "AEI_T": float("nan"),
            "AEI_O": float("nan"),
            "context_len": float("nan"),
            "num_text_tokens": float("nan"),
            "num_image_tokens": float("nan"),
        }

    first_step = gen_attentions[0]
    first_layer = first_step[0]
    context_len = int(first_layer.shape[-1])

    text_mask, image_mask, num_text, num_img = build_context_masks_from_expanded_len(
        prompt_input_ids=prompt_input_ids,
        image_token_id=image_token_id,
        context_len_expanded=context_len,
    )
    num_total = num_text + num_img

    A_text = 0.0
    A_img = 0.0

    for step_attn in gen_attentions:
        for layer_attn in step_attn:
            a = layer_attn[0, :, 0, :context_len]      # [heads, context_len]
            vec = a.sum(dim=0)                          # [context_len]
            A_text += float(vec[text_mask].sum().item())
            A_img += float(vec[image_mask].sum().item())

    total = A_text + A_img
    if total <= 0 or num_img == 0 or num_text == 0:
        return {
            "AT": float("nan"),
            "AO": float("nan"),
            "MDI": float("nan"),
            "AEI_T": float("nan"),
            "AEI_O": float("nan"),
            "context_len": float(context_len),
            "num_text_tokens": float(num_text),
            "num_image_tokens": float(num_img),
        }

    AT = A_text / total
    AO = A_img / total
    mdi = (AT / num_text) / (AO / num_img)

    QT = num_text / num_total
    QO = num_img / num_total
    aei_t = AT / QT
    aei_o = AO / QO

    return {
        "AT": float(AT),
        "AO": float(AO),
        "MDI": float(mdi),
        "AEI_T": float(aei_t),
        "AEI_O": float(aei_o),
        "context_len": float(context_len),
        "num_text_tokens": float(num_text),
        "num_image_tokens": float(num_img),
    }


# -------------------------
# Representation metrics (Effective Rank / CKA / InfoNCE proxy)
# -------------------------
def effective_rank_from_tokens(H: torch.Tensor, eps: float = 1e-12) -> float:
    """
    H: [n_tokens, d]
    Compute effective rank via eigen-spectrum entropy of Gram (H H^T).
    """
    if H is None or H.numel() == 0:
        return float("nan")
    n = H.shape[0]
    if n < 2:
        return 1.0

    X = H.float()
    X = X - X.mean(dim=0, keepdim=True)
    G = X @ X.t()  # [n, n]
    # numerical stability
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
    """
    X: [n, d], Y: [n, d]  (n must match)
    Linear CKA using centered Gram matrices.
    """
    if X is None or Y is None:
        return float("nan")
    if X.numel() == 0 or Y.numel() == 0:
        return float("nan")
    if X.shape[0] != Y.shape[0]:
        return float("nan")
    n = X.shape[0]
    if n < 2:
        return float("nan")

    Xf = X.float()
    Yf = Y.float()
    Xf = Xf - Xf.mean(dim=0, keepdim=True)
    Yf = Yf - Yf.mean(dim=0, keepdim=True)

    K = Xf @ Xf.t()
    L = Yf @ Yf.t()

    Hc = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones((n, n), device=K.device)
    Kc = Hc @ K @ Hc
    Lc = Hc @ L @ Hc

    hsic = torch.sum(Kc * Lc)
    denom = torch.sqrt(torch.sum(Kc * Kc) * torch.sum(Lc * Lc) + eps)
    return float((hsic / (denom + eps)).item())


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.float()
    b = b.float()
    na = torch.norm(a) + eps
    nb = torch.norm(b) + eps
    return float((torch.dot(a, b) / (na * nb)).item())


def info_nce_mi_estimate(
    z_t: torch.Tensor,
    z_v_pos: torch.Tensor,
    z_v_negs: torch.Tensor,  # [K, D]
    tau: float = 0.07,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    InfoNCE MI lower bound (up to constant):
      loss = -log exp(sim_pos/tau) / sum exp(sim_all/tau)
      mi_est ≈ log(K+1) - loss
    Return: (mi_est, loss)
    """
    zt = z_t.float()
    zpos = z_v_pos.float()
    znegs = z_v_negs.float()

    def normed(x):
        return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

    zt_n = normed(zt.unsqueeze(0))             # [1, D]
    zall = torch.cat([zpos.unsqueeze(0), znegs], dim=0)  # [K+1, D]
    zall_n = normed(zall)

    sims = (zt_n @ zall_n.t()).squeeze(0)      # [K+1]
    logits = sims / tau
    # stable log-softmax
    lse = torch.logsumexp(logits, dim=0)
    loss = -(logits[0] - lse)
    mi_est = math.log(zall.shape[0]) - float(loss.item())
    return float(mi_est), float(loss.item())


# -------------------------
# Access vision tower + projector for precomputing vision embeddings
# -------------------------
def get_vision_tower(model):
    if hasattr(model, "get_vision_tower"):
        vt = model.get_vision_tower()
        # sometimes list/tuple
        if isinstance(vt, (list, tuple)):
            vt = vt[0]
        return vt
    if hasattr(model, "vision_tower"):
        vt = model.vision_tower
        if isinstance(vt, (list, tuple)):
            vt = vt[0]
        return vt
    raise RuntimeError("Could not find vision tower in model.")


def get_mm_projector(model):
    for name in ["multi_modal_projector", "mm_projector", "vision_proj"]:
        if hasattr(model, name):
            return getattr(model, name)
    # 일부 버전은 model.model.mm_projector 형태일 수도 있음
    if hasattr(model, "model"):
        for name in ["multi_modal_projector", "mm_projector", "vision_proj"]:
            if hasattr(model.model, name):
                return getattr(model.model, name)
    raise RuntimeError("Could not find multi-modal projector in model.")


@torch.no_grad()
def precompute_vision_embeds(
    images: List[Image.Image],
    image_processor,
    model,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Returns z_v_pool: [N, D] on CPU (float16)
    z_v is mean pooled projected vision tokens (projector output space = LLM hidden size).
    """
    vt = get_vision_tower(model)
    proj = get_mm_projector(model)

    # pick device where vision tower lives
    vt_dev = next(vt.parameters()).device
    proj_dev = next(proj.parameters()).device
    dev = vt_dev if vt_dev == proj_dev else vt_dev

    z_list = []
    for i in range(0, len(images), batch_size):
        batch_imgs = [im.convert("RGB") for im in images[i:i+batch_size]]
        pv = image_processor(images=batch_imgs, return_tensors="pt")["pixel_values"].to(dev)

        # vision forward
        try:
            vout = vt(pixel_values=pv, output_hidden_states=False, return_dict=True)
            feats = vout.last_hidden_state
        except Exception:
            feats = vt(pv).last_hidden_state

        # projector
        feats = feats.to(proj_dev)
        pfeat = proj(feats)  # [B, Nv, D]
        z = pfeat.mean(dim=1)  # [B, D]
        z_list.append(z.detach().to("cpu", dtype=torch.float16))

    return torch.cat(z_list, dim=0)  # [N, D]


# -------------------------
# Separated processing for stability (no processor(images,text))
# -------------------------
def make_inputs_separated(
    tokenizer,
    image_processor,
    model,
    image: Image.Image,
    prompt: str,
    device: str,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    pv = image_processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # safety: ensure exactly one image placeholder token in input_ids
    image_token_id = get_image_token_id(model, tokenizer)
    img_cnt = int((input_ids[0] == image_token_id).sum().item())
    if img_cnt != 1:
        raise ValueError(f"Bad <image> token count in prompt ids: {img_cnt} (expected 1).")

    batch = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "pixel_values": pv.to(device),
    }
    return batch


# -------------------------
# Generation wrapper
# -------------------------
@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    question: str,
    device: str,
    max_new_tokens: int = 16,
    output_attentions: bool = False,
    max_length: int = 512,
) -> Tuple[str, Optional[Any], List[int]]:
    prompt = build_prompt(question)
    inputs = make_inputs_separated(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
        image=image,
        prompt=prompt,
        device=device,
        max_length=max_length,
    )

    prompt_input_ids = inputs["input_ids"][0].tolist()

    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        return_dict_in_generate=True,
        output_attentions=output_attentions,
        output_scores=False,
    )

    seq = gen_out.sequences[0]
    in_len = inputs["input_ids"].shape[1]
    out_ids = seq[in_len:]

    text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    text_norm = normalize_answer(text)

    attn = None
    if output_attentions:
        if hasattr(gen_out, "attentions") and gen_out.attentions is not None:
            attn = gen_out.attentions
        elif hasattr(gen_out, "decoder_attentions") and gen_out.decoder_attentions is not None:
            attn = gen_out.decoder_attentions

    return text_norm, attn, prompt_input_ids


# -------------------------
# Forward to extract hidden + masks (orig/blank) for effrank/CKA
# -------------------------
@torch.no_grad()
def forward_hidden_stats(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    question: str,
    device: str,
    image_token_id: int,
    cka_tokens: int = 64,
    max_length: int = 512,
) -> Dict[str, float]:
    """
    Returns:
      effrank_text, effrank_img,
      cka_tv,
      pooled_text_cos_pooled_img (cos sim)
    """
    prompt = build_prompt(question)
    batch = make_inputs_separated(
        tokenizer=tokenizer,
        image_processor=image_processor,
        model=model,
        image=image,
        prompt=prompt,
        device=device,
        max_length=max_length,
    )
    prompt_ids = batch["input_ids"][0].tolist()

    out = model(
        **batch,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states
    if hs is None or len(hs) == 0:
        return {
            "effrank_text": float("nan"),
            "effrank_img": float("nan"),
            "cka_tv": float("nan"),
            "cos_tv": float("nan"),
            "seq_len": float("nan"),
            "num_text_tokens": float("nan"),
            "num_img_tokens": float("nan"),
        }

    last = hs[-1][0]  # [seq_len, d]
    seq_len = int(last.shape[0])

    text_mask, img_mask, n_text, n_img = build_context_masks_from_expanded_len(
        prompt_input_ids=prompt_ids,
        image_token_id=image_token_id,
        context_len_expanded=seq_len,
    )

    if n_text == 0 or n_img == 0:
        return {
            "effrank_text": float("nan"),
            "effrank_img": float("nan"),
            "cka_tv": float("nan"),
            "cos_tv": float("nan"),
            "seq_len": float(seq_len),
            "num_text_tokens": float(n_text),
            "num_img_tokens": float(n_img),
        }

    Ht = last[text_mask]  # [n_text, d]
    Hv = last[img_mask]   # [n_img, d]

    er_t = effective_rank_from_tokens(Ht)
    er_v = effective_rank_from_tokens(Hv)

    # pooled embeddings (for MI proxy etc.)
    zt = Ht.mean(dim=0)
    zv = Hv.mean(dim=0)
    cos_tv = cosine_sim(zt, zv)

    # CKA between text tokens and image tokens: subsample same n
    m = min(Ht.shape[0], Hv.shape[0], cka_tokens)
    if m >= 2:
        Xt = Ht[:m]
        Xv = Hv[:m]
        cka_tv = linear_cka(Xt, Xv)
    else:
        cka_tv = float("nan")

    return {
        "effrank_text": float(er_t),
        "effrank_img": float(er_v),
        "cka_tv": float(cka_tv),
        "cos_tv": float(cos_tv),
        "seq_len": float(seq_len),
        "num_text_tokens": float(n_text),
        "num_img_tokens": float(n_img),
        # return pooled too (caller can use)
        "_zt": zt.detach().to("cpu", dtype=torch.float32),
        "_zv": zv.detach().to("cpu", dtype=torch.float32),
    }


def pooled_from_hidden_stats(stats: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    zt = stats.get("_zt", None)
    zv = stats.get("_zv", None)
    if isinstance(zt, torch.Tensor) and isinstance(zv, torch.Tensor):
        return zt, zv
    return None, None


# -------------------------
# Output record
# -------------------------
@dataclass
class Record:
    dataset_idx: int
    visual_cued: int
    question: str

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
    context_len_orig: int
    num_text_tokens_orig: int
    num_image_tokens_orig: int

    mdi_blank: float
    aei_t_blank: float
    aei_o_blank: float
    at_blank: float
    ao_blank: float

    # representation-based (new)
    effrank_text_orig: float
    effrank_img_orig: float
    effrank_text_blank: float
    effrank_img_blank: float

    cka_tv_orig: float
    cos_tv_orig: float

    cka_text_orig_blank: float
    cka_img_orig_blank: float

    # MI proxy via InfoNCE against precomputed vision pool (new)
    mi_nce_orig: float
    nce_loss_orig: float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--dataset_name", type=str, default="derek-thomas/ScienceQA")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--max_samples", type=int, default=200)

    ap.add_argument("--swap_k", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    # repr metrics
    ap.add_argument("--cka_tokens", type=int, default=64)
    ap.add_argument("--mi_neg_k", type=int, default=16)
    ap.add_argument("--mi_tau", type=float, default=0.07)
    ap.add_argument("--vision_pool_bs", type=int, default=8)

    # unexpected dominance filter (저장용)
    ap.add_argument("--save_unexpected", action="store_true")
    ap.add_argument("--th_same_rate", type=float, default=0.85)
    ap.add_argument("--th_mdi", type=float, default=3.0)
    ap.add_argument("--require_visual_cue", action="store_true")
    ap.add_argument("--max_save_cases", type=int, default=50)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = Path(args.base_dir).resolve()
    paths = setup_hf_cache(base_dir)

    out_csv = base_dir / "step2_llava_mdi_aei_repr.csv"
    out_json = base_dir / "step2_unexpected_cases.json"
    case_dir = base_dir / "step2_unexpected_cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if LlavaModelClass is None:
        raise RuntimeError("Could not import LLaVA model class from transformers. Update transformers package.")

    # Processor
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]),
        patch_size=32,
        vision_feature_select_strategy="default",
    )
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor

    # Model
    model_kwargs = dict(
        cache_dir=str(paths["HF_TRANSFORMERS_CACHE"]),
        device_map="auto",
    )
    if args.load_in_4bit:
        if not HAS_BNB:
            raise RuntimeError("BitsAndBytesConfig not available. Install bitsandbytes + 최신 transformers.")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = quant_cfg
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    model = LlavaModelClass.from_pretrained(args.model_name, **model_kwargs)
    model.eval()

    image_token_id = get_image_token_id(model, tokenizer)
    print("Image token id:", image_token_id)

    # Dataset load
    download_config = DownloadConfig(resume_download=True, max_retries=50)
    split_expr = f"{args.split}[:{args.max_samples}]"
    ds = load_dataset(
        args.dataset_name,
        split=split_expr,
        cache_dir=str(paths["HF_DATASETS_CACHE"]),
        download_config=download_config,
    )
    print(f"Loaded: {args.dataset_name} {split_expr} rows: {len(ds)}")
    print("Columns:", ds.column_names)

    # image pool for swaps
    images_pool: List[Image.Image] = []
    valid_indices: List[int] = []
    for i in range(len(ds)):
        img = ds[i].get("image", None)
        if isinstance(img, Image.Image):
            images_pool.append(img.convert("RGB"))
            valid_indices.append(i)
    print("Usable samples with image:", len(images_pool))
    if len(images_pool) < max(10, args.swap_k):
        raise RuntimeError("Too few usable images for swaps. Increase max_samples.")

    # Precompute vision embeddings for MI proxy (projector space)
    print("Precomputing vision embeddings (projector space) for MI proxy...")
    z_v_pool = precompute_vision_embeds(
        images=images_pool,
        image_processor=image_processor,
        model=model,
        batch_size=args.vision_pool_bs,
    )  # [N, D] CPU float16
    print("z_v_pool:", tuple(z_v_pool.shape), z_v_pool.dtype)

    records: List[Record] = []
    unexpected_saved = 0
    unexpected_list: List[Dict[str, Any]] = []

    # header
    fieldnames = list(asdict(Record(
        dataset_idx=0, visual_cued=0, question="",
        ans_orig="", ans_blank="", same_orig_blank=0,
        swap_k=0, swap_same_rate=0.0,
        mdi_orig=0.0, aei_t_orig=0.0, aei_o_orig=0.0, at_orig=0.0, ao_orig=0.0,
        context_len_orig=0, num_text_tokens_orig=0, num_image_tokens_orig=0,
        mdi_blank=0.0, aei_t_blank=0.0, aei_o_blank=0.0, at_blank=0.0, ao_blank=0.0,
        effrank_text_orig=0.0, effrank_img_orig=0.0, effrank_text_blank=0.0, effrank_img_blank=0.0,
        cka_tv_orig=0.0, cos_tv_orig=0.0,
        cka_text_orig_blank=0.0, cka_img_orig_blank=0.0,
        mi_nce_orig=0.0, nce_loss_orig=0.0,
    )).keys())

    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for pool_i, ds_i in enumerate(valid_indices):
            ex = ds[ds_i]
            q = ex.get("question", "")
            if not isinstance(q, str) or len(q.strip()) == 0:
                continue

            vcue = is_visual_cued(q)
            img0 = images_pool[pool_i]
            img_blank = blank_like(img0)

            # --- orig generate + attentions (MDI/AEI)
            ans0, attn0, prompt_ids0 = generate_answer(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image=img0,
                question=q,
                device=device,
                max_new_tokens=args.max_new_tokens,
                output_attentions=True,
                max_length=args.max_length,
            )
            mdi0 = compute_mdi_aei_from_generate_attentions(
                gen_attentions=attn0,
                prompt_input_ids=prompt_ids0,
                image_token_id=image_token_id,
            )

            # --- blank generate + attentions
            ansb, attnb, prompt_idsb = generate_answer(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image=img_blank,
                question=q,
                device=device,
                max_new_tokens=args.max_new_tokens,
                output_attentions=True,
                max_length=args.max_length,
            )
            mdib = compute_mdi_aei_from_generate_attentions(
                gen_attentions=attnb,
                prompt_input_ids=prompt_idsb,
                image_token_id=image_token_id,
            )

            same_ob = 1 if ans0 == ansb and len(ans0) > 0 else 0

            # --- swaps (answers only)
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
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    output_attentions=False,
                    max_length=args.max_length,
                )
                if ans_sw == ans0 and len(ans0) > 0:
                    same_cnt += 1
            same_rate = same_cnt / len(swap_ids)

            # --- representation stats (forward hidden) for orig/blank
            st0 = forward_hidden_stats(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image=img0,
                question=q,
                device=device,
                image_token_id=image_token_id,
                cka_tokens=args.cka_tokens,
                max_length=args.max_length,
            )
            stb = forward_hidden_stats(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image=img_blank,
                question=q,
                device=device,
                image_token_id=image_token_id,
                cka_tokens=args.cka_tokens,
                max_length=args.max_length,
            )

            # CKA(orig vs blank) within modalities (subsample to same n)
            # text
            cka_text_ob = float("nan")
            cka_img_ob = float("nan")
            # We need token matrices, but we did not return full token matrices to keep memory low.
            # Approximate with pooled embedding similarity as fallback if you want.
            # Here we compute CKA using pooled embeddings is not meaningful, so we compute cosine instead.
            # If you want true token-level CKA(orig vs blank), enable "return_token_mats" and store them.
            # For now: keep these as NaN to avoid misleading.
            # (If you want, I can give a version that returns token matrices and computes true CKA.)

            # MI proxy via InfoNCE: z_t from orig hidden (text pooled), positives/negatives from precomputed vision pool
            zt0, _ = pooled_from_hidden_stats(st0)
            mi_est = float("nan")
            nce_loss = float("nan")
            if isinstance(zt0, torch.Tensor):
                # positive is current image
                zpos = z_v_pool[pool_i].to(dtype=torch.float32)
                # negatives random
                neg_k = min(args.mi_neg_k, len(images_pool) - 1)
                neg_idx = random.sample([j for j in range(len(images_pool)) if j != pool_i], k=neg_k)
                znegs = z_v_pool[neg_idx].to(dtype=torch.float32)  # [K, D]
                mi_est, nce_loss = info_nce_mi_estimate(
                    z_t=zt0,
                    z_v_pos=zpos,
                    z_v_negs=znegs,
                    tau=args.mi_tau,
                )

            rec = Record(
                dataset_idx=int(ds_i),
                visual_cued=int(vcue),
                question=q,

                ans_orig=ans0,
                ans_blank=ansb,
                same_orig_blank=int(same_ob),

                swap_k=int(len(swap_ids)),
                swap_same_rate=float(same_rate),

                mdi_orig=float(mdi0["MDI"]),
                aei_t_orig=float(mdi0["AEI_T"]),
                aei_o_orig=float(mdi0["AEI_O"]),
                at_orig=float(mdi0["AT"]),
                ao_orig=float(mdi0["AO"]),
                context_len_orig=int(mdi0["context_len"]) if not math.isnan(float(mdi0["context_len"])) else -1,
                num_text_tokens_orig=int(mdi0["num_text_tokens"]) if not math.isnan(float(mdi0["num_text_tokens"])) else -1,
                num_image_tokens_orig=int(mdi0["num_image_tokens"]) if not math.isnan(float(mdi0["num_image_tokens"])) else -1,

                mdi_blank=float(mdib["MDI"]),
                aei_t_blank=float(mdib["AEI_T"]),
                aei_o_blank=float(mdib["AEI_O"]),
                at_blank=float(mdib["AT"]),
                ao_blank=float(mdib["AO"]),

                effrank_text_orig=float(st0["effrank_text"]),
                effrank_img_orig=float(st0["effrank_img"]),
                effrank_text_blank=float(stb["effrank_text"]),
                effrank_img_blank=float(stb["effrank_img"]),

                cka_tv_orig=float(st0["cka_tv"]),
                cos_tv_orig=float(st0["cos_tv"]),

                cka_text_orig_blank=float(cka_text_ob),
                cka_img_orig_blank=float(cka_img_ob),

                mi_nce_orig=float(mi_est),
                nce_loss_orig=float(nce_loss),
            )

            records.append(rec)
            writer.writerow(asdict(rec))
            fcsv.flush()

            # unexpected dominance saving
            if args.save_unexpected:
                consider = True
                if args.require_visual_cue and vcue == 0:
                    consider = False
                if consider:
                    if same_ob == 1 and same_rate >= args.th_same_rate and rec.mdi_orig >= args.th_mdi:
                        if unexpected_saved < args.max_save_cases:
                            cdir = case_dir / f"idx_{ds_i}"
                            cdir.mkdir(parents=True, exist_ok=True)
                            img0.save(cdir / "orig.png")
                            img_blank.save(cdir / "blank.png")
                            for j, sid in enumerate(swap_ids[:min(4, len(swap_ids))]):
                                images_pool[sid].save(cdir / f"swap_{j+1:02d}.png")

                            report = {
                                "dataset_idx": int(ds_i),
                                "visual_cued": int(vcue),
                                "question": q,
                                "ans_orig": ans0,
                                "ans_blank": ansb,
                                "swap_same_rate": float(same_rate),
                                "MDI_orig": float(rec.mdi_orig),
                                "AEI_T_orig": float(rec.aei_t_orig),
                                "AT_orig": float(rec.at_orig),
                                "AO_orig": float(rec.ao_orig),
                                "effrank_text_orig": float(rec.effrank_text_orig),
                                "effrank_img_orig": float(rec.effrank_img_orig),
                                "cka_tv_orig": float(rec.cka_tv_orig),
                                "mi_nce_orig": float(rec.mi_nce_orig),
                            }
                            with open(cdir / "report.json", "w") as fp:
                                json.dump(report, fp, indent=2)
                            unexpected_list.append(report)
                            unexpected_saved += 1

            if (len(records) % 10) == 0:
                print(f"Processed {len(records)} / {len(valid_indices)} | saved_unexpected={unexpected_saved}")

    if args.save_unexpected:
        with open(out_json, "w") as f:
            json.dump(unexpected_list, f, indent=2)
        print("Saved unexpected JSON:", str(out_json))
        print("Saved unexpected dir:", str(case_dir))

    print("Done. CSV:", str(out_csv))


if __name__ == "__main__":
    main()
