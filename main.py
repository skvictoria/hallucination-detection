# main.py
import os
from pathlib import Path

# -------------------------
# 0) Cache dir 고정 (HF 모델/데이터셋 다운로드 위치)
# -------------------------
BASE_DIR = Path("/home/hice1/skim3513/scratch/hallucination-detection").resolve()
HF_HOME = BASE_DIR / "hf_home"
HF_HUB_CACHE = BASE_DIR / "hf_hub"
HF_DATASETS_CACHE = BASE_DIR / "hf_datasets"
HF_TRANSFORMERS_CACHE = BASE_DIR / "hf_transformers"

for p in [HF_HOME, HF_HUB_CACHE, HF_DATASETS_CACHE, HF_TRANSFORMERS_CACHE]:
    p.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_TRANSFORMERS_CACHE)

# (호환용)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_CACHE)

# 다운로드가 느릴 때 hub timeout을 넉넉히
os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

# -------------------------
# 1) Imports (env 변수 설정 이후에 import 권장)
# -------------------------
import random
import torch
import torch.nn.functional as F
from PIL import Image
import aiohttp

from datasets import load_dataset, DownloadConfig
from transformers import ViltProcessor, ViltForQuestionAnswering

# -------------------------
# 2) Config
# -------------------------
MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"

# "이미지 포함 + HF에서 바로 로드"되는 VQA류 데이터셋
DATASET_NAME = "derek-thomas/ScienceQA"  # image/question/choices 포함 :contentReference[oaicite:2]{index=2}
SPLIT = "validation"
N_SAMPLES = 200          # slice 크기
K_SWAPS = 8              # 이미지 swap 횟수
BATCH_SIZE = 8
SEED = 0

# dominance 판정(느슨하게 시작)
DOMINANCE_SAME_RATE_TH = 0.85  # swap해도 top-1 동일 비율
DOMINANCE_KL_TH = 0.05         # 분포 KL도 작으면 "거의 안 변함"

random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -------------------------
# 3) Load model
# -------------------------
processor = ViltProcessor.from_pretrained(MODEL_NAME, cache_dir=str(HF_TRANSFORMERS_CACHE))
model = ViltForQuestionAnswering.from_pretrained(MODEL_NAME, cache_dir=str(HF_TRANSFORMERS_CACHE)).to(device)
model.eval()

# -------------------------
# 4) Load dataset (timeout/resume/retry)
# -------------------------
download_config = DownloadConfig(resume_download=True, max_retries=50)
storage_options = {
    "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
}

ds = load_dataset(
    DATASET_NAME,
    split=f"{SPLIT}[:{N_SAMPLES}]",
    cache_dir=str(HF_DATASETS_CACHE),
    download_config=download_config,
    storage_options=storage_options,
)

print("Loaded:", DATASET_NAME, SPLIT, "rows:", len(ds))
print("Columns:", ds.column_names)

# -------------------------
# 5) Helpers
# -------------------------
def to_pil(img):
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    # datasets Image 타입은 보통 PIL로 디코드되지만, 혹시 몰라 fallback
    if hasattr(img, "convert"):
        return img.convert("RGB")
    return None

def build_question(ex):
    return ex.get("question", "")

MAX_TEXT_LEN = 40  # ViLT VQA fine-tuned 모델의 텍스트 최대 길이

@torch.no_grad()
def predict_probs(images, questions):
    inputs = processor(
        images=images,
        text=questions,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_LEN,
    ).to(device)

    logits = model(**inputs).logits
    return F.softmax(logits, dim=-1).detach().cpu()

def kl(p, q, eps=1e-12):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return float((p * (p.log() - q.log())).sum().item())

def blank_like(img: Image.Image):
    return Image.new("RGB", img.size, color=(128, 128, 128))

def top_answer(probs_1d):
    tid = int(torch.argmax(probs_1d).item())
    return model.config.id2label[tid], tid, float(probs_1d[tid].item())

# -------------------------
# 6) Build image pool (skip None images)
# -------------------------
valid_indices = []
images_pool = []
questions_pool = []

for i in range(len(ds)):
    img = to_pil(ds[i].get("image", None))
    if img is None:
        continue
    valid_indices.append(i)
    images_pool.append(img)
    questions_pool.append(build_question(ds[i]))

print("Usable samples with image:", len(images_pool))
if len(images_pool) < 10:
    raise RuntimeError("Too few usable image samples. Try increasing N_SAMPLES or use another dataset.")

# -------------------------
# 7) Scan for modality dominance
# -------------------------
dominant = []
sensitive = []

for local_i, idx in enumerate(valid_indices):
    ex = ds[idx]
    q = build_question(ex)
    img0 = to_pil(ex["image"])
    if img0 is None:
        continue

    # original
    p0 = predict_probs([img0], [q])[0]
    ans0, tid0, prob0 = top_answer(p0)

    # blank image (text prior proxy)
    pb = predict_probs([blank_like(img0)], [q])[0]
    ansb, _, probb = top_answer(pb)

    # random swaps
    swap_ids = random.sample(range(len(images_pool)), k=min(K_SWAPS, len(images_pool)))
    same = 0
    kls = []

    for b in range(0, len(swap_ids), BATCH_SIZE):
        batch_ids = swap_ids[b:b+BATCH_SIZE]
        batch_imgs = [images_pool[j] for j in batch_ids]
        batch_qs = [q] * len(batch_imgs)

        psw = predict_probs(batch_imgs, batch_qs)  # [B, L]
        top_ids = torch.argmax(psw, dim=-1).tolist()
        same += sum(1 for t in top_ids if t == tid0)
        for r in range(psw.shape[0]):
            kls.append(kl(p0, psw[r]))

    same_rate = same / len(swap_ids)
    mean_kl = sum(kls) / len(kls)

    item = {
        "dataset_idx": idx,
        "question": q[:300],
        "pred": ans0, "pred_p": prob0,
        "blank_pred": ansb, "blank_p": probb,
        "same_rate": same_rate,
        "mean_kl": mean_kl,
    }

    if same_rate >= DOMINANCE_SAME_RATE_TH and mean_kl <= DOMINANCE_KL_TH:
        dominant.append(item)
    if same_rate <= 0.3 and mean_kl >= 0.2:
        sensitive.append(item)

dominant.sort(key=lambda x: (-x["same_rate"], x["mean_kl"]))
sensitive.sort(key=lambda x: (x["same_rate"], -x["mean_kl"]))

print("\nDominant cases:", len(dominant))
print("Sensitive cases:", len(sensitive))

def show_case(x, title):
    print("\n" + "="*90)
    print(title)
    print("dataset_idx:", x["dataset_idx"])
    print("Q:", x["question"])
    print(f"Pred: {x['pred']} (p={x['pred_p']:.3f})")
    print(f"Blank Pred: {x['blank_pred']} (p={x['blank_p']:.3f})")
    print(f"Swap same-rate={x['same_rate']:.2f}, mean KL={x['mean_kl']:.4f}")

for i in range(min(5, len(dominant))):
    show_case(dominant[i], f"[DOMINANT #{i+1}] image swap hardly changes output")

for i in range(min(5, len(sensitive))):
    show_case(sensitive[i], f"[SENSITIVE #{i+1}] image swap changes output")


# -----------------------------
# 어디에 이미지 저장할지
# -----------------------------
VIS_DIR = Path(BASE_DIR) / "case_viz"
VIS_DIR.mkdir(parents=True, exist_ok=True)

def pixel_grad_norm_single(question: str, image_pil):
    """
    예측 로짓(최대 logit)에 대한 pixel_values gradient norm.
    크면 클수록 이미지에 민감(=이미지를 많이 씀).
    """
    model.zero_grad(set_to_none=True)
    inputs = processor(
        images=image_pil,
        text=question,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=40,
    ).to(device)

    inputs["pixel_values"].requires_grad_(True)

    out = model(**inputs)
    logits = out.logits[0]
    pred_id = int(torch.argmax(logits).item())
    score = logits[pred_id]
    score.backward()

    g = inputs["pixel_values"].grad
    return float(g.norm().detach().cpu().item())

@torch.no_grad()
def analyze_case(ds_idx: int, title: str, n_swaps: int = 8, seed: int = 0):
    """
    - 원본 이미지 + 스왑 이미지들에 대해 top-1 답/확률과 KL 출력
    - 원본/blank 예측 비교
    - pixel gradient norm 비교
    - 이미지 파일 저장
    """
    rnd = random.Random(seed)

    ex = ds[ds_idx]
    q = build_question(ex)
    img0 = to_pil(ex["image"])

    case_dir = VIS_DIR / f"idx_{ds_idx}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # 원본/blank probs
    p0 = predict_probs([img0], [q])[0]
    ans0, tid0, prob0 = top_answer(p0)

    img_blank = blank_like(img0)
    pb = predict_probs([img_blank], [q])[0]
    ansb, tidb, probb = top_answer(pb)

    # 이미지 저장
    img0.save(case_dir / "orig.png")
    img_blank.save(case_dir / "blank.png")

    # 스왑 이미지들 선택
    swap_ids = rnd.sample(range(len(images_pool)), k=min(n_swaps, len(images_pool)))
    swap_imgs = [images_pool[j] for j in swap_ids]
    swap_qs = [q] * len(swap_imgs)

    psw = predict_probs(swap_imgs, swap_qs)  # [B, L]

    # 출력
    print("\n" + "="*100)
    print(title)
    print("dataset_idx:", ds_idx)
    print("Q:", q)
    print(f"[ORIG ] pred={ans0} (p={prob0:.3f})")
    print(f"[BLANK] pred={ansb} (p={probb:.3f})")
    print(f"  -> same(orig, blank) = {ans0 == ansb}")

    # gradient norm (원본 vs blank)
    try:
        g_orig = pixel_grad_norm_single(q, img0)
        g_blank = pixel_grad_norm_single(q, img_blank)
        print(f"Pixel-grad norm: orig={g_orig:.6f} | blank={g_blank:.6f}")
    except Exception as e:
        print("Pixel-grad norm failed:", repr(e))

    # swap별 결과 + KL
    same_cnt = 0
    kls = []
    print("\nPer-swap top-1 answers and KL(p_orig || p_swap):")
    for i in range(psw.shape[0]):
        pi = psw[i]
        ansi, tidi, probi = top_answer(pi)
        kli = kl(p0, pi)
        kls.append(kli)
        same = (tidi == tid0)
        same_cnt += int(same)
        tag = f"SWAP{i+1:02d}"
        print(f"{tag}: pred={ansi:>15s} (p={probi:.3f}) | KL={kli:.4f} | same={same}")

        # swap 이미지 저장
        swap_imgs[i].save(case_dir / f"swap_{i+1:02d}.png")

    same_rate = same_cnt / psw.shape[0]
    mean_kl = sum(kls) / len(kls)
    print(f"\nSummary: swap same-rate={same_rate:.2f}, mean KL={mean_kl:.4f}")
    print("Saved images to:", str(case_dir))

def find_unexpected_dominance(max_show: int = 5):
    """
    '이미지가 필요해 보이는 질문'인데도 dominance가 나는 케이스를 별도로 찾고 싶을 때.
    ScienceQA에서는 이게 잘 안 잡힐 수 있음(모델이 꽤 image-sensitive로 동작).
    """
    cues = [
        "which animal", "what animal", "what color", "how many", "in the picture",
        "shown", "this picture", "this image", "these", "figure"
    ]

    cand = []
    for ds_idx in valid_indices:
        q = build_question(ds[ds_idx]).lower()
        if any(c in q for c in cues):
            cand.append(ds_idx)

    print("\nCandidates with visual-cues:", len(cand))
    if len(cand) == 0:
        return []

    # dominance 스캔(빠르게 n_swaps=6 정도로만)
    results = []
    for ds_idx in cand:
        ex = ds[ds_idx]
        q = build_question(ex)
        img0 = to_pil(ex["image"])
        if img0 is None:
            continue

        p0 = predict_probs([img0], [q])[0]
        _, tid0, _ = top_answer(p0)

        swap_ids = random.sample(range(len(images_pool)), k=min(6, len(images_pool)))
        swap_imgs = [images_pool[j] for j in swap_ids]
        psw = predict_probs(swap_imgs, [q]*len(swap_imgs))
        top_ids = torch.argmax(psw, dim=-1).tolist()
        same_rate = sum(1 for t in top_ids if t == tid0) / len(top_ids)

        # blank 비교도 같이
        pb = predict_probs([blank_like(img0)], [q])[0]
        ans0, _, _ = top_answer(p0)
        ansb, _, _ = top_answer(pb)

        results.append((same_rate, ds_idx, ans0, ansb, q[:120]))

    # same_rate 높은 것부터
    results.sort(key=lambda x: -x[0])
    print("\nTop unexpected-dominance candidates (visual-cued but high same-rate):")
    for i, (sr, ds_idx, ans0, ansb, qshort) in enumerate(results[:max_show]):
        print(f"{i+1:02d}. idx={ds_idx}, same_rate={sr:.2f}, pred={ans0}, blank_pred={ansb} | Q={qshort}")

    return results[:max_show]

# ---------------------------------------------------------
# 너가 이미 찾은 케이스들(출력에 나온 idx)로 자세히 분석
# ---------------------------------------------------------
analyze_case(85,  title="[DETAILED] DOMINANT case (should be text-dominant)", n_swaps=8, seed=0)
analyze_case(164, title="[DETAILED] SENSITIVE case (image-dependent)", n_swaps=8, seed=1)

# ---------------------------------------------------------
# (선택) "이미지가 필요해 보이는데도 dominance"가 나는 케이스 탐색
# ---------------------------------------------------------
_ = find_unexpected_dominance(max_show=5)
