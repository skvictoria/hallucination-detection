# make_qual_figure_vizwiz.py
#
# Baseline vs RTF qualitative figure builder for VizWiz (open-ended VQA)
# - baseline_csv / rtf_csv: 당신 eval 스크립트가 만든 CSV (ans_orig/ans_blank/swap_same_rate/etc 포함)
# - VizWiz dataset (HuggingFaceM4/VizWiz)에서 image/question(+가능하면 answers/answerable) 로드
# - dominance_transition(fixed/new/keep/none) 기준으로 샘플 선택해서 그리드 그림 저장
#
# 실행 예:
# python make_qual_figure_vizwiz.py \
#   --baseline_csv /path/to/eval_baseline.csv \
#   --rtf_csv      /path/to/eval_rtf.csv \
#   --out_dir      /path/to/out \
#   --split validation --max_samples 200 \
#   --mode fixed --num 8 --ncols 2
#
import argparse
import math
import textwrap
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from datasets import load_dataset, DownloadConfig
    from PIL import Image
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False


# -------------------------
# VizWiz helpers
# -------------------------
def normalize_ans(s: str) -> str:
    s = (s or "").strip().lower()
    s = " ".join(s.split())
    return s

def choose_vizwiz_target(ex: Dict[str, Any]) -> str:
    """
    HuggingFaceM4/VizWiz (train/validation):
      - answers: list[str] (보통 10개)
      - answerable: int (0/1)
    test split은 answers/answerable가 없을 수 있음 -> 그 경우 빈 문자열 반환
    """
    if "answerable" not in ex and "answers" not in ex:
        return ""

    answerable = ex.get("answerable", 1)
    if isinstance(answerable, (int, float)) and int(answerable) == 0:
        return "unanswerable"

    answers = ex.get("answers", None)
    if not isinstance(answers, list) or len(answers) == 0:
        return "unanswerable"

    norm = [normalize_ans(a) for a in answers if isinstance(a, str) and a.strip()]
    if len(norm) == 0:
        return "unanswerable"
    return Counter(norm).most_common(1)[0][0]

def get_vizwiz_answerable(ex: Dict[str, Any]) -> Optional[int]:
    if "answerable" not in ex:
        return None
    try:
        return int(ex.get("answerable", 1))
    except Exception:
        return None

def load_vizwiz_filtered(dataset_name: str, split: str, max_samples: int, cache_dir: Optional[str]):
    if not HAS_DATASETS:
        raise RuntimeError("datasets/PIL not installed. Install them to load images.")
    dl_cfg = DownloadConfig(resume_download=True, max_retries=50)
    ds = load_dataset(
        dataset_name,
        split=f"{split}[:{max_samples}]",
        download_config=dl_cfg,
        cache_dir=cache_dir,
    )
    ds = ds.filter(lambda ex: ex.get("image", None) is not None and isinstance(ex.get("question", None), str))
    return ds


# -------------------------
# CSV helpers
# -------------------------
def load_df(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    if "dataset_idx" not in df.columns:
        raise ValueError(f"CSV {p} has no dataset_idx column.")
    df["dataset_idx"] = pd.to_numeric(df["dataset_idx"], errors="coerce").astype(int)
    return df


def dominance_flag(
    df: pd.DataFrame,
    th_swap: float,
    require_same_blank: bool,
    th_mdi: Optional[float],
    only_visual: bool,
) -> np.ndarray:
    """
    text-dominant 판정(원하는 대로 조정 가능):
      - swap_same_rate >= th_swap
      - (옵션) same_orig_blank == 1
      - (옵션) mdi_orig >= th_mdi
      - (옵션) only_visual: visual_cued==1만 대상으로
    """
    cond = np.ones(len(df), dtype=bool)

    if only_visual and "visual_cued" in df.columns:
        cond &= (pd.to_numeric(df["visual_cued"], errors="coerce").fillna(0).astype(int).to_numpy() == 1)

    if "swap_same_rate" in df.columns:
        cond &= (pd.to_numeric(df["swap_same_rate"], errors="coerce").fillna(-1).to_numpy() >= th_swap)
    else:
        cond &= False

    if require_same_blank:
        if "same_orig_blank" in df.columns:
            cond &= (pd.to_numeric(df["same_orig_blank"], errors="coerce").fillna(0).to_numpy() == 1)
        else:
            cond &= False

    if th_mdi is not None:
        if "mdi_orig" in df.columns:
            cond &= (pd.to_numeric(df["mdi_orig"], errors="coerce").fillna(-1e9).to_numpy() >= th_mdi)
        else:
            cond &= False

    return cond.astype(int)


def infer_transition(base_flag: np.ndarray, rtf_flag: np.ndarray) -> np.ndarray:
    out = np.array(["none"] * len(base_flag), dtype=object)
    out[(base_flag == 1) & (rtf_flag == 1)] = "keep"
    out[(base_flag == 1) & (rtf_flag == 0)] = "fixed"
    out[(base_flag == 0) & (rtf_flag == 1)] = "new"
    return out


def wrap(s: str, width: int) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\n", " ").strip()
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False))


def fnum(x, fmt=".3g") -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "nan"
        return format(float(x), fmt)
    except Exception:
        return str(x)


def ensure_answer_cols(df: pd.DataFrame, tag: str):
    required = [f"ans_orig_{tag}", f"ans_blank_{tag}"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing answer columns: {missing}\n"
            f"Your eval CSV must include ans_orig/ans_blank."
        )


# -------------------------
# Figure builder
# -------------------------
def make_grid_figure(
    merged: pd.DataFrame,
    ds,
    out_path: Path,
    ncols: int,
    text_width: int,
    title: str,
):
    n = len(merged)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(8.2 * ncols, 6.2 * nrows), dpi=160)
    if title:
        fig.suptitle(title, fontsize=14)

    gs = fig.add_gridspec(nrows * 2, ncols, height_ratios=[3, 2] * nrows, hspace=0.25, wspace=0.15)

    for i in range(n):
        r = merged.iloc[i]
        rr = i // ncols
        cc = i % ncols

        ax_img = fig.add_subplot(gs[2 * rr, cc])
        ax_txt = fig.add_subplot(gs[2 * rr + 1, cc])
        ax_txt.axis("off")

        di = int(r["dataset_idx"])
        ex = ds[di]
        img = ex["image"].convert("RGB")
        q = ex.get("question", "")

        gt = choose_vizwiz_target(ex)  # train/val이면 majority, test면 ""
        answerable = get_vizwiz_answerable(ex)  # None 가능

        ax_img.imshow(img)
        ax_img.axis("off")

        trans = str(r.get("dominance_transition", ""))
        vc = int(r.get("visual_cued_base", r.get("visual_cued_rtf", r.get("visual_cued", 0))))

        b_ans = str(r.get("ans_orig_base", ""))
        b_blk = str(r.get("ans_blank_base", ""))
        r_ans = str(r.get("ans_orig_rtf", ""))
        r_blk = str(r.get("ans_blank_rtf", ""))

        # GT correctness (VizWiz는 open-ended라 exact match는 참고용)
        def eq_norm(a: str, b: str) -> bool:
            if not a or not b:
                return False
            return normalize_ans(a) == normalize_ans(b)

        b_ok = eq_norm(b_ans, gt) if gt else False
        r_ok = eq_norm(r_ans, gt) if gt else False

        def metric_line(tag: str) -> str:
            parts = []
            for k in ["swap_same_rate", "same_orig_blank", "mdi_orig", "aei_t_orig", "aei_o_orig",
                      "effrank_text_orig", "effrank_image_orig", "cka_ti_orig"]:
                col = f"{k}_{tag}"
                if col in r.index:
                    parts.append(f"{k}={fnum(r[col])}")
            return " | ".join(parts)

        m_base = metric_line("base")
        m_rtf = metric_line("rtf")

        lines = []
        lines.append(f"idx={di} | visual_cued={vc} | transition={trans}")
        if answerable is not None:
            lines.append(f"answerable={answerable}")
        lines.append("Q: " + wrap(q, text_width))

        if gt:
            lines.append(f"GT(majority): {wrap(gt, text_width)}")

        lines.append(f"Baseline: orig={wrap(b_ans, text_width)} {'(MATCH)' if b_ok else ''}")
        lines.append(f"          blank={wrap(b_blk, text_width)}")
        if m_base:
            lines.append("Baseline metrics: " + wrap(m_base, text_width))

        lines.append(f"RTF:      orig={wrap(r_ans, text_width)} {'(MATCH)' if r_ok else ''}")
        lines.append(f"          blank={wrap(r_blk, text_width)}")
        if m_rtf:
            lines.append("RTF metrics: " + wrap(m_rtf, text_width))

        ax_txt.text(
            0.0, 1.0, "\n".join(lines),
            va="top", ha="left",
            fontsize=9,
            transform=ax_txt.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.95, edgecolor="black", linewidth=0.6),
        )

        ax_img.set_title(f"idx={di} | {trans}", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", type=str, required=True)
    ap.add_argument("--rtf_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # VizWiz dataset for image/question/GT(answerable/answers)
    ap.add_argument("--dataset_name", type=str, default="HuggingFaceM4/VizWiz")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--hf_datasets_cache", type=str, default="")

    # dominance criteria
    ap.add_argument("--th_swap", type=float, default=0.85)
    ap.add_argument("--require_same_blank", action="store_true")
    ap.add_argument("--th_mdi", type=float, default=-1.0)  # <0이면 off
    ap.add_argument("--only_visual", action="store_true")

    # selection mode
    ap.add_argument("--mode", type=str, default="fixed",
                    choices=["fixed", "new", "keep", "none", "mixed", "random",
                             "blank_sensitive"])
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)

    # layout
    ap.add_argument("--ncols", type=int, default=2)
    ap.add_argument("--text_width", type=int, default=70)

    args = ap.parse_args()
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_b = load_df(Path(args.baseline_csv))
    df_r = load_df(Path(args.rtf_csv))

    m = df_b.merge(df_r, on="dataset_idx", suffixes=("_base", "_rtf"), how="inner")
    if len(m) == 0:
        raise RuntimeError("No overlapping dataset_idx between baseline and rtf CSVs.")

    ensure_answer_cols(m, "base")
    ensure_answer_cols(m, "rtf")

    th_mdi = None if args.th_mdi < 0 else float(args.th_mdi)

    base_view = m.rename(columns={c: c.replace("_base", "") for c in m.columns if c.endswith("_base")})
    rtf_view = m.rename(columns={c: c.replace("_rtf", "") for c in m.columns if c.endswith("_rtf")})

    base_flag = dominance_flag(base_view, args.th_swap, args.require_same_blank, th_mdi, args.only_visual)
    rtf_flag = dominance_flag(rtf_view, args.th_swap, args.require_same_blank, th_mdi, args.only_visual)

    m = m.copy()
    m["dominant_base"] = base_flag
    m["dominant_rtf"] = rtf_flag
    m["dominance_transition"] = infer_transition(base_flag, rtf_flag)

    # dataset load (이미지/질문/GT)
    cache_dir = args.hf_datasets_cache if args.hf_datasets_cache else None
    ds = load_vizwiz_filtered(args.dataset_name, args.split, args.max_samples, cache_dir)

    # selection
    if args.mode == "random":
        sel = m.sample(n=min(args.num, len(m)), random_state=args.seed)

    elif args.mode == "mixed":
        parts = []
        per = max(1, args.num // 4)
        for key in ["fixed", "new", "keep", "none"]:
            sub = m[m["dominance_transition"] == key]
            if len(sub) > 0:
                parts.append(sub.sample(n=min(per, len(sub)), random_state=args.seed))
        sel = pd.concat(parts, axis=0).drop_duplicates(subset=["dataset_idx"]).head(args.num)
        if len(sel) < args.num:
            rest = m[~m["dataset_idx"].isin(sel["dataset_idx"])].sample(
                n=min(args.num - len(sel), len(m)), random_state=args.seed
            )
            sel = pd.concat([sel, rest], axis=0).head(args.num)

    elif args.mode in ["fixed", "new", "keep", "none"]:
        sub = m[m["dominance_transition"] == args.mode]
        if len(sub) == 0:
            raise RuntimeError(f"No samples for mode={args.mode}. Try mixed/random or relax thresholds.")
        sort_key = "swap_same_rate_base" if "swap_same_rate_base" in sub.columns else None
        if sort_key is not None and args.mode in ["fixed", "keep"]:
            sub = sub.sort_values(sort_key, ascending=False)
        sel = sub.head(args.num)

    else:
        # blank_sensitive:
        #   baseline은 blank에도 동일(same_orig_blank_base=1)인데
        #   rtf는 blank에 민감해져서(same_orig_blank_rtf=0) 달라지는 경우
        rows = []
        for _, r in m.iterrows():
            sb = int(r.get("same_orig_blank_base", 0)) if "same_orig_blank_base" in r else 0
            sr = int(r.get("same_orig_blank_rtf", 0)) if "same_orig_blank_rtf" in r else 0
            if sb == 1 and sr == 0:
                rows.append(r)
        if len(rows) == 0:
            raise RuntimeError("No samples found for mode=blank_sensitive.")
        sub = pd.DataFrame(rows)
        sort_key = "swap_same_rate_base" if "swap_same_rate_base" in sub.columns else None
        if sort_key is not None:
            sub = sub.sort_values(sort_key, ascending=False)
        sel = sub.head(args.num)

    # figure save
    out_png = out_dir / f"qual_{args.mode}.png"
    out_pdf = out_dir / f"qual_{args.mode}.pdf"
    title = f"VizWiz qualitative: Baseline vs RTF (mode={args.mode}, n={len(sel)})"

    make_grid_figure(sel, ds, out_png, args.ncols, args.text_width, title)
    make_grid_figure(sel, ds, out_pdf, args.ncols, args.text_width, title)

    print("Saved:", str(out_png))
    print("Saved:", str(out_pdf))


if __name__ == "__main__":
    main()
