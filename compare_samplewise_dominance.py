import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# (옵션) 이미지 저장용
try:
    from datasets import load_dataset, DownloadConfig
    from PIL import Image
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False


PLACEMENTS = ["llm_attn", "llm_mlp", "projector", "all"]


def find_eval_csv(exp_root: Path, placement: str) -> Path:
    # 가장 흔한 형태: exp_root/placement/final/eval_{placement}.csv
    cand1 = exp_root / placement / "final" / f"eval_{placement}.csv"
    if cand1.exists():
        return cand1

    # 그 외: exp_root/placement/**/eval*.csv
    cands = list((exp_root / placement).glob("**/eval*.csv"))
    if len(cands) > 0:
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]

    # 마지막: exp_root 전체에서 placement 포함 eval 찾기
    cands = list(exp_root.glob(f"**/eval*{placement}*.csv")) + list(exp_root.glob(f"**/eval_{placement}.csv"))
    cands = [p for p in cands if p.exists()]
    if len(cands) > 0:
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]

    raise FileNotFoundError(f"Missing eval csv for placement={placement} under {exp_root}")


def load_df(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    # numeric cast
    for c in ["dataset_idx", "visual_cued", "same_orig_blank", "swap_same_rate", "mdi_orig", "aei_t_orig", "aei_o_orig",
              "at_orig", "ao_orig", "effrank_text_orig", "effrank_image_orig"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "dataset_idx" not in df.columns:
        raise ValueError(f"CSV {p} has no dataset_idx column.")
    df["dataset_idx"] = df["dataset_idx"].astype(int)
    return df


def dominance_flag(
    df: pd.DataFrame,
    th_swap: float,
    require_same_blank: bool,
    th_mdi: Optional[float],
    only_visual: bool,
) -> pd.Series:
    """
    dominance(=text-dominant) 판정:
      - swap_same_rate >= th_swap  (이미지를 바꿔도 답이 거의 안 바뀜)
      - (옵션) same_orig_blank == 1 (blank로 바꿔도 답이 같음)
      - (옵션) mdi_orig >= th_mdi   (attention이 text 쪽으로 치우침)
      - (옵션) only_visual: visual_cued==1인 샘플만 비교 대상으로 둠
    """
    cond = pd.Series(True, index=df.index)

    if only_visual and "visual_cued" in df.columns:
        cond &= (df["visual_cued"] == 1)

    if "swap_same_rate" in df.columns:
        cond &= (df["swap_same_rate"] >= th_swap)
    else:
        cond &= False

    if require_same_blank:
        if "same_orig_blank" in df.columns:
            cond &= (df["same_orig_blank"] == 1)
        else:
            cond &= False

    if th_mdi is not None:
        if "mdi_orig" in df.columns:
            cond &= (df["mdi_orig"] >= th_mdi)
        else:
            cond &= False

    return cond.astype(int)


def summarize_confusion(base_flag: np.ndarray, reg_flag: np.ndarray) -> Dict[str, int]:
    # base vs reg
    # 1->1 keep, 1->0 fixed, 0->1 new, 0->0 none
    keep = int(np.sum((base_flag == 1) & (reg_flag == 1)))
    fixed = int(np.sum((base_flag == 1) & (reg_flag == 0)))
    new = int(np.sum((base_flag == 0) & (reg_flag == 1)))
    none = int(np.sum((base_flag == 0) & (reg_flag == 0)))
    return {"keep": keep, "fixed": fixed, "new": new, "none": none, "total": int(len(base_flag))}


def maybe_save_images(
    out_dir: Path,
    dataset_name: str,
    split: str,
    max_samples: int,
    idx_list: List[int],
    prefix: str,
    cache_dir: Optional[str] = None,
):
    """
    eval에서의 dataset_idx가 'filter(image!=None) 후의 인덱스'라는 가정.
    baseline/reg가 동일 split/max_samples/filter로 eval됐으면 같은 이미지가 매칭됨.
    """
    if not HAS_DATASETS:
        print("[WARN] datasets/PIL not installed. Skipping image saving.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    dl_cfg = DownloadConfig(resume_download=True, max_retries=50)
    ds = load_dataset(
        dataset_name,
        split=f"{split}[:{max_samples}]",
        download_config=dl_cfg,
        cache_dir=cache_dir,
    )
    ds = ds.filter(lambda ex: ex.get("image", None) is not None)

    for di in idx_list:
        if di < 0 or di >= len(ds):
            continue
        img = ds[di]["image"]
        if img is None:
            continue
        img = img.convert("RGB")
        img.save(out_dir / f"{prefix}_idx_{di}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_root", type=str, required=True)
    ap.add_argument("--reg_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--placements", type=str, default="all",
                    help="comma sep or 'all' (default: all)")
    ap.add_argument("--th_swap", type=float, default=0.85)
    ap.add_argument("--require_same_blank", action="store_true",
                    help="dominance 판정에 same_orig_blank==1을 포함")
    ap.add_argument("--th_mdi", type=float, default=-1.0,
                    help=">=0이면 mdi_orig >= th_mdi 조건을 포함. (예: 3.0)")
    ap.add_argument("--only_visual", action="store_true",
                    help="visual_cued==1 샘플만 dominance 비교 대상으로 사용")

    ap.add_argument("--topk", type=int, default=30,
                    help="fixed/new 케이스 상위 K개 저장")

    # (옵션) 이미지 저장
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--dataset_name", type=str, default="derek-thomas/ScienceQA")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--hf_datasets_cache", type=str, default="",
                    help="datasets cache_dir (비워두면 default)")

    args = ap.parse_args()

    baseline_root = Path(args.baseline_root).resolve()
    reg_root = Path(args.reg_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.placements.lower() == "all":
        placements = PLACEMENTS
    else:
        placements = [x.strip() for x in args.placements.split(",") if x.strip()]

    th_mdi = None if args.th_mdi < 0 else float(args.th_mdi)

    all_summaries: Dict[str, Any] = {
        "baseline_root": str(baseline_root),
        "reg_root": str(reg_root),
        "th_swap": args.th_swap,
        "require_same_blank": bool(args.require_same_blank),
        "th_mdi": th_mdi,
        "only_visual": bool(args.only_visual),
        "placements": placements,
        "per_placement": {},
    }

    for pl in placements:
        base_csv = find_eval_csv(baseline_root, pl)
        reg_csv = find_eval_csv(reg_root, pl)

        df_b = load_df(base_csv)
        df_r = load_df(reg_csv)

        # merge on dataset_idx (같은 split/max_samples/filter로 eval했다는 가정)
        m = df_b.merge(df_r, on="dataset_idx", suffixes=("_base", "_reg"), how="inner")

        # question mismatch 체크(있으면 경고만)
        if ("question_base" in m.columns) and ("question_reg" in m.columns):
            mismatch = (m["question_base"].fillna("") != m["question_reg"].fillna("")).sum()
            if mismatch > 0:
                print(f"[WARN] {pl}: question mismatch rows = {mismatch} (dataset_idx alignment may be off)")

        # dominance flags
        base_flag = dominance_flag(
            m.rename(columns={c: c.replace("_base", "") for c in m.columns if c.endswith("_base")}),
            th_swap=args.th_swap,
            require_same_blank=args.require_same_blank,
            th_mdi=th_mdi,
            only_visual=args.only_visual,
        ).to_numpy()

        reg_flag = dominance_flag(
            m.rename(columns={c: c.replace("_reg", "") for c in m.columns if c.endswith("_reg")}),
            th_swap=args.th_swap,
            require_same_blank=args.require_same_blank,
            th_mdi=th_mdi,
            only_visual=args.only_visual,
        ).to_numpy()

        conf = summarize_confusion(base_flag, reg_flag)

        # 케이스 테이블 만들기
        m_out = m.copy()
        m_out["dominant_base"] = base_flag
        m_out["dominant_reg"] = reg_flag
        m_out["dominance_transition"] = np.where(
            (base_flag == 1) & (reg_flag == 1), "keep",
            np.where((base_flag == 1) & (reg_flag == 0), "fixed",
                     np.where((base_flag == 0) & (reg_flag == 1), "new", "none"))
        )

        # fixed/new topk: "swap_same_rate_base 높은 것" 우선
        fixed_df = m_out[m_out["dominance_transition"] == "fixed"].copy()
        new_df = m_out[m_out["dominance_transition"] == "new"].copy()

        sort_key = "swap_same_rate_base" if "swap_same_rate_base" in m_out.columns else None
        if sort_key is not None:
            fixed_df = fixed_df.sort_values(sort_key, ascending=False)
            new_df = new_df.sort_values(sort_key, ascending=False)

        fixed_top = fixed_df.head(args.topk)
        new_top = new_df.head(args.topk)

        # 저장
        pl_dir = out_dir / pl
        pl_dir.mkdir(parents=True, exist_ok=True)

        m_out.to_csv(pl_dir / "merged_all.csv", index=False)
        fixed_top.to_csv(pl_dir / "fixed_top.csv", index=False)
        new_top.to_csv(pl_dir / "new_top.csv", index=False)

        # summary 저장
        per = {
            "base_csv": str(base_csv),
            "reg_csv": str(reg_csv),
            "merged_rows": int(len(m_out)),
            "confusion": conf,
        }
        all_summaries["per_placement"][pl] = per

        print(f"[{pl}] merged={len(m_out)} | keep={conf['keep']} fixed={conf['fixed']} new={conf['new']} none={conf['none']}")

        # (옵션) 이미지 저장: fixed/new의 dataset_idx를 뽑아서 저장
        if args.save_images:
            cache_dir = args.hf_datasets_cache if args.hf_datasets_cache else None
            fixed_ids = fixed_top["dataset_idx"].astype(int).tolist()
            new_ids = new_top["dataset_idx"].astype(int).tolist()

            maybe_save_images(
                out_dir=pl_dir / "images_fixed",
                dataset_name=args.dataset_name,
                split=args.split,
                max_samples=args.max_samples,
                idx_list=fixed_ids,
                prefix="fixed",
                cache_dir=cache_dir,
            )
            maybe_save_images(
                out_dir=pl_dir / "images_new",
                dataset_name=args.dataset_name,
                split=args.split,
                max_samples=args.max_samples,
                idx_list=new_ids,
                prefix="new",
                cache_dir=cache_dir,
            )

    with open(out_dir / "summary_all.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("Saved summary:", str(out_dir / "summary_all.json"))
    print("Per-placement outputs in:", str(out_dir))


if __name__ == "__main__":
    main()
