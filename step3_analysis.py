import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# pandas는 사실상 필수 (CSV 다루기 편함)
try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("Please install pandas: pip install pandas") from e

import matplotlib.pyplot as plt


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman = Pearson(cov(rank(x), rank(y))).
    scipy 없이 구현.
    """
    if x.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return pearson_corr(rx, ry)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_eval_csvs(step3_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    step3/<placement>/**/eval_*.csv 또는 eval_<placement>.csv 등을 찾아서 로드.
    """
    placements = ["llm_attn", "llm_mlp", "projector", "all"]
    out: Dict[str, pd.DataFrame] = {}

    for pl in placements:
        pl_dir = step3_dir / pl
        if not pl_dir.exists():
            continue

        # 가장 흔한 패턴: step3/<pl>/final/eval_<pl>.csv
        cand1 = pl_dir / "final" / f"eval_{pl}.csv"
        cands = []
        if cand1.exists():
            cands = [cand1]
        else:
            # 폴더 전체에서 eval_*.csv 탐색
            cands = sorted(pl_dir.glob("**/eval_*.csv"))

        if len(cands) == 0:
            continue

        # 여러 개 있으면 가장 최신 파일(수정시간 기준)을 선택
        cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
        p = cands[0]

        df = pd.read_csv(p)
        df["placement"] = pl
        df["source_csv"] = str(p)
        out[pl] = df

    return out


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    필요한 컬럼들을 numeric으로 변환하고, effrank_ratio 생성.
    """
    # expected columns
    required_any = [
        "visual_cued",
        "swap_same_rate",
        "aei_o_orig",
        "aei_t_orig",
        "effrank_image_orig",
        "effrank_text_orig",
    ]
    for c in required_any:
        if c not in df.columns:
            # 없는 경우도 있을 수 있으니 경고만
            warnings.warn(f"Missing column '{c}' in {df.get('source_csv', 'df')}. Some analyses will be skipped.")

    # numeric conversions
    for c in ["visual_cued", "swap_same_rate", "aei_o_orig", "aei_t_orig", "effrank_image_orig", "effrank_text_orig"]:
        if c in df.columns:
            df[c] = safe_numeric(df[c])

    # effrank_ratio = text / image
    if "effrank_text_orig" in df.columns and "effrank_image_orig" in df.columns:
        denom = df["effrank_image_orig"].replace(0, np.nan)
        df["effrank_ratio"] = df["effrank_text_orig"] / denom
    else:
        df["effrank_ratio"] = np.nan

    return df


def filter_valid(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            return d.iloc[0:0]  # empty
    d = d.dropna(subset=cols)
    # inf 제거
    for c in cols:
        d = d[np.isfinite(d[c].to_numpy())]
    return d


def make_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    if df.shape[0] == 0:
        return

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    r_p = pearson_corr(x, y)
    r_s = spearman_corr(x, y)

    plt.figure()
    plt.scatter(x, y, alpha=0.6, s=20)
    plt.title(f"{title}\nPearson r={r_p:.3f}, Spearman ρ={r_s:.3f}, n={len(x)}")
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize_group(df: pd.DataFrame, group_name: str) -> Dict[str, float]:
    """
    group_name: "all" / "visual" / "non_visual"
    """
    out: Dict[str, float] = {"group": group_name, "n": int(df.shape[0])}

    # 1) effrank_image vs aei_o
    d1 = filter_valid(df, ["effrank_image_orig", "aei_o_orig"])
    if d1.shape[0] >= 2:
        x = d1["effrank_image_orig"].to_numpy()
        y = d1["aei_o_orig"].to_numpy()
        out["pearson_effimg_aeio"] = pearson_corr(x, y)
        out["spearman_effimg_aeio"] = spearman_corr(x, y)
    else:
        out["pearson_effimg_aeio"] = float("nan")
        out["spearman_effimg_aeio"] = float("nan")

    # 2) effrank_ratio vs swap_same_rate
    d2 = filter_valid(df, ["effrank_ratio", "swap_same_rate"])
    if d2.shape[0] >= 2:
        x = d2["effrank_ratio"].to_numpy()
        y = d2["swap_same_rate"].to_numpy()
        out["pearson_ratio_swapsame"] = pearson_corr(x, y)
        out["spearman_ratio_swapsame"] = spearman_corr(x, y)
    else:
        out["pearson_ratio_swapsame"] = float("nan")
        out["spearman_ratio_swapsame"] = float("nan")

    # 참고: ratio vs aei_t 도 같이 뽑아두면 해석이 쉬움
    d3 = filter_valid(df, ["effrank_ratio", "aei_t_orig"])
    if d3.shape[0] >= 2:
        x = d3["effrank_ratio"].to_numpy()
        y = d3["aei_t_orig"].to_numpy()
        out["pearson_ratio_aeit"] = pearson_corr(x, y)
        out["spearman_ratio_aeit"] = spearman_corr(x, y)
    else:
        out["pearson_ratio_aeit"] = float("nan")
        out["spearman_ratio_aeit"] = float("nan")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--step3_dir", type=str, default="", help="default: base_dir/step3")
    ap.add_argument("--out_dir", type=str, default="", help="default: base_dir/effrank_dominance_analysis")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    step3_dir = Path(args.step3_dir).resolve() if args.step3_dir else (base_dir / "step3")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (base_dir / "effrank_dominance_analysis")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dfs_by_pl = load_eval_csvs(step3_dir)
    if len(dfs_by_pl) == 0:
        raise FileNotFoundError(f"No eval_*.csv found under: {step3_dir}")

    # placement별 정리 + 전체 통합
    all_rows = []
    summary_rows = []

    for pl, df in dfs_by_pl.items():
        df = ensure_columns(df)

        # visual/non_visual 분리
        d_all = df
        d_vis = df[df["visual_cued"] == 1] if "visual_cued" in df.columns else df.iloc[0:0]
        d_non = df[df["visual_cued"] == 0] if "visual_cued" in df.columns else df.iloc[0:0]

        # summary
        for name, dd in [("all", d_all), ("visual", d_vis), ("non_visual", d_non)]:
            s = summarize_group(dd, name)
            s["placement"] = pl
            summary_rows.append(s)

        # scatter plots (placement별)
        # effrank_image vs aei_o
        for name, dd in [("visual", d_vis), ("non_visual", d_non)]:
            dd1 = filter_valid(dd, ["effrank_image_orig", "aei_o_orig"])
            make_scatter(
                dd1,
                "effrank_image_orig",
                "aei_o_orig",
                title=f"[{pl}] effrank_image_orig vs aei_o_orig ({name})",
                out_path=fig_dir / f"scatter_effimg_vs_aeio_{pl}_{name}.png",
                xlabel="effrank_image_orig (orig)",
                ylabel="aei_o_orig (orig)",
            )

        # effrank_ratio vs swap_same_rate
        for name, dd in [("visual", d_vis), ("non_visual", d_non)]:
            dd2 = filter_valid(dd, ["effrank_ratio", "swap_same_rate"])
            make_scatter(
                dd2,
                "effrank_ratio",
                "swap_same_rate",
                title=f"[{pl}] effrank_ratio vs swap_same_rate ({name})",
                out_path=fig_dir / f"scatter_ratio_vs_swapsame_{pl}_{name}.png",
                xlabel="effrank_ratio = effrank_text_orig / effrank_image_orig",
                ylabel="swap_same_rate",
            )

        all_rows.append(df)

    # 전체 통합 DF
    df_all = pd.concat(all_rows, axis=0, ignore_index=True)
    df_all = ensure_columns(df_all)

    # 전체 통합 scatter
    if "visual_cued" in df_all.columns:
        d_vis = df_all[df_all["visual_cued"] == 1]
        d_non = df_all[df_all["visual_cued"] == 0]
    else:
        d_vis = df_all.iloc[0:0]
        d_non = df_all.iloc[0:0]

    for name, dd in [("visual", d_vis), ("non_visual", d_non)]:
        dd1 = filter_valid(dd, ["effrank_image_orig", "aei_o_orig"])
        make_scatter(
            dd1,
            "effrank_image_orig",
            "aei_o_orig",
            title=f"[ALL placements] effrank_image_orig vs aei_o_orig ({name})",
            out_path=fig_dir / f"scatter_effimg_vs_aeio_ALL_{name}.png",
        )
        dd2 = filter_valid(dd, ["effrank_ratio", "swap_same_rate"])
        make_scatter(
            dd2,
            "effrank_ratio",
            "swap_same_rate",
            title=f"[ALL placements] effrank_ratio vs swap_same_rate ({name})",
            out_path=fig_dir / f"scatter_ratio_vs_swapsame_ALL_{name}.png",
        )

    # summary table 저장
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "summary_correlations.csv"
    summary_df.to_csv(summary_csv, index=False)

    # 간단히 콘솔 출력(핵심만)
    print("Saved:", str(summary_csv))
    print("Figures dir:", str(fig_dir))
    print()
    print("Top-level view (visual group):")
    v = summary_df[summary_df["group"] == "visual"].copy()
    show_cols = ["placement", "n", "pearson_effimg_aeio", "pearson_ratio_swapsame", "pearson_ratio_aeit"]
    if all(c in v.columns for c in show_cols):
        print(v[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
