import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PLACEMENTS = ["llm_attn", "llm_mlp", "projector", "all"]

METRICS = [
    "swap_same_rate",
    "same_orig_blank",
    "mdi_orig",
    "aei_t_orig",
    "aei_o_orig",
    "at_orig",
    "ao_orig",
    "effrank_text_orig",
    "effrank_image_orig",
]


def find_eval_csv(exp_root: Path, placement: str) -> Path:
    direct = list((exp_root / placement).glob("**/eval*.csv"))
    if len(direct) > 0:
        return direct[0]
    cands = list(exp_root.glob(f"**/eval*{placement}*.csv"))
    if len(cands) > 0:
        return cands[0]
    raise FileNotFoundError(f"Could not find eval csv for placement={placement} under {exp_root}")


def safe_mean(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size > 0 else float("nan")


def safe_median(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    x = x[~np.isnan(x)]
    return float(np.median(x)) if x.size > 0 else float("nan")


def load_eval_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # numeric casting (safe)
    for m in METRICS + ["visual_cued"]:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    return df


def group_stats(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "visual_cued" not in df.columns:
        df = df.copy()
        df["visual_cued"] = -1

    groups = {
        "all": df,
        "visual": df[df["visual_cued"] == 1],
        "non_visual": df[df["visual_cued"] == 0],
    }

    for gname, gdf in groups.items():
        out[f"n_{gname}"] = int(len(gdf))
        for m in metrics:
            if m in gdf.columns:
                arr = gdf[m].to_numpy()
                out[f"{m}_mean_{gname}"] = safe_mean(arr)
                out[f"{m}_median_{gname}"] = safe_median(arr)
            else:
                out[f"{m}_mean_{gname}"] = float("nan")
                out[f"{m}_median_{gname}"] = float("nan")

        # derived metrics
        if ("effrank_text_orig" in gdf.columns) and ("effrank_image_orig" in gdf.columns):
            t = gdf["effrank_text_orig"].to_numpy(dtype=np.float64)
            v = gdf["effrank_image_orig"].to_numpy(dtype=np.float64)
            out[f"effrank_ratio_txt_over_img_mean_{gname}"] = safe_mean(t / (v + 1e-12))
            out[f"effrank_gap_txt_minus_img_mean_{gname}"] = safe_mean(t - v)
        else:
            out[f"effrank_ratio_txt_over_img_mean_{gname}"] = float("nan")
            out[f"effrank_gap_txt_minus_img_mean_{gname}"] = float("nan")

        if ("at_orig" in gdf.columns) and ("ao_orig" in gdf.columns):
            at = gdf["at_orig"].to_numpy(dtype=np.float64)
            ao = gdf["ao_orig"].to_numpy(dtype=np.float64)
            out[f"attn_share_text_minus_img_mean_{gname}"] = safe_mean(at - ao)
        else:
            out[f"attn_share_text_minus_img_mean_{gname}"] = float("nan")

    return out


def save_compare_table(rows: List[Dict[str, Any]], out_csv: Path):
    cols = ["placement", "cond", "n_all", "n_visual", "n_non_visual"]
    for m in METRICS:
        cols += [f"{m}_mean_all", f"{m}_mean_visual", f"{m}_mean_non_visual"]
    cols += [
        "effrank_ratio_txt_over_img_mean_all",
        "effrank_ratio_txt_over_img_mean_visual",
        "effrank_ratio_txt_over_img_mean_non_visual",
        "effrank_gap_txt_minus_img_mean_all",
        "effrank_gap_txt_minus_img_mean_visual",
        "effrank_gap_txt_minus_img_mean_non_visual",
        "attn_share_text_minus_img_mean_all",
        "attn_share_text_minus_img_mean_visual",
        "attn_share_text_minus_img_mean_non_visual",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def bar_plot(deltas: Dict[str, float], title: str, ylabel: str, out_png: Path):
    xs = list(deltas.keys())
    ys = [deltas[k] for k in xs]
    plt.figure()
    plt.bar(xs, ys)
    plt.title(title)
    plt.xlabel("placement")
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def overlay_hist(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str,
                 title: str, xlabel: str, out_png: Path, bins: int = 30, logy: bool = False):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return

    plt.figure()
    plt.hist(a, bins=bins, alpha=0.5, label=label_a)
    plt.hist(b, bins=bins, alpha=0.5, label=label_b)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noreg_dir", type=str, required=True)
    ap.add_argument("--reg_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--bins", type=int, default=30)
    args = ap.parse_args()

    noreg_root = Path(args.noreg_dir).resolve()
    reg_root = Path(args.reg_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    delta_mdi_visual: Dict[str, float] = {}
    delta_swap_visual: Dict[str, float] = {}
    delta_effrank_img_visual: Dict[str, float] = {}
    delta_effrank_ratio_visual: Dict[str, float] = {}

    for pl in PLACEMENTS:
        csv_a = find_eval_csv(noreg_root, pl)
        csv_b = find_eval_csv(reg_root, pl)

        df_a = load_eval_df(csv_a)
        df_b = load_eval_df(csv_b)

        stat_a = group_stats(df_a, METRICS)
        stat_b = group_stats(df_b, METRICS)

        rows.append({"placement": pl, "cond": "noreg", **stat_a})
        rows.append({"placement": pl, "cond": "reg", **stat_b})

        # deltas (reg - noreg) on visual-cued
        delta_mdi_visual[pl] = stat_b.get("mdi_orig_mean_visual", float("nan")) - stat_a.get("mdi_orig_mean_visual", float("nan"))
        delta_swap_visual[pl] = stat_b.get("swap_same_rate_mean_visual", float("nan")) - stat_a.get("swap_same_rate_mean_visual", float("nan"))
        delta_effrank_img_visual[pl] = stat_b.get("effrank_image_orig_mean_visual", float("nan")) - stat_a.get("effrank_image_orig_mean_visual", float("nan"))
        delta_effrank_ratio_visual[pl] = stat_b.get("effrank_ratio_txt_over_img_mean_visual", float("nan")) - stat_a.get("effrank_ratio_txt_over_img_mean_visual", float("nan"))

        # per-placement overlay hists (visual-cued)
        if "visual_cued" in df_a.columns and "visual_cued" in df_b.columns:
            a_vis = df_a[df_a["visual_cued"] == 1]
            b_vis = df_b[df_b["visual_cued"] == 1]

            if "mdi_orig" in a_vis.columns and "mdi_orig" in b_vis.columns:
                overlay_hist(
                    a_vis["mdi_orig"].to_numpy(),
                    b_vis["mdi_orig"].to_numpy(),
                    "noreg", "reg",
                    title=f"{pl}: mdi_orig (visual-cued)",
                    xlabel="mdi_orig",
                    out_png=fig_dir / f"hist_{pl}_mdi_visual_overlay.png",
                    bins=args.bins,
                    logy=True,
                )

            if "effrank_image_orig" in a_vis.columns and "effrank_image_orig" in b_vis.columns:
                overlay_hist(
                    a_vis["effrank_image_orig"].to_numpy(),
                    b_vis["effrank_image_orig"].to_numpy(),
                    "noreg", "reg",
                    title=f"{pl}: effrank_image_orig (visual-cued)",
                    xlabel="effrank_image_orig",
                    out_png=fig_dir / f"hist_{pl}_effrank_image_visual_overlay.png",
                    bins=args.bins,
                    logy=False,
                )

            if "effrank_text_orig" in a_vis.columns and "effrank_text_orig" in b_vis.columns:
                overlay_hist(
                    a_vis["effrank_text_orig"].to_numpy(),
                    b_vis["effrank_text_orig"].to_numpy(),
                    "noreg", "reg",
                    title=f"{pl}: effrank_text_orig (visual-cued)",
                    xlabel="effrank_text_orig",
                    out_png=fig_dir / f"hist_{pl}_effrank_text_visual_overlay.png",
                    bins=args.bins,
                    logy=False,
                )

    save_compare_table(rows, out_dir / "compare_reg_vs_noreg.csv")

    bar_plot(delta_mdi_visual,
             "Delta mdi_orig mean (reg - noreg) on visual-cued",
             "delta_mdi", fig_dir / "bar_delta_mdi_visual.png")

    bar_plot(delta_swap_visual,
             "Delta swap_same_rate mean (reg - noreg) on visual-cued",
             "delta_swap_same_rate", fig_dir / "bar_delta_swap_visual.png")

    bar_plot(delta_effrank_img_visual,
             "Delta effrank_image_orig mean (reg - noreg) on visual-cued",
             "delta_effrank_image", fig_dir / "bar_delta_effrank_image_visual.png")

    bar_plot(delta_effrank_ratio_visual,
             "Delta effrank ratio (txt/img) mean (reg - noreg) on visual-cued",
             "delta_effrank_ratio", fig_dir / "bar_delta_effrank_ratio_visual.png")

    print("Saved table:", str(out_dir / "compare_reg_vs_noreg.csv"))
    print("Figures:", str(fig_dir))


if __name__ == "__main__":
    main()
