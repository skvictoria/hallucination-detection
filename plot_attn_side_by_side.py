# plot_aei_overlay_log.py
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_metric_from_csv(csv_path: Path, metric: str, visual_only: bool) -> List[float]:
    vals = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        if metric not in r.fieldnames:
            raise KeyError(f"{csv_path} missing column: {metric}. fields={r.fieldnames}")
        if "visual_cued" not in r.fieldnames:
            raise KeyError(f"{csv_path} missing column: visual_cued")

        for row in r:
            try:
                vcue = int(row["visual_cued"])
            except Exception:
                continue
            if visual_only and vcue != 1:
                continue
            if (not visual_only) and vcue != 0:
                continue

            s = row.get(metric, "")
            try:
                x = float(s)
            except Exception:
                continue
            if np.isnan(x) or np.isinf(x):
                continue
            vals.append(x)
    return vals


def find_eval_csvs(step3_dir: Path, placements: List[str]) -> Dict[str, Path]:
    out = {}
    for pl in placements:
        # common patterns
        cand1 = step3_dir / pl / "final" / f"eval_{pl}.csv"
        cand2 = step3_dir / pl / "final" / f"eval_{pl}_dominance.csv"
        if cand1.exists():
            out[pl] = cand1
            continue
        if cand2.exists():
            out[pl] = cand2
            continue
        # fallback: first eval_*.csv under that placement
        cands = list((step3_dir / pl).glob("**/eval_*.csv"))
        if len(cands) == 0:
            raise FileNotFoundError(f"Cannot find eval csv for placement={pl} under {step3_dir/pl}")
        out[pl] = cands[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--step3_dir", type=str, default="", help="default: base_dir/step3")
    ap.add_argument("--out_png", type=str, default="", help="default: <step3_dir>/aei_overlay_log_visual.png")
    ap.add_argument("--visual_cued", type=int, default=1, choices=[0, 1], help="1=visual-cued, 0=non-visual")
    ap.add_argument("--placements", type=str, default="llm_attn,llm_mlp,projector,all",
                    help="comma-separated placements to include")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--density", action="store_true", help="plot density instead of raw counts")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    step3_dir = Path(args.step3_dir).resolve() if args.step3_dir else (base_dir / "step3")
    placements = [x.strip() for x in args.placements.split(",") if x.strip()]
    out_png = Path(args.out_png).resolve() if args.out_png else (step3_dir / f"aei_overlay_log_{'visual' if args.visual_cued==1 else 'nonvisual'}.png")

    eval_csvs = find_eval_csvs(step3_dir, placements)

    # concat across placements
    aei_t_all: List[float] = []
    aei_o_all: List[float] = []
    for pl, p in eval_csvs.items():
        aei_t = read_metric_from_csv(p, "aei_t_orig", visual_only=(args.visual_cued == 1))
        aei_o = read_metric_from_csv(p, "aei_o_orig", visual_only=(args.visual_cued == 1))
        aei_t_all.extend(aei_t)
        aei_o_all.extend(aei_o)

    if len(aei_t_all) == 0 or len(aei_o_all) == 0:
        raise RuntimeError(f"No data found. Check CSV paths/columns and visual_cued filter={args.visual_cued}.")

    # log bins (must be >0)
    eps = 1e-12
    min_val = max(min(min(aei_t_all), min(aei_o_all)), eps)
    max_val = max(max(aei_t_all), max(aei_o_all))
    bins = np.logspace(np.log10(min_val), np.log10(max_val), args.bins)

    plt.figure(figsize=(8, 5))
    plt.hist(aei_o_all, bins=bins, alpha=0.55, label="AEI_O_orig (image)", density=args.density)
    plt.hist(aei_t_all, bins=bins, alpha=0.55, label="AEI_T_orig (text)", density=args.density)
    plt.xscale("log")
    plt.xlabel("AEI (log scale)")
    plt.ylabel("density" if args.density else "count")
    plt.title(f"AEI_T vs AEI_O overlay (log-x) | {'visual-cued' if args.visual_cued==1 else 'non-visual'}")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Saved:", str(out_png))


if __name__ == "__main__":
    main()
