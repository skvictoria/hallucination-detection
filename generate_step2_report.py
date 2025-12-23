import argparse
import csv
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt


def to_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def to_float(x, default=float("nan")) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def read_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def split_groups(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    v = []
    nv = []
    for r in rows:
        vc = to_int(r.get("visual_cued", 0))
        if vc == 1:
            v.append(r)
        else:
            nv.append(r)
    return v, nv


def extract_metric(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    vals = [to_float(r.get(key, "nan")) for r in rows]
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr


def clip_by_quantile(arr: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = np.quantile(arr, lo_q)
    hi = np.quantile(arr, hi_q)
    return arr[(arr >= lo) & (arr <= hi)]


def plot_hist_two_groups(
    arr_v: np.ndarray,
    arr_nv: np.ndarray,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 40,
    clip_lo_q: float = 0.01,
    clip_hi_q: float = 0.99,
    log_x: bool = False,
):
    # optional quantile clipping for readability
    a = clip_by_quantile(arr_v, clip_lo_q, clip_hi_q)
    b = clip_by_quantile(arr_nv, clip_lo_q, clip_hi_q)

    if log_x:
        # log transform: add small epsilon to avoid log(0)
        eps = 1e-12
        a = np.log(a + eps)
        b = np.log(b + eps)
        xlabel = f"log({xlabel}+eps)"

    plt.figure()
    plt.hist(a, bins=bins, alpha=0.7, label="visual-cued")
    plt.hist(b, bins=bins, alpha=0.7, label="non-visual")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--csv_path", type=str, default="", help="default: base_dir/step2_llava_mdi_aei.csv")
    ap.add_argument("--out_dir", type=str, default="", help="default: base_dir/report_step2_mdi_aei")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--clip_lo_q", type=float, default=0.01)
    ap.add_argument("--clip_hi_q", type=float, default=0.99)
    ap.add_argument("--include_blank", action="store_true", help="blank metric들도 추가로 그림 생성")
    ap.add_argument("--log_mdi", action="store_true", help="MDI histogram을 log 변환해서도 저장")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    csv_path = Path(args.csv_path).resolve() if args.csv_path else (base_dir / "step2_llava_mdi_aei.csv")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (base_dir / "report_step2_mdi_aei")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = read_rows(csv_path)
    rows_v, rows_nv = split_groups(rows)

    metrics_orig = ["mdi_orig", "aei_t_orig", "aei_o_orig"]
    metrics_blank = ["mdi_blank", "aei_t_blank", "aei_o_blank"]

    # Save summary stats
    summary = {"total": len(rows), "visual_cued": len(rows_v), "non_visual": len(rows_nv), "metrics": {}}

    for m in metrics_orig:
        arr_v = extract_metric(rows_v, m)
        arr_nv = extract_metric(rows_nv, m)

        summary["metrics"][m] = {
            "visual_cued": summarize(arr_v),
            "non_visual": summarize(arr_nv),
        }

        plot_hist_two_groups(
            arr_v, arr_nv,
            title=f"{m} distribution (visual-cued vs non-visual)",
            xlabel=m,
            out_path=fig_dir / f"hist_{m}.png",
            bins=args.bins,
            clip_lo_q=args.clip_lo_q,
            clip_hi_q=args.clip_hi_q,
            log_x=False,
        )

        if args.log_mdi and m == "mdi_orig":
            plot_hist_two_groups(
                arr_v, arr_nv,
                title=f"log({m}) distribution (visual-cued vs non-visual)",
                xlabel=m,
                out_path=fig_dir / f"hist_log_{m}.png",
                bins=args.bins,
                clip_lo_q=args.clip_lo_q,
                clip_hi_q=args.clip_hi_q,
                log_x=True,
            )

    if args.include_blank:
        for m in metrics_blank:
            arr_v = extract_metric(rows_v, m)
            arr_nv = extract_metric(rows_nv, m)

            summary["metrics"][m] = {
                "visual_cued": summarize(arr_v),
                "non_visual": summarize(arr_nv),
            }

            plot_hist_two_groups(
                arr_v, arr_nv,
                title=f"{m} distribution (visual-cued vs non-visual)",
                xlabel=m,
                out_path=fig_dir / f"hist_{m}.png",
                bins=args.bins,
                clip_lo_q=args.clip_lo_q,
                clip_hi_q=args.clip_hi_q,
                log_x=False,
            )

            if args.log_mdi and m == "mdi_blank":
                plot_hist_two_groups(
                    arr_v, arr_nv,
                    title=f"log({m}) distribution (visual-cued vs non-visual)",
                    xlabel=m,
                    out_path=fig_dir / f"hist_log_{m}.png",
                    bins=args.bins,
                    clip_lo_q=args.clip_lo_q,
                    clip_hi_q=args.clip_hi_q,
                    log_x=True,
                )

    # Write summary json
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print("CSV:", str(csv_path))
    print("Figures dir:", str(fig_dir))
    print("Summary:", str(out_dir / "summary.json"))


if __name__ == "__main__":
    main()
