import argparse
import csv
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# -------------------------
# Robust CSV loader (pandas 없이 동작)
# -------------------------
def read_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def to_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def normalize_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        rr = dict(r)
        # numeric columns used
        rr["dataset_idx"] = to_int(r.get("dataset_idx", 0))
        rr["visual_cued"] = to_int(r.get("visual_cued", 0))
        rr["swap_k"] = to_int(r.get("swap_k", 0))
        rr["swap_same_rate"] = to_float(r.get("swap_same_rate", 0.0))
        rr["swap_mean_kl"] = to_float(r.get("swap_mean_kl", 0.0))
        rr["swap_mean_js"] = to_float(r.get("swap_mean_js", 0.0))
        rr["kl_blank"] = to_float(r.get("kl_blank", 0.0))
        rr["js_blank"] = to_float(r.get("js_blank", 0.0))
        rr["delta_p_blank"] = to_float(r.get("delta_p_blank", 0.0))
        rr["same_orig_blank"] = to_int(r.get("same_orig_blank", 0))
        rr["unexpected_dominance"] = to_int(r.get("unexpected_dominance", 0))
        out.append(rr)
    return out


# -------------------------
# Plot helpers
# -------------------------
def hist_two_groups(values_a, values_b, title, xlabel, out_path: Path, bins=30):
    plt.figure()
    plt.hist(values_a, bins=bins, alpha=0.7, label="visual-cued")
    plt.hist(values_b, bins=bins, alpha=0.7, label="non-visual")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def scatter_groups(x_a, y_a, x_b, y_b, title, xlabel, ylabel, out_path: Path):
    plt.figure()
    plt.scatter(x_a, y_a, s=12, alpha=0.7, label="visual-cued")
    plt.scatter(x_b, y_b, s=12, alpha=0.7, label="non-visual")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# Montage 생성
# -------------------------
def safe_open_image(p: Path) -> Optional[Image.Image]:
    if not p.exists():
        return None
    try:
        img = Image.open(p).convert("RGB")
        return img
    except Exception:
        return None

def make_montage(image_paths: List[Path], out_path: Path, tile_size: int = 256, cols: int = 4):
    imgs = []
    for p in image_paths:
        im = safe_open_image(p)
        if im is None:
            continue
        imgs.append(im)

    if len(imgs) == 0:
        return

    # resize to tile
    tiles = [im.resize((tile_size, tile_size), Image.BILINEAR) for im in imgs]
    rows = math.ceil(len(tiles) / cols)
    montage = Image.new("RGB", (cols * tile_size, rows * tile_size), color=(255, 255, 255))

    for i, t in enumerate(tiles):
        r = i // cols
        c = i % cols
        montage.paste(t, (c * tile_size, r * tile_size))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_path)


# -------------------------
# Table writers
# -------------------------
def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})

def write_markdown_table(path: Path, rows: List[Dict[str, Any]], columns: List[str], max_cell_len: int = 120):
    def esc(x):
        s = str(x).replace("\n", " ").replace("\r", " ")
        if len(s) > max_cell_len:
            s = s[:max_cell_len] + "..."
        s = s.replace("|", "\\|")
        return s

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(esc(r.get(c, "")) for c in columns) + " |\n")

def write_html_report(
    path: Path,
    summary: Dict[str, Any],
    top_rows: List[Dict[str, Any]],
    montage_rel_paths: Dict[int, str],
    figures_rel_paths: List[str],
):
    path.parent.mkdir(parents=True, exist_ok=True)

    def h(x):
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    html = []
    html.append("<html><head><meta charset='utf-8'><title>Modality Dominance Report</title></head><body>")
    html.append("<h1>Modality Dominance Step-1 Report</h1>")

    html.append("<h2>Summary</h2><ul>")
    for k, v in summary.items():
        html.append(f"<li><b>{h(k)}</b>: {h(v)}</li>")
    html.append("</ul>")

    html.append("<h2>Figures</h2>")
    for fp in figures_rel_paths:
        html.append(f"<div><img src='{h(fp)}' style='max-width: 100%; height: auto;'></div><br>")

    html.append("<h2>Top Unexpected Dominance Cases</h2>")
    html.append("<table border='1' cellspacing='0' cellpadding='6'>")
    cols = ["dataset_idx", "visual_cued", "swap_same_rate", "swap_mean_js", "pred_orig", "pred_blank", "same_orig_blank", "question"]
    html.append("<tr>" + "".join([f"<th>{h(c)}</th>" for c in cols]) + "</tr>")
    for r in top_rows:
        html.append("<tr>" + "".join([f"<td>{h(r.get(c,''))}</td>" for c in cols]) + "</tr>")
    html.append("</table>")

    html.append("<h2>Montages</h2>")
    for r in top_rows:
        idx = int(r["dataset_idx"])
        m = montage_rel_paths.get(idx, None)
        if m:
            html.append(f"<h3>dataset_idx = {idx}</h3>")
            html.append(f"<p><b>Q</b>: {h(r.get('question',''))}</p>")
            html.append(f"<p><b>orig</b>: {h(r.get('pred_orig',''))} | <b>blank</b>: {h(r.get('pred_blank',''))} | "
                        f"same_rate={h(r.get('swap_same_rate',''))} | mean_js={h(r.get('swap_mean_js',''))}</p>")
            html.append(f"<div><img src='{h(m)}' style='max-width: 100%; height: auto;'></div><br>")

    html.append("</body></html>")

    with open(path, "w") as f:
        f.write("\n".join(html))


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--csv_path", type=str, default="", help="dominance_results.csv path (default: base_dir/dominance_results.csv)")
    ap.add_argument("--artifacts_dir", type=str, default="", help="unexpected_cases dir (default: base_dir/unexpected_cases)")
    ap.add_argument("--out_dir", type=str, default="", help="output report dir (default: base_dir/report)")
    ap.add_argument("--top_n", type=int, default=30)
    ap.add_argument("--only_visual_cued_top", action="store_true", help="Top-N을 visual-cued에서만 뽑기")
    ap.add_argument("--montage_swaps", type=int, default=4, help="몽타주에 포함할 swap 이미지 수")
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--montage_cols", type=int, default=4)
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    csv_path = Path(args.csv_path).resolve() if args.csv_path else (base_dir / "dominance_results.csv")
    artifacts_dir = Path(args.artifacts_dir).resolve() if args.artifacts_dir else (base_dir / "unexpected_cases")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (base_dir / "report")
    fig_dir = out_dir / "figures"
    montage_dir = out_dir / "montages"

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    montage_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows_raw = read_csv_rows(csv_path)
    rows = normalize_records(rows_raw)

    # group split
    rows_v = [r for r in rows if r["visual_cued"] == 1]
    rows_nv = [r for r in rows if r["visual_cued"] == 0]

    # hist: swap_same_rate, swap_mean_js
    hist_two_groups(
        [r["swap_same_rate"] for r in rows_v],
        [r["swap_same_rate"] for r in rows_nv],
        title="swap_same_rate distribution (visual-cued vs non-visual)",
        xlabel="swap_same_rate",
        out_path=fig_dir / "hist_swap_same_rate.png",
        bins=30,
    )

    hist_two_groups(
        [r["swap_mean_js"] for r in rows_v],
        [r["swap_mean_js"] for r in rows_nv],
        title="swap_mean_js distribution (visual-cued vs non-visual)",
        xlabel="swap_mean_js",
        out_path=fig_dir / "hist_swap_mean_js.png",
        bins=30,
    )

    # scatter: same_rate vs mean_js
    scatter_groups(
        x_a=[r["swap_same_rate"] for r in rows_v],
        y_a=[r["swap_mean_js"] for r in rows_v],
        x_b=[r["swap_same_rate"] for r in rows_nv],
        y_b=[r["swap_mean_js"] for r in rows_nv],
        title="swap_same_rate vs swap_mean_js",
        xlabel="swap_same_rate",
        ylabel="swap_mean_js",
        out_path=fig_dir / "scatter_same_rate_vs_js.png",
    )

    # top unexpected cases
    candidates = [r for r in rows if r["unexpected_dominance"] == 1]
    if args.only_visual_cued_top:
        candidates = [r for r in candidates if r["visual_cued"] == 1]

    # 정렬 기준: same_rate 높고, mean_js 낮고, blank에서도 동일 + delta_p_blank 절대값 작을수록 "강한 dominance"
    def key_fn(r):
        return (
            -r["swap_same_rate"],
            r["swap_mean_js"],
            abs(r.get("delta_p_blank", 0.0)),
            r.get("js_blank", 0.0),
        )

    candidates.sort(key=key_fn)
    top = candidates[:args.top_n]

    # save top table
    cols = [
        "dataset_idx", "visual_cued",
        "swap_same_rate", "swap_mean_js", "swap_mean_kl",
        "pred_orig", "p_orig", "pred_blank", "p_blank",
        "same_orig_blank", "kl_blank", "js_blank", "delta_p_blank",
        "question"
    ]
    write_csv(out_dir / "top_unexpected.csv", top, cols)
    write_markdown_table(out_dir / "top_unexpected.md", top, cols)

    # create montages (orig/blank/blur/lowres/crop + swaps)
    montage_rel_paths: Dict[int, str] = {}
    for r in top:
        idx = int(r["dataset_idx"])
        cdir = artifacts_dir / f"idx_{idx}"
        if not cdir.exists():
            # Step1에서 이 idx를 저장하지 않았을 수도 있음
            continue

        image_paths = []
        for name in ["orig.png", "blank.png", "blur.png", "lowres.png", "crop.png"]:
            p = cdir / name
            if p.exists():
                image_paths.append(p)

        for i in range(1, args.montage_swaps + 1):
            sp = cdir / f"swap_{i:02d}.png"
            if sp.exists():
                image_paths.append(sp)

        out_m = montage_dir / f"idx_{idx}.png"
        make_montage(
            image_paths=image_paths,
            out_path=out_m,
            tile_size=args.tile_size,
            cols=args.montage_cols,
        )
        montage_rel_paths[idx] = str(out_m.relative_to(out_dir))

    # summary
    def mean(xs):
        return float(np.mean(xs)) if len(xs) > 0 else float("nan")

    summary = {
        "rows_total": len(rows),
        "rows_visual_cued": len(rows_v),
        "rows_non_visual": len(rows_nv),
        "unexpected_total": sum(1 for r in rows if r["unexpected_dominance"] == 1),
        "unexpected_visual_cued": sum(1 for r in rows if r["unexpected_dominance"] == 1 and r["visual_cued"] == 1),
        "swap_same_rate_mean_visual_cued": mean([r["swap_same_rate"] for r in rows_v]),
        "swap_same_rate_mean_non_visual": mean([r["swap_same_rate"] for r in rows_nv]),
        "swap_mean_js_mean_visual_cued": mean([r["swap_mean_js"] for r in rows_v]),
        "swap_mean_js_mean_non_visual": mean([r["swap_mean_js"] for r in rows_nv]),
        "top_n_requested": args.top_n,
        "top_n_available": len(top),
    }

    figures_rel = [
        str((fig_dir / "hist_swap_same_rate.png").relative_to(out_dir)),
        str((fig_dir / "hist_swap_mean_js.png").relative_to(out_dir)),
        str((fig_dir / "scatter_same_rate_vs_js.png").relative_to(out_dir)),
    ]

    # HTML report
    write_html_report(
        path=out_dir / "report.html",
        summary=summary,
        top_rows=top,
        montage_rel_paths=montage_rel_paths,
        figures_rel_paths=figures_rel,
    )

    print("Done.")
    print("Report dir:", str(out_dir))
    print("Figures:", str(fig_dir))
    print("Top table:", str(out_dir / "top_unexpected.csv"))
    print("HTML report:", str(out_dir / "report.html"))


if __name__ == "__main__":
    main()
