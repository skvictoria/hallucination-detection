import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_eval_csvs(step3_dir: Path):
    placements = ["llm_attn", "llm_mlp", "projector", "all"]
    out = []
    for pl in placements:
        pl_dir = step3_dir / pl
        if not pl_dir.exists():
            continue
        cand = pl_dir / "final" / f"eval_{pl}.csv"
        if cand.exists():
            out.append((pl, cand))
            continue
        cands = sorted(pl_dir.glob("**/eval_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(cands) > 0:
            out.append((pl, cands[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--step3_dir", type=str, default="", help="default: base_dir/step3")
    ap.add_argument("--out_dir", type=str, default="", help="default: base_dir/effrank_vis")
    ap.add_argument("--group", type=str, default="visual", choices=["visual", "non_visual", "all"])
    ap.add_argument("--placement", type=str, default="ALL",
                    choices=["ALL", "llm_attn", "llm_mlp", "projector", "all"])
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=40.0)
    ap.add_argument("--density", action="store_true", help="count 대신 density로")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    step3_dir = Path(args.step3_dir).resolve() if args.step3_dir else (base_dir / "step3")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (base_dir / "effrank_vis")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    evals = find_eval_csvs(step3_dir)
    if len(evals) == 0:
        raise FileNotFoundError(f"No eval_*.csv found under: {step3_dir}")

    dfs = []
    for pl, p in evals:
        if args.placement != "ALL" and pl != args.placement:
            continue
        df = pd.read_csv(p)
        df["placement"] = pl
        dfs.append(df)

    if len(dfs) == 0:
        raise FileNotFoundError(f"No eval csv matched placement={args.placement}")

    df = pd.concat(dfs, axis=0, ignore_index=True)

    for c in ["effrank_image_orig", "effrank_text_orig"]:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in eval csv. Available: {list(df.columns)}")

    df["effrank_image_orig"] = pd.to_numeric(df["effrank_image_orig"], errors="coerce")
    df["effrank_text_orig"] = pd.to_numeric(df["effrank_text_orig"], errors="coerce")

    if args.group != "all":
        if "visual_cued" not in df.columns:
            warnings.warn("Column 'visual_cued' missing. group filter ignored.")
        else:
            df["visual_cued"] = pd.to_numeric(df["visual_cued"], errors="coerce")
            if args.group == "visual":
                df = df[df["visual_cued"] == 1]
            elif args.group == "non_visual":
                df = df[df["visual_cued"] == 0]

    df = df.dropna(subset=["effrank_image_orig", "effrank_text_orig"])
    df = df[np.isfinite(df["effrank_image_orig"].to_numpy())]
    df = df[np.isfinite(df["effrank_text_orig"].to_numpy())]
    if len(df) == 0:
        raise RuntimeError("No valid rows after filtering.")

    img = df["effrank_image_orig"].to_numpy()
    txt = df["effrank_text_orig"].to_numpy()

    bins = np.linspace(args.xmin, args.xmax, args.bins + 1)

    plt.figure(figsize=(7.5, 4.5))
    plt.hist(img, bins=bins, alpha=0.55, label="image effrank")
    plt.hist(txt, bins=bins, alpha=0.55, label="text effrank")
    plt.xlabel("effective rank")
    plt.ylabel("density" if args.density else "count")
    plt.title(f"EffRank overlay (group={args.group}, placement={args.placement}, n={len(df)})")
    plt.legend()
    plt.tight_layout()

    out_png = fig_dir / f"effrank_overlay_group-{args.group}_pl-{args.placement}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Saved:", out_png)


if __name__ == "__main__":
    main()
