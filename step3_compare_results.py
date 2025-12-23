import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# IO utils
# -------------------------
def read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def read_csv_dicts(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(p, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def to_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def to_int(x, default=0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, int):
            return int(x)
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def safe_mean(xs: List[float]) -> float:
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if len(xs) > 0 else float("nan")


def safe_quantile(xs: List[float], q: float) -> float:
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.quantile(xs, q)) if len(xs) > 0 else float("nan")


# -------------------------
# discovery helpers
# -------------------------
def find_one(glob_list: List[Path]) -> Optional[Path]:
    return glob_list[0] if len(glob_list) > 0 else None


def find_summary(step3_dir: Path, placement: str) -> Optional[Path]:
    # prefer final/summary_{placement}.json
    p = step3_dir / placement / "final" / f"summary_{placement}.json"
    if p.exists():
        return p
    # otherwise search
    cands = list((step3_dir / placement).glob("**/summary_*.json"))
    cands = [c for c in cands if c.is_file()]
    return find_one(sorted(cands))


def find_eval_csv(step3_dir: Path, placement: str) -> Optional[Path]:
    # prefer final/eval_{placement}.csv
    p = step3_dir / placement / "final" / f"eval_{placement}.csv"
    if p.exists():
        return p
    cands = list((step3_dir / placement).glob("**/eval_*.csv"))
    cands = [c for c in cands if c.is_file()]
    return find_one(sorted(cands))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# plotting helpers
# -------------------------
def bar_plot(xs, ys, title, ylabel, out_path: Path):
    plt.figure()
    plt.bar(xs, ys)
    plt.title(title)
    plt.xlabel("placement")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def hist_plot(groups: Dict[str, List[float]], title: str, xlabel: str, out_path: Path, bins: int = 30):
    plt.figure()
    for k, vals in groups.items():
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) == 0:
            continue
        plt.hist(vals, bins=bins, alpha=0.5, label=k)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def scatter_plot(x, y, title, xlabel, ylabel, out_path: Path):
    plt.figure()
    plt.scatter(x, y, s=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# core aggregation
# -------------------------
DEFAULT_PLACEMENTS = ["llm_attn", "llm_mlp", "projector", "all"]

SUMMARY_KEYS = [
    "swap_same_rate_mean",
    "swap_same_rate_mean_visual",
    "swap_same_rate_mean_non_visual",
    "same_orig_blank_rate",
    "same_orig_blank_rate_visual",
    "same_orig_blank_rate_non_visual",
    "MDI_orig_mean",
    "MDI_orig_mean_visual",
    "MDI_orig_mean_non_visual",
    "AEI_T_orig_mean",
    "AEI_T_orig_mean_visual",
    "AEI_T_orig_mean_non_visual",
    # optional prompt diagnostics (if present)
    "effrank_text_orig_mean",
    "effrank_image_orig_mean",
    "cka_ti_orig_mean",
    "effrank_text_orig_mean_visual",
    "effrank_image_orig_mean_visual",
    "cka_ti_orig_mean_visual",
    "effrank_text_orig_mean_non_visual",
    "effrank_image_orig_mean_non_visual",
    "cka_ti_orig_mean_non_visual",
]


def summarize_from_eval_rows(rows: List[Dict[str, Any]], prefix: str = "") -> Dict[str, float]:
    """
    eval csv에서 summary json이 없어도 평균/분위수 등을 다시 계산할 수 있게 만든 보조 요약.
    """
    def col(name): return [to_float(r.get(name, np.nan)) for r in rows]

    out = {}
    if len(rows) == 0:
        return out

    visual = [r for r in rows if to_int(r.get("visual_cued", 0)) == 1]
    nonvis = [r for r in rows if to_int(r.get("visual_cued", 0)) == 0]

    # candidate metrics (있을 때만)
    metrics = [
        "swap_same_rate",
        "same_orig_blank",
        "mdi_orig",
        "aei_t_orig",
        "aei_o_orig",
        "effrank_text_orig",
        "effrank_image_orig",
        "cka_ti_orig",
    ]

    for m in metrics:
        allv = col(m)
        out[f"{prefix}{m}_mean"] = safe_mean(allv)
        out[f"{prefix}{m}_p10"] = safe_quantile(allv, 0.10)
        out[f"{prefix}{m}_p50"] = safe_quantile(allv, 0.50)
        out[f"{prefix}{m}_p90"] = safe_quantile(allv, 0.90)

        if len(visual) > 0:
            vv = [to_float(r.get(m, np.nan)) for r in visual]
            out[f"{prefix}{m}_mean_visual"] = safe_mean(vv)
        if len(nonvis) > 0:
            nv = [to_float(r.get(m, np.nan)) for r in nonvis]
            out[f"{prefix}{m}_mean_non_visual"] = safe_mean(nv)

    out[f"{prefix}n"] = float(len(rows))
    out[f"{prefix}n_visual"] = float(len(visual))
    out[f"{prefix}n_non_visual"] = float(len(nonvis))
    return out


def pick_unexpected_cases(
    rows: List[Dict[str, Any]],
    top_n: int = 20,
    require_visual: bool = True,
    th_swap: float = 0.85,
    th_mdi: float = 3.0,
    th_aei_t: float = 2.0,
    low_effrank_img_q: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    'unexpected dominance' 후보를 자동으로 뽑아 표로 저장.
    기본 기준:
      - (옵션) visual_cued == 1
      - swap_same_rate 높음
      - MDI/AEI_T 높음
      - effrank_image 낮음 (가능하면)
    """
    if len(rows) == 0:
        return []

    # effrank_image가 있으면 "낮음" 기준을 분위수로 잡음
    effs = [to_float(r.get("effrank_image_orig", np.nan)) for r in rows]
    effs = [e for e in effs if np.isfinite(e)]
    eff_th = np.quantile(effs, low_effrank_img_q) if len(effs) > 0 else np.nan

    cands = []
    for r in rows:
        vcue = to_int(r.get("visual_cued", 0))
        if require_visual and vcue != 1:
            continue

        swap = to_float(r.get("swap_same_rate", np.nan))
        mdi = to_float(r.get("mdi_orig", np.nan))
        aei = to_float(r.get("aei_t_orig", np.nan))
        eff = to_float(r.get("effrank_image_orig", np.nan))

        if not np.isfinite(swap) or not np.isfinite(mdi) or not np.isfinite(aei):
            continue
        if swap < th_swap or mdi < th_mdi or aei < th_aei_t:
            continue

        # effrank 조건은 있으면 적용
        eff_ok = True
        if np.isfinite(eff_th) and np.isfinite(eff):
            eff_ok = (eff <= eff_th)

        if not eff_ok:
            continue

        score = swap * (mdi + 0.5 * aei)  # 임의의 종합 점수(정렬용)
        cands.append((score, r))

    cands.sort(key=lambda x: x[0], reverse=True)
    picked = []
    for score, r in cands[:top_n]:
        picked.append({
            "score": float(score),
            "dataset_idx": to_int(r.get("dataset_idx", -1)),
            "visual_cued": to_int(r.get("visual_cued", 0)),
            "question": r.get("question", "")[:300],
            "ans_orig": r.get("ans_orig", ""),
            "ans_blank": r.get("ans_blank", ""),
            "swap_same_rate": to_float(r.get("swap_same_rate", np.nan)),
            "same_orig_blank": to_int(r.get("same_orig_blank", 0)),
            "mdi_orig": to_float(r.get("mdi_orig", np.nan)),
            "aei_t_orig": to_float(r.get("aei_t_orig", np.nan)),
            "effrank_image_orig": to_float(r.get("effrank_image_orig", np.nan)),
            "cka_ti_orig": to_float(r.get("cka_ti_orig", np.nan)),
        })
    return picked


def write_table_csv(out_path: Path, rows: List[Dict[str, Any]]):
    if len(rows) == 0:
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="/home/hice1/skim3513/scratch/hallucination-detection")
    ap.add_argument("--step3_dir", type=str, default="", help="default: base_dir/step3")
    ap.add_argument("--out_dir", type=str, default="", help="default: base_dir/final_report")
    ap.add_argument("--placements", type=str, default=",".join(DEFAULT_PLACEMENTS))

    # step2 baseline optional
    ap.add_argument("--step2_csv", type=str, default="", help="default: base_dir/step2_llava_mdi_aei.csv (if exists)")

    # report knobs
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--bins", type=int, default=30)

    # unexpected filter knobs
    ap.add_argument("--th_swap", type=float, default=0.85)
    ap.add_argument("--th_mdi", type=float, default=3.0)
    ap.add_argument("--th_aei_t", type=float, default=2.0)
    ap.add_argument("--require_visual", action="store_true")

    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    step3_dir = Path(args.step3_dir).resolve() if args.step3_dir else (base_dir / "step3")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (base_dir / "final_report")
    fig_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    placements = [p.strip() for p in args.placements.split(",") if p.strip()]
    if len(placements) == 0:
        placements = DEFAULT_PLACEMENTS

    # -------------------------
    # 0) load step2 baseline (if present)
    # -------------------------
    baseline_rows = []
    step2_csv = Path(args.step2_csv).resolve() if args.step2_csv else (base_dir / "step2_llava_mdi_aei.csv")
    if step2_csv.exists():
        baseline_rows = read_csv_dicts(step2_csv)
        print(f"[step2] loaded baseline csv: {step2_csv} (rows={len(baseline_rows)})")
        base_summary = summarize_from_eval_rows(baseline_rows, prefix="base_")
        with open(out_dir / "step2_baseline_summary.json", "w") as f:
            json.dump(base_summary, f, indent=2)
    else:
        print("[step2] baseline csv not found; skip baseline compare.")

    # -------------------------
    # 1) load step3 placements
    # -------------------------
    placement_infos = []
    for pl in placements:
        s_path = find_summary(step3_dir, pl)
        e_path = find_eval_csv(step3_dir, pl)

        info = {"placement": pl, "summary_path": str(s_path) if s_path else "", "eval_path": str(e_path) if e_path else ""}
        if s_path and Path(s_path).exists():
            summ = read_json(s_path)
            summ["placement"] = pl
            info["summary"] = summ
        else:
            info["summary"] = {"placement": pl}

        rows = []
        if e_path and Path(e_path).exists():
            rows = read_csv_dicts(e_path)
            info["eval_rows"] = rows
            # fallback/extra summary computed from eval csv
            info["eval_summary"] = summarize_from_eval_rows(rows, prefix="")
        else:
            info["eval_rows"] = []
            info["eval_summary"] = {}

        placement_infos.append(info)

    # -------------------------
    # 2) write compare table (summary.json 기반 + eval 기반 백업)
    # -------------------------
    compare_rows = []
    for info in placement_infos:
        pl = info["placement"]
        s = dict(info.get("summary", {}))
        # summary에 없으면 eval_summary로 보강
        es = info.get("eval_summary", {})
        row = {"placement": pl}
        # 통합: SUMMARY_KEYS 우선, 없으면 eval_summary의 *_mean_*를 사용
        for k in SUMMARY_KEYS:
            if k in s:
                row[k] = s.get(k, "")
            else:
                # mapping: summary키 -> eval_summary 키 추정
                # e.g. "MDI_orig_mean_visual" vs eval csv "mdi_orig_mean_visual"
                alt = k.lower()
                row[k] = es.get(alt, "")
        # n
        if "n" not in row or row["n"] == "":
            row["n"] = int(es.get("n", np.nan)) if "n" in es else len(info.get("eval_rows", []))
        compare_rows.append(row)

    compare_csv = out_dir / "step3_compare_table.csv"
    cols = ["placement", "n"] + [k for k in SUMMARY_KEYS if k != "placement"]
    with open(compare_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in compare_rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # -------------------------
    # 3) bar plots (visual/non-visual 둘 다)
    # -------------------------
    def get_metric(pl: str, key: str) -> float:
        for r in compare_rows:
            if r["placement"] == pl:
                return to_float(r.get(key, np.nan))
        return np.nan

    # 대표 4개 (없으면 알아서 nan)
    bar_keys = [
        ("swap_same_rate_mean_visual", "Mean swap_same_rate (visual-cued)", "swap_same_rate"),
        ("swap_same_rate_mean_non_visual", "Mean swap_same_rate (non-visual)", "swap_same_rate"),
        ("same_orig_blank_rate_visual", "Rate same_orig_blank (visual-cued)", "same_orig_blank"),
        ("same_orig_blank_rate_non_visual", "Rate same_orig_blank (non-visual)", "same_orig_blank"),
        ("MDI_orig_mean_visual", "Mean MDI_orig (visual-cued)", "mdi_orig"),
        ("MDI_orig_mean_non_visual", "Mean MDI_orig (non-visual)", "mdi_orig"),
        ("AEI_T_orig_mean_visual", "Mean AEI_T_orig (visual-cued)", "aei_t_orig"),
        ("AEI_T_orig_mean_non_visual", "Mean AEI_T_orig (non-visual)", "aei_t_orig"),
        ("effrank_image_orig_mean_visual", "Mean effrank_image (visual-cued)", "effrank_image_orig"),
        ("effrank_image_orig_mean_non_visual", "Mean effrank_image (non-visual)", "effrank_image_orig"),
        ("cka_ti_orig_mean_visual", "Mean CKA(text,image) (visual-cued)", "cka_ti_orig"),
        ("cka_ti_orig_mean_non_visual", "Mean CKA(text,image) (non-visual)", "cka_ti_orig"),
    ]

    for key, title, ylabel in bar_keys:
        xs = placements
        ys = [get_metric(pl, key) for pl in placements]
        if all(not np.isfinite(y) for y in ys):
            continue
        bar_plot(xs, ys, title, ylabel, fig_dir / f"bar_{key}.png")

    # -------------------------
    # 4) distribution plots from eval csv (hist)
    # -------------------------
    dist_metrics = [
        ("swap_same_rate", "swap_same_rate"),
        ("same_orig_blank", "same_orig_blank"),
        ("mdi_orig", "mdi_orig"),
        ("aei_t_orig", "aei_t_orig"),
        ("aei_o_orig", "aei_o_orig"),
        ("effrank_image_orig", "effrank_image_orig"),
        ("effrank_text_orig", "effrank_text_orig"),
        ("cka_ti_orig", "cka_ti_orig"),
    ]

    for colname, label in dist_metrics:
        groups_all = {}
        groups_visual = {}
        groups_nonvis = {}

        for info in placement_infos:
            pl = info["placement"]
            rows = info.get("eval_rows", [])
            if len(rows) == 0:
                continue

            vals_all = [to_float(r.get(colname, np.nan)) for r in rows]
            groups_all[pl] = vals_all

            vrows = [r for r in rows if to_int(r.get("visual_cued", 0)) == 1]
            nrows = [r for r in rows if to_int(r.get("visual_cued", 0)) == 0]
            groups_visual[pl] = [to_float(r.get(colname, np.nan)) for r in vrows]
            groups_nonvis[pl] = [to_float(r.get(colname, np.nan)) for r in nrows]

        if len(groups_all) > 0:
            hist_plot(groups_all, f"Distribution of {label} (all)", label, fig_dir / f"hist_{colname}_all.png", bins=args.bins)
        if len(groups_visual) > 0:
            hist_plot(groups_visual, f"Distribution of {label} (visual-cued)", label, fig_dir / f"hist_{colname}_visual.png", bins=args.bins)
        if len(groups_nonvis) > 0:
            hist_plot(groups_nonvis, f"Distribution of {label} (non-visual)", label, fig_dir / f"hist_{colname}_nonvisual.png", bins=args.bins)

    # -------------------------
    # 5) unexpected dominance top-N per placement (from eval rows)
    # -------------------------
    unexpected_dir = out_dir / "unexpected"
    ensure_dir(unexpected_dir)

    for info in placement_infos:
        pl = info["placement"]
        rows = info.get("eval_rows", [])
        if len(rows) == 0:
            continue

        picked = pick_unexpected_cases(
            rows=rows,
            top_n=args.top_n,
            require_visual=args.require_visual,
            th_swap=args.th_swap,
            th_mdi=args.th_mdi,
            th_aei_t=args.th_aei_t,
        )
        if len(picked) == 0:
            continue

        write_table_csv(unexpected_dir / f"unexpected_top{args.top_n}_{pl}.csv", picked)
        with open(unexpected_dir / f"unexpected_top{args.top_n}_{pl}.json", "w") as f:
            json.dump(picked, f, indent=2)

    # -------------------------
    # 6) baseline(step2) vs placement(step3) delta table (if baseline exists)
    # -------------------------
    if len(baseline_rows) > 0:
        base_eval_summary = summarize_from_eval_rows(baseline_rows, prefix="base_")

        delta_rows = []
        for info in placement_infos:
            pl = info["placement"]
            es = info.get("eval_summary", {})

            row = {"placement": pl}
            # compare key subset
            for k in ["swap_same_rate_mean", "mdi_orig_mean", "aei_t_orig_mean", "effrank_image_orig_mean", "cka_ti_orig_mean"]:
                base_k = "base_" + k
                # eval_summary는 colname 기반으로 만들어져서 키가 소문자일 수 있음
                # 여기서는 둘 다 시도
                v_base = to_float(base_eval_summary.get(base_k, base_eval_summary.get(base_k.lower(), np.nan)))
                v_pl = to_float(es.get(k, es.get(k.lower(), np.nan)))

                if np.isfinite(v_base) and np.isfinite(v_pl):
                    row[k] = v_pl
                    row[base_k] = v_base
                    row["delta_" + k] = float(v_pl - v_base)
                else:
                    row[k] = ""
                    row[base_k] = ""
                    row["delta_" + k] = ""

            delta_rows.append(row)

        delta_csv = out_dir / "step2_vs_step3_delta.csv"
        delta_cols = sorted({c for r in delta_rows for c in r.keys()})
        with open(delta_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=delta_cols)
            w.writeheader()
            for r in delta_rows:
                w.writerow(r)

    # -------------------------
    # 7) quick scatter examples (optional): swap_same_rate vs mdi_orig / effrank_image
    # -------------------------
    for info in placement_infos:
        pl = info["placement"]
        rows = info.get("eval_rows", [])
        if len(rows) == 0:
            continue

        x = [to_float(r.get("swap_same_rate", np.nan)) for r in rows]
        y1 = [to_float(r.get("mdi_orig", np.nan)) for r in rows]
        y2 = [to_float(r.get("effrank_image_orig", np.nan)) for r in rows]

        # finite mask
        idx = [i for i in range(len(x)) if np.isfinite(x[i]) and np.isfinite(y1[i])]
        if len(idx) > 5:
            scatter_plot([x[i] for i in idx], [y1[i] for i in idx],
                         f"{pl}: swap_same_rate vs mdi_orig", "swap_same_rate", "mdi_orig",
                         fig_dir / f"scatter_swap_vs_mdi_{pl}.png")

        idx = [i for i in range(len(x)) if np.isfinite(x[i]) and np.isfinite(y2[i])]
        if len(idx) > 5:
            scatter_plot([x[i] for i in idx], [y2[i] for i in idx],
                         f"{pl}: swap_same_rate vs effrank_image_orig", "swap_same_rate", "effrank_image_orig",
                         fig_dir / f"scatter_swap_vs_effimg_{pl}.png")

    # -------------------------
    # finish
    # -------------------------
    print("Saved compare table:", str(compare_csv))
    print("Figures:", str(fig_dir))
    print("Unexpected:", str(unexpected_dir))
    if (out_dir / "step2_vs_step3_delta.csv").exists():
        print("Delta table:", str(out_dir / "step2_vs_step3_delta.csv"))


if __name__ == "__main__":
    main()
