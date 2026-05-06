"""Audit follow-up #43: subject-jackknife on §6ss hard subject WHO ratio (§6zz).

§6y did specialist-jackknife on the 5 primaries (full grid) — LOO range
10.37–31.33×. §6gg did it on hard — range 15.58–177.14×. This script
extends the LOO logic to subjects: drop one of 17 filtered subjects
at a time and recompute the §6ss subject × helper hard variance ratio.

Hypothesis: range 30-90× across 17 LOO drops, with biology subjects
(hs_biology n=22, college_biology n=9) causing the largest drops
because they dominate the high-W2C tail.

Output: results/verified_pair_grid_qwen3_1p7b/subject_jackknife_hard_who.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/subject_jackknife_hard_who.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_MIN_HARD = 4


def two_way_anova_decomposition(
    cell_means: np.ndarray,
    cell_ns: np.ndarray | None = None,
) -> dict:
    if cell_ns is None:
        weights = np.ones_like(cell_means)
    else:
        weights = cell_ns.astype(float)
    valid = ~np.isnan(cell_means)
    cm = np.where(valid, cell_means, 0.0)
    w = np.where(valid, weights, 0.0)
    grand = (cm * w).sum() / w.sum() if w.sum() > 0 else 0.0
    row_w = w.sum(axis=1)
    row_mean = np.where(
        row_w > 0,
        (cm * w).sum(axis=1) / np.where(row_w > 0, row_w, 1.0),
        grand,
    )
    col_w = w.sum(axis=0)
    col_mean = np.where(
        col_w > 0,
        (cm * w).sum(axis=0) / np.where(col_w > 0, col_w, 1.0),
        grand,
    )
    pred_additive = row_mean[:, None] + col_mean[None, :] - grand
    interaction = cm - pred_additive
    ss_row = (w * ((row_mean[:, None] - grand) ** 2)).sum()
    ss_col = (w * ((col_mean[None, :] - grand) ** 2)).sum()
    ss_inter = (w * (interaction ** 2)).sum()
    ss_total = ss_row + ss_col + ss_inter
    return {
        "SS_row": float(ss_row),
        "SS_col": float(ss_col),
        "SS_interaction": float(ss_inter),
        "SS_total": float(ss_total),
        "frac_row": float(ss_row / ss_total) if ss_total > 0 else 0.0,
        "frac_col": float(ss_col / ss_total) if ss_total > 0 else 0.0,
        "frac_interaction": float(ss_inter / ss_total) if ss_total > 0 else 0.0,
    }


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]

    subject_to_primary: dict[str, str] = {}
    primary_subject_hard_idxs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for p in DOMAINS:
        solo_pq = cond[f"solo_{p}"]["per_q"]
        for q in solo_pq:
            s = q["subject"]
            subject_to_primary[s] = p
            if not q["correct"]:
                primary_subject_hard_idxs[(p, s)].append(q["idx"])

    # Build subject × helper hard W2C cell-mean matrix (filtered)
    subjects_all = sorted(
        subject_to_primary.keys(),
        key=lambda s: (DOMAINS.index(subject_to_primary[s]), s),
    )

    cell_w2c = {}
    cell_n_hard = {}
    for s in subjects_all:
        p = subject_to_primary[s]
        idxs = primary_subject_hard_idxs[(p, s)]
        if len(idxs) < N_MIN_HARD:
            continue
        row = np.zeros(len(HELPERS))
        for j, h in enumerate(HELPERS):
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            row[j] = np.mean([pair_pq[idx]["correct"] for idx in idxs])
        cell_w2c[s] = row
        cell_n_hard[s] = len(idxs)

    filt_subjects = list(cell_w2c.keys())
    print(f"Filtered subjects (n_hard >= {N_MIN_HARD}): {len(filt_subjects)}")

    # Baseline (all filtered subjects): §6ss reproduction
    cell_mat = np.array([cell_w2c[s] for s in filt_subjects])
    n_mat = np.array([[cell_n_hard[s]] * len(HELPERS) for s in filt_subjects])
    baseline = two_way_anova_decomposition(cell_mat, n_mat)
    baseline_ratio = baseline["frac_row"] / baseline["frac_col"] if baseline["frac_col"] > 0 else float("inf")
    print(f"Baseline (all 17 subjects, n-weighted): subject/helper = {baseline_ratio:.2f}×")

    # LOO: for each subject, drop it and recompute
    loo_results = []
    for s_drop in filt_subjects:
        keep = [s for s in filt_subjects if s != s_drop]
        cell_mat_loo = np.array([cell_w2c[s] for s in keep])
        n_mat_loo = np.array([[cell_n_hard[s]] * len(HELPERS) for s in keep])
        anova = two_way_anova_decomposition(cell_mat_loo, n_mat_loo)
        ratio = anova["frac_row"] / anova["frac_col"] if anova["frac_col"] > 0 else float("inf")
        delta = ratio - baseline_ratio
        loo_results.append({
            "dropped": s_drop,
            "primary": subject_to_primary[s_drop],
            "n_hard_dropped": cell_n_hard[s_drop],
            "subject_helper_ratio": ratio,
            "delta_vs_baseline": delta,
            "frac_subject": anova["frac_row"],
            "frac_helper": anova["frac_col"],
            "frac_interaction": anova["frac_interaction"],
        })

    # Sort by delta (most-leverage first)
    loo_results.sort(key=lambda r: r["delta_vs_baseline"])

    out = {
        "n_filtered_subjects": len(filt_subjects),
        "baseline": {
            "subject_helper_ratio": baseline_ratio,
            "frac_subject": baseline["frac_row"],
            "frac_helper": baseline["frac_col"],
            "frac_interaction": baseline["frac_interaction"],
        },
        "loo_results": loo_results,
        "min_ratio": float(min(r["subject_helper_ratio"] for r in loo_results)),
        "max_ratio": float(max(r["subject_helper_ratio"] for r in loo_results)),
        "min_delta": float(min(r["delta_vs_baseline"] for r in loo_results)),
        "max_delta": float(max(r["delta_vs_baseline"] for r in loo_results)),
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 90)
    print("§6zz. Subject-jackknife on §6ss hard subject/helper ratio")
    print("=" * 90)
    print(f"Baseline (all 17 subjects): subject/helper = {baseline_ratio:.2f}×")
    print(f"LOO range: {out['min_ratio']:.2f}× to {out['max_ratio']:.2f}×")
    print(f"LOO delta range: {out['min_delta']:+.2f} to {out['max_delta']:+.2f}")
    print()
    print("Subjects sorted by leverage (most-decreasing first):")
    print(f"{'rank':>4s} {'subject':32s} {'primary':9s} {'n_hard':>7s} "
          f"{'ratio':>9s} {'Δ vs base':>10s} {'frac_subj':>10s} {'frac_help':>10s}")
    for i, r in enumerate(loo_results):
        print(f"{i+1:>4d} {r['dropped']:32s} {r['primary']:9s} {r['n_hard_dropped']:>7d} "
              f"{r['subject_helper_ratio']:>8.2f}× {r['delta_vs_baseline']:>+9.2f} "
              f"{r['frac_subject']*100:>9.1f}% {r['frac_helper']*100:>9.1f}%")


if __name__ == "__main__":
    main()
