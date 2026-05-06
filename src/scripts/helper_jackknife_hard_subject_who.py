"""Audit follow-up #44: helper-jackknife on §6ss hard subject WHO ratio (§6aaa).

§6zz did subject-LOO on §6ss (range 25.87×–85.72× across 17 subjects).
This script completes the jackknife family at the HELPER axis: drop
one of 6 helpers at a time and recompute the §6ss subject × helper
hard variance ratio.

Hypothesis: range 40-80× across 6 helper drops, with base helper
drop producing the biggest decrease (since base is the lone helper-
side outlier per §6jj/§6yy).

Output: results/verified_pair_grid_qwen3_1p7b/helper_jackknife_hard_subject_who.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/helper_jackknife_hard_subject_who.json"

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

    # Baseline: §6ss reproduction with all 6 helpers
    cell_mat = np.array([cell_w2c[s] for s in filt_subjects])
    n_mat = np.array([[cell_n_hard[s]] * len(HELPERS) for s in filt_subjects])
    baseline = two_way_anova_decomposition(cell_mat, n_mat)
    baseline_ratio = baseline["frac_row"] / baseline["frac_col"] if baseline["frac_col"] > 0 else float("inf")
    print(f"Baseline (all 6 helpers, n-weighted): subject/helper = {baseline_ratio:.2f}×")

    # LOO: for each helper, drop the column and recompute
    loo_results = []
    for j_drop, h_drop in enumerate(HELPERS):
        keep_helper_idx = [j for j in range(len(HELPERS)) if j != j_drop]
        cell_mat_loo = cell_mat[:, keep_helper_idx]
        n_mat_loo = n_mat[:, keep_helper_idx]
        anova = two_way_anova_decomposition(cell_mat_loo, n_mat_loo)
        ratio = anova["frac_row"] / anova["frac_col"] if anova["frac_col"] > 0 else float("inf")
        delta = ratio - baseline_ratio
        # Also: helper col mean to interpret what we dropped
        helper_col_mean = float(np.mean(cell_mat[:, j_drop]))
        loo_results.append({
            "dropped": h_drop,
            "subject_helper_ratio": ratio,
            "delta_vs_baseline": delta,
            "frac_subject": anova["frac_row"],
            "frac_helper": anova["frac_col"],
            "frac_interaction": anova["frac_interaction"],
            "dropped_helper_col_mean_w2c": helper_col_mean,
        })

    loo_results.sort(key=lambda r: r["delta_vs_baseline"])

    out = {
        "n_filtered_subjects": len(filt_subjects),
        "n_helpers": len(HELPERS),
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
    print("=" * 80)
    print("§6aaa. Helper-jackknife on §6ss hard subject/helper ratio")
    print("=" * 80)
    print(f"Baseline (6 helpers): {baseline_ratio:.2f}×")
    print(f"LOO range: {out['min_ratio']:.2f}× to {out['max_ratio']:.2f}×")
    print(f"LOO delta range: {out['min_delta']:+.2f} to {out['max_delta']:+.2f}")
    print()
    print("Helper drops sorted by leverage (most-decreasing first):")
    print(f"{'rank':>4s} {'helper':10s} {'col_mean':>10s} {'ratio':>9s} "
          f"{'Δ vs base':>10s} {'frac_subj':>10s} {'frac_help':>10s}")
    for i, r in enumerate(loo_results):
        print(f"{i+1:>4d} {r['dropped']:10s} {r['dropped_helper_col_mean_w2c']*100:>9.1f}% "
              f"{r['subject_helper_ratio']:>8.2f}× {r['delta_vs_baseline']:>+9.2f} "
              f"{r['frac_subject']*100:>9.1f}% {r['frac_helper']*100:>9.1f}%")


if __name__ == "__main__":
    main()
