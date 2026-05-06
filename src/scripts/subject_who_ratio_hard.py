"""Audit follow-up #36: Hard-subset subject-stratified WHO ratio (§6ss).

§6cc reported the hard-subset primary × helper WHO ratio as 67.69×
(vs full-grid 22.11× from §6m). §6rr showed the full-grid ratio
survives subject-stratification (subject/helper = 21.73× ≈ 22.11×).
This script extends §6rr to the hard subset: build the (subject, helper)
hard-only cell-mean delta matrix and decompose variance.

If subject-stratification preserves the hard ratio (~50–70×), then
§6cc's hard primary effect is primary-identity-driven (the same
identity that generates the §6m full-grid effect). If subject-
stratification collapses it (e.g. <10×), then §6cc was largely
subject-mix-driven (within-primary subjects vary in recoverability).

Cell-mean delta on hard questions is computed as:
  pair_acc - solo_acc, restricted to questions where solo was wrong.
Note: solo_acc on the hard subset is by construction 0 (all solo wrong),
so the cell-mean delta on hard equals pair_acc on hard = W2C rate.

This is the more interpretable formulation and matches §6hh's W2C rates.

Filter to subjects with n_hard >= 4 to keep cell estimates reasonable.

Output: results/verified_pair_grid_qwen3_1p7b/subject_who_ratio_hard.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/subject_who_ratio_hard.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS


def two_way_anova_decomposition(
    cell_means: np.ndarray,
    cell_ns: np.ndarray | None = None,
) -> dict:
    """Compute additive-model SS decomposition for a R × C cell-mean matrix."""
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
        "grand_mean": float(grand),
        "row_means": row_mean.tolist(),
        "col_means": col_mean.tolist(),
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

    # Map (primary, subject, idx) for hard-only set
    subject_to_primary: dict[str, str] = {}
    primary_subject_hard_idxs: dict[tuple[str, str], list[int]] = defaultdict(list)

    for p in DOMAINS:
        solo_pq = cond[f"solo_{p}"]["per_q"]
        for q in solo_pq:
            s = q["subject"]
            subject_to_primary[s] = p
            if not q["correct"]:
                primary_subject_hard_idxs[(p, s)].append(q["idx"])

    subjects = sorted(
        subject_to_primary.keys(),
        key=lambda s: (DOMAINS.index(subject_to_primary[s]), s),
    )

    # Hard-only subject × helper cell mean (= W2C rate, since solo_acc on
    # this restricted set is 0 by construction)
    n_subj = len(subjects)
    cell_w2c = np.full((n_subj, len(HELPERS)), np.nan)
    cell_n_hard = np.zeros((n_subj, len(HELPERS)), dtype=int)

    for i, s in enumerate(subjects):
        p = subject_to_primary[s]
        idxs = primary_subject_hard_idxs[(p, s)]
        if not idxs:
            continue
        for j, h in enumerate(HELPERS):
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            pair_correct = np.array([pair_pq[idx]["correct"] for idx in idxs])
            cell_w2c[i, j] = pair_correct.mean()
            cell_n_hard[i, j] = len(idxs)

    # Filter to subjects with n_hard >= 4 (per-subject across helpers,
    # use first column count which is uniform across helpers)
    n_min_hard = 4
    mask = np.array([cell_n_hard[i, 0] >= n_min_hard for i in range(n_subj)])
    filt_subjects = [s for s, m in zip(subjects, mask) if m]
    filt_idx = [i for i, m in enumerate(mask) if m]
    filt_cell = cell_w2c[filt_idx, :]
    filt_n = cell_n_hard[filt_idx, :]

    # Reference §6cc primary × helper hard W2C rate
    primary_helper_w2c = np.zeros((len(DOMAINS), len(HELPERS)))
    primary_helper_n = np.zeros((len(DOMAINS), len(HELPERS)), dtype=int)
    for i, p in enumerate(DOMAINS):
        solo_pq = cond[f"solo_{p}"]["per_q"]
        hard_idxs = [q["idx"] for q in solo_pq if not q["correct"]]
        for j, h in enumerate(HELPERS):
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            primary_helper_w2c[i, j] = np.mean([pair_pq[idx]["correct"] for idx in hard_idxs])
            primary_helper_n[i, j] = len(hard_idxs)

    anova_primary = two_way_anova_decomposition(primary_helper_w2c, primary_helper_n)
    anova_full_uniform = two_way_anova_decomposition(cell_w2c)
    anova_full_weighted = two_way_anova_decomposition(cell_w2c, cell_n_hard)
    anova_filt_uniform = two_way_anova_decomposition(filt_cell)
    anova_filt_weighted = two_way_anova_decomposition(filt_cell, filt_n)

    a = anova_filt_weighted
    f_subject = float(np.sqrt(a["frac_row"] / (1 - a["frac_row"]))) if a["frac_row"] < 1 else float("inf")
    f_helper = float(np.sqrt(a["frac_col"] / (1 - a["frac_col"]))) if a["frac_col"] < 1 else float("inf")
    a_p = anova_primary
    f_primary_hard = float(np.sqrt(a_p["frac_row"] / (1 - a_p["frac_row"]))) if a_p["frac_row"] < 1 else float("inf")
    f_helper_primary_hard = float(np.sqrt(a_p["frac_col"] / (1 - a_p["frac_col"]))) if a_p["frac_col"] < 1 else float("inf")

    # Subject row means on hard W2C
    subject_row_means_hard = []
    for i, s in enumerate(subjects):
        valid = ~np.isnan(cell_w2c[i, :])
        if valid.any():
            subject_row_means_hard.append(
                (s, subject_to_primary[s],
                 float(cell_w2c[i, valid].mean()),
                 int(cell_n_hard[i, 0]))
            )

    out = {
        "subjects": subjects,
        "filtered_subjects": filt_subjects,
        "subject_to_primary": subject_to_primary,
        "helpers": HELPERS,
        "cell_w2c_full": cell_w2c.tolist(),
        "cell_n_hard_full": cell_n_hard.tolist(),
        "cell_w2c_filtered": filt_cell.tolist(),
        "cell_n_hard_filtered": filt_n.tolist(),
        "anova_full_uniform": anova_full_uniform,
        "anova_full_weighted": anova_full_weighted,
        "anova_filtered_uniform": anova_filt_uniform,
        "anova_filtered_weighted": anova_filt_weighted,
        "anova_primary_helper_hard": anova_primary,
        "cohen_f_subject_filtered_weighted": f_subject,
        "cohen_f_helper_filtered_weighted": f_helper,
        "cohen_f_primary_hard": f_primary_hard,
        "cohen_f_helper_primary_hard": f_helper_primary_hard,
        "subject_row_means_hard": subject_row_means_hard,
        "headline": {
            "subject_helper_ratio_filtered_weighted": (
                a["frac_row"] / a["frac_col"] if a["frac_col"] > 0 else float("inf")
            ),
            "primary_helper_ratio_hard": (
                a_p["frac_row"] / a_p["frac_col"] if a_p["frac_col"] > 0 else float("inf")
            ),
        },
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 80)
    print("§6ss. Hard-subset subject-stratified WHO ratio")
    print("=" * 80)

    print(f"\nReference §6cc primary × helper hard:")
    a_p = anova_primary
    print(f"  SS_primary = {a_p['SS_row']:.4f} ({a_p['frac_row']*100:.1f}%)")
    print(f"  SS_helper  = {a_p['SS_col']:.4f} ({a_p['frac_col']*100:.1f}%)")
    print(f"  SS_inter   = {a_p['SS_interaction']:.4f} ({a_p['frac_interaction']*100:.1f}%)")
    print(f"  primary/helper ratio = {out['headline']['primary_helper_ratio_hard']:.2f}×")
    print(f"  Cohen's f primary = {f_primary_hard:.3f}, helper = {f_helper_primary_hard:.3f}")

    print(f"\nNew §6ss subject × helper hard (full {n_subj}, uniform):")
    a = anova_full_uniform
    print(f"  SS_subject = {a['SS_row']:.4f} ({a['frac_row']*100:.1f}%)")
    print(f"  SS_helper  = {a['SS_col']:.4f} ({a['frac_col']*100:.1f}%)")
    print(f"  SS_inter   = {a['SS_interaction']:.4f} ({a['frac_interaction']*100:.1f}%)")

    print(f"\nNew §6ss subject × helper hard (full {n_subj}, n-weighted):")
    a = anova_full_weighted
    print(f"  SS_subject = {a['SS_row']:.4f} ({a['frac_row']*100:.1f}%)")
    print(f"  SS_helper  = {a['SS_col']:.4f} ({a['frac_col']*100:.1f}%)")
    print(f"  SS_inter   = {a['SS_interaction']:.4f} ({a['frac_interaction']*100:.1f}%)")

    print(f"\nNew §6ss subject × helper hard (FILTERED n_hard>=4, n-weighted):")
    a = anova_filt_weighted
    print(f"  SS_subject = {a['SS_row']:.4f} ({a['frac_row']*100:.1f}%)")
    print(f"  SS_helper  = {a['SS_col']:.4f} ({a['frac_col']*100:.1f}%)")
    print(f"  SS_inter   = {a['SS_interaction']:.4f} ({a['frac_interaction']*100:.1f}%)")
    print(f"  subject/helper ratio = {out['headline']['subject_helper_ratio_filtered_weighted']:.2f}×")
    print(f"  Cohen's f subject = {f_subject:.3f}, helper = {f_helper:.3f}")

    print(f"\nFiltered subjects (n_hard >= {n_min_hard}, n-weighted row mean = W2C rate on hard):")
    for s, p, m, n in subject_row_means_hard:
        if s in filt_subjects:
            print(f"  {p:10s} {s:32s} n_hard={n:>2d}  W2C_rate={m*100:>5.1f}%")
    print()
    print(f"Excluded (n_hard < {n_min_hard}):")
    for s, p, m, n in subject_row_means_hard:
        if s not in filt_subjects:
            print(f"  {p:10s} {s:32s} n_hard={n}")

    print()
    print(f"COMPARISON:")
    print(f"  §6cc primary × helper hard  : ratio = {out['headline']['primary_helper_ratio_hard']:.1f}×")
    print(f"  §6ss subject × helper hard  : ratio = {out['headline']['subject_helper_ratio_filtered_weighted']:.1f}×")
    print(f"  §6m  primary × helper full  : ratio = 22.1×  (§6rr subject = 21.7×)")


if __name__ == "__main__":
    main()
