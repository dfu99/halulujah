"""Audit follow-up #35: Subject-stratified WHO ratio (§6rr).

§6m attributed 83.3% of variance to primary-identity, 3.8% to helper-
identity, and 12.9% to interaction on the 5×6 primary × helper cell-
mean delta matrix. §6qq showed within-primary subject heterogeneity
(math 20–75% mutual-unrec, medicine 0–60%, biology 14–33%) matches
between-primary spread, raising the subject-mix confound.

This script tests whether the §6m primary-effect-dominates finding
survives when primary-identity (5 levels) is replaced by subject-
identity (19 levels). Build the subject × helper cell-mean delta
matrix, run two-way ANOVA, and compare:

  Original §6m   : SS_primary    = 83.3%  SS_helper = 3.8%  SS_interaction = 12.9%
  New §6rr       : SS_subject    = ?      SS_helper = ?     SS_interaction = ?

If SS_subject > SS_primary, subject identity is a more predictive
factor than primary identity — directly confirming the §6qq confound.

Filter to subjects with n_questions >= 5 to keep cell deltas reasonably
estimated. Smaller subjects (jurisprudence n=4, college_mathematics
n=6) get pooled into a "small-subject residual" or excluded; we'll
report both filtered and full-grid versions.

Output: results/verified_pair_grid_qwen3_1p7b/subject_who_ratio.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/subject_who_ratio.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS


def two_way_anova_decomposition(
    cell_means: np.ndarray,
    cell_ns: np.ndarray | None = None,
) -> dict:
    """Compute SS decomposition for a two-way (R × C) cell-mean matrix.

    Cells are weighted by n if cell_ns provided, else equally.

    Returns dict with SS_row, SS_col, SS_interaction, SS_total, and
    fractions thereof. The decomposition uses the "anova-on-cell-means"
    convention used in §6m: total SS is sum of squared deviations of
    each cell mean from the grand mean.
    """
    if cell_ns is None:
        weights = np.ones_like(cell_means)
    else:
        weights = cell_ns.astype(float)
    valid = ~np.isnan(cell_means)
    # Keep zeros where invalid so they don't contribute
    cm = np.where(valid, cell_means, 0.0)
    w = np.where(valid, weights, 0.0)
    # Grand mean weighted
    grand = (cm * w).sum() / w.sum() if w.sum() > 0 else 0.0
    # Row means weighted
    row_w = w.sum(axis=1)
    row_mean = np.where(row_w > 0, (cm * w).sum(axis=1) / np.where(row_w > 0, row_w, 1.0), grand)
    # Col means weighted
    col_w = w.sum(axis=0)
    col_mean = np.where(col_w > 0, (cm * w).sum(axis=0) / np.where(col_w > 0, col_w, 1.0), grand)

    # Predicted under additive model
    pred_additive = row_mean[:, None] + col_mean[None, :] - grand
    interaction = cm - pred_additive
    # SS components (weighted)
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

    # First: compute (primary, subject, helper) cell delta = mean(pair correct)
    # − mean(solo correct) restricted to questions in `subject`.
    # Solo cell value: solo accuracy on questions of subject s in primary p.
    subject_to_primary: dict[str, str] = {}
    primary_subject_idxs: dict[tuple[str, str], list[int]] = defaultdict(list)
    primary_subject_solo_correct: dict[tuple[str, str], list[bool]] = defaultdict(list)

    for p in DOMAINS:
        solo_pq = cond[f"solo_{p}"]["per_q"]
        for q in solo_pq:
            s = q["subject"]
            subject_to_primary[s] = p
            primary_subject_idxs[(p, s)].append(q["idx"])
            primary_subject_solo_correct[(p, s)].append(q["correct"])

    subjects = sorted(subject_to_primary.keys(),
                      key=lambda s: (DOMAINS.index(subject_to_primary[s]), s))

    # Now build subject × helper cell mean delta and cell n
    n_subj = len(subjects)
    cell_delta = np.full((n_subj, len(HELPERS)), np.nan)
    cell_n = np.zeros((n_subj, len(HELPERS)), dtype=int)

    for i, s in enumerate(subjects):
        p = subject_to_primary[s]
        idxs = primary_subject_idxs[(p, s)]
        solo_correct = np.array(primary_subject_solo_correct[(p, s)])
        solo_acc = solo_correct.mean()
        for j, h in enumerate(HELPERS):
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            pair_correct = np.array([pair_pq[idx]["correct"] for idx in idxs])
            pair_acc = pair_correct.mean()
            cell_delta[i, j] = pair_acc - solo_acc
            cell_n[i, j] = len(idxs)

    # Filter to subjects with n_total >= 5
    n_min = 5
    mask = np.array([cell_n[i, 0] >= n_min for i in range(n_subj)])
    filt_subjects = [s for s, m in zip(subjects, mask) if m]
    filt_idx = [i for i, m in enumerate(mask) if m]
    filt_delta = cell_delta[filt_idx, :]
    filt_n = cell_n[filt_idx, :]

    # Run two-way ANOVA on full (19×6) and filtered (~14×6)
    anova_full_uniform = two_way_anova_decomposition(cell_delta)
    anova_full_weighted = two_way_anova_decomposition(cell_delta, cell_n)
    anova_filt_uniform = two_way_anova_decomposition(filt_delta)
    anova_filt_weighted = two_way_anova_decomposition(filt_delta, filt_n)

    # Reference §6m primary × helper decomposition (recomputed for sanity)
    primary_helper_delta = np.zeros((len(DOMAINS), len(HELPERS)))
    primary_helper_n = np.zeros((len(DOMAINS), len(HELPERS)), dtype=int)
    for i, p in enumerate(DOMAINS):
        solo_pq = cond[f"solo_{p}"]["per_q"]
        solo_acc = np.mean([q["correct"] for q in solo_pq])
        n_total = len(solo_pq)
        for j, h in enumerate(HELPERS):
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            pair_acc = np.mean([q["correct"] for q in pair_pq])
            primary_helper_delta[i, j] = pair_acc - solo_acc
            primary_helper_n[i, j] = n_total

    anova_primary = two_way_anova_decomposition(primary_helper_delta)

    # Subject row means, helper col means
    subject_row_means = []
    for i, s in enumerate(subjects):
        valid = ~np.isnan(cell_delta[i, :])
        if valid.any():
            subject_row_means.append((s, subject_to_primary[s],
                                       float(cell_delta[i, valid].mean()),
                                       int(cell_n[i, 0])))

    # Compute Cohen's f for subject and helper from filtered weighted ANOVA
    a = anova_filt_weighted
    f_subject = float(np.sqrt(a["frac_row"] / (1 - a["frac_row"]))) if a["frac_row"] < 1 else float("inf")
    f_helper = float(np.sqrt(a["frac_col"] / (1 - a["frac_col"]))) if a["frac_col"] < 1 else float("inf")
    a_p = anova_primary
    f_primary = float(np.sqrt(a_p["frac_row"] / (1 - a_p["frac_row"]))) if a_p["frac_row"] < 1 else float("inf")
    f_helper_orig = float(np.sqrt(a_p["frac_col"] / (1 - a_p["frac_col"]))) if a_p["frac_col"] < 1 else float("inf")

    out = {
        "subjects": subjects,
        "filtered_subjects": filt_subjects,
        "subject_to_primary": subject_to_primary,
        "helpers": HELPERS,
        "cell_delta_full": cell_delta.tolist(),
        "cell_n_full": cell_n.tolist(),
        "cell_delta_filtered": filt_delta.tolist(),
        "cell_n_filtered": filt_n.tolist(),
        "anova_full_uniform": anova_full_uniform,
        "anova_full_weighted": anova_full_weighted,
        "anova_filtered_uniform": anova_filt_uniform,
        "anova_filtered_weighted": anova_filt_weighted,
        "anova_primary_helper_recomputed": anova_primary,
        "cohen_f_subject_filtered_weighted": f_subject,
        "cohen_f_helper_filtered_weighted": f_helper,
        "cohen_f_primary_recomputed": f_primary,
        "cohen_f_helper_recomputed": f_helper_orig,
        "subject_row_means": subject_row_means,
        "headline": {
            "subject_helper_ratio_filtered_weighted": (
                a["frac_row"] / a["frac_col"] if a["frac_col"] > 0 else float("inf")
            ),
            "primary_helper_ratio_original": (
                a_p["frac_row"] / a_p["frac_col"] if a_p["frac_col"] > 0 else float("inf")
            ),
        },
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 80)
    print("§6rr. Subject-stratified WHO ratio")
    print("=" * 80)

    print(f"\nReference §6m primary × helper recompute (sanity check):")
    a_p = anova_primary
    print(f"  SS_primary = {a_p['SS_row']:.4f} ({a_p['frac_row']*100:.1f}%)")
    print(f"  SS_helper  = {a_p['SS_col']:.4f} ({a_p['frac_col']*100:.1f}%)")
    print(f"  SS_inter   = {a_p['SS_interaction']:.4f} ({a_p['frac_interaction']*100:.1f}%)")
    print(f"  primary/helper ratio = {out['headline']['primary_helper_ratio_original']:.2f}×")
    print(f"  Cohen's f primary = {f_primary:.3f}, helper = {f_helper_orig:.3f}")

    print(f"\nNew §6rr subject × helper (full {n_subj} subjects, uniform weights):")
    a = anova_full_uniform
    print(f"  SS_subject = {a['SS_row']:.4f} ({a['frac_row']*100:.1f}%)")
    print(f"  SS_helper  = {a['SS_col']:.4f} ({a['frac_col']*100:.1f}%)")
    print(f"  SS_inter   = {a['SS_interaction']:.4f} ({a['frac_interaction']*100:.1f}%)")

    print(f"\nNew §6rr subject × helper (full {n_subj} subjects, n-weighted):")
    a = anova_full_weighted
    print(f"  SS_subject = {a['SS_row']:.4f} ({a['frac_row']*100:.1f}%)")
    print(f"  SS_helper  = {a['SS_col']:.4f} ({a['frac_col']*100:.1f}%)")
    print(f"  SS_inter   = {a['SS_interaction']:.4f} ({a['frac_interaction']*100:.1f}%)")

    print(f"\nNew §6rr subject × helper (FILTERED {len(filt_subjects)} subjects with n>=5, n-weighted):")
    a = anova_filt_weighted
    print(f"  SS_subject = {a['SS_row']:.4f} ({a['frac_row']*100:.1f}%)")
    print(f"  SS_helper  = {a['SS_col']:.4f} ({a['frac_col']*100:.1f}%)")
    print(f"  SS_inter   = {a['SS_interaction']:.4f} ({a['frac_interaction']*100:.1f}%)")
    print(f"  subject/helper ratio = {out['headline']['subject_helper_ratio_filtered_weighted']:.2f}×")
    print(f"  Cohen's f subject = {f_subject:.3f}, helper = {f_helper:.3f}")

    print(f"\nFiltered subjects (n>=5):")
    for s in filt_subjects:
        p = subject_to_primary[s]
        i = subjects.index(s)
        n = int(cell_n[i, 0])
        rmean = float(np.nanmean(cell_delta[i, :])) * 100
        print(f"  {p:10s} {s:32s} n={n:>2d}  row_mean={rmean:+.1f} pp")

    print(f"\nExcluded (n<5):")
    for s in subjects:
        if s not in filt_subjects:
            p = subject_to_primary[s]
            i = subjects.index(s)
            n = int(cell_n[i, 0])
            print(f"  {p:10s} {s:32s} n={n}")

    print()
    print(f"COMPARISON:")
    print(f"  §6m primary × helper:  SS_primary = {a_p['frac_row']*100:.1f}%, "
          f"ratio = {out['headline']['primary_helper_ratio_original']:.1f}×")
    print(f"  §6rr subject × helper: SS_subject = {anova_filt_weighted['frac_row']*100:.1f}%, "
          f"ratio = {out['headline']['subject_helper_ratio_filtered_weighted']:.1f}×")


if __name__ == "__main__":
    main()
