"""Subject-stratified WHO-asymmetry computation (audit follow-up #12).

Re-aggregates `results/verified_pair_grid_qwen3_1p7b/matrix_results.json`
by (primary, subject, helper) instead of (primary, helper).

The §6n audit finding is that within-primary subject delta spread
(math 43 pp, law 43 pp) rivals between-primary row spread (34 pp). This
script tests whether the §6m variance decomposition flips from
primary-dominated to subject-within-primary-dominated when we slice
the rows finer.

Outputs:
  results/verified_pair_grid_qwen3_1p7b/subject_stratified_who.json

Variance decomposition (2-way ANOVA without replicates on the
subject × helper grid):
  SS_total       = sum of squared deviations of cells from grand mean
  SS_subject     = n_helper * sum of squared row-mean deviations
  SS_helper      = n_subject * sum of squared col-mean deviations
  SS_residual    = SS_total - SS_subject - SS_helper

Note: SS_subject here folds together "subject identity" + "primary
identity" (each subject belongs to exactly one primary). To separate
them we compute a hierarchical decomposition:
  SS_primary_marginal = pool subjects within primary, then compute
                        primary-level row-spread.
  SS_subject_within_primary = SS_subject - SS_primary_marginal.

This isolates how much of the row-level variance is between primaries
vs within-primary across subjects.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/subject_stratified_who.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]


def load_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text())


def build_subject_helper_acc(matrix: dict) -> dict:
    """Returns:
      acc[(primary, subject, helper)] = (n_correct, n_total, mean_acc)
      solo[(primary, subject)]        = (n_correct, n_total, mean_acc)
    """
    pair_acc = defaultdict(lambda: [0, 0])
    solo_acc = defaultdict(lambda: [0, 0])

    cond = matrix["conditions"]
    for primary in DOMAINS:
        # Solo
        sk = f"solo_{primary}"
        for q in cond[sk]["per_q"]:
            subj = q["subject"]
            solo_acc[(primary, subj)][1] += 1
            if q["correct"]:
                solo_acc[(primary, subj)][0] += 1
        # Pair
        for helper in HELPERS:
            pk = f"pair_{primary}_{helper}"
            for q in cond[pk]["per_q"]:
                subj = q["subject"]
                pair_acc[(primary, subj, helper)][1] += 1
                if q["correct"]:
                    pair_acc[(primary, subj, helper)][0] += 1

    pair_out = {
        k: (n_c, n, n_c / n if n else 0.0)
        for k, (n_c, n) in pair_acc.items()
    }
    solo_out = {
        k: (n_c, n, n_c / n if n else 0.0)
        for k, (n_c, n) in solo_acc.items()
    }
    return {"pair": pair_out, "solo": solo_out}


def variance_decomp(grid: np.ndarray) -> dict:
    """2-way ANOVA without replicates.

    grid[i, j] is the cell value (delta or raw acc) for row i, col j.
    """
    grand = grid.mean()
    row_means = grid.mean(axis=1)
    col_means = grid.mean(axis=0)
    n_rows, n_cols = grid.shape

    ss_total = ((grid - grand) ** 2).sum()
    ss_rows = n_cols * ((row_means - grand) ** 2).sum()
    ss_cols = n_rows * ((col_means - grand) ** 2).sum()
    ss_residual = ss_total - ss_rows - ss_cols

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "grand_mean": float(grand),
        "ss_total": float(ss_total),
        "ss_rows": float(ss_rows),
        "ss_cols": float(ss_cols),
        "ss_residual": float(ss_residual),
        "frac_rows": float(ss_rows / ss_total) if ss_total else 0.0,
        "frac_cols": float(ss_cols / ss_total) if ss_total else 0.0,
        "frac_residual": float(ss_residual / ss_total) if ss_total else 0.0,
        "ratio_rows_cols": float(ss_rows / ss_cols) if ss_cols else float("inf"),
    }


def hierarchical_subject_decomp(
    delta_by_sh: dict[tuple[str, str, str], float],
    primary_of_subject: dict[str, str],
) -> dict:
    """Decompose SS_rows on the (subject × helper) grid into:

      SS_primary_marginal   = variance attributable to primary identity
                              after pooling all subjects within each primary
      SS_subject_in_primary = additional variance from subject heterogeneity
                              within primaries (residual at the row level)

    SS_rows from variance_decomp(subject_grid) = SS_primary_marginal + SS_subject_in_primary.
    """
    subjects = sorted({s for (p, s, h) in delta_by_sh.keys()})
    helpers = sorted({h for (p, s, h) in delta_by_sh.keys()})

    # subject-level row means (averaged over helpers)
    subj_row_means = {}
    for s in subjects:
        vals = [delta_by_sh[(primary_of_subject[s], s, h)] for h in helpers]
        subj_row_means[s] = float(np.mean(vals))

    # primary-level row means (averaged over all subjects within primary, all helpers)
    primary_row_means = {}
    for p in DOMAINS:
        vals = []
        for s in subjects:
            if primary_of_subject[s] != p:
                continue
            for h in helpers:
                vals.append(delta_by_sh[(p, s, h)])
        primary_row_means[p] = float(np.mean(vals)) if vals else 0.0

    # grand mean across all (subj, helper) cells
    all_vals = list(delta_by_sh.values())
    grand = float(np.mean(all_vals))
    n_helper = len(helpers)
    n_subject = len(subjects)

    # SS_subject (= SS_rows on the subject × helper grid)
    ss_subject = n_helper * sum((v - grand) ** 2 for v in subj_row_means.values())

    # SS_primary_marginal: each primary contributes (n_subject_within_primary × n_helper) cells
    ss_primary_marginal = 0.0
    for p in DOMAINS:
        n_subj_p = sum(1 for s in subjects if primary_of_subject[s] == p)
        ss_primary_marginal += n_subj_p * n_helper * (primary_row_means[p] - grand) ** 2

    ss_subject_within_primary = ss_subject - ss_primary_marginal

    return {
        "ss_subject_total": float(ss_subject),
        "ss_primary_marginal": float(ss_primary_marginal),
        "ss_subject_within_primary": float(ss_subject_within_primary),
        "frac_primary_of_subject_total": (
            float(ss_primary_marginal / ss_subject) if ss_subject else 0.0
        ),
        "frac_within_primary_of_subject_total": (
            float(ss_subject_within_primary / ss_subject) if ss_subject else 0.0
        ),
    }


def main() -> None:
    print(f"Loading {MATRIX_PATH}")
    matrix = load_matrix()
    accs = build_subject_helper_acc(matrix)
    pair = accs["pair"]
    solo = accs["solo"]

    # Pick subjects with at least n_min total records summed over helpers
    # to exclude super-tiny subjects (e.g. 1-2 questions) that would
    # dominate variance from sampling noise.
    subj_records = defaultdict(int)
    for (p, s, h), (_, n, _) in pair.items():
        subj_records[(p, s)] += n
    n_min = 12  # at least ~2 records per cell averaged across 6 helpers
    keep = {ps for ps, n in subj_records.items() if n >= n_min}
    print(f"Keeping {len(keep)} (primary, subject) tuples with >= {n_min} total pair records")

    primary_of_subject = {s: p for (p, s) in keep}
    helpers = list(HELPERS)
    subjects_kept = sorted({s for (p, s) in keep})

    # Build delta cells (pair − solo) on (subject, helper) grid
    delta_by_sh = {}
    raw_by_sh = {}
    for (p, s) in keep:
        if (p, s) not in solo:
            continue
        solo_acc = solo[(p, s)][2]
        for h in helpers:
            if (p, s, h) not in pair:
                continue
            pair_acc = pair[(p, s, h)][2]
            delta_by_sh[(p, s, h)] = pair_acc - solo_acc
            raw_by_sh[(p, s, h)] = pair_acc

    # Build the n_subject × n_helper grid for ANOVA
    grid_delta = np.array([
        [delta_by_sh[(primary_of_subject[s], s, h)] for h in helpers]
        for s in subjects_kept
    ])
    grid_raw = np.array([
        [raw_by_sh[(primary_of_subject[s], s, h)] for h in helpers]
        for s in subjects_kept
    ])

    decomp_delta = variance_decomp(grid_delta)
    decomp_raw = variance_decomp(grid_raw)

    hier_delta = hierarchical_subject_decomp(delta_by_sh, primary_of_subject)
    hier_raw = hierarchical_subject_decomp(raw_by_sh, primary_of_subject)

    # Compare with §6m headline (primary-only 5×6 grid)
    prim_grid_delta = []
    prim_grid_raw = []
    for p in DOMAINS:
        # primary delta = pair_acc(primary, helper) − solo_acc(primary)
        # use pair/solo aggregate over all subjects within primary
        solo_correct = sum(solo[(p, s)][0] for s in subjects_kept if primary_of_subject[s] == p)
        solo_total = sum(solo[(p, s)][1] for s in subjects_kept if primary_of_subject[s] == p)
        solo_p_acc = solo_correct / solo_total if solo_total else 0.0
        row_d = []
        row_r = []
        for h in helpers:
            n_c = sum(pair[(p, s, h)][0] for s in subjects_kept if primary_of_subject[s] == p)
            n_t = sum(pair[(p, s, h)][1] for s in subjects_kept if primary_of_subject[s] == p)
            pair_p_acc = n_c / n_t if n_t else 0.0
            row_d.append(pair_p_acc - solo_p_acc)
            row_r.append(pair_p_acc)
        prim_grid_delta.append(row_d)
        prim_grid_raw.append(row_r)
    prim_grid_delta = np.array(prim_grid_delta)
    prim_grid_raw = np.array(prim_grid_raw)
    decomp_primary_delta = variance_decomp(prim_grid_delta)
    decomp_primary_raw = variance_decomp(prim_grid_raw)

    out = {
        "n_subjects_kept": len(subjects_kept),
        "subjects_kept_by_primary": {
            p: sorted([s for s in subjects_kept if primary_of_subject[s] == p])
            for p in DOMAINS
        },
        "subject_grid_decomposition_delta": decomp_delta,
        "subject_grid_decomposition_raw": decomp_raw,
        "primary_only_decomposition_delta": decomp_primary_delta,
        "primary_only_decomposition_raw": decomp_primary_raw,
        "hierarchical_decomp_delta": hier_delta,
        "hierarchical_decomp_raw": hier_raw,
        "interpretation": {
            "subject_grid_rows": "n_subjects_kept",
            "subject_grid_cols": "6 helpers",
            "primary_grid_rows": "5 primaries",
            "primary_grid_cols": "6 helpers",
        },
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")

    # Print a punch-line summary
    print()
    print("=" * 64)
    print("Subject-stratified WHO ratio (audit follow-up #12)")
    print("=" * 64)
    print(f"Kept subjects (>= {n_min} records total): {len(subjects_kept)}")
    print()
    print(f"{'metric':40s} {'subj-grid':>10s} {'prim-grid':>10s}")
    print("-" * 64)

    def fmt(d, k, p=3):
        return f"{d[k]:.{p}f}"

    for k, label in [
        ("frac_rows", "rows-of-total"),
        ("frac_cols", "cols-of-total"),
        ("frac_residual", "residual"),
        ("ratio_rows_cols", "rows/cols ratio"),
    ]:
        s_sub = (
            f"{decomp_delta[k]:.3f}" if k != "ratio_rows_cols"
            else f"{decomp_delta[k]:.2f}×"
        )
        s_prim = (
            f"{decomp_primary_delta[k]:.3f}" if k != "ratio_rows_cols"
            else f"{decomp_primary_delta[k]:.2f}×"
        )
        print(f"{label:40s} {s_sub:>10s} {s_prim:>10s}")

    print()
    print("Hierarchical decomp on the subject grid (delta):")
    print(f"  SS_subject_total            = {hier_delta['ss_subject_total']:.4f}")
    print(f"  SS_primary_marginal         = {hier_delta['ss_primary_marginal']:.4f} "
          f"({hier_delta['frac_primary_of_subject_total']:.1%} of subject_total)")
    print(f"  SS_subject_within_primary   = {hier_delta['ss_subject_within_primary']:.4f} "
          f"({hier_delta['frac_within_primary_of_subject_total']:.1%} of subject_total)")


if __name__ == "__main__":
    main()
