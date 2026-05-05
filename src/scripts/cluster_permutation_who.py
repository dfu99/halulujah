"""Cluster-permutation test on the WHO variance ratio (audit follow-up #15).

The §6t strict-shuffle test reassigned all 30 cells across the 5×6 grid,
ignoring the fact that within a primary row the same 50 questions flow
through all 6 helper conditions. Two more clustering-aware permutation
tests live here:

A. **Within-row shuffle.** For each primary row, permute the 6 helper-
   cell values across columns. Preserves SS_rows (same row means) and
   preserves question-clustering. Tests "is the observed SS_cols /
   SS_rows ratio unusual under within-row exchangeability of helpers?"

B. **Within-column shuffle.** For each helper column, permute the 5
   primary-cell values across rows. Preserves SS_cols (same col means).
   Mixes question pools (anti-conservative for primary effect testing
   when there is genuine within-primary clustering). Tests "is the
   observed SS_rows / SS_cols ratio unusual under within-column
   exchangeability of primaries?"

Outputs:
  results/verified_pair_grid_qwen3_1p7b/permutation_who_clustered.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/permutation_who_clustered.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]


def load_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text())


def variance_decomp(grid: np.ndarray) -> dict:
    grand = grid.mean()
    row_means = grid.mean(axis=1)
    col_means = grid.mean(axis=0)
    n_rows, n_cols = grid.shape
    ss_total = ((grid - grand) ** 2).sum()
    ss_rows = n_cols * ((row_means - grand) ** 2).sum()
    ss_cols = n_rows * ((col_means - grand) ** 2).sum()
    ratio = ss_rows / ss_cols if ss_cols > 0 else float("inf")
    return {
        "ss_total": float(ss_total),
        "ss_rows": float(ss_rows),
        "ss_cols": float(ss_cols),
        "ratio_rows_cols": float(ratio),
        "frac_rows": float(ss_rows / ss_total) if ss_total else 0.0,
        "frac_cols": float(ss_cols / ss_total) if ss_total else 0.0,
    }


def within_row_shuffle(grid: np.ndarray, n_iter: int, seed: int) -> dict:
    """Within each row, permute the column values. SS_rows is invariant."""
    rng = np.random.default_rng(seed)
    obs = variance_decomp(grid)
    null_ratios = []
    null_ss_cols = []
    null_frac_rows = []
    for _ in range(n_iter):
        ng = grid.copy()
        for r in range(grid.shape[0]):
            rng.shuffle(ng[r])
        d = variance_decomp(ng)
        null_ratios.append(d["ratio_rows_cols"])
        null_ss_cols.append(d["ss_cols"])
        null_frac_rows.append(d["frac_rows"])
    null_ratios = np.array(null_ratios)
    null_ss_cols = np.array(null_ss_cols)
    null_frac_rows = np.array(null_frac_rows)
    p_ratio = float((null_ratios >= obs["ratio_rows_cols"]).mean())
    p_ss_cols_low = float((null_ss_cols <= obs["ss_cols"]).mean())
    p_frac = float((null_frac_rows >= obs["frac_rows"]).mean())
    return {
        "n_iter": n_iter,
        "seed": seed,
        "observed": obs,
        "ss_rows_invariant": True,
        "ss_rows": obs["ss_rows"],
        "null_ratio": {
            "median": float(np.median(null_ratios)),
            "p2.5": float(np.percentile(null_ratios, 2.5)),
            "p97.5": float(np.percentile(null_ratios, 97.5)),
            "p99": float(np.percentile(null_ratios, 99)),
            "max": float(null_ratios.max()),
            "p_value_observed_geq": p_ratio,
        },
        "null_ss_cols": {
            "median": float(np.median(null_ss_cols)),
            "p2.5": float(np.percentile(null_ss_cols, 2.5)),
            "p97.5": float(np.percentile(null_ss_cols, 97.5)),
            "p_value_observed_leq": p_ss_cols_low,
        },
        "null_frac_rows": {
            "median": float(np.median(null_frac_rows)),
            "p_value_observed_geq": p_frac,
        },
    }


def within_column_shuffle(grid: np.ndarray, n_iter: int, seed: int) -> dict:
    """Within each column, permute the row values. SS_cols is invariant."""
    rng = np.random.default_rng(seed)
    obs = variance_decomp(grid)
    null_ratios = []
    null_ss_rows = []
    null_frac_rows = []
    for _ in range(n_iter):
        ng = grid.copy()
        for c in range(grid.shape[1]):
            rng.shuffle(ng[:, c])
        d = variance_decomp(ng)
        null_ratios.append(d["ratio_rows_cols"])
        null_ss_rows.append(d["ss_rows"])
        null_frac_rows.append(d["frac_rows"])
    null_ratios = np.array(null_ratios)
    null_ss_rows = np.array(null_ss_rows)
    null_frac_rows = np.array(null_frac_rows)
    p_ratio = float((null_ratios >= obs["ratio_rows_cols"]).mean())
    p_ss_rows_high = float((null_ss_rows >= obs["ss_rows"]).mean())
    p_frac = float((null_frac_rows >= obs["frac_rows"]).mean())
    return {
        "n_iter": n_iter,
        "seed": seed,
        "observed": obs,
        "ss_cols_invariant": True,
        "ss_cols": obs["ss_cols"],
        "null_ratio": {
            "median": float(np.median(null_ratios)),
            "p2.5": float(np.percentile(null_ratios, 2.5)),
            "p97.5": float(np.percentile(null_ratios, 97.5)),
            "p99": float(np.percentile(null_ratios, 99)),
            "max": float(null_ratios.max()),
            "p_value_observed_geq": p_ratio,
        },
        "null_ss_rows": {
            "median": float(np.median(null_ss_rows)),
            "p2.5": float(np.percentile(null_ss_rows, 2.5)),
            "p97.5": float(np.percentile(null_ss_rows, 97.5)),
            "p_value_observed_geq": p_ss_rows_high,
        },
        "null_frac_rows": {
            "median": float(np.median(null_frac_rows)),
            "p_value_observed_geq": p_frac,
        },
    }


def main() -> None:
    matrix = load_matrix()
    cond = matrix["conditions"]
    grid = np.zeros((5, 6))
    for i, p in enumerate(DOMAINS):
        solo_a = cond[f"solo_{p}"]["accuracy"]
        for j, h in enumerate(HELPERS):
            grid[i, j] = cond[f"pair_{p}_{h}"]["accuracy"] - solo_a

    n_iter = 5000

    print("=" * 64)
    print("Cluster-permutation test on WHO ratio (audit follow-up #15)")
    print("=" * 64)
    obs = variance_decomp(grid)
    print(f"Observed SS_rows={obs['ss_rows']:.4f}, SS_cols={obs['ss_cols']:.4f}, "
          f"ratio={obs['ratio_rows_cols']:.2f}×, frac_rows={obs['frac_rows']:.1%}")
    print()

    print(f"A. Within-row shuffle (preserves SS_rows, preserves Q-clustering)")
    print("-" * 64)
    a = within_row_shuffle(grid, n_iter=n_iter, seed=11)
    nul = a["null_ratio"]
    print(f"  null ratio median = {nul['median']:.2f}, 95% [{nul['p2.5']:.2f}, {nul['p97.5']:.2f}], "
          f"99th = {nul['p99']:.2f}, max = {nul['max']:.2f}")
    print(f"  p-value (ratio >= observed)        = {nul['p_value_observed_geq']:.4f}")
    print(f"  p-value (SS_cols <= observed)      = {a['null_ss_cols']['p_value_observed_leq']:.4f}")
    print(f"  p-value (frac_rows >= observed)    = {a['null_frac_rows']['p_value_observed_geq']:.4f}")
    print()

    print(f"B. Within-column shuffle (preserves SS_cols, mixes Q-pools)")
    print("-" * 64)
    b = within_column_shuffle(grid, n_iter=n_iter, seed=13)
    nul = b["null_ratio"]
    print(f"  null ratio median = {nul['median']:.2f}, 95% [{nul['p2.5']:.2f}, {nul['p97.5']:.2f}], "
          f"99th = {nul['p99']:.2f}, max = {nul['max']:.2f}")
    print(f"  p-value (ratio >= observed)        = {nul['p_value_observed_geq']:.4f}")
    print(f"  p-value (SS_rows >= observed)      = {b['null_ss_rows']['p_value_observed_geq']:.4f}")
    print(f"  p-value (frac_rows >= observed)    = {b['null_frac_rows']['p_value_observed_geq']:.4f}")
    print()

    out = {
        "observed": obs,
        "within_row_shuffle": a,
        "within_column_shuffle": b,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
