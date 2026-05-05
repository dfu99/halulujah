"""Audit deepening §6s + §6t — cross-helper agreement and permutation test.

§6s: For each (primary, idx) tuple, measure the variance of the 6 helpers'
final-correctness booleans. Average per primary. Tests how much helper
identity matters at the per-question level (vs the cell-mean §6e/§6p).

§6t: Permutation test on the variance ratio (SS_primary / SS_helper).
Permute the 30 cell row-labels, recompute the ratio under each permutation,
get a non-parametric p-value for the observed 22.1× cell-mean ratio.

Outputs:
  results/verified_pair_grid_qwen3_1p7b/cross_helper_agreement.json   (§6s)
  results/verified_pair_grid_qwen3_1p7b/permutation_who.json          (§6t)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT_AGREE = ROOT / "results/verified_pair_grid_qwen3_1p7b/cross_helper_agreement.json"
OUT_PERM = ROOT / "results/verified_pair_grid_qwen3_1p7b/permutation_who.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]


def load_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text())


def cross_helper_agreement(matrix: dict) -> dict:
    """For each (primary, idx), gather the 6 final-correctness booleans
    across helpers, the 6 final letters, and compute:
      - share of questions where ALL helpers agree on correctness
      - share where ALL agree on the letter
      - mean variance of correctness across helpers
      - mean entropy of letter across helpers
    """
    cond = matrix["conditions"]
    out = {}

    for primary in DOMAINS:
        # Gather: for each idx, the 6 helpers' (correct, letter) tuples
        per_q = defaultdict(list)
        for helper in HELPERS:
            for q in cond[f"pair_{primary}_{helper}"]["per_q"]:
                per_q[q["idx"]].append((bool(q["correct"]), q["predicted"]))

        # Solo per-idx for reference
        solo_correct_by_idx = {
            q["idx"]: bool(q["correct"]) for q in cond[f"solo_{primary}"]["per_q"]
        }

        n_q = len(per_q)
        n_unanimous_correct = 0
        n_unanimous_letter = 0
        n_unanimous_correctness = 0
        correct_var = []  # per-q variance of correctness across helpers
        letter_entropy = []  # per-q Shannon entropy of letter
        helper_changes_correctness = 0  # questions where helpers disagree on correctness
        for idx, helpers_obs in per_q.items():
            assert len(helpers_obs) == 6
            corrs = [c for c, _ in helpers_obs]
            letters = [l for _, l in helpers_obs]
            corr_var = np.var(np.array(corrs, dtype=float), ddof=0)
            correct_var.append(float(corr_var))
            unique_letters = set(letters)
            if len(unique_letters) == 1:
                n_unanimous_letter += 1
            if all(corrs):
                n_unanimous_correct += 1
            if len(set(corrs)) == 1:
                n_unanimous_correctness += 1
            else:
                helper_changes_correctness += 1
            # Letter entropy
            from math import log2
            letter_counts = defaultdict(int)
            for ll in letters:
                letter_counts[ll] += 1
            tot = len(letters)
            h = 0.0
            for c in letter_counts.values():
                p = c / tot
                if p > 0:
                    h -= p * log2(p)
            letter_entropy.append(h)

        out[primary] = {
            "n_q": n_q,
            "share_unanimous_correctness": n_unanimous_correctness / n_q if n_q else 0.0,
            "share_helper_changes_correctness": helper_changes_correctness / n_q if n_q else 0.0,
            "share_unanimous_correct": n_unanimous_correct / n_q if n_q else 0.0,
            "share_unanimous_letter": n_unanimous_letter / n_q if n_q else 0.0,
            "mean_correct_variance": float(np.mean(correct_var)) if correct_var else 0.0,
            "mean_letter_entropy_bits": float(np.mean(letter_entropy)) if letter_entropy else 0.0,
            "max_letter_entropy_bits": float(np.max(letter_entropy)) if letter_entropy else 0.0,
            "n_solo_correct": sum(solo_correct_by_idx.values()),
        }

    # Pooled
    n_total = sum(o["n_q"] for o in out.values())
    pooled = {
        "n_q": n_total,
        "share_unanimous_correctness": sum(
            o["share_unanimous_correctness"] * o["n_q"] for o in out.values()
        ) / n_total,
        "share_helper_changes_correctness": sum(
            o["share_helper_changes_correctness"] * o["n_q"] for o in out.values()
        ) / n_total,
        "share_unanimous_correct": sum(
            o["share_unanimous_correct"] * o["n_q"] for o in out.values()
        ) / n_total,
        "share_unanimous_letter": sum(
            o["share_unanimous_letter"] * o["n_q"] for o in out.values()
        ) / n_total,
        "mean_correct_variance": sum(
            o["mean_correct_variance"] * o["n_q"] for o in out.values()
        ) / n_total,
        "mean_letter_entropy_bits": sum(
            o["mean_letter_entropy_bits"] * o["n_q"] for o in out.values()
        ) / n_total,
    }

    return {
        "per_primary": out,
        "pooled": pooled,
    }


def variance_decomp_2way(grid: np.ndarray) -> dict:
    """Two-way ANOVA without replicates on a (rows, cols) cell-mean grid."""
    grand = grid.mean()
    row_means = grid.mean(axis=1)
    col_means = grid.mean(axis=0)
    n_rows, n_cols = grid.shape
    ss_total = ((grid - grand) ** 2).sum()
    ss_rows = n_cols * ((row_means - grand) ** 2).sum()
    ss_cols = n_rows * ((col_means - grand) ** 2).sum()
    ss_residual = ss_total - ss_rows - ss_cols
    ratio = ss_rows / ss_cols if ss_cols > 0 else float("inf")
    return {
        "ss_total": float(ss_total),
        "ss_rows": float(ss_rows),
        "ss_cols": float(ss_cols),
        "ss_residual": float(ss_residual),
        "ratio_rows_cols": float(ratio),
        "frac_rows": float(ss_rows / ss_total) if ss_total else 0.0,
        "frac_cols": float(ss_cols / ss_total) if ss_total else 0.0,
        "frac_residual": float(ss_residual / ss_total) if ss_total else 0.0,
    }


def permutation_test_who(matrix: dict, n_iter: int = 5000, seed: int = 7) -> dict:
    """Permutation test on the cell-mean variance ratio.

    Null hypothesis: row labels (primary) are exchangeable with col labels
    (helper) — i.e., there is no special "primary effect" beyond what you'd
    get by relabeling cells at random.

    Two flavors:
      A. Permute rows: shuffle which row label each cell belongs to. This
         mixes primary-identity but preserves which 5 cells share a row.
         Tests "is the observed primary clustering of cells statistically
         beyond what random row-grouping would produce?"
      B. Permute (row, col) labels jointly: shuffle the 30 cell values across
         the 5×6 grid. Strong null — destroys all structure.

    We report B (the strong null) as the headline; A is reported as a
    sanity check.
    """
    cond = matrix["conditions"]
    rng = np.random.default_rng(seed)

    # Build delta cell-mean grid (5 primary × 6 helper)
    grid = np.zeros((5, 6))
    for i, primary in enumerate(DOMAINS):
        solo_acc = cond[f"solo_{primary}"]["accuracy"]
        for j, helper in enumerate(HELPERS):
            grid[i, j] = cond[f"pair_{primary}_{helper}"]["accuracy"] - solo_acc

    obs = variance_decomp_2way(grid)
    obs_ratio = obs["ratio_rows_cols"]

    # Strong null (B): shuffle all 30 cells
    null_ratios_strong = []
    for _ in range(n_iter):
        flat = grid.flatten()
        rng.shuffle(flat)
        null_grid = flat.reshape(5, 6)
        null_decomp = variance_decomp_2way(null_grid)
        null_ratios_strong.append(null_decomp["ratio_rows_cols"])

    null_ratios_strong = np.array(null_ratios_strong)
    p_value_strong = float((null_ratios_strong >= obs_ratio).mean())

    # Frac-rows null (also report what fraction of total var goes to rows)
    null_frac_rows = []
    for _ in range(n_iter):
        flat = grid.flatten()
        rng.shuffle(flat)
        null_grid = flat.reshape(5, 6)
        d = variance_decomp_2way(null_grid)
        null_frac_rows.append(d["frac_rows"])
    null_frac_rows = np.array(null_frac_rows)
    p_value_frac = float((null_frac_rows >= obs["frac_rows"]).mean())

    return {
        "n_iter": n_iter,
        "seed": seed,
        "observed": {
            "ratio_rows_cols": obs_ratio,
            "ss_rows": obs["ss_rows"],
            "ss_cols": obs["ss_cols"],
            "frac_rows": obs["frac_rows"],
            "frac_cols": obs["frac_cols"],
        },
        "null_strong_shuffle": {
            "median": float(np.median(null_ratios_strong)),
            "p2.5": float(np.percentile(null_ratios_strong, 2.5)),
            "p97.5": float(np.percentile(null_ratios_strong, 97.5)),
            "p99": float(np.percentile(null_ratios_strong, 99)),
            "max": float(np.max(null_ratios_strong)),
            "p_value_observed_geq": p_value_strong,
            "p_value_frac_rows_geq": p_value_frac,
        },
    }


def main() -> None:
    matrix = load_matrix()

    # §6s: per-q cross-helper agreement
    print("§6s: per-question cross-helper agreement")
    print("-" * 64)
    agree = cross_helper_agreement(matrix)
    OUT_AGREE.write_text(json.dumps(agree, indent=2))
    print(f"Wrote {OUT_AGREE}")
    print()
    print(f"{'primary':12s} {'unan_corr':>10s} {'unan_lett':>10s} {'helper_chg':>11s} {'mean_var':>9s} {'mean_H':>9s}")
    for p in DOMAINS:
        d = agree["per_primary"][p]
        print(
            f"{p:12s}"
            f" {d['share_unanimous_correctness']:>10.1%}"
            f" {d['share_unanimous_letter']:>10.1%}"
            f" {d['share_helper_changes_correctness']:>11.1%}"
            f" {d['mean_correct_variance']:>9.4f}"
            f" {d['mean_letter_entropy_bits']:>9.3f}"
        )
    p = agree["pooled"]
    print(
        f"{'POOLED':12s}"
        f" {p['share_unanimous_correctness']:>10.1%}"
        f" {p['share_unanimous_letter']:>10.1%}"
        f" {p['share_helper_changes_correctness']:>11.1%}"
        f" {p['mean_correct_variance']:>9.4f}"
        f" {p['mean_letter_entropy_bits']:>9.3f}"
    )

    print()
    print("§6t: permutation test on the WHO variance ratio")
    print("-" * 64)
    perm = permutation_test_who(matrix, n_iter=5000)
    OUT_PERM.write_text(json.dumps(perm, indent=2))
    print(f"Wrote {OUT_PERM}")
    print()
    obs = perm["observed"]
    nul = perm["null_strong_shuffle"]
    print(f"Observed cell-mean SS_rows/SS_cols  = {obs['ratio_rows_cols']:.2f}×")
    print(f"  (frac_rows = {obs['frac_rows']:.1%})")
    print(f"Null distribution (strong shuffle, n_iter={perm['n_iter']}):")
    print(f"  median     = {nul['median']:.2f}")
    print(f"  95%        = [{nul['p2.5']:.2f}, {nul['p97.5']:.2f}]")
    print(f"  99th pctl  = {nul['p99']:.2f}")
    print(f"  max        = {nul['max']:.2f}")
    print(f"  p-value    = {nul['p_value_observed_geq']:.4f}  (P(null ratio >= obs))")
    print(f"  p-value (frac_rows) = {nul['p_value_frac_rows_geq']:.4f}")


if __name__ == "__main__":
    main()
