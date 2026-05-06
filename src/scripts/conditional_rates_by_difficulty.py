"""Audit follow-up #25: Per-cell conditional rates by difficulty (§6hh).

§6cc/§6dd/§6ee characterized the *cell-mean* WHO ratio's behavior on
easy and hard subsets. This script computes the underlying *per-cell
conditional rates* that drive those means:

  C2W_rate (per cell, easy questions only) =
      P(post wrong | solo correct) = #{post wrong & solo correct} / #{solo correct}

  W2C_rate (per cell, hard questions only) =
      P(post correct | solo wrong) = #{post correct & solo wrong} / #{solo wrong}

These are the proper Reviewer-D-respecting "conditional switch rates"
addressed in §6a, now stratified by difficulty.

Hypotheses:
1. C2W_rate (destruction-on-easy) is small and roughly uniform across
   cells (no helper *destroys* correct answers preferentially).
2. W2C_rate (recovery-on-hard) varies enormously across primaries,
   with law primary having the lowest mean W2C across helpers
   (consistent with §6bb 0/6 corrector cells).

Output: results/verified_pair_grid_qwen3_1p7b/conditional_rates_by_difficulty.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/conditional_rates_by_difficulty.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS


def variance_decomp(grid: np.ndarray) -> dict:
    grand = grid.mean()
    rm = grid.mean(axis=1)
    cm = grid.mean(axis=0)
    nrows, ncols = grid.shape
    ss_total = ((grid - grand) ** 2).sum()
    ss_rows = ncols * ((rm - grand) ** 2).sum()
    ss_cols = nrows * ((cm - grand) ** 2).sum()
    ss_residual = ss_total - ss_rows - ss_cols
    return {
        "grand_mean": float(grand),
        "ss_total": float(ss_total),
        "ss_rows": float(ss_rows),
        "ss_cols": float(ss_cols),
        "ss_residual": float(ss_residual),
        "frac_rows": float(ss_rows / ss_total) if ss_total > 0 else 0.0,
        "frac_cols": float(ss_cols / ss_total) if ss_total > 0 else 0.0,
        "frac_residual": float(ss_residual / ss_total) if ss_total > 0 else 0.0,
        "ratio_rows_cols": float(ss_rows / ss_cols) if ss_cols > 0 else float("inf"),
        "row_spread_pp": float((rm.max() - rm.min()) * 100),
        "col_spread_pp": float((cm.max() - cm.min()) * 100),
        "row_means": {p: float(rm[i]) for i, p in enumerate(DOMAINS)},
        "col_means": {h: float(cm[j]) for j, h in enumerate(HELPERS)},
    }


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]

    # Build solo-correctness-by-idx per primary
    solo_correct = {}
    for p in DOMAINS:
        solo_correct[p] = {q["idx"]: bool(q["correct"]) for q in cond[f"solo_{p}"]["per_q"]}

    # Per-cell conditional rates
    c2w_rate_grid = np.zeros((5, 6))   # P(post wrong | solo correct)
    w2c_rate_grid = np.zeros((5, 6))   # P(post correct | solo wrong)
    hold_correct_grid = np.zeros((5, 6))  # P(post correct | solo correct)
    hold_wrong_grid = np.zeros((5, 6))    # P(post wrong | solo wrong)

    cells = []
    for i, p in enumerate(DOMAINS):
        sc = solo_correct[p]
        easy_idx = {idx for idx, c in sc.items() if c}
        hard_idx = {idx for idx, c in sc.items() if not c}
        n_easy = len(easy_idx)
        n_hard = len(hard_idx)
        for j, h in enumerate(HELPERS):
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            post_correct_by_idx = {q["idx"]: bool(q["correct"]) for q in pair_pq}
            c2w_count = sum(1 for idx in easy_idx if not post_correct_by_idx[idx])
            w2c_count = sum(1 for idx in hard_idx if post_correct_by_idx[idx])
            hold_c = n_easy - c2w_count
            hold_w = n_hard - w2c_count
            c2w_rate = c2w_count / n_easy if n_easy else 0.0
            w2c_rate = w2c_count / n_hard if n_hard else 0.0
            c2w_rate_grid[i, j] = c2w_rate
            w2c_rate_grid[i, j] = w2c_rate
            hold_correct_grid[i, j] = hold_c / n_easy if n_easy else 0.0
            hold_wrong_grid[i, j] = hold_w / n_hard if n_hard else 0.0
            cells.append({
                "primary": p,
                "helper": h,
                "n_easy": n_easy,
                "n_hard": n_hard,
                "c2w_count": c2w_count,
                "w2c_count": w2c_count,
                "c2w_rate_easy": c2w_rate,
                "w2c_rate_hard": w2c_rate,
                "hold_correct_rate_easy": hold_c / n_easy if n_easy else 0.0,
                "hold_wrong_rate_hard": hold_w / n_hard if n_hard else 0.0,
                "net_corrector_score": w2c_rate - c2w_rate,
            })

    decomp_c2w = variance_decomp(c2w_rate_grid)
    decomp_w2c = variance_decomp(w2c_rate_grid)
    decomp_hold_c = variance_decomp(hold_correct_grid)
    decomp_hold_w = variance_decomp(hold_wrong_grid)

    # Per-primary aggregation (mean across helpers)
    per_primary = {}
    for i, p in enumerate(DOMAINS):
        per_primary[p] = {
            "mean_c2w_rate_easy": float(c2w_rate_grid[i, :].mean()),
            "mean_w2c_rate_hard": float(w2c_rate_grid[i, :].mean()),
            "mean_hold_correct_rate_easy": float(hold_correct_grid[i, :].mean()),
            "mean_hold_wrong_rate_hard": float(hold_wrong_grid[i, :].mean()),
            "std_c2w_rate_easy": float(c2w_rate_grid[i, :].std()),
            "std_w2c_rate_hard": float(w2c_rate_grid[i, :].std()),
        }

    # Per-helper aggregation
    per_helper = {}
    for j, h in enumerate(HELPERS):
        per_helper[h] = {
            "mean_c2w_rate_easy": float(c2w_rate_grid[:, j].mean()),
            "mean_w2c_rate_hard": float(w2c_rate_grid[:, j].mean()),
            "mean_hold_correct_rate_easy": float(hold_correct_grid[:, j].mean()),
            "mean_hold_wrong_rate_hard": float(hold_wrong_grid[:, j].mean()),
        }

    out = {
        "n_easy_per_primary": {p: cells[i * 6]["n_easy"] for i, p in enumerate(DOMAINS)},
        "n_hard_per_primary": {p: cells[i * 6]["n_hard"] for i, p in enumerate(DOMAINS)},
        "cells": cells,
        "decomp_c2w_rate_easy": decomp_c2w,
        "decomp_w2c_rate_hard": decomp_w2c,
        "decomp_hold_correct_rate_easy": decomp_hold_c,
        "decomp_hold_wrong_rate_hard": decomp_hold_w,
        "per_primary": per_primary,
        "per_helper": per_helper,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 78)
    print("§6hh. Per-cell conditional rates by difficulty")
    print("=" * 78)
    print(f"\nVariance decomposition (grand_mean / row_spread / col_spread / row_frac / WHO ratio):\n")
    print(f"{'metric':28s}  {'grand':>7s} {'rowsp':>8s} {'colsp':>8s} {'rowfrac':>8s} {'WHO':>9s}")
    for label, d in [
        ("C2W rate (easy → wrong)", decomp_c2w),
        ("W2C rate (hard → correct)", decomp_w2c),
        ("Hold-correct rate (easy)", decomp_hold_c),
        ("Hold-wrong rate (hard)", decomp_hold_w),
    ]:
        print(f"{label:28s}  {d['grand_mean']*100:>5.1f}% {d['row_spread_pp']:>7.1f}pp {d['col_spread_pp']:>7.1f}pp"
              f" {d['frac_rows']:>7.1%} {d['ratio_rows_cols']:>8.2f}×")
    print()
    print("Per-primary mean rates (averaged across 6 helpers):")
    print(f"{'primary':10s} {'mean C2W (easy)':>16s} {'mean W2C (hard)':>16s} {'hold_c (easy)':>14s} {'hold_w (hard)':>14s}")
    for p in DOMAINS:
        pp = per_primary[p]
        print(f"{p:10s} {pp['mean_c2w_rate_easy']*100:>15.1f}% {pp['mean_w2c_rate_hard']*100:>15.1f}% "
              f"{pp['mean_hold_correct_rate_easy']*100:>13.1f}% {pp['mean_hold_wrong_rate_hard']*100:>13.1f}%")
    print()
    print("Per-helper mean rates (averaged across 5 primaries):")
    print(f"{'helper':10s} {'mean C2W (easy)':>16s} {'mean W2C (hard)':>16s}")
    for h in HELPERS:
        ph = per_helper[h]
        print(f"{h:10s} {ph['mean_c2w_rate_easy']*100:>15.1f}% {ph['mean_w2c_rate_hard']*100:>15.1f}%")


if __name__ == "__main__":
    main()
