"""Audit follow-up #18: cell-level interaction residual testing.

For each of the 30 (primary, helper) cells, compute the additive-model
prediction y_hat_ij = grand_mean + (row_mean_i - grand) + (col_mean_j - grand)
and test whether the observed cell mean differs from y_hat under the
within-cell sampling variability.

Standard error of the additive-model deviation:
  Var(y_obs_ij) = sigma² / R           (within-cell mean SE)
  Var(y_hat_ij) is more complex; for our purposes we approximate the
  cell's deviation from y_hat as having SE = sqrt(sigma² / R) — this
  is a slightly conservative test (real SE is somewhat smaller because
  the additive-model prediction also uses the same data).

Multiple testing: 30 simultaneous tests. Apply Bonferroni correction
(divide α by 30) for the family-wise error rate. Also report a
Benjamini-Hochberg FDR correction.

Outputs:
  results/verified_pair_grid_qwen3_1p7b/cell_interaction_residuals.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
ANOVA_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/anova_replicates.json"
OUT_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/cell_interaction_residuals.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]


def normal_pvalue_two_sided(z: float) -> float:
    """Two-sided p-value from a z-statistic via erfc."""
    return float(math.erfc(abs(z) / math.sqrt(2)))


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Returns a boolean list: True for tests that pass BH-FDR at level q."""
    n = len(p_values)
    indexed = sorted(range(n), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in indexed]
    pass_mask = [False] * n
    # Find the largest k such that p_(k) <= k/n * q
    threshold_k = -1
    for k in range(n):
        if sorted_p[k] <= (k + 1) / n * q:
            threshold_k = k
    if threshold_k >= 0:
        for k in range(threshold_k + 1):
            pass_mask[indexed[k]] = True
    return pass_mask


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    anova = json.loads(ANOVA_PATH.read_text())
    cond = matrix["conditions"]

    # Build paired-delta array Y[i, j, k] in {-1, 0, +1}
    a, b, R = 5, 6, 50
    Y = np.zeros((a, b, R), dtype=np.int8)
    for i, primary in enumerate(DOMAINS):
        solo_correct_by_idx = {
            q["idx"]: int(q["correct"]) for q in cond[f"solo_{primary}"]["per_q"]
        }
        for j, helper in enumerate(HELPERS):
            for q in cond[f"pair_{primary}_{helper}"]["per_q"]:
                idx = q["idx"]
                Y[i, j, idx] = int(q["correct"]) - solo_correct_by_idx[idx]

    cell_means = Y.mean(axis=2)  # (5, 6)
    grand = cell_means.mean()
    row_means = cell_means.mean(axis=1)  # (5,)
    col_means = cell_means.mean(axis=0)  # (6,)

    # Within-cell variance (mean squared)
    ss_within = float(((Y - cell_means[:, :, None]) ** 2).sum())
    df_within = a * b * (R - 1)
    ms_within = ss_within / df_within

    # SE of cell mean (sample of R observations per cell)
    se_cell = math.sqrt(ms_within / R)

    rows: list[dict] = []
    p_values: list[float] = []
    for i, primary in enumerate(DOMAINS):
        for j, helper in enumerate(HELPERS):
            obs = float(cell_means[i, j])
            pred = float(grand + (row_means[i] - grand) + (col_means[j] - grand))
            resid = obs - pred
            z = resid / se_cell if se_cell > 0 else 0.0
            p = normal_pvalue_two_sided(z)
            p_values.append(p)
            rows.append({
                "primary": primary,
                "helper": helper,
                "observed": obs,
                "predicted_additive": pred,
                "residual": resid,
                "z": z,
                "p_uncorrected": p,
                "se_cell": se_cell,
            })

    # Bonferroni: alpha_corrected = 0.05 / n_tests
    n = len(rows)
    alpha = 0.05
    bonf_threshold = alpha / n
    bh_pass = benjamini_hochberg(p_values, q=alpha)

    for r, b_pass in zip(rows, bh_pass):
        r["bonferroni_pass"] = bool(r["p_uncorrected"] < bonf_threshold)
        r["bh_fdr_pass"] = bool(b_pass)
        r["uncorrected_pass"] = bool(r["p_uncorrected"] < alpha)

    # Summary
    n_uncorr = sum(1 for r in rows if r["uncorrected_pass"])
    n_bonf = sum(1 for r in rows if r["bonferroni_pass"])
    n_bh = sum(1 for r in rows if r["bh_fdr_pass"])

    out = {
        "anova_F_interaction": anova["F"]["interaction_AB"],
        "anova_p_interaction": anova["p"]["interaction_AB"],
        "ms_within": ms_within,
        "se_cell": se_cell,
        "n_cells": n,
        "alpha": alpha,
        "bonferroni_threshold": bonf_threshold,
        "n_uncorrected_significant": n_uncorr,
        "n_bonferroni_significant": n_bonf,
        "n_bh_fdr_significant": n_bh,
        "cells": rows,
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")
    print()
    print("=" * 80)
    print("Tukey-style cell-level interaction residual test (audit follow-up #18)")
    print("=" * 80)
    print(f"ANOVA interaction p (overall) = {anova['p']['interaction_AB']:.3f}")
    print(f"MS_within = {ms_within:.4f}, SE_cell = {se_cell:.4f}")
    print(f"Bonferroni threshold (alpha/{n}) = {bonf_threshold:.5f}")
    print(f"BH-FDR threshold (q={alpha})")
    print()
    print(f"Cells with significant interaction residual:")
    print(f"  uncorrected (p < 0.05): {n_uncorr}/{n}")
    print(f"  Bonferroni-corrected:   {n_bonf}/{n}")
    print(f"  BH-FDR-corrected:       {n_bh}/{n}")
    print()

    # Sort by absolute residual to highlight extremes
    sorted_rows = sorted(rows, key=lambda r: abs(r["residual"]), reverse=True)
    print(f"Top-10 cells by |residual|:")
    print(f"{'primary':10s} {'helper':10s} {'obs':>7s} {'pred':>7s} {'resid':>7s} {'z':>6s} {'p_unc':>8s} {'sig':>15s}")
    for r in sorted_rows[:10]:
        flags = []
        if r["uncorrected_pass"]:
            flags.append("p<.05")
        if r["bonferroni_pass"]:
            flags.append("Bonf")
        if r["bh_fdr_pass"]:
            flags.append("BH")
        flag_str = ",".join(flags) if flags else "—"
        print(
            f"{r['primary']:10s} {r['helper']:10s}"
            f" {r['observed']:>+7.3f} {r['predicted_additive']:>+7.3f}"
            f" {r['residual']:>+7.3f} {r['z']:>+6.2f} {r['p_uncorrected']:>8.4f}"
            f" {flag_str:>15s}"
        )

    # Per-primary breakdown
    print()
    print(f"Per-primary count of significant cells:")
    print(f"{'primary':10s} {'unc':>5s} {'Bonf':>5s} {'BH':>5s}")
    for primary in DOMAINS:
        prim_rows = [r for r in rows if r["primary"] == primary]
        print(
            f"{primary:10s}"
            f" {sum(1 for r in prim_rows if r['uncorrected_pass']):>5d}"
            f" {sum(1 for r in prim_rows if r['bonferroni_pass']):>5d}"
            f" {sum(1 for r in prim_rows if r['bh_fdr_pass']):>5d}"
        )


if __name__ == "__main__":
    main()
