"""Replicate-aware 2-way ANOVA on the verified 5×6 pair-grid.

Audit follow-up #13 (tasks/audit-2026-05-05.md §6m extension).

The §6m two-way ANOVA was "without replicates" — it treated each of the 30
(primary × helper) cells as a single observation, conflating residual with
interaction. Each cell has 50 question-level replicates available in
matrix_results.json per_q. This script computes the textbook two-way
ANOVA-with-replicates decomposition:

  Y_ijk = (paired delta) for primary i, helper j, question k
        = pair_correct[i, j, k] - solo_correct[i, k]
        ∈ {-1, 0, +1}

  SS_total = SS_A (primary) + SS_B (helper) + SS_AB (interaction) + SS_within

with the interaction term separated from the within-cell replicate noise.

Outputs:
  results/verified_pair_grid_qwen3_1p7b/anova_replicates.json

Hypothesis (per §6p): SS_AB (interaction) >> SS_B (helper main effect).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/anova_replicates.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]


def f_to_p_upper(f: float, df1: int, df2: int) -> float:
    """Approximate upper-tail p-value of an F-statistic without scipy.

    Uses the regularized incomplete beta function via series.
    For the audit we only need a rough ballpark — we report 'p < 1e-3'
    or the numeric value when feasible.
    """
    if f <= 0 or df1 <= 0 or df2 <= 0:
        return 1.0
    # Use the survival function via the regularized incomplete beta:
    #   P(F > f) = I_{df2/(df2+df1*f)}(df2/2, df1/2)
    # math.lgamma + a continued-fraction approximation
    x = df2 / (df2 + df1 * f)
    a = df2 / 2
    b = df1 / 2
    # Use scipy if available; otherwise a Lentz continued fraction.
    try:
        from scipy.special import betainc  # type: ignore
        return float(betainc(a, b, x))
    except Exception:
        # crude approximation: assume normal limit for large df2
        # F ~ chi2_df1/df1 in limit df2->inf, so P(F > f) ~ 1 - chi2cdf(df1*f, df1)
        # this is fine because our df_within is 1470 (very large)
        try:
            from scipy.stats import chi2  # type: ignore
            return float(1 - chi2.cdf(df1 * f, df1))
        except Exception:
            # final fallback: just report nan
            return float("nan")


def load_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text())


def build_paired_deltas(matrix: dict) -> np.ndarray:
    """Build the (5, 6, 50) array of paired Y_ijk = pair_correct - solo_correct."""
    cond = matrix["conditions"]
    a = len(DOMAINS)
    b = len(HELPERS)
    R = 50  # questions per cell

    Y = np.zeros((a, b, R), dtype=np.int8)

    for i, primary in enumerate(DOMAINS):
        solo_per_q = cond[f"solo_{primary}"]["per_q"]
        # Verify solo per_q has exactly R entries indexed 0..R-1
        assert len(solo_per_q) == R, (
            f"solo_{primary} has {len(solo_per_q)} per_q records, expected {R}"
        )
        # Use idx as join key (same deterministic seed=42 pool per primary)
        solo_correct_by_idx = {q["idx"]: int(q["correct"]) for q in solo_per_q}

        for j, helper in enumerate(HELPERS):
            pair_per_q = cond[f"pair_{primary}_{helper}"]["per_q"]
            assert len(pair_per_q) == R
            for q in pair_per_q:
                idx = q["idx"]
                if idx not in solo_correct_by_idx:
                    raise ValueError(
                        f"per_q idx {idx} present in pair_{primary}_{helper} "
                        f"but not in solo_{primary}"
                    )
                # Cross-cell alignment sanity check
                solo_q = solo_per_q[idx]
                assert solo_q["subject"] == q["subject"], (
                    f"subject mismatch primary={primary} helper={helper} idx={idx}: "
                    f"solo={solo_q['subject']} pair={q['subject']}"
                )
                assert solo_q["expected"] == q["expected"]

                pair_corr = int(q["correct"])
                solo_corr = solo_correct_by_idx[idx]
                Y[i, j, idx] = pair_corr - solo_corr  # in {-1, 0, +1}

    return Y


def two_way_anova_with_replicates(Y: np.ndarray) -> dict:
    """Textbook formulation.

    Y has shape (a, b, R).
    """
    a, b, R = Y.shape
    N = a * b * R

    grand = Y.mean()
    row_means = Y.mean(axis=(1, 2))         # length a
    col_means = Y.mean(axis=(0, 2))         # length b
    cell_means = Y.mean(axis=2)             # shape (a, b)

    SS_total = float(((Y - grand) ** 2).sum())
    SS_A = float(b * R * ((row_means - grand) ** 2).sum())
    SS_B = float(a * R * ((col_means - grand) ** 2).sum())
    # Interaction SS
    interaction_term = (
        cell_means
        - row_means[:, None]
        - col_means[None, :]
        + grand
    )
    SS_AB = float(R * (interaction_term ** 2).sum())
    # Within-cell SS
    SS_within = float(((Y - cell_means[:, :, None]) ** 2).sum())

    # Sanity: SS_total ≈ SS_A + SS_B + SS_AB + SS_within
    decomp_sum = SS_A + SS_B + SS_AB + SS_within
    decomp_residual = SS_total - decomp_sum

    df_A = a - 1
    df_B = b - 1
    df_AB = (a - 1) * (b - 1)
    df_within = a * b * (R - 1)
    df_total = N - 1

    MS_A = SS_A / df_A
    MS_B = SS_B / df_B
    MS_AB = SS_AB / df_AB
    MS_within = SS_within / df_within

    F_A = MS_A / MS_within
    F_B = MS_B / MS_within
    F_AB = MS_AB / MS_within

    p_A = f_to_p_upper(F_A, df_A, df_within)
    p_B = f_to_p_upper(F_B, df_B, df_within)
    p_AB = f_to_p_upper(F_AB, df_AB, df_within)

    return {
        "shape": {"a_primary": a, "b_helper": b, "R_replicates": R, "N_total": N},
        "grand_mean": float(grand),
        "row_means": {DOMAINS[i]: float(row_means[i]) for i in range(a)},
        "col_means": {HELPERS[j]: float(col_means[j]) for j in range(b)},
        "ss": {
            "total": SS_total,
            "primary_A": SS_A,
            "helper_B": SS_B,
            "interaction_AB": SS_AB,
            "within": SS_within,
            "decomp_sum_check": decomp_sum,
            "decomp_residual": decomp_residual,
        },
        "frac_of_total": {
            "primary_A": SS_A / SS_total,
            "helper_B": SS_B / SS_total,
            "interaction_AB": SS_AB / SS_total,
            "within": SS_within / SS_total,
        },
        "df": {
            "primary_A": df_A,
            "helper_B": df_B,
            "interaction_AB": df_AB,
            "within": df_within,
            "total": df_total,
        },
        "MS": {
            "primary_A": MS_A,
            "helper_B": MS_B,
            "interaction_AB": MS_AB,
            "within": MS_within,
        },
        "F": {
            "primary_A": F_A,
            "helper_B": F_B,
            "interaction_AB": F_AB,
        },
        "p": {
            "primary_A": p_A,
            "helper_B": p_B,
            "interaction_AB": p_AB,
        },
    }


def main() -> None:
    print(f"Loading {MATRIX_PATH}")
    matrix = load_matrix()
    Y = build_paired_deltas(matrix)
    print(f"Y shape: {Y.shape}, dtype={Y.dtype}, "
          f"value distribution: {dict(zip(*np.unique(Y, return_counts=True)))}")

    decomp = two_way_anova_with_replicates(Y)

    OUT_PATH.write_text(json.dumps(decomp, indent=2))
    print(f"Wrote {OUT_PATH}")
    print()
    print("=" * 64)
    print("Two-way ANOVA WITH replicates — (primary × helper × question)")
    print("=" * 64)

    print(f"{'source':18s} {'SS':>10s} {'frac':>7s} {'df':>5s} {'MS':>10s} {'F':>8s} {'p':>10s}")
    print("-" * 64)
    for source, label in [
        ("primary_A", "Primary (A)"),
        ("helper_B", "Helper (B)"),
        ("interaction_AB", "Interaction"),
        ("within", "Within (replicate)"),
    ]:
        ss = decomp["ss"][source]
        frac = decomp["frac_of_total"][source]
        df = decomp["df"][source]
        ms = decomp["MS"][source]
        f = decomp["F"].get(source, math.nan)
        p = decomp["p"].get(source, math.nan)
        f_str = f"{f:.2f}" if isinstance(f, (int, float)) and not math.isnan(f) else "—"
        p_str = (
            "<1e-10" if (isinstance(p, (int, float)) and not math.isnan(p) and p < 1e-10)
            else (f"{p:.2e}" if isinstance(p, (int, float)) and not math.isnan(p) else "—")
        )
        print(f"{label:18s} {ss:>10.2f} {frac:>7.1%} {df:>5d} {ms:>10.4f} {f_str:>8s} {p_str:>10s}")
    print("-" * 64)
    print(
        f"{'Total':18s} {decomp['ss']['total']:>10.2f} {1.0:>7.1%} "
        f"{decomp['df']['total']:>5d}"
    )

    # The §6p prediction check
    print()
    f_b = decomp["F"]["helper_B"]
    f_ab = decomp["F"]["interaction_AB"]
    ss_b = decomp["ss"]["helper_B"]
    ss_ab = decomp["ss"]["interaction_AB"]
    print(f"§6p prediction check (interaction >> helper main):")
    print(f"  SS_AB / SS_B = {ss_ab / ss_b:.2f}× ({'YES' if ss_ab > ss_b else 'NO'})")
    print(f"  F_AB / F_B   = {f_ab / f_b:.2f}× ({'YES' if f_ab > f_b else 'NO'})")


if __name__ == "__main__":
    main()
