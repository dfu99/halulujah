"""Audit follow-up #21: replicate-aware 2-way ANOVA on hard-only subset.

The §6cc finding (hard-question WHO ratio 67.69× vs all-question 22.11×)
predicts that restricting the §6r ANOVA to hard-question replicates
(primary solo wrong) should give an even larger F-statistic for the
primary main effect.

Design is unbalanced: each primary has a different number of hard
questions (math 32, medicine 35, biology 31, law 34, physics 44 →
total 176). Within each primary, the same hard-question set flows
through all 6 helper conditions (seed=42 in run_verified_pair_grid),
so the cell-replicate count n_ij = R_hard_i is constant across
helpers for fixed primary. This is a "proportional" unbalanced design;
the Type I SS decomposition is unambiguous.

Hypothesis (from §6cc): F_primary ≫ 26.55 (the §6r baseline);
F_helper ≈ 1; F_interaction insignificant.

Output: results/verified_pair_grid_qwen3_1p7b/anova_replicates_hard_only.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/anova_replicates_hard_only.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]

    # Build the hard-question set per primary from solo per_q "correct" labels.
    hard_idx_per_primary: dict[str, set[int]] = {}
    for primary in DOMAINS:
        hard = {
            q["idx"]
            for q in cond[f"solo_{primary}"]["per_q"]
            if not q.get("correct")
        }
        hard_idx_per_primary[primary] = hard

    # Build long-format observations Y_ijk = pair_correct - solo_correct
    # restricted to k in hard_idx_per_primary[i]. Use deterministic idx
    # alignment (the same key as the §6f clustered bootstrap).
    obs = []
    for i, primary in enumerate(DOMAINS):
        keep = hard_idx_per_primary[primary]
        # solo per_q indexed by idx
        solo_correct_by_idx = {
            q["idx"]: int(q["correct"]) for q in cond[f"solo_{primary}"]["per_q"]
        }
        for j, helper in enumerate(HELPERS):
            pair_correct_by_idx = {
                q["idx"]: int(q["correct"])
                for q in cond[f"pair_{primary}_{helper}"]["per_q"]
            }
            for idx in keep:
                y = pair_correct_by_idx[idx] - solo_correct_by_idx[idx]
                obs.append((i, j, idx, y))

    arr = np.array(obs, dtype=float)
    Y = arr[:, 3]
    iA = arr[:, 0].astype(int)
    iB = arr[:, 1].astype(int)
    n = len(Y)

    grand = Y.mean()

    # Cell means Y_bar_ij and group means
    a, b = 5, 6
    cell_means = np.zeros((a, b))
    cell_n = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            mask = (iA == i) & (iB == j)
            cell_n[i, j] = mask.sum()
            cell_means[i, j] = Y[mask].mean() if mask.sum() else 0.0

    # Row means (primary): unweighted across helpers since cell_n[i,:] is
    # constant in j within each row.
    row_means = cell_means.mean(axis=1)
    n_per_row = cell_n.sum(axis=1)  # = 6 * R_hard_i

    # Col means (helper): weighted by n_ij over rows. Since n_ij is the
    # same for each helper at a fixed row, helper-pooled mean uses each
    # row's n_per_row contribution evenly across helpers.
    n_per_col = cell_n.sum(axis=0)  # = sum over i of R_hard_i = 176 for every j
    col_means = np.zeros(b)
    for j in range(b):
        col_means[j] = (cell_means[:, j] * (cell_n[:, j] / n_per_col[j])).sum()

    # SS decomposition (Type I, primary first then helper)
    ss_total = ((Y - grand) ** 2).sum()
    ss_primary = (n_per_row * (row_means - grand) ** 2).sum()
    ss_helper = (n_per_col * (col_means - grand) ** 2).sum()
    ss_interaction = 0.0
    for i in range(a):
        for j in range(b):
            mu_ij_hat = row_means[i] + col_means[j] - grand
            ss_interaction += cell_n[i, j] * (cell_means[i, j] - mu_ij_hat) ** 2
    # Within-cell SS: sum over (i,j,k) of (y_ijk - cell_mean_ij)^2
    ss_within = 0.0
    for i in range(a):
        for j in range(b):
            mask = (iA == i) & (iB == j)
            cm = cell_means[i, j]
            ss_within += ((Y[mask] - cm) ** 2).sum()

    # Degrees of freedom
    df_primary = a - 1
    df_helper = b - 1
    df_interaction = (a - 1) * (b - 1)
    df_within = int(n) - a * b
    df_total = int(n) - 1

    ms_primary = ss_primary / df_primary
    ms_helper = ss_helper / df_helper
    ms_interaction = ss_interaction / df_interaction
    ms_within = ss_within / df_within

    F_primary = ms_primary / ms_within
    F_helper = ms_helper / ms_within
    F_interaction = ms_interaction / ms_within

    p_primary = 1.0 - stats.f.cdf(F_primary, df_primary, df_within)
    p_helper = 1.0 - stats.f.cdf(F_helper, df_helper, df_within)
    p_interaction = 1.0 - stats.f.cdf(F_interaction, df_interaction, df_within)

    # Comparison to the §6r baseline (all 50 questions)
    baseline_F = {
        "primary_A": 26.547,
        "helper_B": 0.961,
        "interaction_AB": 0.822,
    }

    out = {
        "subset": "hard_only (primary solo wrong)",
        "n_total": int(n),
        "n_per_primary_hard": {p: int(n_per_row[i] / 6) for i, p in enumerate(DOMAINS)},
        "grand_mean_delta": float(grand),
        "row_means": {p: float(row_means[i]) for i, p in enumerate(DOMAINS)},
        "col_means": {h: float(col_means[j]) for j, h in enumerate(HELPERS)},
        "ss": {
            "total": float(ss_total),
            "primary_A": float(ss_primary),
            "helper_B": float(ss_helper),
            "interaction_AB": float(ss_interaction),
            "within": float(ss_within),
        },
        "frac_of_total": {
            "primary_A": float(ss_primary / ss_total) if ss_total else 0.0,
            "helper_B": float(ss_helper / ss_total) if ss_total else 0.0,
            "interaction_AB": float(ss_interaction / ss_total) if ss_total else 0.0,
            "within": float(ss_within / ss_total) if ss_total else 0.0,
        },
        "df": {
            "primary_A": df_primary,
            "helper_B": df_helper,
            "interaction_AB": df_interaction,
            "within": df_within,
            "total": df_total,
        },
        "MS": {
            "primary_A": float(ms_primary),
            "helper_B": float(ms_helper),
            "interaction_AB": float(ms_interaction),
            "within": float(ms_within),
        },
        "F": {
            "primary_A": float(F_primary),
            "helper_B": float(F_helper),
            "interaction_AB": float(F_interaction),
        },
        "p": {
            "primary_A": float(p_primary),
            "helper_B": float(p_helper),
            "interaction_AB": float(p_interaction),
        },
        "baseline_F_full_50q": baseline_F,
        "F_ratio_hard_to_full": {
            "primary_A": float(F_primary) / baseline_F["primary_A"],
            "helper_B": float(F_helper) / baseline_F["helper_B"],
            "interaction_AB": float(F_interaction) / baseline_F["interaction_AB"],
        },
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 64)
    print("§6dd. Hard-only replicate-aware ANOVA (n_hard = 176)")
    print("=" * 64)
    print(f"n_per_primary_hard: " + ", ".join(
        f"{p}={out['n_per_primary_hard'][p]}" for p in DOMAINS
    ))
    print()
    print(f"{'source':16s} {'F-stat':>10s} {'p-value':>14s} {'SS%':>7s}  {'F-ratio':>9s}")
    for src, label in [
        ("primary_A", "Primary"),
        ("helper_B", "Helper"),
        ("interaction_AB", "Interaction"),
    ]:
        F = out["F"][src]
        p = out["p"][src]
        frac = out["frac_of_total"][src] * 100
        ratio = out["F_ratio_hard_to_full"][src]
        print(f"{label:16s} {F:>10.3f} {p:>14.2e} {frac:>6.1f}%  {ratio:>8.2f}×")
    print(f"{'Within':16s} {'':>10s} {'':>14s} {out['frac_of_total']['within']*100:>6.1f}%")


if __name__ == "__main__":
    main()
