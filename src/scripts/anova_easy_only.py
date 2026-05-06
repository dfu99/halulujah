"""Audit follow-up #41: replicate-aware 2-way ANOVA on easy-only subset (§6xx).

Mirror of §6dd (hard-only ANOVA) on the easy subset (primary solo correct).
Completes the difficulty ANOVA family alongside §6r (full F_primary=26.55)
and §6dd (hard F_primary=31.22).

Design is unbalanced: each primary has a different number of easy
questions (math 18, medicine 15, biology 19, law 16, physics 6 → total 74).

Hypothesis (from §6cc easy WHO 1.50× cell-mean): F_primary modest,
F_helper potentially non-trivial (since SS_helper is ~22% on easy
per §6uu), F_interaction non-significant.

Output: results/verified_pair_grid_qwen3_1p7b/anova_replicates_easy_only.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/anova_replicates_easy_only.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]

    # Build the easy-question set per primary (solo correct)
    easy_idx_per_primary: dict[str, set[int]] = {}
    for primary in DOMAINS:
        easy = {
            q["idx"]
            for q in cond[f"solo_{primary}"]["per_q"]
            if q.get("correct")
        }
        easy_idx_per_primary[primary] = easy

    # Build long-format observations Y_ijk = pair_correct - solo_correct
    # restricted to k in easy_idx_per_primary[i]. On easy, solo_correct=1,
    # so Y_ijk = pair_correct - 1 ∈ {-1, 0}; the response measures C2W
    # rate (with sign flipped).
    obs = []
    for i, primary in enumerate(DOMAINS):
        keep = easy_idx_per_primary[primary]
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

    a, b = 5, 6
    cell_n = np.zeros((a, b), dtype=int)
    cell_means = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            mask = (iA == i) & (iB == j)
            cell_n[i, j] = int(mask.sum())
            cell_means[i, j] = Y[mask].mean() if cell_n[i, j] > 0 else 0.0

    n_per_row = cell_n.sum(axis=1).astype(float)
    n_per_col = cell_n.sum(axis=0).astype(float)
    row_means = np.array([(Y[iA == i].mean() if (iA == i).any() else 0.0) for i in range(a)])
    col_means = np.array([(Y[iB == j].mean() if (iB == j).any() else 0.0) for j in range(b)])

    ss_total = ((Y - grand) ** 2).sum()
    ss_primary = (n_per_row * (row_means - grand) ** 2).sum()
    ss_helper = (n_per_col * (col_means - grand) ** 2).sum()
    ss_interaction = 0.0
    for i in range(a):
        for j in range(b):
            mu_ij_hat = row_means[i] + col_means[j] - grand
            ss_interaction += cell_n[i, j] * (cell_means[i, j] - mu_ij_hat) ** 2
    ss_within = 0.0
    for i in range(a):
        for j in range(b):
            mask = (iA == i) & (iB == j)
            cm = cell_means[i, j]
            ss_within += ((Y[mask] - cm) ** 2).sum()

    df_primary = a - 1
    df_helper = b - 1
    df_interaction = (a - 1) * (b - 1)
    df_within = int(n) - a * b
    df_total = int(n) - 1

    ms_primary = ss_primary / df_primary
    ms_helper = ss_helper / df_helper
    ms_interaction = ss_interaction / df_interaction
    ms_within = ss_within / df_within if df_within > 0 else float("nan")

    F_primary = ms_primary / ms_within if ms_within > 0 else float("nan")
    F_helper = ms_helper / ms_within if ms_within > 0 else float("nan")
    F_interaction = ms_interaction / ms_within if ms_within > 0 else float("nan")

    p_primary = 1.0 - stats.f.cdf(F_primary, df_primary, df_within) if df_within > 0 else float("nan")
    p_helper = 1.0 - stats.f.cdf(F_helper, df_helper, df_within) if df_within > 0 else float("nan")
    p_interaction = 1.0 - stats.f.cdf(F_interaction, df_interaction, df_within) if df_within > 0 else float("nan")

    # Cohen's f from η²
    eta2_primary = ss_primary / ss_total if ss_total > 0 else 0.0
    eta2_helper = ss_helper / ss_total if ss_total > 0 else 0.0
    eta2_interaction = ss_interaction / ss_total if ss_total > 0 else 0.0
    cohen_f_primary = float(np.sqrt(eta2_primary / (1 - eta2_primary))) if eta2_primary < 1 else float("inf")
    cohen_f_helper = float(np.sqrt(eta2_helper / (1 - eta2_helper))) if eta2_helper < 1 else float("inf")
    cohen_f_interaction = float(np.sqrt(eta2_interaction / (1 - eta2_interaction))) if eta2_interaction < 1 else float("inf")

    # ω² (bias-corrected)
    omega2_primary = max(
        0.0,
        (ss_primary - df_primary * ms_within) / (ss_total + ms_within) if ms_within > 0 else 0.0,
    )
    omega2_helper = max(
        0.0,
        (ss_helper - df_helper * ms_within) / (ss_total + ms_within) if ms_within > 0 else 0.0,
    )
    omega2_interaction = max(
        0.0,
        (ss_interaction - df_interaction * ms_within) / (ss_total + ms_within) if ms_within > 0 else 0.0,
    )

    baseline_F = {
        "primary_A_full": 26.547,
        "primary_A_hard": 31.22,
        "helper_B_full": 0.961,
        "helper_B_hard": 0.50,
        "interaction_AB_full": 0.822,
        "interaction_AB_hard": 0.79,
    }

    out = {
        "subset": "easy_only (primary solo correct)",
        "n_total": int(n),
        "n_per_primary_easy": {p: int(n_per_row[i] / 6) for i, p in enumerate(DOMAINS)},
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
        "cohen_f": {
            "primary_A": float(cohen_f_primary),
            "helper_B": float(cohen_f_helper),
            "interaction_AB": float(cohen_f_interaction),
        },
        "omega_sq": {
            "primary_A": float(omega2_primary),
            "helper_B": float(omega2_helper),
            "interaction_AB": float(omega2_interaction),
        },
        "baseline_F_comparison": baseline_F,
        "F_ratio_easy_to_full": {
            "primary_A": float(F_primary) / baseline_F["primary_A_full"],
            "helper_B": float(F_helper) / baseline_F["helper_B_full"],
            "interaction_AB": float(F_interaction) / baseline_F["interaction_AB_full"],
        },
        "F_ratio_easy_to_hard": {
            "primary_A": float(F_primary) / baseline_F["primary_A_hard"],
            "helper_B": float(F_helper) / baseline_F["helper_B_hard"],
            "interaction_AB": float(F_interaction) / baseline_F["interaction_AB_hard"],
        },
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 80)
    print("§6xx. Easy-only replicate-aware ANOVA (n_easy = 74)")
    print("=" * 80)
    print(f"n_per_primary_easy: " + ", ".join(
        f"{p}={out['n_per_primary_easy'][p]}" for p in DOMAINS
    ))
    print()
    print(f"{'source':16s} {'F-stat':>10s} {'p-value':>14s} {'SS%':>7s}  "
          f"{'Cohen f':>9s}  {'ω²':>7s}  {'F-ratio (full→easy)':>22s}")
    for src, label in [
        ("primary_A", "Primary"),
        ("helper_B", "Helper"),
        ("interaction_AB", "Interaction"),
    ]:
        F = out["F"][src]
        p = out["p"][src]
        frac = out["frac_of_total"][src] * 100
        cf = out["cohen_f"][src]
        omega = out["omega_sq"][src]
        ratio = out["F_ratio_easy_to_full"][src]
        print(f"{label:16s} {F:>10.3f} {p:>14.2e} {frac:>6.1f}%  {cf:>8.3f}  "
              f"{omega:>6.3f}  {ratio:>21.3f}×")
    print(f"{'Within':16s} {'':>10s} {'':>14s} {out['frac_of_total']['within']*100:>6.1f}%")
    print()
    print("Comparison across difficulty (F-stat):")
    print(f"  source         §6r full   §6dd hard   §6xx easy")
    print(f"  primary_A      {baseline_F['primary_A_full']:>9.2f}   {baseline_F['primary_A_hard']:>9.2f}   {out['F']['primary_A']:>9.3f}")
    print(f"  helper_B       {baseline_F['helper_B_full']:>9.2f}   {baseline_F['helper_B_hard']:>9.2f}   {out['F']['helper_B']:>9.3f}")
    print(f"  interaction    {baseline_F['interaction_AB_full']:>9.2f}   {baseline_F['interaction_AB_hard']:>9.2f}   {out['F']['interaction_AB']:>9.3f}")


if __name__ == "__main__":
    main()
