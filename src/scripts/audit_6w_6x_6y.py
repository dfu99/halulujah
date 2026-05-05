"""Audit deepening §6w + §6x + §6y.

§6w. Effect-size standardization. Converts §6r ANOVA F-stats into Cohen's f,
     partial η², and ω² (recognized effect-size scales for the paper).

§6x. Oracle ceiling. For each (primary, idx), compute "best-of-6 helpers"
     correctness — what accuracy could be achieved if the optimal helper for
     each question were chosen oracularly. Gives an upper bound on what
     helper-side intervention could deliver.

§6y. Specialist-jackknife on the WHO ratio. Leave-one-out specialist roster
     (5 jackknife replicates), recompute the canonical §6b ratio on each
     4×5 + base = 4×6 sub-grid. Quantifies the leverage of each specialist
     on the headline ratio.

Outputs:
  results/verified_pair_grid_qwen3_1p7b/effect_sizes.json     (§6w)
  results/verified_pair_grid_qwen3_1p7b/oracle_ceiling.json   (§6x)
  results/verified_pair_grid_qwen3_1p7b/specialist_jackknife.json (§6y)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
ANOVA_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/anova_replicates.json"
OUT_W = ROOT / "results/verified_pair_grid_qwen3_1p7b/effect_sizes.json"
OUT_X = ROOT / "results/verified_pair_grid_qwen3_1p7b/oracle_ceiling.json"
OUT_Y = ROOT / "results/verified_pair_grid_qwen3_1p7b/specialist_jackknife.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]


def load_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text())


# ───── §6w: Effect-size standardization ─────────────────────────────────


def cohen_f_partial(F: float, df_num: int, df_den: int) -> float:
    """Cohen's f from F-stat: f² = (df_num × F) / df_den."""
    if F <= 0 or df_den == 0:
        return 0.0
    return float(np.sqrt(df_num * F / df_den))


def effect_sizes_from_anova() -> dict:
    av = json.loads(ANOVA_PATH.read_text())
    ss = av["ss"]
    df = av["df"]
    F = av["F"]
    p = av["p"]

    ss_total = ss["total"]
    ss_within = ss["within"]
    df_within = df["within"]
    ms_within = ss_within / df_within if df_within else 0.0

    out = {"sources": {}}
    for src in ("primary_A", "helper_B", "interaction_AB"):
        ss_src = ss[src]
        df_src = df[src]
        F_src = F[src]
        # Eta squared (proportion of total variance, biased upward)
        eta_sq = ss_src / ss_total if ss_total else 0.0
        # Partial eta squared (excludes other effects from denominator)
        eta_sq_partial = ss_src / (ss_src + ss_within) if (ss_src + ss_within) else 0.0
        # Omega squared (less biased estimate)
        # ω² = (SS_effect − df_effect × MS_within) / (SS_total + MS_within)
        omega_sq = (ss_src - df_src * ms_within) / (ss_total + ms_within) if (ss_total + ms_within) else 0.0
        # Cohen's f (sqrt(eta²/(1-eta²)))
        cohens_f = float(np.sqrt(eta_sq_partial / (1 - eta_sq_partial))) if eta_sq_partial < 1 else float("inf")

        # Cohen's small/medium/large thresholds for f
        if cohens_f < 0.10:
            f_label = "trivial"
        elif cohens_f < 0.25:
            f_label = "small"
        elif cohens_f < 0.40:
            f_label = "medium"
        else:
            f_label = "large"

        out["sources"][src] = {
            "F": F_src,
            "p": p[src],
            "df_num": df_src,
            "df_den": df_within,
            "ss": ss_src,
            "ss_frac_of_total": eta_sq,
            "eta_squared": eta_sq,
            "eta_squared_partial": eta_sq_partial,
            "omega_squared": max(omega_sq, 0.0),
            "cohens_f": cohens_f,
            "cohens_f_label": f_label,
        }
    out["interpretation"] = {
        "cohens_f_thresholds": {"trivial": "<0.10", "small": "0.10–0.25", "medium": "0.25–0.40", "large": "≥0.40"},
        "eta_squared": "ratio of SS_effect / SS_total (proportion-of-variance, biased upward)",
        "eta_squared_partial": "SS_effect / (SS_effect + SS_within); strips other effects from denominator",
        "omega_squared": "less biased version of η²; can be negative if effect is at noise floor",
    }
    return out


# ───── §6x: Oracle ceiling ─────────────────────────────────────────────


def oracle_ceiling(matrix: dict) -> dict:
    cond = matrix["conditions"]
    out = {}
    pooled_actual_mean = []
    pooled_oracle_correct = 0
    pooled_n = 0

    for primary in DOMAINS:
        per_q_correct = {}  # idx -> [bool x 6]
        for h in HELPERS:
            for q in cond[f"pair_{primary}_{h}"]["per_q"]:
                per_q_correct.setdefault(q["idx"], []).append(bool(q["correct"]))
        n_q = len(per_q_correct)
        # Solo
        solo_per_q = cond[f"solo_{primary}"]["per_q"]
        solo_correct = sum(int(q["correct"]) for q in solo_per_q)
        solo_acc = solo_correct / len(solo_per_q) if solo_per_q else 0.0

        # Mean over helpers (per cell, then averaged across helpers)
        actual_mean_acc = float(np.mean([
            np.mean(v) for v in per_q_correct.values()
        ]))
        # Oracle: any helper correct -> count as correct
        oracle_correct = sum(1 for vs in per_q_correct.values() if any(vs))
        # Worst: NO helper correct -> count as wrong
        worst_correct = sum(1 for vs in per_q_correct.values() if all(vs))
        # Mean across (helper, idx)
        all_observations = [c for vs in per_q_correct.values() for c in vs]
        actual_pooled_acc = float(np.mean(all_observations))

        out[primary] = {
            "n_q": n_q,
            "solo_acc": solo_acc,
            "actual_mean_helper_acc": actual_mean_acc,
            "actual_pooled_helper_acc": actual_pooled_acc,
            "oracle_acc": oracle_correct / n_q if n_q else 0.0,  # any helper correct
            "worst_acc": worst_correct / n_q if n_q else 0.0,    # all helpers correct
            "oracle_lift_over_actual": (oracle_correct / n_q) - actual_pooled_acc if n_q else 0.0,
            "oracle_lift_over_solo": (oracle_correct / n_q) - solo_acc if n_q else 0.0,
        }
        pooled_actual_mean.append(actual_pooled_acc)
        pooled_oracle_correct += oracle_correct
        pooled_n += n_q

    pooled = {
        "n_q": pooled_n,
        "actual_pooled_helper_acc": float(np.mean([
            out[p]["actual_pooled_helper_acc"] for p in DOMAINS
        ])),
        "oracle_acc": pooled_oracle_correct / pooled_n if pooled_n else 0.0,
    }
    pooled["oracle_lift_over_actual"] = pooled["oracle_acc"] - pooled["actual_pooled_helper_acc"]
    return {"per_primary": out, "pooled": pooled}


# ───── §6y: Specialist-jackknife on WHO ratio ──────────────────────────


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
        "ratio_rows_cols": float(ratio),
        "ss_rows": float(ss_rows),
        "ss_cols": float(ss_cols),
        "ss_total": float(ss_total),
        "frac_rows": float(ss_rows / ss_total) if ss_total else 0.0,
        "row_spread_pp": float((row_means.max() - row_means.min()) * 100),
        "col_spread_pp": float((col_means.max() - col_means.min()) * 100),
    }


def specialist_jackknife(matrix: dict) -> dict:
    cond = matrix["conditions"]

    # Build the canonical 5×6 delta grid
    full_grid = np.zeros((5, 6))
    for i, p in enumerate(DOMAINS):
        solo_a = cond[f"solo_{p}"]["accuracy"]
        for j, h in enumerate(HELPERS):
            full_grid[i, j] = cond[f"pair_{p}_{h}"]["accuracy"] - solo_a

    full_decomp = variance_decomp(full_grid)
    out = {
        "full_5x6": full_decomp,
        "loo_specialist_dropped": {},
    }
    for drop_idx, drop_dom in enumerate(DOMAINS):
        # Drop both that primary's row AND that specialist's helper column
        # (so the 4-spec jackknife uses 4 primaries × 5 helpers = base + 4 specialists)
        keep_rows = [i for i in range(5) if i != drop_idx]
        # Helpers: keep base + the 4 remaining specialists
        keep_cols = [0] + [j + 1 for j in range(5) if j != drop_idx]
        sub = full_grid[np.ix_(keep_rows, keep_cols)]
        d = variance_decomp(sub)
        out["loo_specialist_dropped"][drop_dom] = {
            "shape": [int(sub.shape[0]), int(sub.shape[1])],
            **d,
            "leverage_on_ratio": (d["ratio_rows_cols"] - full_decomp["ratio_rows_cols"]),
            "leverage_on_frac_rows": (d["frac_rows"] - full_decomp["frac_rows"]),
        }

    # Summary stats across the 5 LOO replicates
    ratios = [out["loo_specialist_dropped"][p]["ratio_rows_cols"] for p in DOMAINS]
    fracs = [out["loo_specialist_dropped"][p]["frac_rows"] for p in DOMAINS]
    out["loo_summary"] = {
        "ratio_min": float(min(ratios)),
        "ratio_max": float(max(ratios)),
        "ratio_mean": float(np.mean(ratios)),
        "ratio_std": float(np.std(ratios, ddof=1)),
        "frac_rows_min": float(min(fracs)),
        "frac_rows_max": float(max(fracs)),
        "frac_rows_mean": float(np.mean(fracs)),
        "max_leverage_specialist": DOMAINS[
            int(np.argmax([abs(r - full_decomp["ratio_rows_cols"]) for r in ratios]))
        ],
        "max_leverage_value": float(
            max(abs(r - full_decomp["ratio_rows_cols"]) for r in ratios)
        ),
    }
    return out


def main() -> None:
    matrix = load_matrix()

    # §6w
    print("=" * 64)
    print("§6w. Effect-size standardization from §6r ANOVA")
    print("=" * 64)
    es = effect_sizes_from_anova()
    OUT_W.write_text(json.dumps(es, indent=2))
    print(f"Wrote {OUT_W}")
    print()
    print(f"{'source':18s} {'F':>7s} {'eta²':>7s} {'eta²_partial':>13s} {'omega²':>8s} {'Cohen f':>9s} {'label':>10s}")
    for src, e in es["sources"].items():
        print(
            f"{src:18s}"
            f" {e['F']:>7.2f}"
            f" {e['eta_squared']:>7.3f}"
            f" {e['eta_squared_partial']:>13.4f}"
            f" {e['omega_squared']:>8.4f}"
            f" {e['cohens_f']:>9.3f}"
            f" {e['cohens_f_label']:>10s}"
        )

    # §6x
    print()
    print("=" * 64)
    print("§6x. Oracle ceiling — best-of-6 helpers")
    print("=" * 64)
    oc = oracle_ceiling(matrix)
    OUT_X.write_text(json.dumps(oc, indent=2))
    print(f"Wrote {OUT_X}")
    print()
    print(f"{'primary':10s} {'solo':>6s} {'actual':>7s} {'oracle':>7s} {'lift_act':>9s} {'lift_solo':>10s}")
    for p in DOMAINS:
        d = oc["per_primary"][p]
        print(
            f"{p:10s}"
            f" {d['solo_acc']:>6.1%}"
            f" {d['actual_pooled_helper_acc']:>7.1%}"
            f" {d['oracle_acc']:>7.1%}"
            f" {d['oracle_lift_over_actual']:>9.1%}"
            f" {d['oracle_lift_over_solo']:>10.1%}"
        )
    p = oc["pooled"]
    print(
        f"{'POOLED':10s}"
        f" {' ':>6s}"
        f" {p['actual_pooled_helper_acc']:>7.1%}"
        f" {p['oracle_acc']:>7.1%}"
        f" {p['oracle_lift_over_actual']:>9.1%}"
    )

    # §6y
    print()
    print("=" * 64)
    print("§6y. Specialist-jackknife on the WHO ratio")
    print("=" * 64)
    jk = specialist_jackknife(matrix)
    OUT_Y.write_text(json.dumps(jk, indent=2))
    print(f"Wrote {OUT_Y}")
    print()
    full = jk["full_5x6"]
    print(f"Full 5×6: ratio = {full['ratio_rows_cols']:.2f}×, frac_rows = {full['frac_rows']:.1%}")
    print()
    print(f"{'dropped':10s} {'shape':>8s} {'ratio':>8s} {'frac_rows':>10s} {'leverage':>10s}")
    for p in DOMAINS:
        d = jk["loo_specialist_dropped"][p]
        print(
            f"{p:10s}"
            f" {str(d['shape']):>8s}"
            f" {d['ratio_rows_cols']:>7.2f}×"
            f" {d['frac_rows']:>10.1%}"
            f" {d['leverage_on_ratio']:>+10.2f}"
        )
    s = jk["loo_summary"]
    print()
    print(f"LOO range: {s['ratio_min']:.2f} – {s['ratio_max']:.2f}× (mean {s['ratio_mean']:.2f}, std {s['ratio_std']:.2f})")
    print(f"Max-leverage specialist: {s['max_leverage_specialist']} (Δ = {s['max_leverage_value']:+.2f})")


if __name__ == "__main__":
    main()
