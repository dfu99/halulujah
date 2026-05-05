"""Per-question helper-correctness correlation analysis (audit follow-up #16).

For each (primary, idx) tuple in the verified Qwen3-1.7B 5×6 pair-grid,
compute the Spearman ρ between the 6 helpers' correctness on that
question and the 6 helpers' self-solo accuracy. Then aggregate per
primary and per "difficulty" bucket.

Self-solo definition (queue specified "helper's self-solo on the
primary domain"; we don't have cross-domain self-solo for non-base
helpers, so we use TWO interpretations):

  A. Helper own-domain self-solo. solo_base = mean(base_solo over 5
     primaries) for the base helper; solo_<dom> for specialist helpers.
     Interprets "self-solo" as "helper quality as a specialist."

  B. Helper competence on the primary's domain (only available for
     base and primary-as-helper):
       base on primary    = base_solo_<primary>
       <primary> on primary = solo_<primary>
       cross-domain        = (NOT measured; treated as a missing covariate)

Outputs:
  results/verified_pair_grid_qwen3_1p7b/per_q_helper_correlation.json
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/per_q_helper_correlation.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]


def load_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text())


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation. Returns NaN if either vector has zero variance."""
    n = len(x)
    if n < 2:
        return float("nan")
    rx = ranks(x)
    ry = ranks(y)
    rx_mean = sum(rx) / n
    ry_mean = sum(ry) / n
    num = sum((rx[i] - rx_mean) * (ry[i] - ry_mean) for i in range(n))
    dx2 = sum((rx[i] - rx_mean) ** 2 for i in range(n))
    dy2 = sum((ry[i] - ry_mean) ** 2 for i in range(n))
    denom = (dx2 * dy2) ** 0.5
    if denom == 0:
        return float("nan")
    return num / denom


def ranks(x: list[float]) -> list[float]:
    """Average-ties rank."""
    n = len(x)
    indexed = sorted(range(n), key=lambda i: x[i])
    rs = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[indexed[j + 1]] == x[indexed[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rs[indexed[k]] = avg
        i = j + 1
    return rs


def main() -> None:
    matrix = load_matrix()
    cond = matrix["conditions"]

    # Self-solo interpretations
    solo_self = {h: cond[f"solo_{h}"]["accuracy"] for h in HELPERS if h != "base"}
    base_solo_per_primary = {p: cond[f"base_solo_{p}"]["accuracy"] for p in DOMAINS}
    base_solo_self = float(np.mean(list(base_solo_per_primary.values())))
    solo_self["base"] = base_solo_self

    # Interpretation A vector (constant across primaries)
    interp_A = [solo_self[h] for h in HELPERS]
    # Interpretation B vector (varies by primary; only valid for base + primary-as-helper)

    print("Self-solo (Interp A — own domain):")
    for h in HELPERS:
        print(f"  {h:10s} {solo_self[h]:.3f}")

    per_primary_corr: dict = {p: {} for p in DOMAINS}
    per_primary_corr_disagree: dict = {p: {} for p in DOMAINS}
    pooled_rhos_A_all: list[float] = []
    pooled_rhos_A_disagree: list[float] = []
    pooled_rhos_A_easy: list[float] = []  # questions where all 6 correct
    pooled_rhos_A_hard: list[float] = []  # questions where 1-3 of 6 correct (most informative)

    per_q_records = []

    for primary in DOMAINS:
        # Build per-q correctness across 6 helpers
        per_q = defaultdict(list)
        for j, h in enumerate(HELPERS):
            for q in cond[f"pair_{primary}_{h}"]["per_q"]:
                per_q[q["idx"]].append((j, q["subject"], int(q["correct"])))

        rhos_A_all = []
        rhos_A_disagree = []
        rhos_A_easy = []
        rhos_A_hard = []
        n_correct_dist = []

        for idx, observations in per_q.items():
            assert len(observations) == 6
            # Sort by helper index j
            obs_sorted = sorted(observations, key=lambda t: t[0])
            corrects = [c for _, _, c in obs_sorted]
            n_correct = sum(corrects)
            n_correct_dist.append(n_correct)

            # Compute rho across 6 helpers between correctness and self-solo (Interp A)
            if n_correct in (0, 6):
                rho_A = float("nan")  # all agree -> zero variance -> rho undefined
            else:
                rho_A = spearman_rho([float(c) for c in corrects], interp_A)
            rhos_A_all.append(rho_A)
            if 0 < n_correct < 6:
                rhos_A_disagree.append(rho_A)
            if n_correct == 6:
                rhos_A_easy.append(0.0)  # all correct, no info
            if 1 <= n_correct <= 3:
                rhos_A_hard.append(rho_A)

            per_q_records.append({
                "primary": primary,
                "idx": idx,
                "subject": obs_sorted[0][1],
                "n_correct_helpers": n_correct,
                "rho_A": rho_A if not np.isnan(rho_A) else None,
            })

        # Aggregate per primary
        valid_A_all = [r for r in rhos_A_all if not np.isnan(r)]
        valid_A_disagree = [r for r in rhos_A_disagree if not np.isnan(r)]
        valid_A_hard = [r for r in rhos_A_hard if not np.isnan(r)]

        per_primary_corr[primary] = {
            "n_q_total": len(rhos_A_all),
            "n_q_all_correct": sum(1 for v in n_correct_dist if v == 6),
            "n_q_all_wrong": sum(1 for v in n_correct_dist if v == 0),
            "n_q_disagree": sum(1 for v in n_correct_dist if 0 < v < 6),
            "n_q_hard_1_to_3": sum(1 for v in n_correct_dist if 1 <= v <= 3),
            "rho_A_mean_all": float(np.mean(valid_A_all)) if valid_A_all else None,
            "rho_A_mean_disagree": float(np.mean(valid_A_disagree)) if valid_A_disagree else None,
            "rho_A_mean_hard": float(np.mean(valid_A_hard)) if valid_A_hard else None,
            "rho_A_median_disagree": (
                float(np.median(valid_A_disagree)) if valid_A_disagree else None
            ),
            "n_correct_distribution": {
                str(k): sum(1 for v in n_correct_dist if v == k)
                for k in range(7)
            },
        }
        per_primary_corr_disagree[primary] = valid_A_disagree

        pooled_rhos_A_all.extend(valid_A_all)
        pooled_rhos_A_disagree.extend(valid_A_disagree)
        pooled_rhos_A_hard.extend(valid_A_hard)

    pooled = {
        "n_disagree": len(pooled_rhos_A_disagree),
        "n_hard_1_to_3": len(pooled_rhos_A_hard),
        "rho_A_mean_disagree": float(np.mean(pooled_rhos_A_disagree)) if pooled_rhos_A_disagree else None,
        "rho_A_mean_hard": float(np.mean(pooled_rhos_A_hard)) if pooled_rhos_A_hard else None,
        "rho_A_median_disagree": (
            float(np.median(pooled_rhos_A_disagree)) if pooled_rhos_A_disagree else None
        ),
        "rho_A_p_above_zero_disagree": (
            float(sum(1 for r in pooled_rhos_A_disagree if r > 0) / len(pooled_rhos_A_disagree))
            if pooled_rhos_A_disagree else None
        ),
    }

    out = {
        "self_solo_interp_A": solo_self,
        "per_primary": per_primary_corr,
        "pooled": pooled,
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print()
    print("=" * 64)
    print("Per-q helper-correctness vs helper self-solo (Interp A: own-domain)")
    print("=" * 64)
    print(f"{'primary':10s} {'all-corr':>9s} {'all-wr':>7s} {'disagree':>9s} {'rho_disag':>10s} {'rho_hard':>10s} {'rho_med':>9s}")
    for p in DOMAINS:
        d = per_primary_corr[p]
        rd = d["rho_A_mean_disagree"]
        rh = d["rho_A_mean_hard"]
        rm = d["rho_A_median_disagree"]
        rd_s = f"{rd:+.3f}" if rd is not None else "—"
        rh_s = f"{rh:+.3f}" if rh is not None else "—"
        rm_s = f"{rm:+.3f}" if rm is not None else "—"
        print(
            f"{p:10s}"
            f" {d['n_q_all_correct']:>9d}"
            f" {d['n_q_all_wrong']:>7d}"
            f" {d['n_q_disagree']:>9d}"
            f" {rd_s:>10s}"
            f" {rh_s:>10s}"
            f" {rm_s:>9s}"
        )
    print(
        f"{'POOLED':10s}"
        f" {sum(per_primary_corr[p]['n_q_all_correct'] for p in DOMAINS):>9d}"
        f" {sum(per_primary_corr[p]['n_q_all_wrong'] for p in DOMAINS):>7d}"
        f" {pooled['n_disagree']:>9d}"
        f" {pooled['rho_A_mean_disagree']:>+10.3f}"
        f" {pooled['rho_A_mean_hard']:>+10.3f}"
        f" {pooled['rho_A_median_disagree']:>+9.3f}"
    )
    print()
    print(f"Pooled: P(rho > 0 | disagree) = {pooled['rho_A_p_above_zero_disagree']:.1%}")
    print(f"  (under H0 of no effect, this should be ~50%)")


if __name__ == "__main__":
    main()
