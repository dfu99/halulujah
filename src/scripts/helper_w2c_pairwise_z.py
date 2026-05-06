"""Audit follow-up #47: closed-form z-test on helper pairs (§6ddd).

§6yy reported 0/15 helper-pair Bonferroni-25 survivors at both
n_iter=2000 and n_iter=50000 bootstrap. This script applies the
closed-form two-proportion z-test (companion to §6ccc) to test
whether the formal "no helper effect" finding holds under closed-form
parametric testing too.

Method:
- For each of 6 helpers, pool hard-W2C trials across 17 filtered
  subjects (n_hard >= 4): compute (k_h, n_h) pair.
- For each of 15 unordered helper pairs, run two-proportion z-test
  on (k1, n1) vs (k2, n2) — same formulation as §6ccc.
- Apply Bonferroni-15 (α/15 = 0.00333), Holm step-down, BH-FDR.

If the closed-form test gives 0/15 Bonferroni-15 survivors, it
matches the bootstrap and confirms the "no helper effect" finding
is method-agnostic.

Output: results/verified_pair_grid_qwen3_1p7b/helper_w2c_pairwise_z.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/helper_w2c_pairwise_z.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_MIN_HARD = 4


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    survives_sorted = pvals[order] <= (np.arange(1, n + 1) * alpha / n)
    if survives_sorted.any():
        k_max = np.where(survives_sorted)[0].max() + 1
    else:
        k_max = 0
    pass_mask = np.zeros(n, dtype=bool)
    for i, k in enumerate(order):
        if i < k_max:
            pass_mask[k] = True
    return pass_mask


def holm_step_down(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    pass_mask = np.zeros(n, dtype=bool)
    for rank, k in enumerate(order):
        thresh = alpha / (n - rank)
        if pvals[k] <= thresh:
            pass_mask[k] = True
        else:
            break
    return pass_mask


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]

    # Identify hard idxs per primary, then per filtered subject
    primary_subject_hard_idxs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for p in DOMAINS:
        solo_pq = cond[f"solo_{p}"]["per_q"]
        for q in solo_pq:
            if not q["correct"]:
                primary_subject_hard_idxs[(p, q["subject"])].append(q["idx"])

    # Filter to subjects with n_hard >= 4
    keep_subjects = [
        (p, s) for (p, s), idxs in primary_subject_hard_idxs.items()
        if len(idxs) >= N_MIN_HARD
    ]
    print(f"Filtered subjects (n_hard >= {N_MIN_HARD}): {len(keep_subjects)}")

    # Per helper: k = pair_correct count, n = total hard trials across subjects
    helper_kn: dict[str, tuple[int, int]] = {}
    for h in HELPERS:
        k = 0
        n = 0
        for (p, s) in keep_subjects:
            idxs = primary_subject_hard_idxs[(p, s)]
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            for idx in idxs:
                n += 1
                if pair_pq[idx]["correct"]:
                    k += 1
        helper_kn[h] = (k, n)

    # Pairwise z-test on 15 pairs
    pair_stats = []
    for i, h1 in enumerate(HELPERS):
        for j in range(i + 1, len(HELPERS)):
            h2 = HELPERS[j]
            k1, n1 = helper_kn[h1]
            k2, n2 = helper_kn[h2]
            z, p_val = two_prop_z(k1, n1, k2, n2)
            point_diff = (k1 / n1) - (k2 / n2)
            pair_stats.append({
                "h1": h1,
                "h2": h2,
                "k1": k1, "n1": n1,
                "k2": k2, "n2": n2,
                "p1_w2c": k1 / n1,
                "p2_w2c": k2 / n2,
                "point_diff": point_diff,
                "z": z,
                "two_tailed_p": p_val,
            })

    raw_p = np.array([ps["two_tailed_p"] for ps in pair_stats])
    bh_pass = bh_fdr(raw_p, alpha=0.05)
    holm_pass = holm_step_down(raw_p, alpha=0.05)
    bonf_pass = raw_p < 0.05 / len(raw_p)
    for k, ps in enumerate(pair_stats):
        ps["bh_fdr_pass"] = bool(bh_pass[k])
        ps["holm_pass"] = bool(holm_pass[k])
        ps["bonferroni_pass"] = bool(bonf_pass[k])

    out = {
        "n_pairs": len(pair_stats),
        "helper_kn": {h: {"k": helper_kn[h][0], "n": helper_kn[h][1],
                          "p_w2c": helper_kn[h][0] / helper_kn[h][1]}
                      for h in HELPERS},
        "n_uncorrected_pass": int((raw_p < 0.05).sum()),
        "n_bh_pass": int(bh_pass.sum()),
        "n_holm_pass": int(holm_pass.sum()),
        "n_bonferroni_pass": int(bonf_pass.sum()),
        "bonferroni_threshold": 0.05 / len(pair_stats),
        "pair_stats": pair_stats,
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 80)
    print("§6ddd. Helper-pair two-proportion z-test (closed-form)")
    print("=" * 80)
    print(f"Per-helper hard W2C (across 17 filtered subjects):")
    for h in HELPERS:
        k, n = helper_kn[h]
        print(f"  {h:10s}  k={k:>4d} / n={n:>4d} = {k/n*100:>5.1f}%")
    print()
    print(f"Bonferroni-{len(pair_stats)} threshold: {0.05/len(pair_stats):.5f}")
    print(f"Of {len(pair_stats)} helper-pair tests:")
    print(f"  {out['n_uncorrected_pass']} pass uncorrected α=0.05")
    print(f"  {out['n_bh_pass']} pass BH-FDR α=0.05")
    print(f"  {out['n_holm_pass']} pass Holm step-down α=0.05")
    print(f"  {out['n_bonferroni_pass']} pass Bonferroni-{len(pair_stats)}")
    print()
    print("All 15 helper pairs sorted by two-tailed p:")
    sorted_pairs = sorted(pair_stats, key=lambda x: x["two_tailed_p"])
    for ps in sorted_pairs:
        flag = (
            ("BH" if ps["bh_fdr_pass"] else "  ")
            + ("H" if ps["holm_pass"] else " ")
            + ("B" if ps["bonferroni_pass"] else " ")
        )
        print(f"  {flag} {ps['h1']:10s} ({ps['p1_w2c']*100:>5.1f}%) vs "
              f"{ps['h2']:10s} ({ps['p2_w2c']*100:>5.1f}%)  "
              f"Δ={ps['point_diff']*100:+5.1f}% z={ps['z']:>+5.2f}  p={ps['two_tailed_p']:.4f}")


if __name__ == "__main__":
    main()
