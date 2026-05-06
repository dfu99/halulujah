"""Audit follow-up #46: Closed-form two-proportion z-test pairwise (§6ccc).

§6ww at n_iter=50000 reported 11 of 136 subject-pair contrasts surviving
Bonferroni-136 via question-cluster bootstrap. §6bbb showed Wilson CIs
are uniformly tighter than bootstrap (16/17 subjects). This script
applies the closed-form two-proportion z-test pairwise:

z = (p̂1 − p̂2) / sqrt(p̂(1−p̂)·(1/n1 + 1/n2))   (pooled variance)

with p̂ = (k1+k2)/(n1+n2). Two-tailed p = 2·(1 − Φ(|z|)).

Apply Bonferroni-136, Holm step-down, BH-FDR at α=0.05.

If the closed-form result agrees with the §6ww 11/136 Bonferroni count,
the audit's strongest formal claim is robust to bootstrap vs parametric
pairwise method.

Output: results/verified_pair_grid_qwen3_1p7b/subject_w2c_pairwise_z.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/subject_w2c_pairwise_z.json"

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
    """Two-proportion z-test (pooled variance). Returns (z, two_tailed_p)."""
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

    subject_to_primary: dict[str, str] = {}
    primary_subject_hard_idxs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for p in DOMAINS:
        solo_pq = cond[f"solo_{p}"]["per_q"]
        for q in solo_pq:
            s = q["subject"]
            subject_to_primary[s] = p
            if not q["correct"]:
                primary_subject_hard_idxs[(p, s)].append(q["idx"])

    # Per-subject (k, n) — k = pair_correct count, n = n_hard × 6 helpers
    subject_kn: dict[str, tuple[int, int, str]] = {}
    for (p, s), idxs in primary_subject_hard_idxs.items():
        if len(idxs) < N_MIN_HARD:
            continue
        n_trials = len(idxs) * len(HELPERS)
        k = 0
        for h in HELPERS:
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            for idx in idxs:
                if pair_pq[idx]["correct"]:
                    k += 1
        subject_kn[s] = (k, n_trials, p)

    subjects = sorted(
        subject_kn.keys(),
        key=lambda s: (DOMAINS.index(subject_kn[s][2]), s),
    )

    pair_stats = []
    for i, s1 in enumerate(subjects):
        for j in range(i + 1, len(subjects)):
            s2 = subjects[j]
            k1, n1, p1 = subject_kn[s1]
            k2, n2, p2 = subject_kn[s2]
            z, p_val = two_prop_z(k1, n1, k2, n2)
            point_diff = (k1 / n1) - (k2 / n2)
            pair_stats.append({
                "s1": s1,
                "s2": s2,
                "primary1": p1,
                "primary2": p2,
                "k1": k1,
                "n1": n1,
                "k2": k2,
                "n2": n2,
                "z": z,
                "two_tailed_p": p_val,
                "point_diff": point_diff,
            })

    raw_p = np.array([ps["two_tailed_p"] for ps in pair_stats])
    bh_pass = bh_fdr(raw_p, alpha=0.05)
    holm_pass = holm_step_down(raw_p, alpha=0.05)
    bonf_pass = raw_p < 0.05 / len(raw_p)
    for k, ps in enumerate(pair_stats):
        ps["bh_fdr_pass"] = bool(bh_pass[k])
        ps["holm_pass"] = bool(holm_pass[k])
        ps["bonferroni_pass"] = bool(bonf_pass[k])

    # Compare with §6ww n=50k Bonferroni-136 set
    boot_path = ROOT / "results/verified_pair_grid_qwen3_1p7b/subject_w2c_hard_bootstrap_hin.json"
    boot_set: set[frozenset[str]] = set()
    if boot_path.exists():
        boot_data = json.loads(boot_path.read_text())
        for ps in boot_data["pair_stats"]:
            if ps.get("bonferroni_pass"):
                boot_set.add(frozenset([ps["s1"], ps["s2"]]))
    z_set: set[frozenset[str]] = set()
    for ps in pair_stats:
        if ps["bonferroni_pass"]:
            z_set.add(frozenset([ps["s1"], ps["s2"]]))

    intersection = boot_set & z_set
    only_boot = boot_set - z_set
    only_z = z_set - boot_set

    out = {
        "n_pairs": len(pair_stats),
        "n_uncorrected_pass": int((raw_p < 0.05).sum()),
        "n_bh_pass": int(bh_pass.sum()),
        "n_holm_pass": int(holm_pass.sum()),
        "n_bonferroni_pass": int(bonf_pass.sum()),
        "bonferroni_threshold": 0.05 / len(pair_stats),
        "subjects": subjects,
        "subject_kn": {s: {"k": v[0], "n": v[1], "primary": v[2]} for s, v in subject_kn.items()},
        "pair_stats": pair_stats,
        "comparison_with_6ww_bootstrap": {
            "n_bootstrap_bonferroni": len(boot_set),
            "n_z_bonferroni": len(z_set),
            "n_intersection": len(intersection),
            "n_only_bootstrap": len(only_boot),
            "n_only_z": len(only_z),
            "intersection_pairs": [sorted(list(p)) for p in intersection],
            "only_bootstrap_pairs": [sorted(list(p)) for p in only_boot],
            "only_z_pairs": [sorted(list(p)) for p in only_z],
        },
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 90)
    print("§6ccc. Two-proportion z-test pairwise (closed-form)")
    print("=" * 90)
    print(f"Bonferroni-{len(pair_stats)} threshold: {0.05/len(pair_stats):.6f}")
    print()
    print(f"Of {len(pair_stats)} subject-pair tests:")
    print(f"  {out['n_uncorrected_pass']} pass uncorrected α=0.05")
    print(f"  {out['n_bh_pass']} pass BH-FDR α=0.05")
    print(f"  {out['n_holm_pass']} pass Holm step-down α=0.05")
    print(f"  {out['n_bonferroni_pass']} pass Bonferroni-{len(pair_stats)}")
    print()
    cmp = out["comparison_with_6ww_bootstrap"]
    print(f"Comparison with §6ww n=50k bootstrap Bonferroni-136 set:")
    print(f"  bootstrap survivors:  {cmp['n_bootstrap_bonferroni']}")
    print(f"  z-test survivors:     {cmp['n_z_bonferroni']}")
    print(f"  intersection:         {cmp['n_intersection']}")
    print(f"  only-bootstrap:       {cmp['n_only_bootstrap']}")
    print(f"  only-z:               {cmp['n_only_z']}")

    if cmp["only_bootstrap_pairs"]:
        print(f"\nPairs surviving bootstrap but not z-test:")
        for p in cmp["only_bootstrap_pairs"]:
            print(f"  {p[0]} ↔ {p[1]}")
    if cmp["only_z_pairs"]:
        print(f"\nPairs surviving z-test but not bootstrap:")
        for p in cmp["only_z_pairs"]:
            print(f"  {p[0]} ↔ {p[1]}")
    print()
    print("Top 20 by z-test two-tailed p:")
    sorted_pairs = sorted(pair_stats, key=lambda x: x["two_tailed_p"])
    for ps in sorted_pairs[:20]:
        flag = (
            ("BH" if ps["bh_fdr_pass"] else "  ")
            + ("H" if ps["holm_pass"] else " ")
            + ("B" if ps["bonferroni_pass"] else " ")
        )
        print(f"  {flag} {ps['s1']:30s} vs {ps['s2']:30s} Δ={ps['point_diff']*100:+6.1f}% "
              f"z={ps['z']:>+6.2f}  p={ps['two_tailed_p']:.2e}")


if __name__ == "__main__":
    main()
