"""Audit follow-up #40: High-iteration §6tt bootstrap (§6ww).

§6tt at n_iter=2000 gave 32 uncorrected, 20 BH-FDR, 0 Bonferroni-136
survivors. The 0/Bonferroni was a precision-floor artifact: minimum
two-tailed p with n_iter=2000 is ≈ 1/1000 = 0.001, which exceeds
Bonferroni-136 threshold α/136 = 0.000368.

Re-running at n_iter=50000:
  - Precision floor = 1/25000 = 0.00004
  - Bonferroni-136 threshold = 0.000368
  - Pairs with p ≤ 9 / 25000 = 0.00036 will survive Bonferroni
  - Equivalently, ≥ 99.96% of bootstrap iterations on one side

This test reveals which §6tt BH-FDR survivors hold up under the most
conservative family-wise control. Highlights the audit's strongest
subject-pair contrasts.

Output: results/verified_pair_grid_qwen3_1p7b/subject_w2c_hard_bootstrap_hin.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/subject_w2c_hard_bootstrap_hin.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_ITER = 50000
N_MIN_HARD = 4
SEED = 42


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


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]
    rng = np.random.default_rng(SEED)

    subject_to_primary: dict[str, str] = {}
    primary_subject_hard_idxs: dict[tuple[str, str], list[int]] = defaultdict(list)

    for p in DOMAINS:
        solo_pq = cond[f"solo_{p}"]["per_q"]
        for q in solo_pq:
            s = q["subject"]
            subject_to_primary[s] = p
            if not q["correct"]:
                primary_subject_hard_idxs[(p, s)].append(q["idx"])

    subject_cells: dict[str, np.ndarray] = {}
    subject_n_hard: dict[str, int] = {}
    for (p, s), idxs in primary_subject_hard_idxs.items():
        if len(idxs) < N_MIN_HARD:
            continue
        cell = np.zeros((len(idxs), len(HELPERS)), dtype=int)
        for j, h in enumerate(HELPERS):
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            for k, idx in enumerate(idxs):
                cell[k, j] = 1 if pair_pq[idx]["correct"] else 0
        subject_cells[s] = cell
        subject_n_hard[s] = len(idxs)

    subjects = sorted(
        subject_cells.keys(),
        key=lambda s: (DOMAINS.index(subject_to_primary[s]), s),
    )

    print(f"Bootstrap: {len(subjects)} subjects × {N_ITER} iterations...")

    # Vectorize: precompute cells, sample once per subject for all iters
    subject_boot_means: dict[str, np.ndarray] = {}
    for s in subjects:
        cell = subject_cells[s]
        n_hard = cell.shape[0]
        # Sample (N_ITER, n_hard) indices, take rows of cell, mean over both
        idx_mat = rng.integers(0, n_hard, (N_ITER, n_hard))
        # Reshape to flat-index, gather row-wise mean of (n_hard, 6)
        # cell[idx_mat] has shape (N_ITER, n_hard, 6)
        resampled = cell[idx_mat]  # (N_ITER, n_hard, 6)
        subject_boot_means[s] = resampled.mean(axis=(1, 2))

    print("Bootstrap complete; computing pairwise tests...")

    subject_summary = {}
    for s in subjects:
        bm = subject_boot_means[s]
        point = subject_cells[s].mean()
        subject_summary[s] = {
            "primary": subject_to_primary[s],
            "n_hard": subject_n_hard[s],
            "point_w2c": float(point),
            "boot_mean_w2c": float(bm.mean()),
            "ci_lo": float(np.percentile(bm, 2.5)),
            "ci_hi": float(np.percentile(bm, 97.5)),
        }

    pair_stats = []
    for i, s1 in enumerate(subjects):
        for j in range(i + 1, len(subjects)):
            s2 = subjects[j]
            diff = subject_boot_means[s1] - subject_boot_means[s2]
            point_diff = subject_cells[s1].mean() - subject_cells[s2].mean()
            p_above = float((diff > 0).mean())
            p_below = float((diff < 0).mean())
            two_tailed_p = 2 * min(p_above, p_below)
            pair_stats.append({
                "s1": s1,
                "s2": s2,
                "primary1": subject_to_primary[s1],
                "primary2": subject_to_primary[s2],
                "point_diff": float(point_diff),
                "ci_lo": float(np.percentile(diff, 2.5)),
                "ci_hi": float(np.percentile(diff, 97.5)),
                "p_above_zero": float(p_above),
                "two_tailed_p": float(two_tailed_p),
            })

    raw_p = np.array([max(ps["two_tailed_p"], 1.0 / N_ITER) for ps in pair_stats])
    bh_pass = bh_fdr(raw_p, alpha=0.05)
    bonf_pass = raw_p < 0.05 / len(raw_p)
    holm_pass = np.zeros(len(raw_p), dtype=bool)
    # Holm step-down
    order = np.argsort(raw_p)
    for rank, k in enumerate(order):
        thresh = 0.05 / (len(raw_p) - rank)
        if raw_p[k] <= thresh:
            holm_pass[k] = True
        else:
            break  # Holm: stop once one fails

    for k, ps in enumerate(pair_stats):
        ps["bh_fdr_pass"] = bool(bh_pass[k])
        ps["bonferroni_pass"] = bool(bonf_pass[k])
        ps["holm_pass"] = bool(holm_pass[k])

    out = {
        "n_iter": N_ITER,
        "n_min_hard": N_MIN_HARD,
        "subjects": subjects,
        "subject_summary": subject_summary,
        "pair_stats": pair_stats,
        "n_pairs": len(pair_stats),
        "n_bh_pass": int(bh_pass.sum()),
        "n_bonferroni_pass": int(bonf_pass.sum()),
        "n_holm_pass": int(holm_pass.sum()),
        "n_uncorrected_pass": int((raw_p < 0.05).sum()),
        "precision_floor_p": 2.0 / N_ITER,
        "bonferroni_threshold": 0.05 / len(raw_p),
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 90)
    print(f"§6ww. High-iteration per-subject hard W2C bootstrap (n_iter={N_ITER})")
    print("=" * 90)
    print(f"Precision floor (two-tailed): {2.0/N_ITER:.6f}")
    print(f"Bonferroni-{len(pair_stats)} threshold: {0.05/len(pair_stats):.6f}")
    print()

    print(f"Of {len(pair_stats)} subject-pair comparisons:")
    print(f"  {out['n_uncorrected_pass']} pass uncorrected α=0.05")
    print(f"  {out['n_bh_pass']} pass BH-FDR α=0.05")
    print(f"  {out['n_holm_pass']} pass Holm step-down α=0.05")
    print(f"  {out['n_bonferroni_pass']} pass Bonferroni-{len(pair_stats)}")
    print()

    print("Top 30 most-significant pairs:")
    sorted_pairs = sorted(pair_stats, key=lambda x: x["two_tailed_p"])
    for ps in sorted_pairs[:30]:
        flag = (
            ("BH" if ps["bh_fdr_pass"] else "  ")
            + ("H" if ps["holm_pass"] else " ")
            + ("B" if ps["bonferroni_pass"] else " ")
        )
        ci_str = f"[{ps['ci_lo']*100:+.1f}, {ps['ci_hi']*100:+.1f}]"
        print(f"  {flag} {ps['s1']:30s} vs {ps['s2']:30s} Δ={ps['point_diff']*100:+6.1f}% "
              f"CI {ci_str:>20s} p={ps['two_tailed_p']:.5f}")

    if out['n_bonferroni_pass'] > 0:
        print()
        print(f"Bonferroni-{len(pair_stats)} survivors:")
        for ps in pair_stats:
            if ps["bonferroni_pass"]:
                ci_str = f"[{ps['ci_lo']*100:+.1f}, {ps['ci_hi']*100:+.1f}]"
                print(f"  {ps['s1']:30s} vs {ps['s2']:30s} Δ={ps['point_diff']*100:+6.1f}% "
                      f"CI {ci_str:>20s} p={ps['two_tailed_p']:.5f}")


if __name__ == "__main__":
    main()
