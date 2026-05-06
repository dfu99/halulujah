"""Audit follow-up #39: Per-subject easy C2W bootstrap CI (§6vv).

Symmetric companion to §6tt (per-subject hard W2C bootstrap). On the
easy subset, the natural cell-level metric is C2W rate (1 − pair_acc,
since solo_acc = 1 by construction). §6uu reported point C2W rates
for 5 filtered subjects (n_easy ≥ 4): hs_biology 11.1%, elementary_math
13.9%, college_biology 16.7%, professional_medicine 19.4%,
professional_law 33.3%.

This script places 95% CIs on those point estimates and runs all
(5 choose 2) = 10 pairwise tests with BH-FDR + Bonferroni-10.

Output: results/verified_pair_grid_qwen3_1p7b/subject_c2w_easy_bootstrap.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/subject_c2w_easy_bootstrap.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_ITER = 2000
N_MIN_EASY = 4
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
    primary_subject_easy_idxs: dict[tuple[str, str], list[int]] = defaultdict(list)

    for p in DOMAINS:
        solo_pq = cond[f"solo_{p}"]["per_q"]
        for q in solo_pq:
            s = q["subject"]
            subject_to_primary[s] = p
            if q["correct"]:
                primary_subject_easy_idxs[(p, s)].append(q["idx"])

    # Build subject × (n_easy, helpers) cell of pair_correct booleans
    # (C2W = 1 − pair_correct on easy)
    subject_cells: dict[str, np.ndarray] = {}
    subject_n_easy: dict[str, int] = {}
    for (p, s), idxs in primary_subject_easy_idxs.items():
        if len(idxs) < N_MIN_EASY:
            continue
        cell = np.zeros((len(idxs), len(HELPERS)), dtype=int)
        for j, h in enumerate(HELPERS):
            pair_pq = cond[f"pair_{p}_{h}"]["per_q"]
            for k, idx in enumerate(idxs):
                cell[k, j] = 1 if pair_pq[idx]["correct"] else 0
        subject_cells[s] = cell
        subject_n_easy[s] = len(idxs)

    subjects = sorted(
        subject_cells.keys(),
        key=lambda s: (DOMAINS.index(subject_to_primary[s]), s),
    )

    subject_boot_c2w: dict[str, np.ndarray] = {}
    for s in subjects:
        cell = subject_cells[s]
        n_easy = cell.shape[0]
        boot_c2w = np.zeros(N_ITER)
        for it in range(N_ITER):
            idxs = rng.integers(0, n_easy, n_easy)
            resampled = cell[idxs, :]
            # C2W = 1 - pair_correct
            boot_c2w[it] = 1.0 - resampled.mean()
        subject_boot_c2w[s] = boot_c2w

    subject_summary = {}
    for s in subjects:
        bc = subject_boot_c2w[s]
        point = 1.0 - subject_cells[s].mean()
        subject_summary[s] = {
            "primary": subject_to_primary[s],
            "n_easy": subject_n_easy[s],
            "point_c2w": float(point),
            "boot_mean_c2w": float(bc.mean()),
            "ci_lo": float(np.percentile(bc, 2.5)),
            "ci_hi": float(np.percentile(bc, 97.5)),
            "p_above_zero": float((bc > 0).mean()),
            "p_above_25": float((bc > 0.25).mean()),
            "p_above_30": float((bc > 0.30).mean()),
        }

    # Pairwise (5 subjects = 10 pairs)
    pair_stats = []
    for i, s1 in enumerate(subjects):
        for j in range(i + 1, len(subjects)):
            s2 = subjects[j]
            diff = subject_boot_c2w[s1] - subject_boot_c2w[s2]
            point_diff = (1.0 - subject_cells[s1].mean()) - (1.0 - subject_cells[s2].mean())
            p_above = (diff > 0).mean()
            p_below = (diff < 0).mean()
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
    for k, ps in enumerate(pair_stats):
        ps["bh_fdr_pass"] = bool(bh_pass[k])
        ps["bonferroni_pass"] = bool(bonf_pass[k])

    out = {
        "n_iter": N_ITER,
        "n_min_easy": N_MIN_EASY,
        "subjects": subjects,
        "subject_summary": subject_summary,
        "pair_stats": pair_stats,
        "n_pairs": len(pair_stats),
        "n_bh_pass": int(bh_pass.sum()),
        "n_bonferroni_pass": int(bonf_pass.sum()),
        "n_uncorrected_pass": int((raw_p < 0.05).sum()),
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 90)
    print("§6vv. Per-subject easy C2W bootstrap (n_easy >= 4, n_iter=2000)")
    print("=" * 90)

    sorted_subjects = sorted(subjects,
                             key=lambda s: subject_summary[s]["point_c2w"])
    print()
    print(f"{'Subject':36s} {'primary':10s} {'n_easy':>7s} {'point%':>7s} "
          f"{'CI':>20s}  {'P(>0.25)':>9s} {'P(>0.30)':>9s}")
    for s in sorted_subjects:
        b = subject_summary[s]
        ci_str = f"[{b['ci_lo']*100:.1f}, {b['ci_hi']*100:.1f}]"
        print(f"  {s:34s} {b['primary']:10s} {b['n_easy']:>7d} "
              f"{b['point_c2w']*100:>6.1f}% {ci_str:>20s}  "
              f"{b['p_above_25']*100:>8.1f}% {b['p_above_30']*100:>8.1f}%")

    print()
    print(f"Of {len(pair_stats)} subject-pair comparisons (5 choose 2 = 10):")
    print(f"  {out['n_uncorrected_pass']} pass uncorrected α=0.05")
    print(f"  {out['n_bh_pass']} pass BH-FDR α=0.05")
    print(f"  {out['n_bonferroni_pass']} pass Bonferroni-{len(pair_stats)} (α/{len(pair_stats)}={0.05/len(pair_stats):.4f})")

    print()
    print("All 10 pairs sorted by two-tailed p:")
    sorted_pairs = sorted(pair_stats, key=lambda x: x["two_tailed_p"])
    for ps in sorted_pairs:
        flag = ("BH" if ps["bh_fdr_pass"] else "  ") + ("B" if ps["bonferroni_pass"] else " ")
        ci_str = f"[{ps['ci_lo']*100:+.1f}, {ps['ci_hi']*100:+.1f}]"
        print(f"  {flag} {ps['s1']:30s} vs {ps['s2']:30s} ΔC2W={ps['point_diff']*100:+6.1f}% "
              f"CI {ci_str:>20s} p={ps['two_tailed_p']:.4f}")


if __name__ == "__main__":
    main()
