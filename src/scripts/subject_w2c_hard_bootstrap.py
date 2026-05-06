"""Audit follow-up #37: Per-subject hard-subset W2C bootstrap CI (§6tt).

§6ss reported subject row-mean W2C on hard (point estimates) ranging
from 4.2% (college_mathematics) to 66.7% (high_school_biology) — a
16× spread. This script puts formal 95% CIs around those point
estimates by question-cluster bootstrap and runs pairwise tests
across subject pairs.

Method:
  For each filtered subject s (n_hard >= 4):
    Build the hard-question pool for that subject: idxs from primary's
    solo-wrong set restricted to subject s.
    For each bootstrap iteration (n=2000):
      Resample idxs with replacement (preserving alignment across the
      6 helpers via the deterministic seed=42 idx-based join).
      Compute the mean W2C rate over (n_hard × 6 helpers) trials.
    Report 95% CI + percentile distribution.

  Pairwise comparison: for each unordered pair (s_i, s_j), bootstrap
  the difference (W2C_i − W2C_j) and report P(W2C_i > W2C_j) +
  two-tailed p-value. Apply BH-FDR correction across (17 choose 2)
  = 136 pairs.

Output: results/verified_pair_grid_qwen3_1p7b/subject_w2c_hard_bootstrap.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/subject_w2c_hard_bootstrap.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_ITER = 2000
N_MIN_HARD = 4
SEED = 42


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns array of bool: which pass at alpha."""
    n = len(pvals)
    order = np.argsort(pvals)
    survives_sorted = pvals[order] <= (np.arange(1, n + 1) * alpha / n)
    # Find max k where p_(k) <= k*alpha/n
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

    # Build subject -> primary, hard idxs in primary
    subject_to_primary: dict[str, str] = {}
    primary_subject_hard_idxs: dict[tuple[str, str], list[int]] = defaultdict(list)

    for p in DOMAINS:
        solo_pq = cond[f"solo_{p}"]["per_q"]
        for q in solo_pq:
            s = q["subject"]
            subject_to_primary[s] = p
            if not q["correct"]:
                primary_subject_hard_idxs[(p, s)].append(q["idx"])

    # Collect cells: subject -> 2D array of (n_hard × 6 helpers) booleans
    # (pair_correct on the hard-question subset)
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

    # Per-subject bootstrap of mean W2C rate across (n_hard, 6) trials,
    # resampling the n_hard rows (i.e., questions) with replacement and
    # averaging across helpers within each iteration.
    subject_boot_means: dict[str, np.ndarray] = {}
    for s in subjects:
        cell = subject_cells[s]
        n_hard = cell.shape[0]
        boot_means = np.zeros(N_ITER)
        for it in range(N_ITER):
            idxs = rng.integers(0, n_hard, n_hard)
            resampled = cell[idxs, :]  # (n_hard, 6)
            boot_means[it] = resampled.mean()
        subject_boot_means[s] = boot_means

    # Per-subject CIs
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
            "p_above_zero": float((bm > 0).mean()),
            "p_above_50": float((bm > 0.5).mean()),
            "p_above_25": float((bm > 0.25).mean()),
        }

    # Pairwise comparisons (136 pairs, 17 subjects)
    pair_stats = []
    for i, s1 in enumerate(subjects):
        for j in range(i + 1, len(subjects)):
            s2 = subjects[j]
            diff = subject_boot_means[s1] - subject_boot_means[s2]
            point_diff = subject_cells[s1].mean() - subject_cells[s2].mean()
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

    # BH-FDR over the 136 pairs (clip 0-valued ps to bootstrap floor 1/N_ITER)
    raw_p = np.array([max(ps["two_tailed_p"], 1.0 / N_ITER) for ps in pair_stats])
    bh_pass = bh_fdr(raw_p, alpha=0.05)
    bonf_pass = raw_p < 0.05 / len(raw_p)
    for k, ps in enumerate(pair_stats):
        ps["bh_fdr_pass"] = bool(bh_pass[k])
        ps["bonferroni_pass"] = bool(bonf_pass[k])

    out = {
        "n_iter": N_ITER,
        "n_min_hard": N_MIN_HARD,
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
    print("§6tt. Per-subject hard W2C bootstrap (n_hard >= 4, n_iter=2000)")
    print("=" * 90)

    # Per-subject CI table sorted by point
    sorted_subjects = sorted(subjects,
                             key=lambda s: subject_summary[s]["point_w2c"],
                             reverse=True)
    print()
    print(f"{'Subject':36s} {'primary':10s} {'n_hard':>7s} {'point%':>7s} "
          f"{'CI':>20s}  {'P(>0.25)':>9s} {'P(>0.50)':>9s}")
    for s in sorted_subjects:
        b = subject_summary[s]
        ci_str = f"[{b['ci_lo']*100:.1f}, {b['ci_hi']*100:.1f}]"
        print(f"  {s:34s} {b['primary']:10s} {b['n_hard']:>7d} "
              f"{b['point_w2c']*100:>6.1f}% {ci_str:>20s}  "
              f"{b['p_above_25']*100:>8.1f}% {b['p_above_50']*100:>8.1f}%")

    print()
    print(f"Of {len(pair_stats)} subject-pair comparisons (17 choose 2 = 136):")
    print(f"  {out['n_uncorrected_pass']} pass uncorrected α=0.05")
    print(f"  {out['n_bh_pass']} pass BH-FDR α=0.05")
    print(f"  {out['n_bonferroni_pass']} pass Bonferroni-{len(pair_stats)} (α/{len(pair_stats)}={0.05/len(pair_stats):.4f})")

    # Top pairs by significance
    top_pairs = sorted(pair_stats, key=lambda x: x["two_tailed_p"])[:20]
    print()
    print("Top 20 most-significant pairs (sorted by two-tailed p):")
    for ps in top_pairs:
        flag = ("BH" if ps["bh_fdr_pass"] else "  ") + ("B" if ps["bonferroni_pass"] else " ")
        ci_str = f"[{ps['ci_lo']*100:+.1f}, {ps['ci_hi']*100:+.1f}]"
        print(f"  {flag} {ps['s1']:30s} vs {ps['s2']:30s} Δ={ps['point_diff']*100:+6.1f}% "
              f"CI {ci_str:>20s} p={ps['two_tailed_p']:.4f}")

    # Cross-primary biology row supremacy test
    print()
    print("Cross-primary tests (biology subject vs each math/law subject):")
    for ps in pair_stats:
        is_bio_vs_mathlaw = (
            (ps["primary1"] == "biology" and ps["primary2"] in ("math", "law")) or
            (ps["primary2"] == "biology" and ps["primary1"] in ("math", "law"))
        )
        if is_bio_vs_mathlaw:
            flag = ("BH" if ps["bh_fdr_pass"] else "  ") + ("B" if ps["bonferroni_pass"] else " ")
            ci_str = f"[{ps['ci_lo']*100:+.1f}, {ps['ci_hi']*100:+.1f}]"
            print(f"  {flag} {ps['s1']:30s} vs {ps['s2']:30s} Δ={ps['point_diff']*100:+6.1f}% "
                  f"CI {ci_str:>20s} p={ps['two_tailed_p']:.4f}")


if __name__ == "__main__":
    main()
