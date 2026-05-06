"""Audit follow-up #42: High-iteration §6kk re-run (§6yy).

§6kk applied Bonferroni / Holm / BH-FDR to 25 pairwise tests (10
primary + 15 helper) on the §6ii / §6jj net corrector scores at
n_iter=2000. Result: 2 Bonferroni-25 survivors (biology > law,
biology > math).

The bootstrap precision floor at n_iter=2000 (1/2001 ≈ 0.001
two-tailed) is just under Bonferroni-25 threshold (0.002), so the
2/25 result was *not* a precision-floor artifact. But the §6ww
finding (§6tt 0/Bonferroni-136 → §6ww 11/Bonferroni-136 at n=50000)
suggests that running §6kk at higher n could reveal more survivors.

This script re-runs the full §6ii + §6jj + §6kk pipeline at
n_iter=50000:
- Vectorized batch bootstrap of per-primary net corrector score
- Vectorized batch bootstrap of per-helper net corrector score
- 25 pairwise tests with Bonferroni-25 / Holm / BH-FDR

Output: results/verified_pair_grid_qwen3_1p7b/net_corrector_bootstrap_hin.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/net_corrector_bootstrap_hin.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_ITER = 50000
SEED = 2026


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


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]
    rng = np.random.default_rng(SEED)

    # Build solo and pair correctness arrays (per primary, per helper, per question)
    # solo_correct[p] : (50,) int
    # pair_correct[p][h] : (50,) int
    solo_correct: dict[str, np.ndarray] = {}
    pair_correct: dict[str, dict[str, np.ndarray]] = {p: {} for p in DOMAINS}
    for p in DOMAINS:
        solo_correct[p] = np.array(
            [int(q["correct"]) for q in cond[f"solo_{p}"]["per_q"]],
            dtype=np.int8,
        )
        for h in HELPERS:
            pair_correct[p][h] = np.array(
                [int(q["correct"]) for q in cond[f"pair_{p}_{h}"]["per_q"]],
                dtype=np.int8,
            )

    # Per-primary bootstrap of net corrector score
    # For each iteration: resample 50 question idxs per primary, compute
    # mean(W2C - C2W) across 6 helpers
    print(f"Bootstrapping {len(DOMAINS)} primaries × {N_ITER} iters...")
    primary_boot: dict[str, np.ndarray] = {}
    for p in DOMAINS:
        sc = solo_correct[p]  # (50,)
        # Stack pair correctness into (6, 50)
        pc_stack = np.stack([pair_correct[p][h] for h in HELPERS], axis=0)  # (6, 50)
        # Sample idx_mat: (N_ITER, 50)
        idx_mat = rng.integers(0, 50, (N_ITER, 50))
        # For each iter: compute c2w and w2c rates per helper, mean across helpers
        sc_resampled = sc[idx_mat]  # (N_ITER, 50)
        easy_mask = sc_resampled == 1  # (N_ITER, 50)
        hard_mask = sc_resampled == 0
        n_easy = easy_mask.sum(axis=1)  # (N_ITER,)
        n_hard = hard_mask.sum(axis=1)
        # pair_correct resampled: (6, N_ITER, 50)
        pc_resampled = pc_stack[:, idx_mat[None, :, :].squeeze(0)]  # broadcast
        # Actually, reshape: pc_stack[h, idx_mat] has shape (6, N_ITER, 50)
        pc_resampled = pc_stack[:, None, :][:, np.zeros(N_ITER, dtype=int), :]
        # Above is wrong — fix:
        pc_resampled = np.empty((len(HELPERS), N_ITER, 50), dtype=np.int8)
        for j, h in enumerate(HELPERS):
            pc_resampled[j] = pair_correct[p][h][idx_mat]  # (N_ITER, 50)
        # C2W: (sc==1 & pc==0) / n_easy ; W2C: (sc==0 & pc==1) / n_hard
        c2w_per_helper = ((sc_resampled[None] == 1) & (pc_resampled == 0)).sum(axis=2) / np.where(n_easy > 0, n_easy, 1)
        w2c_per_helper = ((sc_resampled[None] == 0) & (pc_resampled == 1)).sum(axis=2) / np.where(n_hard > 0, n_hard, 1)
        # Mean across 6 helpers
        c2w_mean = c2w_per_helper.mean(axis=0)  # (N_ITER,)
        w2c_mean = w2c_per_helper.mean(axis=0)
        net = w2c_mean - c2w_mean
        # Mask iterations where n_easy or n_hard is 0
        valid = (n_easy > 0) & (n_hard > 0)
        primary_boot[p] = np.where(valid, net, np.nan)

    # Per-helper bootstrap: mean across primaries of net corrector
    # Helper net is computed per primary then averaged across primaries
    print(f"Bootstrapping {len(HELPERS)} helpers × {N_ITER} iters...")
    helper_boot: dict[str, np.ndarray] = {}
    for h in HELPERS:
        # For each iter: per primary, resample 50 idxs (independently),
        # compute c2w/w2c for THIS helper, then average across 5 primaries
        helper_net_per_iter = np.zeros(N_ITER)
        valid_count = np.zeros(N_ITER)
        for p in DOMAINS:
            sc = solo_correct[p]
            pc = pair_correct[p][h]
            idx_mat = rng.integers(0, 50, (N_ITER, 50))
            sc_resampled = sc[idx_mat]
            pc_resampled = pc[idx_mat]
            n_easy = (sc_resampled == 1).sum(axis=1)
            n_hard = (sc_resampled == 0).sum(axis=1)
            c2w = ((sc_resampled == 1) & (pc_resampled == 0)).sum(axis=1) / np.where(n_easy > 0, n_easy, 1)
            w2c = ((sc_resampled == 0) & (pc_resampled == 1)).sum(axis=1) / np.where(n_hard > 0, n_hard, 1)
            valid = (n_easy > 0) & (n_hard > 0)
            net_for_primary = np.where(valid, w2c - c2w, 0)
            helper_net_per_iter += net_for_primary
            valid_count += valid.astype(float)
        helper_boot[h] = helper_net_per_iter / np.where(valid_count > 0, valid_count, 1)

    # Build per-primary and per-helper summary
    primary_summary = {}
    for p in DOMAINS:
        bm = primary_boot[p]
        bm = bm[np.isfinite(bm)]
        primary_summary[p] = {
            "boot_mean": float(bm.mean()),
            "ci_lo": float(np.percentile(bm, 2.5)),
            "ci_hi": float(np.percentile(bm, 97.5)),
            "p_above_zero": float((bm > 0).mean()),
        }
    helper_summary = {}
    for h in HELPERS:
        bm = helper_boot[h]
        helper_summary[h] = {
            "boot_mean": float(bm.mean()),
            "ci_lo": float(np.percentile(bm, 2.5)),
            "ci_hi": float(np.percentile(bm, 97.5)),
            "p_above_zero": float((bm > 0).mean()),
        }

    # Pairwise tests: 10 primary + 15 helper = 25
    pair_stats = []
    for i, p1 in enumerate(DOMAINS):
        for j in range(i + 1, len(DOMAINS)):
            p2 = DOMAINS[j]
            diff = primary_boot[p1] - primary_boot[p2]
            diff = diff[np.isfinite(diff)]
            point = primary_summary[p1]["boot_mean"] - primary_summary[p2]["boot_mean"]
            p_above = float((diff > 0).mean())
            two_tailed_p = 2 * min(p_above, 1 - p_above)
            pair_stats.append({
                "axis": "primary",
                "s1": p1,
                "s2": p2,
                "point_diff": point,
                "ci_lo": float(np.percentile(diff, 2.5)),
                "ci_hi": float(np.percentile(diff, 97.5)),
                "p_above_zero": p_above,
                "two_tailed_p": float(two_tailed_p),
            })

    for i, h1 in enumerate(HELPERS):
        for j in range(i + 1, len(HELPERS)):
            h2 = HELPERS[j]
            diff = helper_boot[h1] - helper_boot[h2]
            point = helper_summary[h1]["boot_mean"] - helper_summary[h2]["boot_mean"]
            p_above = float((diff > 0).mean())
            two_tailed_p = 2 * min(p_above, 1 - p_above)
            pair_stats.append({
                "axis": "helper",
                "s1": h1,
                "s2": h2,
                "point_diff": point,
                "ci_lo": float(np.percentile(diff, 2.5)),
                "ci_hi": float(np.percentile(diff, 97.5)),
                "p_above_zero": p_above,
                "two_tailed_p": float(two_tailed_p),
            })

    raw_p = np.array([max(ps["two_tailed_p"], 1.0 / N_ITER) for ps in pair_stats])
    bh_pass = bh_fdr(raw_p, alpha=0.05)
    holm_pass = holm_step_down(raw_p, alpha=0.05)
    bonf_pass = raw_p < 0.05 / len(raw_p)
    for k, ps in enumerate(pair_stats):
        ps["bh_fdr_pass"] = bool(bh_pass[k])
        ps["holm_pass"] = bool(holm_pass[k])
        ps["bonferroni_pass"] = bool(bonf_pass[k])

    out = {
        "n_iter": N_ITER,
        "primary_summary": primary_summary,
        "helper_summary": helper_summary,
        "pair_stats": pair_stats,
        "n_pairs": len(pair_stats),
        "n_uncorrected_pass": int((raw_p < 0.05).sum()),
        "n_bh_pass": int(bh_pass.sum()),
        "n_holm_pass": int(holm_pass.sum()),
        "n_bonferroni_pass": int(bonf_pass.sum()),
        "precision_floor_p": 2.0 / N_ITER,
        "bonferroni_threshold": 0.05 / len(pair_stats),
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 80)
    print(f"§6yy. High-iteration §6kk re-run (n_iter={N_ITER})")
    print("=" * 80)
    print(f"Precision floor (two-tailed): {2.0/N_ITER:.6f}")
    print(f"Bonferroni-{len(pair_stats)} threshold: {0.05/len(pair_stats):.6f}")
    print()
    print(f"Of {len(pair_stats)} pairwise tests (10 primary + 15 helper):")
    print(f"  {out['n_uncorrected_pass']} pass uncorrected α=0.05")
    print(f"  {out['n_bh_pass']} pass BH-FDR α=0.05")
    print(f"  {out['n_holm_pass']} pass Holm step-down α=0.05")
    print(f"  {out['n_bonferroni_pass']} pass Bonferroni-{len(pair_stats)}")
    print()
    print("Per-primary CIs (n_iter=50000):")
    for p in DOMAINS:
        s = primary_summary[p]
        print(f"  {p:10s} mean={s['boot_mean']*100:+.1f} pp  "
              f"95% CI [{s['ci_lo']*100:+.1f}, {s['ci_hi']*100:+.1f}]  "
              f"P>0={s['p_above_zero']*100:.1f}%")
    print()
    print("Per-helper CIs (n_iter=50000):")
    for h in HELPERS:
        s = helper_summary[h]
        print(f"  {h:10s} mean={s['boot_mean']*100:+.1f} pp  "
              f"95% CI [{s['ci_lo']*100:+.1f}, {s['ci_hi']*100:+.1f}]  "
              f"P>0={s['p_above_zero']*100:.1f}%")
    print()
    print("Top 12 most-significant pairs:")
    sorted_pairs = sorted(pair_stats, key=lambda x: x["two_tailed_p"])
    for ps in sorted_pairs[:12]:
        flag = (
            ("BH" if ps["bh_fdr_pass"] else "  ")
            + ("H" if ps["holm_pass"] else " ")
            + ("B" if ps["bonferroni_pass"] else " ")
        )
        ci_str = f"[{ps['ci_lo']*100:+.1f}, {ps['ci_hi']*100:+.1f}]"
        print(f"  {flag} ({ps['axis']:7s}) {ps['s1']:10s} vs {ps['s2']:10s} "
              f"Δ={ps['point_diff']*100:+6.1f}% CI {ci_str:>20s} p={ps['two_tailed_p']:.5f}")

    if out['n_bonferroni_pass'] > 0:
        print()
        print(f"Bonferroni-{len(pair_stats)} survivors:")
        for ps in pair_stats:
            if ps["bonferroni_pass"]:
                ci_str = f"[{ps['ci_lo']*100:+.1f}, {ps['ci_hi']*100:+.1f}]"
                print(f"  ({ps['axis']:7s}) {ps['s1']:10s} vs {ps['s2']:10s} "
                      f"Δ={ps['point_diff']*100:+6.1f}% CI {ci_str:>20s} p={ps['two_tailed_p']:.5f}")


if __name__ == "__main__":
    main()
