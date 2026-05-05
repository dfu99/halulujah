"""Audit follow-up #22: difficulty-stratified clustered bootstrap on §6cc.

The §6cc finding (cell-mean WHO ratio = 1.50× on easy, 67.69× on hard)
is computed as a single point estimate per subset. This script
question-clusters bootstrap each subset separately, with the same
deterministic-seed alignment used in `clustered_bootstrap_who.py`,
and reports CIs for the easy and hard subsets, plus the bootstrap
distribution of the ratio-of-ratios (hard/easy).

Method:
  1. For each primary, partition idx 0..49 into easy (solo correct)
     and hard (solo wrong) sets. Easy set sizes per primary differ
     (math 18, medicine 15, biology 19, law 16, physics 6); hard sizes
     are 50 minus easy.
  2. Resample within each subset separately (n_easy questions with
     replacement → easy bootstrap; n_hard questions with replacement
     → hard bootstrap).
  3. Recompute cell-mean WHO ratio (variance ratio SS_primary/SS_helper)
     on each resampled subset.
  4. The hard/easy ratio is computed from each iter's pair of resampled
     ratios.
  5. 2000 iterations, percentile CIs.

Outputs:
  results/verified_pair_grid_qwen3_1p7b/difficulty_stratified_bootstrap.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/difficulty_stratified_bootstrap.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_ITER = 2000
SEED = 2026


def load_grid() -> tuple[dict, dict, dict]:
    """Returns (per_q_correct_grid, solo_correct_grid, hard_easy_indices).

    grid[primary][helper] is a length-50 array of post-deliberation
    correctness (0/1).
    solo_correct[primary] is a length-50 array of solo correctness.
    hard_easy_indices[primary] = (easy_idx_array, hard_idx_array)
        based on solo per_q correctness.
    """
    data = json.loads(RESULTS.read_text())
    C = data["conditions"]

    grid: dict[str, dict[str, np.ndarray]] = {p: {} for p in DOMAINS}
    solo_correct: dict[str, np.ndarray] = {}
    hard_easy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p in DOMAINS:
        solo_correct[p] = np.array(
            [int(q["correct"]) for q in C[f"solo_{p}"]["per_q"]],
            dtype=np.int8,
        )
        easy_idx = np.where(solo_correct[p] == 1)[0]
        hard_idx = np.where(solo_correct[p] == 0)[0]
        hard_easy[p] = (easy_idx, hard_idx)
        for h in HELPERS:
            cell = C[f"pair_{p}_{h}"]
            grid[p][h] = np.array(
                [int(q["correct"]) for q in cell["per_q"]], dtype=np.int8
            )
    return grid, solo_correct, hard_easy


def who_variance_ratio_subset(
    grid: dict[str, dict[str, np.ndarray]],
    solo_correct: dict[str, np.ndarray],
    sample_idx_per_primary: dict[str, np.ndarray],
) -> tuple[float, float, float, float, float]:
    """Compute the cell-mean WHO variance ratio on a question-subset.

    The cell value is delta = mean(post_correct[idx]) - mean(solo_correct[idx]),
    where the same idx set is applied to all 6 helpers (cluster preserved)
    and the solo accuracy is *also* recomputed on the same idx (so the
    delta is internally consistent on the subset).

    Returns (row_spread_pp, col_spread_pp, spread_ratio,
             variance_ratio, frac_rows_of_total_ss).
    """
    a, b = len(DOMAINS), len(HELPERS)
    cell = np.zeros((a, b), dtype=float)
    for i, p in enumerate(DOMAINS):
        idx = sample_idx_per_primary[p]
        if len(idx) == 0:
            cell[i, :] = 0.0
            continue
        solo_subset_acc = float(solo_correct[p][idx].mean())
        for j, h in enumerate(HELPERS):
            cell_subset_acc = float(grid[p][h][idx].mean())
            cell[i, j] = cell_subset_acc - solo_subset_acc

    row_mean = cell.mean(axis=1)
    col_mean = cell.mean(axis=0)
    grand_mean = float(cell.mean())
    row_spread_pp = float((row_mean.max() - row_mean.min()) * 100)
    col_spread_pp = float((col_mean.max() - col_mean.min()) * 100)
    spread_ratio = row_spread_pp / col_spread_pp if col_spread_pp > 0 else float("inf")
    SS_primary = b * ((row_mean - grand_mean) ** 2).sum()
    SS_helper = a * ((col_mean - grand_mean) ** 2).sum()
    SS_total = ((cell - grand_mean) ** 2).sum()
    variance_ratio = SS_primary / SS_helper if SS_helper > 0 else float("inf")
    frac_rows = SS_primary / SS_total if SS_total > 0 else 0.0
    return row_spread_pp, col_spread_pp, spread_ratio, variance_ratio, frac_rows


def summarize(boot: list[float]) -> dict:
    a = np.array(boot)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n_finite": 0}
    pcts = np.percentile(a, [2.5, 5, 50, 95, 97.5])
    return {
        "n_finite": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p2.5": float(pcts[0]),
        "p5": float(pcts[1]),
        "p50": float(pcts[2]),
        "p95": float(pcts[3]),
        "p97.5": float(pcts[4]),
    }


def main() -> None:
    grid, solo_correct, hard_easy = load_grid()

    # Point estimates (no resampling)
    easy_full_idx = {p: hard_easy[p][0] for p in DOMAINS}
    hard_full_idx = {p: hard_easy[p][1] for p in DOMAINS}
    rs_e, cs_e, sr_e, vr_e, fr_e = who_variance_ratio_subset(
        grid, solo_correct, easy_full_idx
    )
    rs_h, cs_h, sr_h, vr_h, fr_h = who_variance_ratio_subset(
        grid, solo_correct, hard_full_idx
    )
    print(f"easy point: row={rs_e:.1f}pp col={cs_e:.1f}pp WHO={vr_e:.2f}× rows={fr_e*100:.1f}%")
    print(f"hard point: row={rs_h:.1f}pp col={cs_h:.1f}pp WHO={vr_h:.2f}× rows={fr_h*100:.1f}%")
    print(f"hard/easy ratio (point): {vr_h/vr_e:.2f}×")
    print()

    n_easy_per_primary = {p: int(len(easy_full_idx[p])) for p in DOMAINS}
    n_hard_per_primary = {p: int(len(hard_full_idx[p])) for p in DOMAINS}

    np_rng = np.random.default_rng(SEED)
    boot_easy_vr: list[float] = []
    boot_hard_vr: list[float] = []
    boot_easy_sr: list[float] = []
    boot_hard_sr: list[float] = []
    boot_ratio_of_ratios: list[float] = []

    n_trivial_easy = 0  # iterations where easy subset is degenerate (e.g. physics has only 6 easy)
    for _ in range(N_ITER):
        easy_sample = {}
        hard_sample = {}
        for p in DOMAINS:
            ei = easy_full_idx[p]
            hi = hard_full_idx[p]
            if len(ei) > 0:
                easy_sample[p] = np_rng.choice(ei, size=len(ei), replace=True)
            else:
                easy_sample[p] = ei  # empty
            if len(hi) > 0:
                hard_sample[p] = np_rng.choice(hi, size=len(hi), replace=True)
            else:
                hard_sample[p] = hi
        _, _, sr_easy, vr_easy, _ = who_variance_ratio_subset(
            grid, solo_correct, easy_sample
        )
        _, _, sr_hard, vr_hard, _ = who_variance_ratio_subset(
            grid, solo_correct, hard_sample
        )
        boot_easy_vr.append(vr_easy)
        boot_hard_vr.append(vr_hard)
        boot_easy_sr.append(sr_easy)
        boot_hard_sr.append(sr_hard)
        if np.isfinite(vr_easy) and vr_easy > 0 and np.isfinite(vr_hard):
            boot_ratio_of_ratios.append(vr_hard / vr_easy)
        else:
            n_trivial_easy += 1

    # P(hard > easy) under bootstrap
    paired = [
        (h, e)
        for h, e in zip(boot_hard_vr, boot_easy_vr)
        if np.isfinite(h) and np.isfinite(e)
    ]
    n_paired = len(paired)
    n_hard_gt_easy = sum(1 for h, e in paired if h > e)
    p_hard_gt_easy = n_hard_gt_easy / n_paired if n_paired else 0.0

    out = {
        "n_iter": N_ITER,
        "seed": SEED,
        "n_easy_per_primary": n_easy_per_primary,
        "n_hard_per_primary": n_hard_per_primary,
        "n_easy_pooled": int(sum(n_easy_per_primary.values())),
        "n_hard_pooled": int(sum(n_hard_per_primary.values())),
        "point_estimates": {
            "easy": {
                "row_spread_pp": rs_e, "col_spread_pp": cs_e,
                "spread_ratio": sr_e, "variance_ratio": vr_e,
                "frac_rows_of_total": fr_e,
            },
            "hard": {
                "row_spread_pp": rs_h, "col_spread_pp": cs_h,
                "spread_ratio": sr_h, "variance_ratio": vr_h,
                "frac_rows_of_total": fr_h,
            },
            "hard_to_easy_ratio_point": float(vr_h / vr_e) if vr_e > 0 else float("inf"),
        },
        "bootstrap": {
            "easy_variance_ratio": summarize(boot_easy_vr),
            "hard_variance_ratio": summarize(boot_hard_vr),
            "easy_spread_ratio": summarize(boot_easy_sr),
            "hard_spread_ratio": summarize(boot_hard_sr),
            "ratio_of_ratios_hard_over_easy": summarize(boot_ratio_of_ratios),
            "n_trivial_easy_iter": n_trivial_easy,
            "p_hard_gt_easy": p_hard_gt_easy,
            "n_paired": n_paired,
        },
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 64)
    print("§6ee. Difficulty-stratified bootstrap (n_iter = {})".format(N_ITER))
    print("=" * 64)
    print(f"easy WHO ratio: 95% CI [{out['bootstrap']['easy_variance_ratio']['p2.5']:.2f}, "
          f"{out['bootstrap']['easy_variance_ratio']['p97.5']:.2f}]; "
          f"median {out['bootstrap']['easy_variance_ratio']['p50']:.2f}; "
          f"point {vr_e:.2f}")
    print(f"hard WHO ratio: 95% CI [{out['bootstrap']['hard_variance_ratio']['p2.5']:.2f}, "
          f"{out['bootstrap']['hard_variance_ratio']['p97.5']:.2f}]; "
          f"median {out['bootstrap']['hard_variance_ratio']['p50']:.2f}; "
          f"point {vr_h:.2f}")
    rr = out['bootstrap']['ratio_of_ratios_hard_over_easy']
    print(f"hard/easy ratio: 95% CI [{rr['p2.5']:.2f}, {rr['p97.5']:.2f}]; "
          f"median {rr['p50']:.2f}; point {vr_h/vr_e:.2f}")
    print(f"P(hard_ratio > easy_ratio) under bootstrap = {p_hard_gt_easy:.4f}")
    print(f"n_trivial_easy_iter = {n_trivial_easy} of {N_ITER}")


if __name__ == "__main__":
    main()
