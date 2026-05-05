"""Question-clustered bootstrap on the canonical WHO-asymmetry ratio.

Audit context: tasks/audit-2026-05-05.md §6f (within-cell bootstrap CI
[1.89, 8.39] is an under-estimate of uncertainty) and §12 / follow-up #9.

The verified `matrix_results.json` was emitted with deterministic
seed=42 question shuffling, so the same 50 questions appear in the
same order across all 6 helper conditions for a given primary. This
means we *can* run a question-clustered bootstrap on the existing
data using `(primary, subject, idx)` as the join key, even though the
runner did not previously emit explicit `qid` fields. (The runner has
since been patched in this commit to emit `qid` for future runs.)

Method:
  1. For each (primary, idx in 0..n-1), read the question's *delta*
     contribution to each helper's column under the canonical
     aggregator: contribution_h = post_correct[primary, helper, idx] -
     solo_primary_acc.
  2. To resample at the question-cluster level: draw n questions with
     replacement *per primary*, applying the same drawn indices to all
     6 helper columns. This propagates question-identity dependence
     across the row.
  3. Recompute row spread and col spread on the resampled grid using
     the canonical aggregator (delta cells, mean-row vs mean-col).
  4. Repeat 2000 times, percentile CI.

Outputs:
  results/verified_pair_grid_qwen3_1p7b/clustered_bootstrap.json
  Updates audit §6f / §12 with the cluster-corrected CI (manual).

Run: python -m src.scripts.clustered_bootstrap_who
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/clustered_bootstrap.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_ITER = 2000
SEED = 2026


def load_grid() -> tuple[dict, dict]:
    """Returns (per_q_correct_grid, solo_primary_accs).

    per_q_correct_grid[primary][helper] is a length-n array of 0/1 for
    that cell's per-question post-deliberation correctness. The arrays
    are aligned by index across helpers (deterministic question
    sampling in the runner guarantees this).
    """
    data = json.loads(RESULTS.read_text())
    C = data["conditions"]

    grid: dict[str, dict[str, np.ndarray]] = {p: {} for p in DOMAINS}
    solo_acc: dict[str, float] = {}
    for p in DOMAINS:
        solo_acc[p] = C[f"solo_{p}"]["accuracy"]
        for h in HELPERS:
            cell = C[f"pair_{p}_{h}"]
            per_q = cell["per_q"]
            grid[p][h] = np.array(
                [int(q["correct"]) for q in per_q], dtype=np.int8
            )

    # Sanity-check question alignment: per_q[i].subject and .expected
    # must be identical across helpers for a fixed primary.
    for p in DOMAINS:
        ref = C[f"pair_{p}_base"]["per_q"]
        for h in HELPERS[1:]:
            other = C[f"pair_{p}_{h}"]["per_q"]
            for i, (r, o) in enumerate(zip(ref, other)):
                assert r["subject"] == o["subject"], (
                    f"question alignment broken at primary={p} "
                    f"helper={h} idx={i}: "
                    f"{r['subject']} != {o['subject']}"
                )
                assert r["expected"] == o["expected"], (
                    f"answer mismatch primary={p} helper={h} idx={i}"
                )
    return grid, solo_acc


def who_ratio_from_grid(
    grid: dict[str, dict[str, np.ndarray]],
    solo_acc: dict[str, float],
    sample_idx: dict[str, np.ndarray] | None = None,
) -> tuple[float, float, float]:
    """Compute (row_spread, col_spread, ratio) under canonical aggregator.

    Cell value = mean(post_correct[sample_idx]) - solo_acc[primary]
    Row spread = max row mean − min row mean (mean across 6 helpers).
    Col spread = max col mean − min col mean (mean across 5 primaries).
    """
    cell = np.zeros((len(DOMAINS), len(HELPERS)), dtype=float)
    for i, p in enumerate(DOMAINS):
        idx = sample_idx[p] if sample_idx is not None else None
        for j, h in enumerate(HELPERS):
            arr = grid[p][h]
            if idx is None:
                cell_acc = float(arr.mean())
            else:
                cell_acc = float(arr[idx].mean())
            cell[i, j] = cell_acc - solo_acc[p]

    row_mean = cell.mean(axis=1)
    col_mean = cell.mean(axis=0)
    row_spread = float(row_mean.max() - row_mean.min())
    col_spread = float(col_mean.max() - col_mean.min())
    ratio = row_spread / col_spread if col_spread > 0 else float("inf")
    return row_spread, col_spread, ratio


def main() -> None:
    grid, solo_acc = load_grid()
    n_per_primary = {p: len(grid[p]["base"]) for p in DOMAINS}
    print(f"loaded grid: {n_per_primary}")

    # Point estimate (no resampling)
    row_pt, col_pt, ratio_pt = who_ratio_from_grid(grid, solo_acc)
    print(
        f"point estimate: row_spread={row_pt*100:.1f}pp "
        f"col_spread={col_pt*100:.1f}pp ratio={ratio_pt:.3f}"
    )

    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    boot_ratios: list[float] = []
    boot_row: list[float] = []
    boot_col: list[float] = []
    for k in range(N_ITER):
        # Draw n indices with replacement per primary; the SAME drawn
        # indices apply to all 6 helper columns for that primary —
        # this is the question-clustered bootstrap.
        sample_idx = {
            p: np_rng.integers(0, n_per_primary[p], size=n_per_primary[p])
            for p in DOMAINS
        }
        rs, cs, rt = who_ratio_from_grid(grid, solo_acc, sample_idx)
        boot_ratios.append(rt)
        boot_row.append(rs)
        boot_col.append(cs)

    arr = np.array(boot_ratios)
    arr = arr[np.isfinite(arr)]
    pcts = np.percentile(arr, [2.5, 5, 50, 95, 97.5])
    summary = {
        "method": "question-clustered bootstrap on canonical aggregator",
        "n_iterations": N_ITER,
        "seed": SEED,
        "point": {
            "row_spread_pp": row_pt * 100,
            "col_spread_pp": col_pt * 100,
            "ratio": ratio_pt,
        },
        "bootstrap": {
            "n_finite": int(len(arr)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p2.5": float(pcts[0]),
            "p5": float(pcts[1]),
            "p50": float(pcts[2]),
            "p95": float(pcts[3]),
            "p97.5": float(pcts[4]),
            "ci_95": [float(pcts[0]), float(pcts[4])],
            "ci_90": [float(pcts[1]), float(pcts[3])],
            "p_ratio_gt_1": float((arr > 1.0).mean()),
            "p_ratio_gt_2": float((arr > 2.0).mean()),
            "p_ratio_gt_3": float((arr > 3.0).mean()),
        },
        "row_spread_pp_bootstrap": {
            "mean": float(np.mean(boot_row) * 100),
            "ci_95_pp": [
                float(np.percentile(boot_row, 2.5) * 100),
                float(np.percentile(boot_row, 97.5) * 100),
            ],
        },
        "col_spread_pp_bootstrap": {
            "mean": float(np.mean(boot_col) * 100),
            "ci_95_pp": [
                float(np.percentile(boot_col, 2.5) * 100),
                float(np.percentile(boot_col, 97.5) * 100),
            ],
        },
    }

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")
    print(f"point ratio: {ratio_pt:.3f}")
    print(
        f"clustered bootstrap 95% CI: "
        f"[{summary['bootstrap']['p2.5']:.3f}, "
        f"{summary['bootstrap']['p97.5']:.3f}]"
    )
    print(f"P(ratio > 1) = {summary['bootstrap']['p_ratio_gt_1']:.3f}")
    print(f"P(ratio > 2) = {summary['bootstrap']['p_ratio_gt_2']:.3f}")


if __name__ == "__main__":
    main()
