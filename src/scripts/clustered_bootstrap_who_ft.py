"""Question-clustered bootstrap on the FT pair-grid WHO-asymmetry ratio.

Audit context: tasks/audit-2026-05-05.md §14d (point estimate 52.37×
under simple var-of-row-means/col-means) and §14f caveat #2 — cluster-
respecting bootstrap on the FT grid is the matched complement to the
LoRA grid's audit §6f bootstrap (see clustered_bootstrap_who.py).

Reads results/ft_pair_grid_2026-05-08/matrix_results_v2.json (the
parser-repaired matrix). Same question-clustering strategy as
clustered_bootstrap_who.py: draw n_per_primary indices with
replacement per primary, apply the same drawn indices across all 6
helper columns (preserves question-identity dependence). Repeat
2000x, percentile CI.

Outputs:
  results/ft_pair_grid_2026-05-08/clustered_bootstrap.json
  Update audit §14d / §14f with the cluster-corrected CI (manual).

Run: python -m src.scripts.clustered_bootstrap_who_ft
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/ft_pair_grid_2026-05-08/matrix_results_v2.json"
OUT = ROOT / "results/ft_pair_grid_2026-05-08/clustered_bootstrap.json"

DOMAINS = ["medicine", "math", "biology", "law", "physics"]
HELPERS = DOMAINS + ["base"]

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
        # Use v2 accuracy if available (FT grid has both); fall back to orig
        solo = C[f"solo_{p}"]
        solo_acc[p] = solo.get("accuracy_v2", solo.get("accuracy", 0.0))
        for h in HELPERS:
            cell = C[f"pair_{p}_{h}"]
            per_q = cell["per_q"]
            # Use correct_v2 if present (FT grid); else 'correct' (LoRA)
            grid[p][h] = np.array(
                [int(q.get("correct_v2", q.get("correct", False)))
                 for q in per_q], dtype=np.int8
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
) -> tuple[float, float, float, float]:
    """Compute (row_spread, col_spread, spread_ratio, variance_ratio).

    Cell value = mean(post_correct[sample_idx]) - solo_acc[primary]
    spread_ratio   = (max-min row mean) / (max-min col mean) — §6b
    variance_ratio = SS_primary / SS_helper                  — §6m
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
    grand_mean = float(cell.mean())
    row_spread = float(row_mean.max() - row_mean.min())
    col_spread = float(col_mean.max() - col_mean.min())
    spread_ratio = (
        row_spread / col_spread if col_spread > 0 else float("inf")
    )
    SS_primary = float(len(HELPERS) * ((row_mean - grand_mean) ** 2).sum())
    SS_helper = float(len(DOMAINS) * ((col_mean - grand_mean) ** 2).sum())
    variance_ratio = (
        SS_primary / SS_helper if SS_helper > 0 else float("inf")
    )
    return row_spread, col_spread, spread_ratio, variance_ratio


def main() -> None:
    grid, solo_acc = load_grid()
    n_per_primary = {p: len(grid[p]["base"]) for p in DOMAINS}
    print(f"loaded grid: {n_per_primary}")

    # Point estimate (no resampling)
    row_pt, col_pt, sratio_pt, vratio_pt = who_ratio_from_grid(grid, solo_acc)
    print(
        f"point estimate: row_spread={row_pt*100:.1f}pp "
        f"col_spread={col_pt*100:.1f}pp"
    )
    print(f"  spread ratio   (§6b): {sratio_pt:.3f}")
    print(f"  variance ratio (§6m): {vratio_pt:.3f}")

    np_rng = np.random.default_rng(SEED)
    boot_sratio: list[float] = []
    boot_vratio: list[float] = []
    boot_row: list[float] = []
    boot_col: list[float] = []
    for _ in range(N_ITER):
        sample_idx = {
            p: np_rng.integers(0, n_per_primary[p], size=n_per_primary[p])
            for p in DOMAINS
        }
        rs, cs, sr, vr = who_ratio_from_grid(grid, solo_acc, sample_idx)
        boot_sratio.append(sr)
        boot_vratio.append(vr)
        boot_row.append(rs)
        boot_col.append(cs)

    def summarize(boot: list[float]) -> dict:
        a = np.array(boot)
        a = a[np.isfinite(a)]
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
            "ci_95": [float(pcts[0]), float(pcts[4])],
            "ci_90": [float(pcts[1]), float(pcts[3])],
            "p_gt_1": float((a > 1.0).mean()),
            "p_gt_2": float((a > 2.0).mean()),
            "p_gt_3": float((a > 3.0).mean()),
            "p_gt_5": float((a > 5.0).mean()),
        }

    summary = {
        "method": "question-clustered bootstrap on canonical aggregator",
        "n_iterations": N_ITER,
        "seed": SEED,
        "point": {
            "row_spread_pp": row_pt * 100,
            "col_spread_pp": col_pt * 100,
            "spread_ratio_§6b": sratio_pt,
            "variance_ratio_§6m": vratio_pt,
        },
        "bootstrap_spread_ratio": summarize(boot_sratio),
        "bootstrap_variance_ratio": summarize(boot_vratio),
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
    print(f"\nspread ratio (§6b)   point {sratio_pt:.3f}, "
          f"95% CI [{summary['bootstrap_spread_ratio']['p2.5']:.2f}, "
          f"{summary['bootstrap_spread_ratio']['p97.5']:.2f}]")
    print(f"variance ratio (§6m) point {vratio_pt:.3f}, "
          f"95% CI [{summary['bootstrap_variance_ratio']['p2.5']:.2f}, "
          f"{summary['bootstrap_variance_ratio']['p97.5']:.2f}]")
    print(f"P(spread ratio > 1) = {summary['bootstrap_spread_ratio']['p_gt_1']:.3f}")
    print(f"P(variance ratio > 5) = {summary['bootstrap_variance_ratio']['p_gt_5']:.3f}")


if __name__ == "__main__":
    main()
