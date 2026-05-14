"""Question-clustered bootstrap on the 1.7B FT WHO ratio AFTER parser recovery.

The 2026-05-13 parser audit (`who_recovered_2026-05-13.json`,
commit `f60d5e0`) re-extracted `predicted` from `final_raw` in every
pair cell of the 1.7B FT pair-grid using the permissive parser. X-rate
dropped 36% -> 20.5% (233 letters recovered, 15.5% predictions
changed). The variance ratio strengthened from 38.79x (strict) to
54.74x (permissive).

This script is the matched complement to `clustered_bootstrap_who_ft.py`
on the *recovered* cells (cells_v3/). Convention matches
`who_recovered_2026-05-13.json`: solos=0 for all primaries (variance
ratio computed on raw post-collab accuracy, NOT on delta cells, because
the FT solo cells have no `final_raw` and cannot be parser-recovered).

Same question-clustering strategy as `clustered_bootstrap_who.py`:
draw n indices with replacement per primary, apply the same drawn
indices across all 6 helper columns (preserves question-identity
dependence within a primary row). Repeat 2000x, percentile CI.

Outputs:
  results/ft_pair_grid_2026-05-08/clustered_bootstrap_v3.json
  figures/clustered_bootstrap_ft_recovered_2026-05-13.png  (separate)

Run: python -m src.scripts.clustered_bootstrap_who_ft_recovered
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CELLS = ROOT / "results/ft_pair_grid_2026-05-08/cells_v3"
OUT = ROOT / "results/ft_pair_grid_2026-05-08/clustered_bootstrap_v3.json"

PRIMARIES = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]

N_ITER = 2000
SEED = 2026


def load_grid() -> dict[str, dict[str, np.ndarray]]:
    """grid[primary][helper] is a length-n int8 array of post-deliberation
    `correct` flags from cells_v3 (recovered parser).

    Asserts question alignment by (subject, expected) across the 6 helper
    columns within each primary row.
    """
    grid: dict[str, dict[str, np.ndarray]] = {p: {} for p in PRIMARIES}
    for p in PRIMARIES:
        for h in HELPERS:
            path = CELLS / f"pair_{p}_{h}.json"
            cell = json.loads(path.read_text())
            grid[p][h] = np.array(
                [int(q["correct"]) for q in cell["per_q"]], dtype=np.int8
            )

    for p in PRIMARIES:
        ref = json.loads((CELLS / f"pair_{p}_base.json").read_text())["per_q"]
        for h in HELPERS[1:]:
            other = json.loads(
                (CELLS / f"pair_{p}_{h}.json").read_text()
            )["per_q"]
            assert len(ref) == len(other), (
                f"length mismatch primary={p} helper={h}: "
                f"{len(ref)} vs {len(other)}"
            )
            for i, (r, o) in enumerate(zip(ref, other)):
                assert r["subject"] == o["subject"], (
                    f"question alignment broken at primary={p} "
                    f"helper={h} idx={i}: "
                    f"{r['subject']} != {o['subject']}"
                )
                assert r["expected"] == o["expected"], (
                    f"answer mismatch primary={p} helper={h} idx={i}"
                )
    return grid


def who_metrics(
    grid: dict[str, dict[str, np.ndarray]],
    sample_idx: dict[str, np.ndarray] | None = None,
) -> tuple[float, float, float, float]:
    """Return (row_spread, col_spread, spread_ratio, variance_ratio).

    Cell value = mean(correct[sample_idx]).  Solos = 0 by convention
    (matches `who_recovered_2026-05-13.json`).
    """
    cell = np.zeros((len(PRIMARIES), len(HELPERS)), dtype=float)
    for i, p in enumerate(PRIMARIES):
        idx = sample_idx[p] if sample_idx is not None else None
        for j, h in enumerate(HELPERS):
            arr = grid[p][h]
            cell[i, j] = float(arr.mean() if idx is None else arr[idx].mean())

    row_mean = cell.mean(axis=1)
    col_mean = cell.mean(axis=0)
    grand_mean = float(cell.mean())
    row_spread = float(row_mean.max() - row_mean.min())
    col_spread = float(col_mean.max() - col_mean.min())
    spread_ratio = (
        row_spread / col_spread if col_spread > 0 else float("inf")
    )
    ss_primary = float(len(HELPERS) * ((row_mean - grand_mean) ** 2).sum())
    ss_helper = float(len(PRIMARIES) * ((col_mean - grand_mean) ** 2).sum())
    variance_ratio = (
        ss_primary / ss_helper if ss_helper > 0 else float("inf")
    )
    return row_spread, col_spread, spread_ratio, variance_ratio


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
        "p_gt_5": float((a > 5.0).mean()),
        "p_gt_10": float((a > 10.0).mean()),
        "p_gt_20": float((a > 20.0).mean()),
        "p_gt_30": float((a > 30.0).mean()),
    }


def main() -> None:
    grid = load_grid()
    n_per_primary = {p: len(grid[p]["base"]) for p in PRIMARIES}
    print(f"loaded cells_v3 grid: {n_per_primary}")

    row_pt, col_pt, sratio_pt, vratio_pt = who_metrics(grid)
    print(
        f"point: row_spread={row_pt * 100:.2f}pp "
        f"col_spread={col_pt * 100:.2f}pp"
    )
    print(f"  spread ratio   (row/col, max-min):     {sratio_pt:.3f}")
    print(f"  variance ratio (SS_primary/SS_helper): {vratio_pt:.3f}")

    np_rng = np.random.default_rng(SEED)
    b_row, b_col, b_sratio, b_vratio = [], [], [], []
    for _ in range(N_ITER):
        sample_idx = {
            p: np_rng.integers(0, n_per_primary[p], size=n_per_primary[p])
            for p in PRIMARIES
        }
        rs, cs, sr, vr = who_metrics(grid, sample_idx)
        b_row.append(rs)
        b_col.append(cs)
        b_sratio.append(sr)
        b_vratio.append(vr)

    summary = {
        "method": (
            "question-clustered bootstrap on recovered cells_v3 grid; "
            "solos=0 convention matching who_recovered_2026-05-13.json"
        ),
        "n_iterations": N_ITER,
        "seed": SEED,
        "point": {
            "row_spread_pp": row_pt * 100,
            "col_spread_pp": col_pt * 100,
            "spread_ratio_row_over_col": sratio_pt,
            "variance_ratio_ss_primary_over_ss_helper": vratio_pt,
        },
        "bootstrap_spread_ratio": summarize(b_sratio),
        "bootstrap_variance_ratio": summarize(b_vratio),
        "row_spread_pp_bootstrap": {
            "mean": float(np.mean(b_row) * 100),
            "ci_95_pp": [
                float(np.percentile(b_row, 2.5) * 100),
                float(np.percentile(b_row, 97.5) * 100),
            ],
        },
        "col_spread_pp_bootstrap": {
            "mean": float(np.mean(b_col) * 100),
            "ci_95_pp": [
                float(np.percentile(b_col, 2.5) * 100),
                float(np.percentile(b_col, 97.5) * 100),
            ],
        },
    }

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}\n")

    print(
        f"spread ratio   point {sratio_pt:.3f}, "
        f"95% CI [{summary['bootstrap_spread_ratio']['p2.5']:.2f}, "
        f"{summary['bootstrap_spread_ratio']['p97.5']:.2f}]"
    )
    print(
        f"variance ratio point {vratio_pt:.3f}, "
        f"95% CI [{summary['bootstrap_variance_ratio']['p2.5']:.2f}, "
        f"{summary['bootstrap_variance_ratio']['p97.5']:.2f}]"
    )
    print(
        f"P(spread > 1)   = "
        f"{summary['bootstrap_spread_ratio']['p_gt_1']:.3f}"
    )
    print(
        f"P(variance > 1) = "
        f"{summary['bootstrap_variance_ratio']['p_gt_1']:.3f}"
    )
    print(
        f"P(variance > 5) = "
        f"{summary['bootstrap_variance_ratio']['p_gt_5']:.3f}"
    )
    print(
        f"P(variance > 10) = "
        f"{summary['bootstrap_variance_ratio']['p_gt_10']:.3f}"
    )


if __name__ == "__main__":
    main()
