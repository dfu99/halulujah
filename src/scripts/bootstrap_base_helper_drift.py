"""Bootstrap CI on the base-helper drift delta (audit §15c finding).

The §15c headline: Δ accuracy from cp1500 → final on the 5 base-helper
cells (pair_<primary>_base) averages exactly 0.0 pp across the 5
primaries. CI on that 0.0 pp delta is the missing piece per audit §15f
caveat #2.

Method: question-paired bootstrap. For each primary, the 50 questions
in pair_<primary>_base are aligned across the cp1500 and final grids
(same seed=42 sampling). Resample 50 indices with replacement per
primary, recompute (correct_v2_final[idx].mean() - correct_v2_cp1500[idx].mean())
for each base-helper cell, then take the mean across 5 primaries.
Repeat 2000x, percentile CI.

Reads:
  results/ft_pair_grid_2026-05-08/matrix_results_v2.json (final)
  results/ft_pair_grid_step1500_2026-05-08/matrix_results_v2.json (cp1500)

Writes:
  results/ft_pair_grid_2026-05-08/base_helper_drift_bootstrap.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "results/ft_pair_grid_2026-05-08/matrix_results_v2.json"
EARLY = ROOT / "results/ft_pair_grid_step1500_2026-05-08/matrix_results_v2.json"
OUT = ROOT / "results/ft_pair_grid_2026-05-08/base_helper_drift_bootstrap.json"

DOMAINS = ["medicine", "math", "biology", "law", "physics"]
N_ITER = 2000
SEED = 2026


def per_q_correct(grid: dict, primary: str, helper: str) -> np.ndarray:
    cell = grid["conditions"][f"pair_{primary}_{helper}"]
    return np.array(
        [int(q.get("correct_v2", q.get("correct", False)))
         for q in cell["per_q"]],
        dtype=np.int8,
    )


def main() -> None:
    final = json.loads(FINAL.read_text())
    early = json.loads(EARLY.read_text())

    # Sanity-check question alignment: cp1500 and final must use the
    # same seed=42 question pool per primary.
    for p in DOMAINS:
        f_pq = final["conditions"][f"pair_{p}_base"]["per_q"]
        e_pq = early["conditions"][f"pair_{p}_base"]["per_q"]
        for i, (fq, eq) in enumerate(zip(f_pq, e_pq)):
            assert fq["subject"] == eq["subject"], (
                f"alignment broken {p} idx={i}")
            assert fq["expected"] == eq["expected"], (
                f"answer mismatch {p} idx={i}")

    # Per-cell point delta and bootstrap
    point_deltas = {}
    bootstrap_deltas: dict[str, list[float]] = {p: [] for p in DOMAINS}
    rng = np.random.default_rng(SEED)
    for p in DOMAINS:
        f_arr = per_q_correct(final, p, "base")
        e_arr = per_q_correct(early, p, "base")
        n = len(f_arr)
        point_deltas[p] = float(f_arr.mean() - e_arr.mean())
    print(f"point per-primary base-helper Δ acc:")
    for p, v in point_deltas.items():
        print(f"  {p:9s}: {v:+.3f}")

    # Now bootstrap: resample indices per primary
    for it in range(N_ITER):
        for p in DOMAINS:
            f_arr = per_q_correct(final, p, "base")
            e_arr = per_q_correct(early, p, "base")
            n = len(f_arr)
            idx = rng.integers(0, n, size=n)
            d = float(f_arr[idx].mean() - e_arr[idx].mean())
            bootstrap_deltas[p].append(d)

    # Mean-across-5-primaries bootstrap distribution
    n_iter = N_ITER
    mean_boot = []
    for it in range(n_iter):
        m = float(np.mean([bootstrap_deltas[p][it] for p in DOMAINS]))
        mean_boot.append(m)

    arr = np.array(mean_boot)
    pcts = np.percentile(arr, [2.5, 5, 50, 95, 97.5])
    summary = {
        "method": "question-paired bootstrap on per-primary base-helper Δ acc",
        "n_iterations": N_ITER,
        "seed": SEED,
        "point_per_primary": point_deltas,
        "point_mean": float(np.mean(list(point_deltas.values()))),
        "mean_across_5_primaries": {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "ci_95": [float(pcts[0]), float(pcts[4])],
            "ci_90": [float(pcts[1]), float(pcts[3])],
            "p_lt_0": float((arr < 0.0).mean()),
            "p_lt_minus_2pp": float((arr < -0.02).mean()),
            "p_gt_2pp": float((arr > 0.02).mean()),
            "p_gt_5pp": float((arr > 0.05).mean()),
            "p_abs_gt_5pp": float((np.abs(arr) > 0.05).mean()),
        },
        "per_primary_bootstrap": {
            p: {
                "mean": float(np.mean(bootstrap_deltas[p])),
                "ci_95": [
                    float(np.percentile(bootstrap_deltas[p], 2.5)),
                    float(np.percentile(bootstrap_deltas[p], 97.5)),
                ],
            }
            for p in DOMAINS
        },
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}")
    print(f"\nMean Δ acc base-helper across 5 primaries:")
    print(f"  point: {summary['point_mean']:+.3f}")
    print(f"  bootstrap mean: {arr.mean():+.3f}")
    print(f"  95% CI: [{pcts[0]:+.3f}, {pcts[4]:+.3f}]")
    print(f"  P(Δ < 0)   = {summary['mean_across_5_primaries']['p_lt_0']:.3f}")
    print(f"  P(|Δ|>5pp) = {summary['mean_across_5_primaries']['p_abs_gt_5pp']:.3f}")


if __name__ == "__main__":
    main()
