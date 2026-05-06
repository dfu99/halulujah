"""Audit follow-up #29: Bootstrap stability of best-by-col-mean (§6ll).

§6aa found that the "best-by-col-mean per primary" rule closes 33% of
the §6x oracle gap. Per-primary best helpers (point estimate):
  math       → base    (col delta +0.160)
  medicine   → law     (col delta +0.260)
  biology    → law     (col delta +0.420)
  law        → medicine (col delta +0.100)
  physics    → biology (col delta +0.480)

Question: is this best-helper assignment STABLE under question
resampling? Or is it overfitting to the 50-question sample?

Method: question-cluster bootstrap (n=2000). For each iteration:
  1. Resample 50 questions per primary with replacement (preserving
     alignment across helpers).
  2. Compute the col-mean delta (vs solo) for each cell on the
     resampled questions.
  3. Identify the best helper per primary (argmax col-mean).
  4. Record the assignment.

Then for each primary, compute the bootstrap distribution of
best-helper choices. Stability score = max over helpers of
P(helper is best for this primary).

Hypothesis: math's best (base) is unstable (base barely beats
specialist alternatives); biology's best (law) is stable
(huge margin in the point estimate).

Output: results/verified_pair_grid_qwen3_1p7b/best_helper_stability.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/best_helper_stability.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_ITER = 2000
SEED = 2026


def load_grid() -> tuple[dict, dict]:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]
    post_correct: dict[str, dict[str, np.ndarray]] = {p: {} for p in DOMAINS}
    solo_correct: dict[str, np.ndarray] = {}
    for p in DOMAINS:
        solo_correct[p] = np.array(
            [int(q["correct"]) for q in cond[f"solo_{p}"]["per_q"]],
            dtype=np.int8,
        )
        for h in HELPERS:
            post_correct[p][h] = np.array(
                [int(q["correct"]) for q in cond[f"pair_{p}_{h}"]["per_q"]],
                dtype=np.int8,
            )
    return post_correct, solo_correct


def best_helper_per_primary(
    post_correct: dict,
    solo_correct: dict,
    sample_idx_per_primary: dict[str, np.ndarray],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """For each primary, find the helper with the highest col-mean
    delta on the resampled question set."""
    best = {}
    deltas = {}
    for p in DOMAINS:
        idx = sample_idx_per_primary[p]
        solo_acc = float(solo_correct[p][idx].mean())
        helper_deltas = {}
        for h in HELPERS:
            pair_acc = float(post_correct[p][h][idx].mean())
            helper_deltas[h] = pair_acc - solo_acc
        # Best helper = argmax delta
        best_h = max(helper_deltas, key=lambda h: helper_deltas[h])
        best[p] = best_h
        deltas[p] = helper_deltas
    return best, deltas


def main() -> None:
    post_correct, solo_correct = load_grid()
    rng = np.random.default_rng(SEED)

    n_per_primary = {p: len(solo_correct[p]) for p in DOMAINS}

    # Point estimate
    full_idx = {p: np.arange(n_per_primary[p]) for p in DOMAINS}
    point_best, point_deltas = best_helper_per_primary(post_correct, solo_correct, full_idx)
    print("Point estimates (best helper per primary):")
    for p in DOMAINS:
        print(f"  {p:10s} → {point_best[p]:10s} (delta = {point_deltas[p][point_best[p]]*100:+.1f} pp)")
    print()

    # Bootstrap
    bootstrap_best: dict[str, list[str]] = {p: [] for p in DOMAINS}
    helper_delta_dist: dict[str, dict[str, list[float]]] = {
        p: {h: [] for h in HELPERS} for p in DOMAINS
    }
    for _ in range(N_ITER):
        sample_idx = {
            p: rng.integers(0, n_per_primary[p], size=n_per_primary[p])
            for p in DOMAINS
        }
        best, deltas = best_helper_per_primary(post_correct, solo_correct, sample_idx)
        for p in DOMAINS:
            bootstrap_best[p].append(best[p])
            for h in HELPERS:
                helper_delta_dist[p][h].append(deltas[p][h])

    # Per-primary stability summary
    out = {
        "n_iter": N_ITER,
        "seed": SEED,
        "point_estimate": {
            p: {
                "best_helper": point_best[p],
                "best_delta_pp": float(point_deltas[p][point_best[p]] * 100),
                "all_deltas_pp": {h: float(point_deltas[p][h] * 100) for h in HELPERS},
            }
            for p in DOMAINS
        },
        "bootstrap_per_primary": {},
    }

    for p in DOMAINS:
        counter = Counter(bootstrap_best[p])
        total = sum(counter.values())
        prop = {h: counter[h] / total for h in HELPERS}
        # Stability = P(best helper is the point-estimate best helper)
        stability_at_point = prop[point_best[p]]
        # Maximum-probability helper under bootstrap (might differ from point)
        max_p_helper = max(prop, key=lambda h: prop[h])
        max_prob = prop[max_p_helper]

        # Per-helper delta CIs
        delta_cis = {}
        for h in HELPERS:
            arr = np.array(helper_delta_dist[p][h])
            delta_cis[h] = {
                "median_pp": float(np.median(arr) * 100),
                "p2.5_pp": float(np.percentile(arr, 2.5) * 100),
                "p97.5_pp": float(np.percentile(arr, 97.5) * 100),
            }

        out["bootstrap_per_primary"][p] = {
            "point_best_helper": point_best[p],
            "best_helper_distribution": prop,
            "stability_at_point_best": stability_at_point,
            "max_probability_helper": max_p_helper,
            "max_probability": max_prob,
            "stable": stability_at_point >= 0.5,
            "delta_cis_per_helper": delta_cis,
        }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 78)
    print(f"§6ll. Bootstrap stability of best-by-col-mean (n_iter = {N_ITER})")
    print("=" * 78)
    print(f"{'primary':10s} {'point best':>15s} {'P(point best)':>15s} {'max-prob helper':>18s} {'max prob':>10s}")
    for p in DOMAINS:
        b = out["bootstrap_per_primary"][p]
        print(f"{p:10s} {b['point_best_helper']:>15s} {b['stability_at_point_best']*100:>13.1f}% "
              f"{b['max_probability_helper']:>18s} {b['max_probability']*100:>9.1f}%")
    print()
    print("Detailed best-helper distribution per primary:")
    for p in DOMAINS:
        b = out["bootstrap_per_primary"][p]
        print(f"  {p}:")
        sorted_helpers = sorted(b["best_helper_distribution"].items(),
                                 key=lambda x: x[1], reverse=True)
        for h, prob in sorted_helpers:
            if prob < 0.005:
                continue
            star = " ★" if h == point_best[p] else ""
            print(f"    {h:12s}: {prob*100:>5.1f}%{star}")
    print()
    # Aggregate stability score: average over primaries
    avg_stab = np.mean([
        out["bootstrap_per_primary"][p]["stability_at_point_best"]
        for p in DOMAINS
    ])
    print(f"Average stability across 5 primaries: {avg_stab*100:.1f}%")
    n_stable = sum(1 for p in DOMAINS if out["bootstrap_per_primary"][p]["stable"])
    print(f"Number of stable primaries (P(point best) >= 50%): {n_stable}/5")


if __name__ == "__main__":
    main()
