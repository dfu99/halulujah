"""Audit follow-up #30: LOO-CV on best-helper selection (§6mm).

§6aa reported +7.0 pp gap closure (in-sample) for the
"best-by-col-mean per primary" rule. §6ll showed the per-primary
best-helper assignment is stable for only 3/5 primaries under
bootstrap. This script directly tests the out-of-sample
generalization via leave-one-question-out cross-validation.

Method (per primary):
  For each idx in 0..49:
    1. Hold out question idx
    2. On the remaining 49 questions, compute the best-helper-by-
       col-mean (i.e., max over helpers of mean post_correct on the
       49 held-in questions; equivalent under fixed solo).
    3. Use that chosen helper's prediction on the held-out idx as
       the LOO prediction.
    4. Record whether the LOO prediction is correct.
  Pool LOO accuracy across the 50 held-out questions per primary,
  then across primaries.

Hypotheses:
  - LOO-CV pooled accuracy will be LOWER than in-sample 58.0% but
    HIGHER than random 51.0%, since the best-helper rule
    generalizes partially.
  - Predicted LOO-CV: somewhere in 53-56% range (closing 10-25%
    of the oracle gap, vs 33% in-sample).
  - Per-primary LOO-CV: physics (most stable) should generalize
    best; medicine/law (unstable) should generalize worst.

Output: results/verified_pair_grid_qwen3_1p7b/best_helper_loo_cv.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
ORACLE_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/oracle_ceiling.json"
ORCH_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/helper_orchestration.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/best_helper_loo_cv.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]
    oracle = json.loads(ORACLE_PATH.read_text())
    orch = json.loads(ORCH_PATH.read_text())

    # Build post_correct[primary][helper] = length-50 array
    post_correct: dict[str, dict[str, np.ndarray]] = {p: {} for p in DOMAINS}
    for p in DOMAINS:
        for h in HELPERS:
            post_correct[p][h] = np.array(
                [int(q["correct"]) for q in cond[f"pair_{p}_{h}"]["per_q"]],
                dtype=np.int8,
            )

    # In-sample best helper (replicates §6aa)
    in_sample_best: dict[str, str] = {}
    for p in DOMAINS:
        helper_means = {h: float(post_correct[p][h].mean()) for h in HELPERS}
        in_sample_best[p] = max(helper_means, key=lambda h: helper_means[h])

    # LOO-CV per (primary, idx)
    loo_predictions: dict[str, list[dict]] = {p: [] for p in DOMAINS}
    for p in DOMAINS:
        n = len(post_correct[p]["base"])
        for held in range(n):
            mask = np.ones(n, dtype=bool)
            mask[held] = False
            # Best helper on held-in 49
            helper_means = {h: float(post_correct[p][h][mask].mean()) for h in HELPERS}
            picked = max(helper_means, key=lambda h: helper_means[h])
            # Apply picked helper's prediction on held-out idx
            loo_correct = int(post_correct[p][picked][held])
            loo_predictions[p].append({
                "idx": held,
                "picked_helper": picked,
                "correct": loo_correct,
            })

    # Aggregate per-primary
    per_primary = {}
    for p in DOMAINS:
        records = loo_predictions[p]
        n = len(records)
        n_correct = sum(r["correct"] for r in records)
        loo_acc = n_correct / n if n else 0.0
        # Picked-helper stability under LOO: how often is the
        # in-sample best helper picked?
        picked_dist = {}
        for r in records:
            h = r["picked_helper"]
            picked_dist[h] = picked_dist.get(h, 0) + 1
        per_primary[p] = {
            "n": n,
            "n_correct": n_correct,
            "loo_accuracy": loo_acc,
            "in_sample_best_helper": in_sample_best[p],
            "picked_helper_dist": {h: picked_dist[h] / n for h in picked_dist},
            "in_sample_best_picked_frac": picked_dist.get(in_sample_best[p], 0) / n,
        }

    # Pooled LOO accuracy (uniform weight over primaries since each has 50)
    pooled_loo_n = sum(per_primary[p]["n"] for p in DOMAINS)
    pooled_loo_correct = sum(per_primary[p]["n_correct"] for p in DOMAINS)
    pooled_loo_acc = pooled_loo_correct / pooled_loo_n

    # Comparison metrics
    in_sample_best_acc = orch["best_by_col_mean"]["pooled_accuracy"]
    random_acc = orch["random_actual_mean"]["pooled_accuracy"]
    oracle_acc = oracle["pooled"]["oracle_acc"]
    gap_random_to_oracle = oracle_acc - random_acc

    out = {
        "method": "leave-one-question-out cross-validation",
        "n_total_loo_predictions": pooled_loo_n,
        "pooled_loo_accuracy": pooled_loo_acc,
        "comparison": {
            "random_baseline_acc": random_acc,
            "in_sample_best_acc": in_sample_best_acc,
            "oracle_acc": oracle_acc,
            "loo_acc": pooled_loo_acc,
            "in_sample_lift_over_random_pp": (in_sample_best_acc - random_acc) * 100,
            "loo_lift_over_random_pp": (pooled_loo_acc - random_acc) * 100,
            "in_sample_gap_closed_pct": (in_sample_best_acc - random_acc) / gap_random_to_oracle * 100
                if gap_random_to_oracle > 0 else 0.0,
            "loo_gap_closed_pct": (pooled_loo_acc - random_acc) / gap_random_to_oracle * 100
                if gap_random_to_oracle > 0 else 0.0,
            "overfitting_penalty_pp": (in_sample_best_acc - pooled_loo_acc) * 100,
        },
        "per_primary": per_primary,
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 78)
    print("§6mm. LOO-CV on best-helper selection")
    print("=" * 78)
    print(f"Random baseline (mean across 6 helpers): {random_acc*100:5.1f}%")
    print(f"In-sample best-by-col-mean (§6aa):       {in_sample_best_acc*100:5.1f}%")
    print(f"LOO-CV best-by-col-mean:                 {pooled_loo_acc*100:5.1f}%")
    print(f"Oracle ceiling (best-of-6 per question): {oracle_acc*100:5.1f}%")
    print()
    c = out["comparison"]
    print(f"In-sample lift over random:   {c['in_sample_lift_over_random_pp']:+5.1f} pp ({c['in_sample_gap_closed_pct']:.0f}% of gap)")
    print(f"LOO-CV    lift over random:   {c['loo_lift_over_random_pp']:+5.1f} pp ({c['loo_gap_closed_pct']:.0f}% of gap)")
    print(f"Overfitting penalty (in-LOO): {c['overfitting_penalty_pp']:+5.1f} pp")
    print()
    print(f"{'primary':10s} {'LOO acc':>10s} {'pt best':>15s} {'in-best picked':>15s}")
    for p in DOMAINS:
        b = per_primary[p]
        print(f"{p:10s} {b['loo_accuracy']*100:>9.1f}% {b['in_sample_best_helper']:>15s} {b['in_sample_best_picked_frac']*100:>14.1f}%")
    print()
    # Show how often the LOO-picked helper matches the in-sample best
    print("LOO-picked helper distribution per primary:")
    for p in DOMAINS:
        d = per_primary[p]["picked_helper_dist"]
        sorted_h = sorted(d.items(), key=lambda x: x[1], reverse=True)
        print(f"  {p}: " + ", ".join(f"{h}={pct*100:.0f}%" for h, pct in sorted_h if pct >= 0.05))


if __name__ == "__main__":
    main()
