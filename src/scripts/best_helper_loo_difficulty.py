"""Audit follow-up #31: Difficulty-stratified LOO-CV (§6nn).

§6mm computed LOO-CV on the full 50-question set per primary:
in-sample +7.0 pp drops to LOO-CV +1.8 pp (overfitting penalty 5.2 pp).
This script restricts the LOO to either:
  - hard subset only (primary solo wrong; n=176 pooled): does
    orchestration generalize on the questions where it matters most?
  - easy subset only (primary solo correct; n=74 pooled): are
    helpers more interchangeable on easy questions?

For each subset, compute LOO-CV pooled accuracy and compare to
in-sample best-by-col-mean restricted to the same subset, to oracle
ceiling on the same subset, and to random baseline.

Hypothesis (from §6cc): hard questions have larger primary-side
spread (43.1 pp vs 18.3 pp on easy). So orchestration on hard
should have MORE potential lift, but generalization might still be
limited by within-cell sampling noise (especially for primaries
with few hard questions like physics, n_hard=44 of 50).

Output: results/verified_pair_grid_qwen3_1p7b/best_helper_loo_difficulty.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/best_helper_loo_difficulty.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]

    # Build solo_correct[primary][idx] and post_correct[primary][helper][idx]
    solo_correct: dict[str, np.ndarray] = {}
    post_correct: dict[str, dict[str, np.ndarray]] = {p: {} for p in DOMAINS}
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

    def loo_on_subset(
        subset_filter: str  # "all" / "hard" / "easy"
    ) -> dict:
        """LOO-CV restricted to held-out indices in the chosen subset.
        For each held-out idx in the chosen subset, compute best-helper
        on the held-in 49 (full grid, ALL questions for the held-in set
        — same as §6mm). Apply on held-out idx. Pool only over held-out
        in the subset.
        """
        per_primary = {}
        total_n = 0
        total_correct = 0
        total_in_sample_n = 0
        total_in_sample_correct = 0
        total_oracle_n = 0
        total_oracle_correct = 0
        total_random_n = 0
        total_random_correct = 0
        for p in DOMAINS:
            sc = solo_correct[p]
            n = len(sc)
            if subset_filter == "hard":
                subset_idx = np.where(sc == 0)[0]
            elif subset_filter == "easy":
                subset_idx = np.where(sc == 1)[0]
            else:
                subset_idx = np.arange(n)

            n_correct = 0
            for held in subset_idx:
                mask = np.ones(n, dtype=bool)
                mask[held] = False
                helper_means = {h: float(post_correct[p][h][mask].mean())
                                for h in HELPERS}
                picked = max(helper_means, key=lambda h: helper_means[h])
                n_correct += int(post_correct[p][picked][held])
            loo_acc = n_correct / len(subset_idx) if len(subset_idx) else 0.0

            # In-sample best-by-col-mean restricted to subset:
            # for each helper, compute mean(post_correct on subset);
            # max over helpers gives in-sample best.
            helper_subset_acc = {h: float(post_correct[p][h][subset_idx].mean())
                                  if len(subset_idx) else 0.0 for h in HELPERS}
            in_sample_best = max(helper_subset_acc, key=lambda h: helper_subset_acc[h])
            in_sample_best_acc = helper_subset_acc[in_sample_best]
            in_sample_correct = int(post_correct[p][in_sample_best][subset_idx].sum())

            # Random baseline: mean across 6 helpers' subset accuracies
            random_acc = float(np.mean(list(helper_subset_acc.values())))
            random_correct = int(round(random_acc * len(subset_idx)))

            # Oracle ceiling on subset: best-of-6 per question
            oracle_correct = 0
            for held in subset_idx:
                anyc = max(int(post_correct[p][h][held]) for h in HELPERS)
                oracle_correct += anyc
            oracle_acc = oracle_correct / len(subset_idx) if len(subset_idx) else 0.0

            per_primary[p] = {
                "n_subset": int(len(subset_idx)),
                "loo_n_correct": int(n_correct),
                "loo_accuracy": loo_acc,
                "in_sample_best_helper": in_sample_best,
                "in_sample_best_acc": in_sample_best_acc,
                "in_sample_n_correct": in_sample_correct,
                "random_baseline_acc": random_acc,
                "oracle_acc": oracle_acc,
                "oracle_n_correct": oracle_correct,
            }
            total_n += len(subset_idx)
            total_correct += n_correct
            total_in_sample_n += len(subset_idx)
            total_in_sample_correct += in_sample_correct
            total_oracle_n += len(subset_idx)
            total_oracle_correct += oracle_correct
            total_random_n += len(subset_idx)
            total_random_correct += random_correct

        pooled_loo_acc = total_correct / total_n if total_n else 0.0
        pooled_in_sample_acc = total_in_sample_correct / total_in_sample_n if total_in_sample_n else 0.0
        pooled_oracle_acc = total_oracle_correct / total_oracle_n if total_oracle_n else 0.0
        # Random pooled is mean of per-primary (each primary equally weighted)
        pooled_random_acc = float(np.mean([per_primary[p]["random_baseline_acc"] for p in DOMAINS]))

        gap_random_to_oracle = pooled_oracle_acc - pooled_random_acc
        return {
            "subset": subset_filter,
            "n_total": total_n,
            "pooled_random_acc": pooled_random_acc,
            "pooled_in_sample_acc": pooled_in_sample_acc,
            "pooled_loo_acc": pooled_loo_acc,
            "pooled_oracle_acc": pooled_oracle_acc,
            "in_sample_lift_pp": (pooled_in_sample_acc - pooled_random_acc) * 100,
            "loo_lift_pp": (pooled_loo_acc - pooled_random_acc) * 100,
            "in_sample_gap_closed_pct": (pooled_in_sample_acc - pooled_random_acc)
                / gap_random_to_oracle * 100 if gap_random_to_oracle > 0 else 0.0,
            "loo_gap_closed_pct": (pooled_loo_acc - pooled_random_acc)
                / gap_random_to_oracle * 100 if gap_random_to_oracle > 0 else 0.0,
            "overfitting_penalty_pp": (pooled_in_sample_acc - pooled_loo_acc) * 100,
            "per_primary": per_primary,
        }

    out = {
        "all": loo_on_subset("all"),
        "hard": loo_on_subset("hard"),
        "easy": loo_on_subset("easy"),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 78)
    print("§6nn. Difficulty-stratified LOO-CV on best-helper selection")
    print("=" * 78)
    for subset in ["all", "easy", "hard"]:
        s = out[subset]
        print()
        print(f"--- subset = {subset.upper()} (n_total = {s['n_total']}) ---")
        print(f"  random baseline:       {s['pooled_random_acc']*100:5.1f}%")
        print(f"  LOO-CV pooled:         {s['pooled_loo_acc']*100:5.1f}%  "
              f"(lift {s['loo_lift_pp']:+5.1f} pp, {s['loo_gap_closed_pct']:.0f}% of gap)")
        print(f"  in-sample best-by-cm:  {s['pooled_in_sample_acc']*100:5.1f}%  "
              f"(lift {s['in_sample_lift_pp']:+5.1f} pp, {s['in_sample_gap_closed_pct']:.0f}% of gap)")
        print(f"  oracle ceiling:        {s['pooled_oracle_acc']*100:5.1f}%")
        print(f"  overfitting penalty:   {s['overfitting_penalty_pp']:+5.1f} pp")
    print()
    print("Per-primary LOO accuracy by subset:")
    print(f"{'primary':10s}  {'all (n=50)':>12s}  {'easy':>14s}  {'hard':>14s}")
    for p in DOMAINS:
        a_pp = out["all"]["per_primary"][p]
        e_pp = out["easy"]["per_primary"][p]
        h_pp = out["hard"]["per_primary"][p]
        print(f"{p:10s}  "
              f"{a_pp['loo_accuracy']*100:>10.1f}%  "
              f"{e_pp['loo_accuracy']*100:>5.1f}% (n={e_pp['n_subset']:>2d})  "
              f"{h_pp['loo_accuracy']*100:>5.1f}% (n={h_pp['n_subset']:>2d})")


if __name__ == "__main__":
    main()
