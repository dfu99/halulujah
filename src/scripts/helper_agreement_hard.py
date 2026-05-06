"""Audit follow-up #33: Helper agreement distribution on hard questions (§6pp).

§6nn showed orchestration LOO-CV closes only 4% of hard-question oracle
gap. §6cc/§6dd/§6oo all converge on "WHO-asymmetry on hard is a primary-
level finding." This script characterizes WHY: for each hard question
(primary solo wrong), how many of the 6 helper conditions lead to
correct post-collab answer?

Bin each (primary, idx) hard question by recoverability:
  k=0: no helper recovers (primary stays wrong with all 6 helpers)
  k=1-5: partial — some helpers recover, others don't
  k=6: all 6 helpers recover (primary always recovers)

Per-primary distribution shows the structural recoverability of
each primary's hard questions.

Hypothesis (from §6hh recovery rates):
  - biology: many 6/6 recoverable (64% W2C → most hard qs are recoverable
    under most helpers)
  - math, law: many 0/6 recoverable (~21% W2C → many hard qs are
    fundamentally unrecoverable by the helper pool)

Output: results/verified_pair_grid_qwen3_1p7b/helper_agreement_hard.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/helper_agreement_hard.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]

    out = {"per_primary": {}, "pooled": {}}
    pooled_counts: dict[int, int] = {k: 0 for k in range(7)}
    pooled_n_hard = 0

    for p in DOMAINS:
        solo = cond[f"solo_{p}"]["per_q"]
        # Hard idx = solo wrong
        hard_idx = [q["idx"] for q in solo if not q["correct"]]
        n_hard = len(hard_idx)

        # For each hard question, count how many of 6 helpers recover
        recovery_count_per_q = []
        for idx in hard_idx:
            n_recover = 0
            for h in HELPERS:
                pq = cond[f"pair_{p}_{h}"]["per_q"][idx]
                # Confirm idx alignment
                assert pq["idx"] == idx
                if pq["correct"]:
                    n_recover += 1
            recovery_count_per_q.append(n_recover)

        # Bin counts
        counts = Counter(recovery_count_per_q)
        bin_counts = {k: counts[k] for k in range(7)}
        for k, v in bin_counts.items():
            pooled_counts[k] += v

        # Summary stats
        rec_arr = np.array(recovery_count_per_q)
        mean_recovery = float(rec_arr.mean()) if n_hard else 0.0
        # Mean recovery / 6 = mean W2C rate across helpers (for that primary's hard subset)
        # Verify: should match §6hh per-primary W2C
        out["per_primary"][p] = {
            "n_hard": n_hard,
            "bin_counts": bin_counts,
            "mean_recovery_count_of_6": mean_recovery,
            "mean_w2c_rate": mean_recovery / 6 if n_hard else 0.0,
            "frac_0_of_6": bin_counts[0] / n_hard if n_hard else 0.0,
            "frac_6_of_6": bin_counts[6] / n_hard if n_hard else 0.0,
            "frac_majority_recoverable_4_5_6": (
                (bin_counts[4] + bin_counts[5] + bin_counts[6]) / n_hard
                if n_hard else 0.0
            ),
            "frac_minority_recoverable_1_2_3": (
                (bin_counts[1] + bin_counts[2] + bin_counts[3]) / n_hard
                if n_hard else 0.0
            ),
        }
        pooled_n_hard += n_hard

    # Pooled
    out["pooled"] = {
        "n_hard_total": pooled_n_hard,
        "bin_counts": pooled_counts,
        "frac_0_of_6": pooled_counts[0] / pooled_n_hard if pooled_n_hard else 0.0,
        "frac_6_of_6": pooled_counts[6] / pooled_n_hard if pooled_n_hard else 0.0,
        "frac_majority_recoverable_4_5_6": (
            (pooled_counts[4] + pooled_counts[5] + pooled_counts[6]) / pooled_n_hard
            if pooled_n_hard else 0.0
        ),
        "frac_minority_recoverable_1_2_3": (
            (pooled_counts[1] + pooled_counts[2] + pooled_counts[3]) / pooled_n_hard
            if pooled_n_hard else 0.0
        ),
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 78)
    print("§6pp. Helper agreement on hard questions (recovery count of 6 helpers)")
    print("=" * 78)
    print(f"Pooled n_hard = {pooled_n_hard}")
    print()
    print("Recovery count distribution per primary (counts of 0-6 helpers recovering):")
    print(f"{'primary':10s} {'n_hard':>7s}  " + " ".join(f"{k}/6".rjust(6) for k in range(7)) +
          f"  {'mean':>7s}  {'0/6 frac':>9s} {'6/6 frac':>9s}")
    for p in DOMAINS:
        b = out["per_primary"][p]
        print(f"{p:10s} {b['n_hard']:>7d}  " +
              " ".join(f"{b['bin_counts'][k]:>6d}" for k in range(7)) +
              f"  {b['mean_recovery_count_of_6']:>6.2f}  "
              f"{b['frac_0_of_6']*100:>7.1f}% {b['frac_6_of_6']*100:>7.1f}%")
    print()
    print("POOLED:")
    p = out["pooled"]
    print(f"  0 helpers recover: {p['bin_counts'][0]:>4d} ({p['frac_0_of_6']*100:>5.1f}%)  "
          f"← 'mutually unrecoverable' hard questions")
    print(f"  1-3 helpers:       {sum(p['bin_counts'][k] for k in [1,2,3]):>4d} "
          f"({p['frac_minority_recoverable_1_2_3']*100:>5.1f}%)")
    print(f"  4-5 helpers:       {sum(p['bin_counts'][k] for k in [4,5]):>4d} "
          f"({(p['bin_counts'][4]+p['bin_counts'][5])/p['n_hard_total']*100:>5.1f}%)")
    print(f"  6 helpers:         {p['bin_counts'][6]:>4d} ({p['frac_6_of_6']*100:>5.1f}%)  "
          f"← 'universally recoverable' hard questions")
    print()
    print(f"  fraction recoverable by majority of helpers (4+ of 6):  "
          f"{p['frac_majority_recoverable_4_5_6']*100:>5.1f}%")
    print()
    # Implication: of the n_hard pooled, how many are fundamentally
    # unrecoverable (0/6 or 1/6)?
    n_basically_unrecoverable = pooled_counts[0] + pooled_counts[1]
    print(f"  fraction nearly-unrecoverable (0-1 of 6): "
          f"{n_basically_unrecoverable/pooled_n_hard*100:.1f}% "
          f"({n_basically_unrecoverable} of {pooled_n_hard})")


if __name__ == "__main__":
    main()
