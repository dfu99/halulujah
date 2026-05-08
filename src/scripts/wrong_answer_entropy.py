"""Audit follow-up #34: Wrong-answer entropy on unrecoverable hard questions (§6qq).

§6pp showed 47.7% of hard questions are nearly unrecoverable
(0-1 of 6 helpers help). Mechanistic question: on these questions,
is the primary STUCK (same wrong letter regardless of helper) or
BUFFETED (helpers move the primary between different wrong
answers)?

For each hard question (primary solo wrong), compute the
distribution of the primary's POST letter across the 6 helper
conditions. Bin by:
  - "stuck": only 1 distinct post letter across 6 helpers
  - "lightly buffeted": 2 distinct letters
  - "moderately buffeted": 3 distinct letters
  - "highly buffeted": 4-5 distinct letters

Cross-tabulate by recovery count (§6pp): are mutually-unrecoverable
questions (k=0) more often stuck or more often buffeted?

If stuck: helpers don't change the primary's reasoning; the primary
locks onto a fixed wrong answer.
If buffeted: helpers do change the primary's reasoning, but never
toward correct.

Hypothesis: stuck questions are more common because solo-wrong
+ confident answers are likely to persist. But the "rank-constraint
destroys collaboration" framing predicts buffeted (helpers do
nudge but never enough to correct).

Output: results/verified_pair_grid_qwen3_1p7b/wrong_answer_entropy.json
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/wrong_answer_entropy.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS


def shannon_entropy(letters: list[str]) -> float:
    n = len(letters)
    if n == 0:
        return 0.0
    cnt = Counter(letters)
    return -sum((v / n) * math.log2(v / n) for v in cnt.values())


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]

    out = {"per_primary": {}, "pooled_by_recovery_bin": {}}
    pooled_records = []

    for p in DOMAINS:
        solo = cond[f"solo_{p}"]["per_q"]
        hard_idx = [q["idx"] for q in solo if not q["correct"]]
        primary_records = []
        for idx in hard_idx:
            expected = cond[f"pair_{p}_base"]["per_q"][idx]["expected"]
            post_letters = []
            recovery_count = 0
            for h in HELPERS:
                pq = cond[f"pair_{p}_{h}"]["per_q"][idx]
                post_letters.append(pq["predicted"])
                if pq["correct"]:
                    recovery_count += 1
            # Distinct post letters
            distinct = len(set(post_letters))
            entropy = shannon_entropy(post_letters)
            # When NOT recovered (i.e., post != expected), what's the wrong-letter
            # entropy? Filter to only the helpers where post is wrong.
            wrong_letters = [l for l in post_letters if l != expected]
            wrong_distinct = len(set(wrong_letters))
            wrong_entropy = shannon_entropy(wrong_letters) if wrong_letters else 0.0
            # Most-common letter and its count
            cnt = Counter(post_letters)
            mode_letter, mode_count = cnt.most_common(1)[0]
            # "Stuck" = mode_count = 6 OR (mode_count = 5-6 AND mode is wrong AND
            # mode != expected)
            # Define more carefully:
            # all_same: all 6 helpers gave same letter (regardless of correct)
            all_same = (distinct == 1)
            primary_records.append({
                "idx": idx,
                "expected": expected,
                "post_letters": post_letters,
                "distinct_count": distinct,
                "entropy": entropy,
                "recovery_count": recovery_count,
                "wrong_distinct_count": wrong_distinct,
                "wrong_entropy": wrong_entropy,
                "mode_letter": mode_letter,
                "mode_count": mode_count,
                "all_same": all_same,
            })

        # Bin by recovery_count
        bin_breakdown = {k: {"n": 0, "stuck": 0, "buffeted_2": 0,
                             "buffeted_3": 0, "buffeted_4plus": 0,
                             "mean_distinct": 0.0, "mean_entropy": 0.0}
                         for k in range(7)}
        for r in primary_records:
            k = r["recovery_count"]
            d = r["distinct_count"]
            bin_breakdown[k]["n"] += 1
            if d == 1:
                bin_breakdown[k]["stuck"] += 1
            elif d == 2:
                bin_breakdown[k]["buffeted_2"] += 1
            elif d == 3:
                bin_breakdown[k]["buffeted_3"] += 1
            else:
                bin_breakdown[k]["buffeted_4plus"] += 1
        # Means per bin
        for k in range(7):
            bin_records = [r for r in primary_records if r["recovery_count"] == k]
            n = len(bin_records)
            if n:
                bin_breakdown[k]["mean_distinct"] = sum(r["distinct_count"] for r in bin_records) / n
                bin_breakdown[k]["mean_entropy"] = sum(r["entropy"] for r in bin_records) / n

        out["per_primary"][p] = {
            "n_hard": len(primary_records),
            "bin_breakdown": bin_breakdown,
            "all_records": primary_records,
        }
        pooled_records.extend([{**r, "primary": p} for r in primary_records])

    # Pooled by recovery bin
    pooled_bins = {k: {"n": 0, "stuck": 0, "buffeted_2": 0,
                       "buffeted_3": 0, "buffeted_4plus": 0,
                       "mean_distinct": 0.0, "mean_entropy": 0.0}
                   for k in range(7)}
    for r in pooled_records:
        k = r["recovery_count"]
        d = r["distinct_count"]
        pooled_bins[k]["n"] += 1
        if d == 1:
            pooled_bins[k]["stuck"] += 1
        elif d == 2:
            pooled_bins[k]["buffeted_2"] += 1
        elif d == 3:
            pooled_bins[k]["buffeted_3"] += 1
        else:
            pooled_bins[k]["buffeted_4plus"] += 1
    for k in range(7):
        bin_records = [r for r in pooled_records if r["recovery_count"] == k]
        n = len(bin_records)
        if n:
            pooled_bins[k]["mean_distinct"] = sum(r["distinct_count"] for r in bin_records) / n
            pooled_bins[k]["mean_entropy"] = sum(r["entropy"] for r in bin_records) / n
    out["pooled_by_recovery_bin"] = pooled_bins

    # Headline: across all 176 hard questions, what fraction are "stuck"
    # (all 6 helpers same letter)?
    n_total_hard = len(pooled_records)
    n_stuck = sum(1 for r in pooled_records if r["all_same"])
    n_stuck_wrong = sum(1 for r in pooled_records if r["all_same"] and r["mode_letter"] != r["expected"])
    n_stuck_correct = n_stuck - n_stuck_wrong  # hard but somehow all 6 helpers got it right and identical letter
    out["headline"] = {
        "n_total_hard": n_total_hard,
        "n_stuck_all_same_letter": n_stuck,
        "frac_stuck": n_stuck / n_total_hard,
        "n_stuck_wrong_letter": n_stuck_wrong,
        "n_stuck_correct_letter": n_stuck_correct,
        "frac_stuck_wrong": n_stuck_wrong / n_total_hard,
        "frac_stuck_correct": n_stuck_correct / n_total_hard,
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 78)
    print(f"§6qq. Wrong-answer entropy on hard questions (n_total = {n_total_hard})")
    print("=" * 78)
    print()
    h = out["headline"]
    print(f"Pooled headline:")
    print(f"  All 6 helpers same letter ('stuck'):              "
          f"{h['n_stuck_all_same_letter']:>3d} / {n_total_hard} = {h['frac_stuck']*100:.1f}%")
    print(f"    of which: stuck on WRONG letter:                "
          f"{h['n_stuck_wrong_letter']:>3d}  ({h['frac_stuck_wrong']*100:.1f}% of hard)")
    print(f"    of which: stuck on CORRECT letter (all helpers): "
          f"{h['n_stuck_correct_letter']:>3d}  ({h['frac_stuck_correct']*100:.1f}% of hard)")
    print()
    print(f"Pooled distribution by recovery bin (k = #helpers recover):")
    print(f"{'k':>3} {'n':>5} {'stuck (1L)':>11s} {'buff-2L':>9s} {'buff-3L':>9s} {'buff-4+L':>10s} {'mean dist':>10s} {'mean H':>8s}")
    for k in range(7):
        b = pooled_bins[k]
        if b["n"] == 0:
            continue
        print(f"{k:>3} {b['n']:>5d} {b['stuck']:>10d}  {b['buffeted_2']:>8d}  {b['buffeted_3']:>8d}  "
              f"{b['buffeted_4plus']:>9d}  {b['mean_distinct']:>9.2f}  {b['mean_entropy']:>7.2f}")
    print()
    # Stuck rate among k=0 vs k=6
    k0 = pooled_bins[0]
    k6 = pooled_bins[6]
    print(f"k=0 (mutually unrecoverable): stuck rate = "
          f"{k0['stuck']}/{k0['n']} = {k0['stuck']/k0['n']*100:.1f}% (mean distinct {k0['mean_distinct']:.2f})")
    print(f"k=6 (universally recoverable): stuck rate = "
          f"{k6['stuck']}/{k6['n']} = {k6['stuck']/k6['n']*100:.1f}% (mean distinct {k6['mean_distinct']:.2f})")


if __name__ == "__main__":
    main()
