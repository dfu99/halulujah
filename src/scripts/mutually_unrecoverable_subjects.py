"""Audit follow-up #34: Subject decomposition of mutually-unrecoverable hard
questions (§6qq).

§6pp identified that 47.7% of hard questions are nearly-unrecoverable (0-1
of 6 helpers help) and that math/law primaries have 53% mutually-
unrecoverable hard rates vs biology's 19%. This script decomposes that
finding by MMLU subject within each primary.

Hypothesis: math primary's 53% mutually-unrecoverable rate is concentrated
in particular MMLU subjects (e.g. abstract_algebra, college_mathematics)
rather than uniformly distributed; elementary_mathematics may show lower
mutual-unrecoverability than its solo accuracy suggests.

This addresses Reviewer I's "subject-mix" concern restricted to the hard
subset: if mutual-unrecoverability is a subject-level phenomenon, then the
§6cc 67.69× hard WHO-asymmetry is partly a "which subjects MMLU treats as
hard" question rather than a primary-identity question.

Output: results/verified_pair_grid_qwen3_1p7b/mutually_unrecoverable_subjects.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/mutually_unrecoverable_subjects.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]

    out: dict = {"per_primary_subject": {}, "pooled_subject": {}}

    pooled_subject_n: dict[str, int] = defaultdict(int)
    pooled_subject_hard: dict[str, int] = defaultdict(int)
    pooled_subject_unrec: dict[str, int] = defaultdict(int)
    pooled_subject_univ: dict[str, int] = defaultdict(int)
    pooled_subject_recsum: dict[str, int] = defaultdict(int)

    for p in DOMAINS:
        solo = cond[f"solo_{p}"]["per_q"]

        # All idx (any difficulty) by subject
        subject_total: dict[str, int] = defaultdict(int)
        # Hard (solo wrong) idx by subject
        subject_hard: dict[str, list[int]] = defaultdict(list)

        for q in solo:
            subject_total[q["subject"]] += 1
            if not q["correct"]:
                subject_hard[q["subject"]].append(q["idx"])

        per_subject = {}
        for subj, hard_idxs in subject_hard.items():
            n_hard_subj = len(hard_idxs)
            n_subj_total = subject_total[subj]
            recovery_counts = []
            for idx in hard_idxs:
                n_recover = 0
                for h in HELPERS:
                    pq = cond[f"pair_{p}_{h}"]["per_q"][idx]
                    assert pq["idx"] == idx
                    if pq["correct"]:
                        n_recover += 1
                recovery_counts.append(n_recover)

            unrec_count = sum(1 for k in recovery_counts if k == 0)
            univ_count = sum(1 for k in recovery_counts if k == 6)
            mean_rec = float(np.mean(recovery_counts)) if recovery_counts else 0.0

            per_subject[subj] = {
                "n_total": n_subj_total,
                "n_hard": n_hard_subj,
                "hard_rate": n_hard_subj / n_subj_total if n_subj_total else 0.0,
                "n_mutually_unrecoverable": unrec_count,
                "n_universally_recoverable": univ_count,
                "frac_unrec_of_hard": unrec_count / n_hard_subj if n_hard_subj else 0.0,
                "frac_univ_of_hard": univ_count / n_hard_subj if n_hard_subj else 0.0,
                "mean_recovery_count_of_6": mean_rec,
                "recovery_distribution": dict(Counter(recovery_counts)),
            }

            pooled_subject_n[subj] += n_subj_total
            pooled_subject_hard[subj] += n_hard_subj
            pooled_subject_unrec[subj] += unrec_count
            pooled_subject_univ[subj] += univ_count
            pooled_subject_recsum[subj] += sum(recovery_counts)

        # Add subjects that had no hard questions
        for subj, n in subject_total.items():
            if subj not in per_subject:
                per_subject[subj] = {
                    "n_total": n,
                    "n_hard": 0,
                    "hard_rate": 0.0,
                    "n_mutually_unrecoverable": 0,
                    "n_universally_recoverable": 0,
                    "frac_unrec_of_hard": 0.0,
                    "frac_univ_of_hard": 0.0,
                    "mean_recovery_count_of_6": float("nan"),
                    "recovery_distribution": {},
                }
                pooled_subject_n[subj] += n

        out["per_primary_subject"][p] = per_subject

    # Pooled across primaries (a subject can appear in multiple primaries
    # only if MMLU sampler overlapped — in practice each subject is
    # primary-specific because the 5 MMLU domain splits are disjoint)
    pooled = {}
    for subj in pooled_subject_n:
        n_hard = pooled_subject_hard[subj]
        pooled[subj] = {
            "n_total": pooled_subject_n[subj],
            "n_hard": n_hard,
            "n_mutually_unrecoverable": pooled_subject_unrec[subj],
            "n_universally_recoverable": pooled_subject_univ[subj],
            "frac_unrec_of_hard": (
                pooled_subject_unrec[subj] / n_hard if n_hard else 0.0
            ),
            "frac_univ_of_hard": (
                pooled_subject_univ[subj] / n_hard if n_hard else 0.0
            ),
            "mean_recovery_count_of_6": (
                pooled_subject_recsum[subj] / n_hard if n_hard else float("nan")
            ),
        }
    out["pooled_subject"] = pooled

    # Overall summary stats
    all_unrec = sum(pooled_subject_unrec.values())
    all_hard = sum(pooled_subject_hard.values())
    out["summary"] = {
        "n_subjects": len(pooled),
        "n_hard_total": all_hard,
        "n_unrec_total": all_unrec,
        "frac_unrec_overall": all_unrec / all_hard if all_hard else 0.0,
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 90)
    print("§6qq. Subject decomposition of mutually-unrecoverable hard questions")
    print("=" * 90)

    for p in DOMAINS:
        per_subj = out["per_primary_subject"][p]
        # Sort subjects by frac_unrec_of_hard desc, but only those with n_hard > 0
        subjs_sorted = sorted(
            [(s, b) for s, b in per_subj.items() if b["n_hard"] > 0],
            key=lambda x: (-x[1]["frac_unrec_of_hard"], -x[1]["n_hard"]),
        )
        print()
        print(f"PRIMARY: {p}")
        print(f"  {'subject':<35s} {'n_tot':>5s} {'n_hard':>6s} {'unrec':>6s} "
              f"{'univ':>5s} {'unrec_frac':>11s} {'mean_rec/6':>11s}")
        for subj, b in subjs_sorted:
            print(f"  {subj:<35s} {b['n_total']:>5d} {b['n_hard']:>6d} "
                  f"{b['n_mutually_unrecoverable']:>6d} {b['n_universally_recoverable']:>5d} "
                  f"{b['frac_unrec_of_hard']*100:>10.1f}% "
                  f"{b['mean_recovery_count_of_6']:>10.2f}")

    # Top-10 subjects globally by unrec rate (n_hard >= 3)
    print()
    print("=" * 90)
    print("Top subjects globally by mutually-unrecoverable rate (n_hard >= 3):")
    print("=" * 90)
    pooled_filt = sorted(
        [(s, b) for s, b in pooled.items() if b["n_hard"] >= 3],
        key=lambda x: (-x[1]["frac_unrec_of_hard"], -x[1]["n_hard"]),
    )
    print(f"  {'subject':<35s} {'n_hard':>6s} {'unrec':>6s} {'unrec_frac':>11s} {'mean_rec/6':>11s}")
    for subj, b in pooled_filt[:15]:
        print(f"  {subj:<35s} {b['n_hard']:>6d} {b['n_mutually_unrecoverable']:>6d} "
              f"{b['frac_unrec_of_hard']*100:>10.1f}% {b['mean_recovery_count_of_6']:>10.2f}")

    # Bottom (most recoverable)
    print()
    print("Bottom subjects globally (most recoverable, n_hard >= 3):")
    for subj, b in pooled_filt[-10:]:
        print(f"  {subj:<35s} {b['n_hard']:>6d} {b['n_mutually_unrecoverable']:>6d} "
              f"{b['frac_unrec_of_hard']*100:>10.1f}% {b['mean_recovery_count_of_6']:>10.2f}")

    print()
    print(f"Overall: {all_unrec} of {all_hard} hard questions ({all_unrec/all_hard*100:.1f}%) "
          f"are mutually unrecoverable.")


if __name__ == "__main__":
    main()
