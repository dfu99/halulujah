"""Helper-aware orchestration predictor evaluation (audit follow-up #19).

The §6x oracle ceiling is 72.4% pooled vs 51.0% actual random-helper
mean = +21.4 pp gap.  Test simple predictors of "which helper to use
for this question" and report how much of that gap each closes.

Predictors:

  Random:    actual mean across 6 helpers (the §6x baseline).
  Best-base: always use the base helper.
  Best-self: always use the helper matching the primary domain.
  Best-by-col-mean: always use the per-primary best helper as ranked
                     by col-mean delta (a single fixed helper per
                     primary; the strongest cell-mean candidate).
  Best-by-self-solo: always use the helper with the highest own-domain
                     self-solo accuracy (the "smartest specialist").
  Subject-aware:    for each question's MMLU subject, pick the helper
                     whose own-domain best matches the subject (manual
                     mapping from subject to helper domain).
  Vote-of-6:        majority answer across the 6 helpers.
  Oracle:           best-of-6 (audit §6x; upper bound).
  Worst-of-6:       worst-of-6 (lower bound; "all helpers correct" cells).

Outputs:
  results/verified_pair_grid_qwen3_1p7b/helper_orchestration.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/helper_orchestration.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]


# Manual mapping from MMLU subject to "best matching helper" for the
# subject-aware predictor. Reasonable choices given our 5 specialists:
SUBJECT_TO_HELPER = {
    # math subjects
    "abstract_algebra": "math",
    "college_mathematics": "math",
    "elementary_mathematics": "math",
    "high_school_mathematics": "math",
    # medicine subjects
    "anatomy": "medicine",
    "clinical_knowledge": "medicine",
    "college_medicine": "medicine",
    "medical_genetics": "medicine",
    "professional_medicine": "medicine",
    "virology": "medicine",
    # biology subjects
    "college_biology": "biology",
    "high_school_biology": "biology",
    # law subjects
    "international_law": "law",
    "jurisprudence": "law",
    "professional_law": "law",
    # physics subjects
    "astronomy": "physics",
    "college_physics": "physics",
    "conceptual_physics": "physics",
    "high_school_physics": "physics",
}


def load_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text())


def build_grid(matrix: dict):
    """Returns:
      correctness[i, j, k]  shape (5 primary, 6 helper, 50 q): bool
      letters[i, j, k]      shape (5, 6, 50): string letter
      subjects[i, k]        shape (5, 50): string
      idx_map[(i, k)]       shape (5, 50): int idx
    """
    cond = matrix["conditions"]
    R = 50
    C = np.zeros((5, 6, R), dtype=np.int8)
    L = np.full((5, 6, R), "", dtype=object)
    S = np.full((5, R), "", dtype=object)
    for i, primary in enumerate(DOMAINS):
        # Use the solo per_q to lock subjects
        for q in cond[f"solo_{primary}"]["per_q"]:
            S[i, q["idx"]] = q["subject"]
        for j, helper in enumerate(HELPERS):
            for q in cond[f"pair_{primary}_{helper}"]["per_q"]:
                C[i, j, q["idx"]] = int(q["correct"])
                L[i, j, q["idx"]] = q["predicted"]
    return C, L, S


def main() -> None:
    matrix = load_matrix()
    cond = matrix["conditions"]

    C, L, S = build_grid(matrix)
    a, b, R = C.shape

    # Self-solo per helper
    self_solo = {h: cond[f"solo_{h}"]["accuracy"] for h in HELPERS if h != "base"}
    self_solo["base"] = float(np.mean([
        cond[f"base_solo_{p}"]["accuracy"] for p in DOMAINS
    ]))

    # Helper col-mean delta per primary (which helper is best per primary on average)
    primary_solo = {
        p: cond[f"solo_{p}"]["accuracy"] for p in DOMAINS
    }
    primary_helper_delta = {}
    for i, p in enumerate(DOMAINS):
        for j, h in enumerate(HELPERS):
            primary_helper_delta[(p, h)] = (
                cond[f"pair_{p}_{h}"]["accuracy"] - primary_solo[p]
            )

    # Best helper per primary by col-mean delta (within that primary's row)
    best_helper_per_primary = {
        p: HELPERS[int(np.argmax([primary_helper_delta[(p, h)] for h in HELPERS]))]
        for p in DOMAINS
    }
    print(f"Best-helper-per-primary (by row-mean delta):")
    for p, h in best_helper_per_primary.items():
        print(f"  {p:10s} → {h}  (delta={primary_helper_delta[(p, h)]:+.3f})")

    # Single helper with highest own-domain self-solo
    best_self_solo_helper = max(self_solo.keys(), key=lambda h: self_solo[h])
    print(f"\nBest-self-solo helper (single, fixed across primaries): "
          f"{best_self_solo_helper} (self-solo {self_solo[best_self_solo_helper]:.3f})")

    # Run predictors per (primary, idx)
    results: dict = {}

    def evaluate_predictor(name: str, helper_chooser):
        """helper_chooser(primary_idx, q_idx, subject) -> helper_idx in 0..5"""
        n_correct = 0
        n_total = 0
        per_primary = {p: {"n_correct": 0, "n_total": 0} for p in DOMAINS}
        for i in range(a):
            for k in range(R):
                primary = DOMAINS[i]
                subject = S[i, k]
                j = helper_chooser(i, k, subject)
                c = int(C[i, j, k])
                n_correct += c
                n_total += 1
                per_primary[primary]["n_correct"] += c
                per_primary[primary]["n_total"] += 1
        results[name] = {
            "pooled_accuracy": n_correct / n_total if n_total else 0.0,
            "n_correct": n_correct,
            "n_total": n_total,
            "per_primary": {
                p: d["n_correct"] / d["n_total"] if d["n_total"] else 0.0
                for p, d in per_primary.items()
            },
        }

    # Random (actual mean across helpers)
    def random_chooser(i, k, subject):
        # Average accuracy across 6 helpers for this (primary, idx)
        # We don't sample; we just return the helper-averaged correctness
        return -1  # sentinel
    n_correct = 0
    for i in range(a):
        for k in range(R):
            n_correct += int(C[i, :, k].sum())
    results["random_actual_mean"] = {
        "pooled_accuracy": n_correct / (a * b * R),
        "n_correct": n_correct,
        "n_total": a * b * R,
        "per_primary": {
            DOMAINS[i]: float(C[i, :, :].mean()) for i in range(a)
        },
    }

    # Always use base
    evaluate_predictor("always_base", lambda i, k, s: HELPERS.index("base"))

    # Always use primary-as-helper (same-domain)
    evaluate_predictor(
        "always_self_match",
        lambda i, k, s: HELPERS.index(DOMAINS[i]),
    )

    # Always use the per-primary best-by-col-mean helper
    evaluate_predictor(
        "best_by_col_mean",
        lambda i, k, s: HELPERS.index(best_helper_per_primary[DOMAINS[i]]),
    )

    # Always use the helper with highest self-solo
    evaluate_predictor(
        "best_self_solo",
        lambda i, k, s: HELPERS.index(best_self_solo_helper),
    )

    # Subject-aware (helper matches subject's mapped helper)
    def subject_aware_chooser(i, k, s):
        h = SUBJECT_TO_HELPER.get(s, "base")
        return HELPERS.index(h)
    evaluate_predictor("subject_aware", subject_aware_chooser)

    # Majority-vote across 6 helpers (per question, take the modal letter)
    n_correct_vote = 0
    n_total_vote = 0
    per_primary_vote = {p: {"n_correct": 0, "n_total": 0} for p in DOMAINS}
    for i in range(a):
        for k in range(R):
            primary = DOMAINS[i]
            letters = [L[i, j, k] for j in range(b)]
            # Modal letter
            counter = Counter(letters)
            modal_letter, _ = counter.most_common(1)[0]
            # Find the expected
            expected = None
            for q in cond[f"solo_{primary}"]["per_q"]:
                if q["idx"] == k:
                    expected = q["expected"]
                    break
            assert expected is not None
            c = int(modal_letter == expected)
            n_correct_vote += c
            n_total_vote += 1
            per_primary_vote[primary]["n_correct"] += c
            per_primary_vote[primary]["n_total"] += 1
    results["majority_vote_of_6"] = {
        "pooled_accuracy": n_correct_vote / n_total_vote if n_total_vote else 0.0,
        "n_correct": n_correct_vote,
        "n_total": n_total_vote,
        "per_primary": {
            p: d["n_correct"] / d["n_total"] if d["n_total"] else 0.0
            for p, d in per_primary_vote.items()
        },
    }

    # Oracle (best-of-6 — at least one helper correct = correct)
    n_oracle = 0
    n_worst = 0
    for i in range(a):
        for k in range(R):
            corrects = C[i, :, k]
            n_oracle += int(corrects.sum() > 0)
            n_worst += int(corrects.sum() == b)  # all helpers correct
    results["oracle_best_of_6"] = {
        "pooled_accuracy": n_oracle / (a * R),
        "n_correct": n_oracle,
        "n_total": a * R,
        "per_primary": {
            DOMAINS[i]: sum(1 for k in range(R) if C[i, :, k].sum() > 0) / R
            for i in range(a)
        },
    }
    results["worst_only_unanimous"] = {
        "pooled_accuracy": n_worst / (a * R),
        "n_correct": n_worst,
        "n_total": a * R,
        "per_primary": {
            DOMAINS[i]: sum(1 for k in range(R) if C[i, :, k].sum() == b) / R
            for i in range(a)
        },
    }

    # Compute gap closure
    actual = results["random_actual_mean"]["pooled_accuracy"]
    oracle = results["oracle_best_of_6"]["pooled_accuracy"]
    gap = oracle - actual
    for name, r in results.items():
        if name in ("random_actual_mean", "oracle_best_of_6", "worst_only_unanimous"):
            r["gap_closed"] = None
            continue
        lift = r["pooled_accuracy"] - actual
        r["lift_over_random"] = lift
        r["gap_closed_pp"] = (lift / gap * 100) if gap > 0 else 0.0

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print()
    print("=" * 80)
    print("Helper-aware orchestration predictor evaluation (audit §6x extension)")
    print("=" * 80)
    print(f"§6x oracle gap: actual {actual:.1%} → oracle {oracle:.1%} = {gap*100:+.1f} pp gap")
    print()
    print(f"{'predictor':25s} {'pooled':>8s} {'lift':>7s} {'gap%':>6s}")
    print("-" * 60)
    order = [
        "worst_only_unanimous",
        "random_actual_mean",
        "always_base",
        "always_self_match",
        "best_self_solo",
        "best_by_col_mean",
        "subject_aware",
        "majority_vote_of_6",
        "oracle_best_of_6",
    ]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        acc = r["pooled_accuracy"]
        lift_str = (
            f"{r.get('lift_over_random', 0)*100:+.1f}"
            if "lift_over_random" in r else "—"
        )
        gap_str = (
            f"{r.get('gap_closed_pp', 0):.0f}%"
            if "gap_closed_pp" in r and r["gap_closed_pp"] is not None else "—"
        )
        print(f"{name:25s} {acc:>8.1%} {lift_str:>7s} {gap_str:>6s}")


if __name__ == "__main__":
    main()
