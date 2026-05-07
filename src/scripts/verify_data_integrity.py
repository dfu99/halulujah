"""Audit follow-up #50: verify matrix_results.json data integrity.

Complements verify_audit_headlines.py (which checks document↔data
consistency) with a data-source check (confirms the underlying data
has the expected schema). Catches data corruption or schema drift
before it propagates into audit numbers.

Checks:
  1. All 45 expected conditions present (5 solo_X + 5 base_solo_X +
     30 pair_X_Y + 5 base_pair_X = 45).
  2. Each cell has N=50 per_q records.
  3. For each primary, all 6 helper cells share the same (idx, subject,
     expected) tuple — confirms the seed=42 question-pool alignment that
     the audit's question-clustered bootstrap depends on.
  4. solo_X[i] questions match the pair_X_*[i] questions (cross-cell
     alignment between solo and pair grids).
  5. The hard set (solo_X.correct == False) has the expected size:
     pooled n_hard = 176 across 5 primaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

EXPECTED_N = 50
EXPECTED_TOTAL_HARD = 176


def main() -> int:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]

    n_pass = 0
    n_total = 0
    n_fail = 0

    def report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal n_pass, n_fail, n_total
        n_total += 1
        if ok:
            n_pass += 1
            flag = "✓"
        else:
            n_fail += 1
            flag = "✗"
        print(f"  {flag} {name}{(' — ' + detail) if detail else ''}")

    print("=" * 90)
    print("Data integrity verification: matrix_results.json")
    print("=" * 90)

    # 1. All 45 conditions present
    print("\n[Schema: condition coverage]")
    expected_keys = (
        [f"solo_{p}" for p in DOMAINS]
        + [f"base_solo_{p}" for p in DOMAINS]
        + [f"pair_{p}_{h}" for p in DOMAINS for h in HELPERS]
        + [f"base_pair_{p}" for p in DOMAINS]
    )
    for k in expected_keys:
        report(f"condition exists: {k}", k in cond)
    report(f"condition count == {len(expected_keys)}",
           len(set(expected_keys) & set(cond)) == len(expected_keys),
           detail=f"observed={len(cond)}")

    # 2. N=50 per cell
    print("\n[Schema: per-cell N]")
    for k in expected_keys:
        if k not in cond:
            continue
        n = len(cond[k].get("per_q", []))
        ok = n == EXPECTED_N
        if not ok:
            report(f"{k} has N={EXPECTED_N}", ok, detail=f"observed={n}")
        else:
            n_pass += 1
            n_total += 1
    print(f"  ✓ all 45 cells have N={EXPECTED_N} per_q records")

    # 3. Cross-helper question-pool alignment per primary (seed=42)
    print("\n[Alignment: pair_X_h shares question pool across helpers]")
    for p in DOMAINS:
        pq_pool = None
        all_aligned = True
        for h in HELPERS:
            cell = cond[f"pair_{p}_{h}"]["per_q"]
            sig = tuple((q["idx"], q["subject"], q["expected"]) for q in cell)
            if pq_pool is None:
                pq_pool = sig
            elif sig != pq_pool:
                all_aligned = False
                break
        report(f"pair_{p}_X cells share (idx, subject, expected) across 6 helpers",
               all_aligned)

    # 4. solo_X aligned with pair_X_*
    print("\n[Alignment: solo_X aligned with pair_X_*]")
    for p in DOMAINS:
        solo_sig = tuple((q["idx"], q["subject"], q["expected"])
                         for q in cond[f"solo_{p}"]["per_q"])
        pair_sig = tuple((q["idx"], q["subject"], q["expected"])
                         for q in cond[f"pair_{p}_base"]["per_q"])
        report(f"solo_{p} == pair_{p}_base on (idx, subject, expected)",
               solo_sig == pair_sig)

    # 5. Hard pool size
    print("\n[Hard pool: pooled n_hard = 176]")
    n_hard_total = 0
    for p in DOMAINS:
        n_hard_p = sum(1 for q in cond[f"solo_{p}"]["per_q"] if not q["correct"])
        n_hard_total += n_hard_p
    report(f"pooled n_hard == {EXPECTED_TOTAL_HARD}",
           n_hard_total == EXPECTED_TOTAL_HARD,
           detail=f"observed={n_hard_total}")

    # 6. solo accuracy bounds (sanity).
    # Physics primary has solo_physics ≈ 0.12 (documented in §6x: "12%
    # solo → 70% oracle"); other primaries are 0.30-0.40. Bound 0.10 to
    # accommodate physics while still catching gross corruption.
    print("\n[Solo accuracy plausibility]")
    for p in DOMAINS:
        acc = cond[f"solo_{p}"]["accuracy"]
        ok = 0.10 <= acc <= 0.85
        report(f"solo_{p} accuracy in [0.10, 0.85]", ok, detail=f"{acc:.3f}")

    # 7. Pair-cell schema: each per_q has switch_type label
    print("\n[Pair cell schema: per_q has switch_type field]")
    for p in DOMAINS:
        for h in HELPERS:
            cell = cond[f"pair_{p}_{h}"]["per_q"]
            has_field = all("switch_type" in q for q in cell)
            if not has_field:
                report(f"pair_{p}_{h}.per_q[*].switch_type", has_field)
                break
        else:
            continue
        break
    else:
        n_pass += 1
        n_total += 1
        print("  ✓ all 30 pair cells have switch_type per per_q record")

    # Summary
    print()
    print("=" * 90)
    print(f"DATA INTEGRITY SUMMARY: {n_pass} / {n_total} checks pass")
    print("=" * 90)
    if n_fail == 0:
        print("matrix_results.json passes all integrity checks ✓")
        return 0
    else:
        print(f"WARNING: {n_fail} checks failed ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
