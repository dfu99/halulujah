"""Audit follow-up #49: verify §10 11th-revision headline numbers.

Loads each JSON referenced by the audit's §10 11th-revision and checks
that the claimed headline numbers are internally consistent. Outputs
a pass/fail report.

Headlines to check:
  - ANOVA F-stat triple (full / hard / easy):  26.55 / 31.22 / 3.13
  - F_helper triple:                            0.96 / 0.50 / 1.91
  - Variance ratio (primary, full / hard / easy): 22.1 / 50.3 / 1.31
  - Variance ratio (subject, full / hard / easy): 21.7 / 56.5 / 1.64
  - Bonferroni-136 (subject, n=50k bootstrap):  11
  - Bonferroni-136 (subject, z-test):           28
  - Bonferroni-25  (primary, n=50k):             2
  - Bonferroni-15  (helper, z-test):             0
  - Per-primary mean W2C range:  math 21% / biology 64%
  - Per-helper mean W2C range:   ~35-41%
  - 47.7% mutually unrecoverable (hard)
  - hs_biology W2C CI [50.8, 81.8]
  - college_math  W2C CI [0.0, 12.5]

Output: print a pass/fail line for each headline.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/verified_pair_grid_qwen3_1p7b"


def check(name: str, claimed: float, observed: float, tol: float = 0.05) -> bool:
    """Check that observed matches claimed within relative tolerance tol."""
    if claimed == 0:
        ok = abs(observed) < tol
    else:
        rel_err = abs(observed - claimed) / abs(claimed)
        ok = rel_err < tol
    flag = "✓" if ok else "✗"
    print(f"  {flag} {name:50s} claimed={claimed:>8.3f}  observed={observed:>8.3f}")
    return ok


def check_int(name: str, claimed: int, observed: int) -> bool:
    ok = claimed == observed
    flag = "✓" if ok else "✗"
    print(f"  {flag} {name:50s} claimed={claimed:>4d}     observed={observed:>4d}")
    return ok


def main() -> None:
    results = []

    print("=" * 90)
    print("§10 11th-revision audit headline verification")
    print("=" * 90)

    # ANOVA F-stat triple
    print("\n[ANOVA F-stat triple]")
    a_full = json.loads((RESULTS / "anova_replicates.json").read_text())
    a_hard = json.loads((RESULTS / "anova_replicates_hard_only.json").read_text())
    a_easy = json.loads((RESULTS / "anova_replicates_easy_only.json").read_text())
    results.append(check("F_primary full",  26.55, a_full["F"]["primary_A"]))
    results.append(check("F_primary hard",  31.22, a_hard["F"]["primary_A"]))
    results.append(check("F_primary easy",   3.128, a_easy["F"]["primary_A"]))
    results.append(check("F_helper full",    0.961, a_full["F"]["helper_B"]))
    results.append(check("F_helper hard",    0.50,  a_hard["F"]["helper_B"]))
    results.append(check("F_helper easy",    1.910, a_easy["F"]["helper_B"]))

    # Variance ratio family
    print("\n[Variance ratio family]")
    swr = json.loads((RESULTS / "subject_who_ratio.json").read_text())
    swrh = json.loads((RESULTS / "subject_who_ratio_hard.json").read_text())
    swre = json.loads((RESULTS / "subject_who_ratio_easy.json").read_text())
    results.append(check("primary/helper full (§6m)",  22.11,
                         swr["headline"]["primary_helper_ratio_original"]))
    results.append(check("subject/helper full (§6rr)", 21.73,
                         swr["headline"]["subject_helper_ratio_filtered_weighted"]))
    results.append(check("primary/helper hard (§6cc)", 50.34,
                         swrh["headline"]["primary_helper_ratio_hard"]))
    results.append(check("subject/helper hard (§6ss)", 56.54,
                         swrh["headline"]["subject_helper_ratio_filtered_weighted"]))
    results.append(check("primary/helper easy",         1.31,
                         swre["headline"]["primary_helper_ratio_easy"]))
    results.append(check("subject/helper easy (§6uu)",  1.64,
                         swre["headline"]["subject_helper_ratio_filtered_weighted"]))

    # Bonferroni hierarchy at n=50k bootstrap
    print("\n[Bonferroni hierarchy: n=50k bootstrap]")
    swhin = json.loads((RESULTS / "subject_w2c_hard_bootstrap_hin.json").read_text())
    ncbh = json.loads((RESULTS / "net_corrector_bootstrap_hin.json").read_text())
    results.append(check_int("subject Bonferroni-136 (§6ww)", 11,
                             swhin["n_bonferroni_pass"]))
    n_prim_bonf = sum(1 for ps in ncbh["pair_stats"]
                      if ps["axis"] == "primary" and ps["bonferroni_pass"])
    n_help_bonf = sum(1 for ps in ncbh["pair_stats"]
                      if ps["axis"] == "helper" and ps["bonferroni_pass"])
    results.append(check_int("primary Bonferroni-25 (§6yy)",   2, n_prim_bonf))
    results.append(check_int("helper Bonferroni-25 (§6yy)",    0, n_help_bonf))

    # Bonferroni hierarchy: z-test
    print("\n[Bonferroni hierarchy: z-test]")
    swpz = json.loads((RESULTS / "subject_w2c_pairwise_z.json").read_text())
    hwpz = json.loads((RESULTS / "helper_w2c_pairwise_z.json").read_text())
    results.append(check_int("subject Bonferroni-136 z-test (§6ccc)", 28,
                             swpz["n_bonferroni_pass"]))
    results.append(check_int("helper  Bonferroni-15  z-test (§6ddd)",  0,
                             hwpz["n_bonferroni_pass"]))
    results.append(check_int("helper  uncorrected α=0.05 (§6ddd)",     0,
                             hwpz["n_uncorrected_pass"]))

    # Per-primary W2C range
    print("\n[Per-primary mean W2C on hard]")
    cond = json.loads((RESULTS / "matrix_results.json").read_text())["conditions"]
    DOMAINS = ["math", "medicine", "biology", "law", "physics"]
    HELPERS = ["base"] + DOMAINS
    primary_w2c = {}
    for p in DOMAINS:
        solo_pq = cond[f"solo_{p}"]["per_q"]
        hard = [q["idx"] for q in solo_pq if not q["correct"]]
        if not hard:
            continue
        rates = []
        for h in HELPERS:
            pp = cond[f"pair_{p}_{h}"]["per_q"]
            rates.append(sum(1 for i in hard if pp[i]["correct"]) / len(hard))
        primary_w2c[p] = sum(rates) / len(rates)
    results.append(check("math primary mean W2C",     0.208, primary_w2c["math"], tol=0.10))
    results.append(check("biology primary mean W2C",  0.640, primary_w2c["biology"], tol=0.05))
    results.append(check("law primary mean W2C",      0.211, primary_w2c["law"], tol=0.10))

    # 47.7% nearly unrecoverable
    print("\n[Mutually-unrecoverable rate]")
    ha = json.loads((RESULTS / "helper_agreement_hard.json").read_text())
    nearly_unrec = (ha["pooled"]["bin_counts"][str(0)] +
                    ha["pooled"]["bin_counts"][str(1)])
    pooled_n = ha["pooled"]["n_hard_total"]
    rate = nearly_unrec / pooled_n
    results.append(check("47.7% nearly unrecoverable (≤1/6)", 0.477, rate, tol=0.02))

    # Per-subject hard W2C CI: hs_biology and college_math
    print("\n[Per-subject hard W2C 95% bootstrap CI]")
    sw = json.loads((RESULTS / "subject_w2c_hard_bootstrap.json").read_text())
    hsb = sw["subject_summary"]["high_school_biology"]
    cm = sw["subject_summary"]["college_mathematics"]
    results.append(check("hs_biology CI lo",  0.508, hsb["ci_lo"], tol=0.05))
    results.append(check("hs_biology CI hi",  0.818, hsb["ci_hi"], tol=0.05))
    results.append(check("college_math CI lo", 0.000, cm["ci_lo"], tol=0.05))
    results.append(check("college_math CI hi", 0.125, cm["ci_hi"], tol=0.10))

    # 8-pair ironclad intersection (bootstrap ∩ z-test Bonferroni-136)
    # Catches the kind of enumeration drift fixed in commit c34fcec.
    print("\n[8-pair ironclad intersection (§6ccc bootstrap ∩ z-test)]")
    boot = json.loads((RESULTS / "subject_w2c_hard_bootstrap_hin.json").read_text())
    zt = json.loads((RESULTS / "subject_w2c_pairwise_z.json").read_text())
    boot_set: set[frozenset[str]] = set()
    for ps in boot["pair_stats"]:
        if ps["bonferroni_pass"]:
            boot_set.add(frozenset([ps["s1"], ps["s2"]]))
    zt_set: set[frozenset[str]] = set()
    for ps in zt["pair_stats"]:
        if ps["bonferroni_pass"]:
            zt_set.add(frozenset([ps["s1"], ps["s2"]]))
    intersection = boot_set & zt_set
    results.append(check_int("intersection size", 8, len(intersection)))
    results.append(check_int("bootstrap-only size", 3, len(boot_set - zt_set)))
    results.append(check_int("z-test-only size", 20, len(zt_set - boot_set)))

    # Audit document claims these 8 specific pairs in §6ccc enumeration
    expected_intersection = {
        frozenset(["high_school_biology", "professional_law"]),
        frozenset(["high_school_mathematics", "high_school_biology"]),
        frozenset(["college_mathematics", "high_school_biology"]),
        frozenset(["high_school_mathematics", "college_medicine"]),
        frozenset(["college_mathematics", "college_medicine"]),
        frozenset(["college_biology", "college_mathematics"]),
        frozenset(["college_mathematics", "high_school_physics"]),
        frozenset(["college_mathematics", "professional_medicine"]),
    }
    set_match = (intersection == expected_intersection)
    flag = "✓" if set_match else "✗"
    print(f"  {flag} {'enumeration matches audit §6ccc list':50s} "
          f"{'OK' if set_match else 'DRIFT DETECTED'}")
    results.append(set_match)

    if not set_match:
        # Show the diff
        only_actual = intersection - expected_intersection
        only_expected = expected_intersection - intersection
        if only_actual:
            print(f"    in data but not in audit list: {[sorted(p) for p in only_actual]}")
        if only_expected:
            print(f"    in audit list but not in data: {[sorted(p) for p in only_expected]}")

    # Final summary
    print()
    print("=" * 90)
    n_pass = sum(results)
    n_total = len(results)
    print(f"VERIFICATION SUMMARY: {n_pass} / {n_total} headline numbers consistent")
    print("=" * 90)
    if n_pass == n_total:
        print("All §10 11th-revision headlines pass internal consistency check ✓")
    else:
        print(f"WARNING: {n_total - n_pass} headlines fail consistency check ✗")


if __name__ == "__main__":
    main()
