"""Audit follow-up #28: Multiple-comparison corrections on §6ii + §6jj pairwise.

§6ii had 10 pairwise comparisons on the primary axis; §6jj had 15
pairwise comparisons on the helper axis. Total = 25 pairwise tests.

Without correction, the §6ii+§6jj report:
  primary axis: 5/10 are 95% sig (alpha = 0.05)
  helper axis:  3/15 are 95% sig
  combined:     8/25 are 95% sig

This script applies two standard corrections:
  Bonferroni:        alpha_corrected = 0.05/25 = 0.002 per test
  Holm step-down:    sort p-values, reject in increasing order while
                     p[k] < alpha / (n - k + 1)

The bootstrap precision is 1/2000 = 0.0005 per tail, which means we
can reliably claim p < 0.001 (two-tailed) for any comparison where
P(diff > 0) is exactly 0% or 100%. Comparisons whose P is between
2.5% and 97.5% have p > 0.05 (uncorrected); those at 99.8% or above
have p < 0.004 two-tailed (potentially Bonferroni-significant).

Output: results/verified_pair_grid_qwen3_1p7b/multiple_comparison_corrections.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/net_corrector_bootstrap.json"
HELPER_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/net_corrector_bootstrap_helper.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/multiple_comparison_corrections.json"


def two_tailed_p(p_diff_gt_0: float, n_iter: int) -> float:
    """Convert bootstrap P(diff > 0) into a conservative two-tailed p.

    For an exact one-tailed bootstrap probability p, the two-tailed
    p-value (under H0: diff = 0) is 2 * min(p, 1 - p). We bound the
    bootstrap precision: any P that is exactly 0 or 1 is treated as
    1/(n_iter+1), since with n_iter samples we can only resolve
    fractions to that precision.
    """
    p = p_diff_gt_0
    p_clamped = max(min(p, 1 - 1.0 / (n_iter + 1)), 1.0 / (n_iter + 1))
    one_tailed = min(p_clamped, 1 - p_clamped)
    return 2 * one_tailed


def main() -> None:
    primary = json.loads(PRIMARY_PATH.read_text())
    helper = json.loads(HELPER_PATH.read_text())
    n_iter = primary["n_iter"]
    assert n_iter == helper["n_iter"], "iter count mismatch"

    # Collect all 25 pairwise tests
    all_tests = []
    for k, v in primary["pairwise_diffs"].items():
        p1, p2 = k.split("_vs_")
        pt_diff = (primary["point_estimates"][p1]["net_corrector_score"]
                   - primary["point_estimates"][p2]["net_corrector_score"]) * 100
        all_tests.append({
            "axis": "primary",
            "pair": k,
            "point_diff_pp": pt_diff,
            "p_diff_gt_0": v["p_diff_gt_0"],
            "two_tailed_p": two_tailed_p(v["p_diff_gt_0"], n_iter),
            "ci_low_pp": v["summary"]["p2.5"] * 100,
            "ci_hi_pp": v["summary"]["p97.5"] * 100,
        })
    for k, v in helper["pairwise_diffs"].items():
        h1, h2 = k.split("_vs_")
        pt_diff = (helper["point_estimates"][h1]["net"]
                   - helper["point_estimates"][h2]["net"]) * 100
        all_tests.append({
            "axis": "helper",
            "pair": k,
            "point_diff_pp": pt_diff,
            "p_diff_gt_0": v["p_diff_gt_0"],
            "two_tailed_p": two_tailed_p(v["p_diff_gt_0"], n_iter),
            "ci_low_pp": v["summary"]["p2.5"] * 100,
            "ci_hi_pp": v["summary"]["p97.5"] * 100,
        })

    n_total = len(all_tests)
    alpha = 0.05

    # Sort by two-tailed p (ascending)
    all_tests_sorted = sorted(all_tests, key=lambda t: t["two_tailed_p"])

    # Uncorrected: which pairs are p < 0.05?
    n_uncorrected = sum(1 for t in all_tests_sorted if t["two_tailed_p"] < alpha)

    # Bonferroni: alpha_corrected = alpha / n
    bonf_alpha = alpha / n_total
    n_bonferroni = sum(1 for t in all_tests_sorted if t["two_tailed_p"] < bonf_alpha)

    # Holm step-down: sort p ascending; reject p[k] if p[k] < alpha / (n - k)
    holm_survivors = []
    for k, t in enumerate(all_tests_sorted):
        threshold = alpha / (n_total - k)
        survives = t["two_tailed_p"] < threshold
        t["holm_threshold"] = threshold
        t["holm_survives"] = survives
        if survives:
            holm_survivors.append(t["pair"])
        else:
            # Once one fails, all subsequent fail too
            for t2 in all_tests_sorted[k:]:
                t2["holm_threshold"] = alpha / (n_total - all_tests_sorted.index(t2))
                t2["holm_survives"] = False
            break

    # BH-FDR (Benjamini-Hochberg)
    bh_survivors = []
    bh_alpha = alpha
    # walk in REVERSE order (largest p first); find largest k where p[k] <= k/n * alpha
    bh_largest_k = None
    for k_inv in range(n_total - 1, -1, -1):
        rank = k_inv + 1  # 1-indexed
        threshold = rank / n_total * bh_alpha
        if all_tests_sorted[k_inv]["two_tailed_p"] <= threshold:
            bh_largest_k = k_inv
            break
    if bh_largest_k is not None:
        for k in range(bh_largest_k + 1):
            all_tests_sorted[k]["bh_survives"] = True
            bh_survivors.append(all_tests_sorted[k]["pair"])
        for k in range(bh_largest_k + 1, n_total):
            all_tests_sorted[k]["bh_survives"] = False
    else:
        for t in all_tests_sorted:
            t["bh_survives"] = False

    out = {
        "n_total_tests": n_total,
        "n_iter_bootstrap": n_iter,
        "alpha": alpha,
        "bonferroni_alpha": bonf_alpha,
        "n_uncorrected_significant": n_uncorrected,
        "n_bonferroni_significant": n_bonferroni,
        "n_holm_significant": len(holm_survivors),
        "n_bh_significant": len(bh_survivors),
        "holm_survivors": holm_survivors,
        "bh_survivors": bh_survivors,
        "all_tests_sorted_by_p": all_tests_sorted,
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 78)
    print(f"§6kk. Multiple-comparison corrections on {n_total} pairwise tests")
    print("=" * 78)
    print(f"Bootstrap n_iter = {n_iter} (precision floor: p ~ 1/{n_iter+1} = {1/(n_iter+1):.5f})")
    print(f"Per-test alpha = {alpha}")
    print(f"Bonferroni alpha_corrected = {bonf_alpha:.4f}")
    print()
    print(f"Uncorrected (p < 0.05):       {n_uncorrected:>2d} of {n_total} significant")
    print(f"Bonferroni (p < {bonf_alpha:.4f}):    {n_bonferroni:>2d} of {n_total} significant")
    print(f"Holm step-down:               {len(holm_survivors):>2d} of {n_total} significant")
    print(f"BH-FDR (q < 0.05):            {len(bh_survivors):>2d} of {n_total} significant")
    print()
    print("All 25 tests, sorted by two-tailed p (ascending):")
    print(f"{'#':>3} {'axis':>9s} {'pair':>28s} {'pt_diff':>9s} {'p_diff>0':>9s} {'2tail_p':>10s} {'sig':>16s}")
    for i, t in enumerate(all_tests_sorted, start=1):
        sig_marks = []
        if t["two_tailed_p"] < alpha:
            sig_marks.append("u")
        if t["two_tailed_p"] < bonf_alpha:
            sig_marks.append("B")
        if t.get("holm_survives", False):
            sig_marks.append("H")
        if t.get("bh_survives", False):
            sig_marks.append("F")
        sig_str = " ".join(sig_marks) if sig_marks else "-"
        print(f"{i:>3} {t['axis']:>9s} {t['pair']:>28s} "
              f"{t['point_diff_pp']:>+8.1f}pp "
              f"{t['p_diff_gt_0']*100:>7.1f}% "
              f"{t['two_tailed_p']:>9.4f}  "
              f"{sig_str:>16s}")
    print()
    print("Legend: u = uncorrected p<0.05; B = Bonferroni p<0.002;")
    print("        H = Holm step-down survivor; F = BH-FDR survivor.")


if __name__ == "__main__":
    main()
