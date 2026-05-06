"""Audit follow-up #32: Per-cell bootstrap CI on net corrector score (§6oo).

§6bb classified each of 30 cells as corrector (24/30) / distractor (5/30) /
neutral (1/30) based on point counts. §6ii bootstrapped per *primary*
(averaging across 6 helpers). §6jj bootstrapped per *helper*. This script
gives the most granular bootstrap: per-cell (5 primaries × 6 helpers).

For each (primary, helper) cell, question-cluster bootstrap (n=2000):
  W2C rate (cell-level) = #{post correct & solo wrong} / #{solo wrong}
  C2W rate (cell-level) = #{post wrong & solo correct} / #{solo correct}
  net = W2C - C2W

Per cell report: 95% CI on net corrector score, and indicator
sig_pos = (lower CI > 0); sig_neg = (upper CI < 0).

Hypothesis: most cells will have CIs that cross 0 (small subset
sizes — math has 32 hard / 18 easy, etc., bootstrap has high variance
per cell). The cleanest cells will be biology rows (high net score
+~50pp) and law rows (low net score ~-10pp).

Output: results/verified_pair_grid_qwen3_1p7b/per_cell_net_corrector_ci.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/per_cell_net_corrector_ci.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

N_ITER = 2000
SEED = 2026


def cell_net_score(
    pre_corr: np.ndarray, post_corr: np.ndarray, sample_idx: np.ndarray,
) -> tuple[float, float, float]:
    """C2W = #{post wrong & solo correct} / #{solo correct};
    W2C = #{post correct & solo wrong} / #{solo wrong};
    net = W2C - C2W."""
    sc = pre_corr[sample_idx]
    pc = post_corr[sample_idx]
    n_easy = int((sc == 1).sum())
    n_hard = int((sc == 0).sum())
    if n_easy == 0 or n_hard == 0:
        return float("nan"), float("nan"), float("nan")
    c2w = ((sc == 1) & (pc == 0)).sum() / n_easy
    w2c = ((sc == 0) & (pc == 1)).sum() / n_hard
    return float(c2w), float(w2c), float(w2c - c2w)


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]
    rng = np.random.default_rng(SEED)

    solo_correct = {}
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

    cells = []
    sig_pos = 0
    sig_neg = 0
    sig_zero = 0
    for p in DOMAINS:
        n = len(solo_correct[p])
        for h in HELPERS:
            # Point estimate
            full_idx = np.arange(n)
            c2w_pt, w2c_pt, net_pt = cell_net_score(
                solo_correct[p], post_correct[p][h], full_idx
            )
            # Bootstrap
            net_dist = []
            c2w_dist = []
            w2c_dist = []
            for _ in range(N_ITER):
                idx = rng.integers(0, n, size=n)
                c2w, w2c, net = cell_net_score(solo_correct[p], post_correct[p][h], idx)
                net_dist.append(net)
                c2w_dist.append(c2w)
                w2c_dist.append(w2c)
            net_arr = np.array([x for x in net_dist if np.isfinite(x)])
            c2w_arr = np.array([x for x in c2w_dist if np.isfinite(x)])
            w2c_arr = np.array([x for x in w2c_dist if np.isfinite(x)])
            net_p2_5 = float(np.percentile(net_arr, 2.5))
            net_p97_5 = float(np.percentile(net_arr, 97.5))
            net_p50 = float(np.percentile(net_arr, 50))
            p_gt_0 = float((net_arr > 0).mean())
            sig_class = "pos" if net_p2_5 > 0 else ("neg" if net_p97_5 < 0 else "zero")
            if sig_class == "pos":
                sig_pos += 1
            elif sig_class == "neg":
                sig_neg += 1
            else:
                sig_zero += 1
            cells.append({
                "primary": p,
                "helper": h,
                "n_easy_pt": int((solo_correct[p] == 1).sum()),
                "n_hard_pt": int((solo_correct[p] == 0).sum()),
                "c2w_rate_pt": c2w_pt,
                "w2c_rate_pt": w2c_pt,
                "net_pt": net_pt,
                "net_p2.5": net_p2_5,
                "net_p50": net_p50,
                "net_p97.5": net_p97_5,
                "p_net_gt_0": p_gt_0,
                "sig_class": sig_class,
                "c2w_rate_p2.5": float(np.percentile(c2w_arr, 2.5)),
                "c2w_rate_p97.5": float(np.percentile(c2w_arr, 97.5)),
                "w2c_rate_p2.5": float(np.percentile(w2c_arr, 2.5)),
                "w2c_rate_p97.5": float(np.percentile(w2c_arr, 97.5)),
            })

    n_total = len(cells)
    out = {
        "n_iter": N_ITER,
        "seed": SEED,
        "n_cells_total": n_total,
        "n_cells_sig_positive": sig_pos,
        "n_cells_sig_negative": sig_neg,
        "n_cells_sig_zero": sig_zero,
        "cells": cells,
    }

    # Summary by primary and helper
    by_primary = {}
    for p in DOMAINS:
        cells_p = [c for c in cells if c["primary"] == p]
        by_primary[p] = {
            "n_pos": sum(1 for c in cells_p if c["sig_class"] == "pos"),
            "n_neg": sum(1 for c in cells_p if c["sig_class"] == "neg"),
            "n_zero": sum(1 for c in cells_p if c["sig_class"] == "zero"),
            "mean_net_pp": float(np.mean([c["net_pt"] for c in cells_p]) * 100),
        }
    by_helper = {}
    for h in HELPERS:
        cells_h = [c for c in cells if c["helper"] == h]
        by_helper[h] = {
            "n_pos": sum(1 for c in cells_h if c["sig_class"] == "pos"),
            "n_neg": sum(1 for c in cells_h if c["sig_class"] == "neg"),
            "n_zero": sum(1 for c in cells_h if c["sig_class"] == "zero"),
            "mean_net_pp": float(np.mean([c["net_pt"] for c in cells_h]) * 100),
        }
    out["by_primary"] = by_primary
    out["by_helper"] = by_helper

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 78)
    print(f"§6oo. Per-cell bootstrap CI on net corrector score (n_iter = {N_ITER})")
    print("=" * 78)
    print(f"Total cells: {n_total}")
    print(f"  CI excludes 0 (positive — robust corrector): {sig_pos} ({sig_pos/n_total*100:.0f}%)")
    print(f"  CI excludes 0 (negative — robust distractor): {sig_neg} ({sig_neg/n_total*100:.0f}%)")
    print(f"  CI crosses 0 (uncertain): {sig_zero} ({sig_zero/n_total*100:.0f}%)")
    print()
    print("By primary:")
    print(f"{'primary':10s}  {'pos':>5s} {'neg':>5s} {'zero':>5s}  {'mean net (pt)':>15s}")
    for p in DOMAINS:
        b = by_primary[p]
        print(f"{p:10s}  {b['n_pos']:>5d} {b['n_neg']:>5d} {b['n_zero']:>5d}  {b['mean_net_pp']:>+13.1f} pp")
    print()
    print("By helper:")
    print(f"{'helper':10s}  {'pos':>5s} {'neg':>5s} {'zero':>5s}  {'mean net (pt)':>15s}")
    for h in HELPERS:
        b = by_helper[h]
        print(f"{h:10s}  {b['n_pos']:>5d} {b['n_neg']:>5d} {b['n_zero']:>5d}  {b['mean_net_pp']:>+13.1f} pp")
    print()
    print("All 30 cells, sorted by point net (descending):")
    cells_sorted = sorted(cells, key=lambda c: c["net_pt"], reverse=True)
    print(f"{'#':>3} {'primary':>10s} {'helper':>10s}  {'net (pt)':>10s} {'95% CI':>22s} {'sig':>6s}")
    for i, c in enumerate(cells_sorted, 1):
        ci = f"[{c['net_p2.5']*100:+5.1f}, {c['net_p97.5']*100:+5.1f}] pp"
        sig = c["sig_class"]
        marker = " ★+" if sig == "pos" else (" ★-" if sig == "neg" else "    ")
        print(f"{i:>3} {c['primary']:>10s} {c['helper']:>10s}  {c['net_pt']*100:>+8.1f}pp "
              f"{ci:>22s} {marker:>6s}")


if __name__ == "__main__":
    main()
