"""Audit deepening §6bb + §6cc.

§6bb: For each (primary, helper) cell, the existing matrix_results.json
records c2w, w2c, switches, c2w_w2c_ratio counts. Classify the helper's
role at this cell as:
  CORRECTOR  if w2c > c2w (helper drove primary toward correct answers)
  DISTRACTOR if c2w > w2c (helper drove primary toward wrong answers)
  NEUTRAL    if c2w == w2c
Per-primary aggregation: count corrector / distractor / neutral cells
per primary. The §6aa best-by-col-mean rule should preferentially pair
each primary with its corrector helpers.

§6cc: Split each primary's 50 questions into "easy" (pre_a_correct=True
in solo) and "hard" (pre_a_correct=False in solo). Recompute the cell-
mean pair_acc on each subset. Then variance-decompose the resulting
hard-only and easy-only grids. Hypothesis: WHO-asymmetry is concentrated
in hard questions because easy questions saturate at high accuracy
across all helpers.

Outputs:
  results/verified_pair_grid_qwen3_1p7b/helper_role_per_cell.json    (§6bb)
  results/verified_pair_grid_qwen3_1p7b/difficulty_stratified_who.json (§6cc)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT_BB = ROOT / "results/verified_pair_grid_qwen3_1p7b/helper_role_per_cell.json"
OUT_CC = ROOT / "results/verified_pair_grid_qwen3_1p7b/difficulty_stratified_who.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]


def load_matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text())


# ───── §6bb: helper-as-corrector vs distractor ──────────────────────────


def classify_helper_role(matrix: dict) -> dict:
    cond = matrix["conditions"]
    cells = []
    role_counts = {p: {"corrector": 0, "distractor": 0, "neutral": 0} for p in DOMAINS}
    role_counts_by_helper = {h: {"corrector": 0, "distractor": 0, "neutral": 0} for h in HELPERS}
    pooled = {"corrector": 0, "distractor": 0, "neutral": 0}

    for primary in DOMAINS:
        for helper in HELPERS:
            cell = cond[f"pair_{primary}_{helper}"]
            c2w = cell["c2w"]
            w2c = cell["w2c"]
            if w2c > c2w:
                role = "corrector"
            elif c2w > w2c:
                role = "distractor"
            else:
                role = "neutral"
            net = w2c - c2w
            cells.append({
                "primary": primary,
                "helper": helper,
                "c2w": c2w,
                "w2c": w2c,
                "net_w2c": net,
                "role": role,
            })
            role_counts[primary][role] += 1
            role_counts_by_helper[helper][role] += 1
            pooled[role] += 1

    return {
        "n_cells": len(cells),
        "pooled_role_counts": pooled,
        "role_counts_per_primary": role_counts,
        "role_counts_per_helper": role_counts_by_helper,
        "cells": cells,
    }


# ───── §6cc: difficulty-stratified WHO ratio ────────────────────────────


def variance_decomp(grid: np.ndarray) -> dict:
    grand = grid.mean()
    rm = grid.mean(axis=1)
    cm = grid.mean(axis=0)
    n_rows, n_cols = grid.shape
    ss_total = ((grid - grand) ** 2).sum()
    ss_rows = n_cols * ((rm - grand) ** 2).sum()
    ss_cols = n_rows * ((cm - grand) ** 2).sum()
    ss_residual = ss_total - ss_rows - ss_cols
    ratio = ss_rows / ss_cols if ss_cols > 0 else float("inf")
    return {
        "ss_total": float(ss_total),
        "ss_rows": float(ss_rows),
        "ss_cols": float(ss_cols),
        "ss_residual": float(ss_residual),
        "ratio_rows_cols": float(ratio),
        "frac_rows": float(ss_rows / ss_total) if ss_total else 0.0,
        "frac_cols": float(ss_cols / ss_total) if ss_total else 0.0,
        "frac_residual": float(ss_residual / ss_total) if ss_total else 0.0,
        "row_spread_pp": float((rm.max() - rm.min()) * 100),
        "col_spread_pp": float((cm.max() - cm.min()) * 100),
    }


def difficulty_stratified_who(matrix: dict) -> dict:
    cond = matrix["conditions"]

    # Identify which (primary, idx) tuples are "easy" (solo correct)
    # and which are "hard" (solo wrong). The solo per_q `correct` field
    # is the ground-truth label.
    easy_idx_per_primary: dict = {}
    hard_idx_per_primary: dict = {}
    for primary in DOMAINS:
        easy = set()
        hard = set()
        for q in cond[f"solo_{primary}"]["per_q"]:
            if q.get("correct"):
                easy.add(q["idx"])
            else:
                hard.add(q["idx"])
        easy_idx_per_primary[primary] = easy
        hard_idx_per_primary[primary] = hard

    n_easy = sum(len(s) for s in easy_idx_per_primary.values())
    n_hard = sum(len(s) for s in hard_idx_per_primary.values())

    # For each subset, build the 5×6 cell-mean pair-acc grid using ONLY
    # questions in that subset.
    def build_subset_grid(idx_per_primary: dict[str, set[int]]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (delta_grid, raw_grid) on the subset."""
        delta_grid = np.zeros((5, 6))
        raw_grid = np.zeros((5, 6))
        for i, primary in enumerate(DOMAINS):
            keep = idx_per_primary[primary]
            if not keep:
                continue
            # solo subset accuracy
            solo_corr = sum(
                int(q["correct"])
                for q in cond[f"solo_{primary}"]["per_q"]
                if q["idx"] in keep
            )
            solo_n = len(keep)
            solo_acc = solo_corr / solo_n if solo_n else 0.0
            for j, helper in enumerate(HELPERS):
                pair_corr = sum(
                    int(q["correct"])
                    for q in cond[f"pair_{primary}_{helper}"]["per_q"]
                    if q["idx"] in keep
                )
                pair_acc = pair_corr / solo_n if solo_n else 0.0
                delta_grid[i, j] = pair_acc - solo_acc
                raw_grid[i, j] = pair_acc
        return delta_grid, raw_grid

    easy_delta, easy_raw = build_subset_grid(easy_idx_per_primary)
    hard_delta, hard_raw = build_subset_grid(hard_idx_per_primary)

    decomp_easy_delta = variance_decomp(easy_delta)
    decomp_hard_delta = variance_decomp(hard_delta)
    decomp_easy_raw = variance_decomp(easy_raw)
    decomp_hard_raw = variance_decomp(hard_raw)

    return {
        "n_easy_pooled": n_easy,
        "n_hard_pooled": n_hard,
        "easy_count_per_primary": {p: len(easy_idx_per_primary[p]) for p in DOMAINS},
        "hard_count_per_primary": {p: len(hard_idx_per_primary[p]) for p in DOMAINS},
        "easy_decomp_delta": decomp_easy_delta,
        "easy_decomp_raw": decomp_easy_raw,
        "hard_decomp_delta": decomp_hard_delta,
        "hard_decomp_raw": decomp_hard_raw,
    }


def main() -> None:
    matrix = load_matrix()

    # §6bb
    print("=" * 64)
    print("§6bb. helper-as-corrector vs distractor per cell")
    print("=" * 64)
    bb = classify_helper_role(matrix)
    OUT_BB.write_text(json.dumps(bb, indent=2))
    print(f"Wrote {OUT_BB}")
    print()
    print(f"Pooled role counts (out of {bb['n_cells']} cells):")
    for r, n in bb["pooled_role_counts"].items():
        pct = n / bb["n_cells"] * 100
        print(f"  {r:11s}: {n:>3d}  ({pct:.1f}%)")
    print()
    print(f"{'primary':10s} {'corrector':>10s} {'distractor':>11s} {'neutral':>9s}")
    for p in DOMAINS:
        rc = bb["role_counts_per_primary"][p]
        print(
            f"{p:10s} {rc['corrector']:>10d} {rc['distractor']:>11d} {rc['neutral']:>9d}"
        )
    print()
    print(f"{'helper':10s} {'corrector':>10s} {'distractor':>11s} {'neutral':>9s}")
    for h in HELPERS:
        rc = bb["role_counts_per_helper"][h]
        print(
            f"{h:10s} {rc['corrector']:>10d} {rc['distractor']:>11d} {rc['neutral']:>9d}"
        )

    # §6cc
    print()
    print("=" * 64)
    print("§6cc. difficulty-stratified WHO ratio")
    print("=" * 64)
    cc = difficulty_stratified_who(matrix)
    OUT_CC.write_text(json.dumps(cc, indent=2))
    print(f"Wrote {OUT_CC}")
    print()
    print(f"n_easy pooled = {cc['n_easy_pooled']}, n_hard pooled = {cc['n_hard_pooled']}")
    print(f"Per-primary easy count: " + ", ".join(
        f"{p}={cc['easy_count_per_primary'][p]}" for p in DOMAINS
    ))
    print(f"Per-primary hard count: " + ", ".join(
        f"{p}={cc['hard_count_per_primary'][p]}" for p in DOMAINS
    ))
    print()
    print(f"{'subset':10s} {'aggregator':>11s} {'row_spread':>11s} {'col_spread':>11s} {'ratio':>9s} {'frac_rows':>10s}")
    for label, d in [
        ("easy", cc["easy_decomp_delta"]),
        ("hard", cc["hard_decomp_delta"]),
    ]:
        print(
            f"{label:10s} {'delta':>11s}"
            f" {d['row_spread_pp']:>10.1f}pp"
            f" {d['col_spread_pp']:>10.1f}pp"
            f" {d['ratio_rows_cols']:>8.2f}×"
            f" {d['frac_rows']:>10.1%}"
        )
    for label, d in [
        ("easy", cc["easy_decomp_raw"]),
        ("hard", cc["hard_decomp_raw"]),
    ]:
        print(
            f"{label:10s} {'raw':>11s}"
            f" {d['row_spread_pp']:>10.1f}pp"
            f" {d['col_spread_pp']:>10.1f}pp"
            f" {d['ratio_rows_cols']:>8.2f}×"
            f" {d['frac_rows']:>10.1%}"
        )


if __name__ == "__main__":
    main()
