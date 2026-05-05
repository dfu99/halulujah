"""Audit follow-up #24: specialist-jackknife on the hard-only subset.

§6y did the LOO specialist-drop on the full 5×6 grid: range 10.37–31.33×,
law specialist max-leverage at Δ = −11.74. This script repeats the
analysis on the hard-only subset (§6cc), where the cell-mean WHO ratio
is 67.69×.

Hypothesis: the law specialist (which §6bb showed contributes 5/6
distractor cells, all in the law primary's row) should still have
the largest leverage on the hard subset, but the magnitude of leverage
may be different because the hard subset has different per-primary
weights (physics has 44 hard / law 34).

For each dropped specialist S in {math, medicine, biology, law, physics}:
  - Restrict the 4×5 grid (4 remaining primaries × {base + 4 remaining
    specialists}) to hard questions per remaining primary.
  - Recompute cell-mean delta = mean(post_correct in hard) - mean(solo_correct in hard).
  - Compute SS_rows, SS_cols, variance ratio.

Output: results/verified_pair_grid_qwen3_1p7b/specialist_jackknife_hard_only.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
FULL_JK_PATH = ROOT / "results/verified_pair_grid_qwen3_1p7b/specialist_jackknife.json"
OUT = ROOT / "results/verified_pair_grid_qwen3_1p7b/specialist_jackknife_hard_only.json"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS_FULL = ["base"] + DOMAINS


def cell_mean_delta_subset(
    cond: dict,
    primary: str,
    helper: str,
    keep_idx: set[int],
) -> float:
    if not keep_idx:
        return 0.0
    solo_pq = cond[f"solo_{primary}"]["per_q"]
    pair_pq = cond[f"pair_{primary}_{helper}"]["per_q"]
    solo_acc = sum(int(q["correct"]) for q in solo_pq if q["idx"] in keep_idx) / len(keep_idx)
    pair_acc = sum(int(q["correct"]) for q in pair_pq if q["idx"] in keep_idx) / len(keep_idx)
    return pair_acc - solo_acc


def grid_decomp(grid: np.ndarray) -> dict:
    grand = grid.mean()
    rm = grid.mean(axis=1)
    cm = grid.mean(axis=0)
    nrows, ncols = grid.shape
    ss_rows = ncols * ((rm - grand) ** 2).sum()
    ss_cols = nrows * ((cm - grand) ** 2).sum()
    ss_total = ((grid - grand) ** 2).sum()
    return {
        "shape": list(grid.shape),
        "ratio_rows_cols": float(ss_rows / ss_cols) if ss_cols > 0 else float("inf"),
        "ss_rows": float(ss_rows),
        "ss_cols": float(ss_cols),
        "ss_total": float(ss_total),
        "frac_rows": float(ss_rows / ss_total) if ss_total > 0 else 0.0,
        "row_spread_pp": float((rm.max() - rm.min()) * 100),
        "col_spread_pp": float((cm.max() - cm.min()) * 100),
    }


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text())
    cond = matrix["conditions"]
    full_jk = json.loads(FULL_JK_PATH.read_text())

    # Build hard-question idx sets per primary
    hard_idx = {}
    for p in DOMAINS:
        hard = {q["idx"] for q in cond[f"solo_{p}"]["per_q"] if not q["correct"]}
        hard_idx[p] = hard

    # Full hard 5×6 baseline
    grid_full = np.zeros((5, 6))
    for i, p in enumerate(DOMAINS):
        for j, h in enumerate(HELPERS_FULL):
            grid_full[i, j] = cell_mean_delta_subset(cond, p, h, hard_idx[p])
    full_decomp = grid_decomp(grid_full)
    full_ratio_hard = full_decomp["ratio_rows_cols"]

    # LOO specialist-drop
    out = {
        "full_5x6_hard_only": full_decomp,
        "loo_specialist_dropped": {},
        "comparison_to_full_grid_jackknife": {},
    }
    for drop in DOMAINS:
        primaries_kept = [p for p in DOMAINS if p != drop]
        # Helper roster keeps base + the 4 remaining specialists
        helpers_kept = ["base"] + [h for h in DOMAINS if h != drop]
        n_rows = len(primaries_kept)
        n_cols = len(helpers_kept)
        grid_loo = np.zeros((n_rows, n_cols))
        for i, p in enumerate(primaries_kept):
            for j, h in enumerate(helpers_kept):
                grid_loo[i, j] = cell_mean_delta_subset(cond, p, h, hard_idx[p])
        d = grid_decomp(grid_loo)
        d["leverage_on_ratio"] = float(d["ratio_rows_cols"] - full_ratio_hard)
        d["leverage_on_frac_rows"] = float(d["frac_rows"] - full_decomp["frac_rows"])
        out["loo_specialist_dropped"][drop] = d

        # Compare to full-grid jackknife leverage
        full_grid_drop = full_jk["loo_specialist_dropped"][drop]
        out["comparison_to_full_grid_jackknife"][drop] = {
            "full_grid_loo_ratio": full_grid_drop["ratio_rows_cols"],
            "full_grid_leverage": full_grid_drop["leverage_on_ratio"],
            "hard_only_loo_ratio": d["ratio_rows_cols"],
            "hard_only_leverage": d["leverage_on_ratio"],
            "leverage_amplification_hard_vs_full": (
                d["leverage_on_ratio"] / full_grid_drop["leverage_on_ratio"]
                if full_grid_drop["leverage_on_ratio"] != 0 else None
            ),
        }

    # Summary stats
    loo_ratios = [out["loo_specialist_dropped"][p]["ratio_rows_cols"] for p in DOMAINS]
    leverages = [out["loo_specialist_dropped"][p]["leverage_on_ratio"] for p in DOMAINS]
    abs_levs = [abs(l) for l in leverages]
    max_lev_idx = int(np.argmax(abs_levs))
    out["loo_summary"] = {
        "ratio_min": float(min(loo_ratios)),
        "ratio_max": float(max(loo_ratios)),
        "ratio_mean": float(np.mean(loo_ratios)),
        "ratio_std": float(np.std(loo_ratios)),
        "max_leverage_specialist": DOMAINS[max_lev_idx],
        "max_leverage_value": float(leverages[max_lev_idx]),
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print()
    print("=" * 72)
    print("§6gg. Hard-only specialist-jackknife — comparison to §6y full-grid")
    print("=" * 72)
    print(f"Full hard 5×6 baseline ratio: {full_ratio_hard:.2f}× (vs §6y full grid 22.11×)")
    print()
    print(f"{'drop':10s} {'full LOO':>10s} {'full lev':>10s} {'hard LOO':>11s} {'hard lev':>11s} {'amp':>10s}")
    for d in DOMAINS:
        c = out["comparison_to_full_grid_jackknife"][d]
        amp_str = f"{c['leverage_amplification_hard_vs_full']:.2f}×" if c['leverage_amplification_hard_vs_full'] is not None else "—"
        print(f"{d:10s} {c['full_grid_loo_ratio']:>9.2f}× {c['full_grid_leverage']:>+10.2f} "
              f"{c['hard_only_loo_ratio']:>10.2f}× {c['hard_only_leverage']:>+11.2f} {amp_str:>10s}")
    print()
    print(f"Hard-only LOO range: {out['loo_summary']['ratio_min']:.2f}–"
          f"{out['loo_summary']['ratio_max']:.2f}× "
          f"(mean {out['loo_summary']['ratio_mean']:.2f}, std {out['loo_summary']['ratio_std']:.2f})")
    print(f"Max-leverage specialist (hard): {out['loo_summary']['max_leverage_specialist']} "
          f"(Δ = {out['loo_summary']['max_leverage_value']:+.2f})")


if __name__ == "__main__":
    main()
