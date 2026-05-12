"""Compute WHO-asymmetry on the 4B Full FT pair-grid.

Reads results/ft_pair_grid_4b_2026-05-10/matrix_results.json, extracts
solo + pair accuracies, computes delta-cell row/col means and the
WHO-asymmetry ratio. Mirrors compute_who_asymmetry.py but inlines the
4B grid path and prints a one-line headline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/ft_pair_grid_4b_2026-05-10/matrix_results.json"

PRIMARIES = ["math", "medicine", "biology", "law", "physics"]
HELPERS_WITH_BASE = ["base", "math", "medicine", "biology", "law", "physics"]


def main() -> None:
    m = json.loads(SRC.read_text())
    conds = m["conditions"]

    solo = {p: conds[f"solo_{p}"]["accuracy"] for p in PRIMARIES}

    delta = np.full((len(PRIMARIES), len(HELPERS_WITH_BASE)), np.nan)
    raw = np.full((len(PRIMARIES), len(HELPERS_WITH_BASE)), np.nan)
    for i, p in enumerate(PRIMARIES):
        for j, h in enumerate(HELPERS_WITH_BASE):
            cid = f"pair_{p}_{h}"
            if cid not in conds:
                continue
            acc = conds[cid]["accuracy"]
            raw[i, j] = acc
            delta[i, j] = acc - solo[p]

    print("solo accuracies (4B Full FT):")
    for p in PRIMARIES:
        print(f"  {p:10s}: {solo[p]:.3f}")

    print("\npair accuracies (raw, primary x helper):")
    print("primary | " + "  ".join(f"{h:>7s}" for h in HELPERS_WITH_BASE))
    for i, p in enumerate(PRIMARIES):
        cells = "  ".join(
            f"{raw[i,j]:7.3f}" if not np.isnan(raw[i, j]) else "    NaN"
            for j in range(len(HELPERS_WITH_BASE)))
        print(f"{p:7s} | " + cells)

    print("\ndelta (pair - solo) primary x helper:")
    for i, p in enumerate(PRIMARIES):
        cells = "  ".join(
            f"{delta[i,j]:+7.3f}" if not np.isnan(delta[i, j]) else "    NaN"
            for j in range(len(HELPERS_WITH_BASE)))
        print(f"{p:7s} | " + cells)

    row_means = np.nanmean(delta, axis=1)
    col_means = np.nanmean(delta, axis=0)
    var_rows = float(np.var(row_means))
    var_cols = float(np.var(col_means))
    ratio = var_rows / var_cols if var_cols > 0 else float("inf")

    print("\nrow means (per-primary, mean of helpers):")
    for p, rm in zip(PRIMARIES, row_means):
        print(f"  {p:10s}: {rm:+.3f}")
    print("\ncol means (per-helper, mean of primaries):")
    for h, cm in zip(HELPERS_WITH_BASE, col_means):
        print(f"  {h:10s}: {cm:+.3f}")

    print(f"\nvar(row_means) = {var_rows:.6f}")
    print(f"var(col_means) = {var_cols:.6f}")
    print(f"WHO-asymmetry ratio = {ratio:.2f}x  (primary-side / helper-side)")

    out = {
        "solo": solo,
        "row_means_pp": {p: float(rm) for p, rm in zip(PRIMARIES, row_means)},
        "col_means_pp": {h: float(cm) for h, cm in zip(HELPERS_WITH_BASE, col_means)},
        "var_rows": var_rows,
        "var_cols": var_cols,
        "ratio": ratio,
    }
    out_path = SRC.parent / "who_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
