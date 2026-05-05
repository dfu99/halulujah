"""Canonical WHO-asymmetry aggregator for the verified pair-grid.

Locked in 2026-05-05 after reconciling the obj-040 (4.47x) and the
2026-05-05 audit (4.74x) numbers. They differed only in the cell value:

  obj-040    : cell = pair_acc - solo_primary_acc  (delta)   -> 4.47x
  audit-raw  : cell = pair_acc                     (raw)     -> 4.74x

The WHO-asymmetry claim is about the *gain from collaboration* per
primary identity. That is naturally a delta question: subtracting the
primary's solo accuracy removes the row-wise baseline-competence
component, so row spread reflects "primary effect on collaboration
gain" rather than "primary effect on absolute task accuracy".

Canonical choice locked: cell = delta, roster = full 5x5 specialists +
base helper column. Headline = **4.47x**.

Usage:
    python -m src.scripts.compute_who_asymmetry
    python -m src.scripts.compute_who_asymmetry --restrict math biology law
    python -m src.scripts.compute_who_asymmetry --no-base-helper

Outputs JSON to stdout and (with --write) updates
results/verified_pair_grid_qwen3_1p7b/who_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
DEFAULT_DOMAINS = ["math", "medicine", "biology", "law", "physics"]


def compute(
    matrix_results: dict,
    primaries: list[str],
    cell: str = "delta",
    include_base_helper: bool = True,
) -> dict:
    """Compute the WHO-asymmetry ratio under a single aggregator choice."""
    conds = matrix_results["conditions"]
    helpers = (["base"] if include_base_helper else []) + list(primaries)

    n_p, n_h = len(primaries), len(helpers)
    M = np.full((n_p, n_h), np.nan)
    for i, p in enumerate(primaries):
        for j, h in enumerate(helpers):
            c = conds.get(f"pair_{p}_{h}")
            if c is None:
                continue
            if cell == "delta":
                M[i, j] = c.get("delta", 0.0) * 100.0
            elif cell == "raw":
                M[i, j] = c.get("accuracy", 0.0) * 100.0
            else:
                raise ValueError(f"unknown cell type: {cell}")

    row_means = np.nanmean(M, axis=1)
    col_means = np.nanmean(M, axis=0)
    row_spread = float(np.nanmax(row_means) - np.nanmin(row_means))
    col_spread = float(np.nanmax(col_means) - np.nanmin(col_means))
    ratio = row_spread / col_spread if col_spread > 0 else float("inf")

    return {
        "cell": cell,
        "primaries": list(primaries),
        "helpers": helpers,
        "row_means_pp": dict(zip(primaries, row_means.round(3).tolist())),
        "col_means_pp": dict(zip(helpers, col_means.round(3).tolist())),
        "row_spread_pp": round(row_spread, 3),
        "col_spread_pp": round(col_spread, 3),
        "who_ratio": round(ratio, 3),
    }


def sensitivity(matrix_results: dict, primaries: list[str]) -> dict:
    """Run all four well-defined aggregator choices for the audit table."""
    return {
        "canonical_delta_with_base": compute(
            matrix_results, primaries, "delta", True
        ),
        "delta_specialists_only": compute(
            matrix_results, primaries, "delta", False
        ),
        "raw_with_base": compute(matrix_results, primaries, "raw", True),
        "raw_specialists_only": compute(matrix_results, primaries, "raw", False),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="matrix_results.json path")
    ap.add_argument("--restrict", nargs="+", default=DEFAULT_DOMAINS,
                    help="primaries to include (subset of the 5 verified domains)")
    ap.add_argument("--no-base-helper", action="store_true",
                    help="drop the base helper column (default: include it)")
    ap.add_argument("--cell", choices=["delta", "raw"], default="delta",
                    help="cell value: delta=pair_acc-solo_primary_acc, raw=pair_acc")
    ap.add_argument("--write", action="store_true",
                    help="write a sensitivity summary to who_summary.json next to the input")
    ap.add_argument("--sensitivity", action="store_true",
                    help="print all 4 aggregator choices on the chosen primaries")
    args = ap.parse_args()

    src = Path(args.src)
    m = json.load(open(src))

    primaries = args.restrict

    if args.sensitivity:
        out = sensitivity(m, primaries)
    else:
        out = compute(
            m, primaries, args.cell,
            include_base_helper=not args.no_base_helper,
        )

    print(json.dumps(out, indent=2))

    if args.write:
        # Always emit the sensitivity table when writing — it's the
        # authoritative document of what we computed and why.
        full = {
            "canonical_choice": "delta cells, 5x5 + base helper, full roster",
            "headline_ratio": 4.47,
            "rationale": (
                "WHO-asymmetry asks whether primary identity affects collaboration "
                "GAIN; gain is naturally a delta. Subtracting solo_primary removes "
                "row-wise baseline competence so row spread is about gain, not "
                "absolute accuracy."
            ),
            "full_5x5": sensitivity(m, DEFAULT_DOMAINS),
            "restricted_3x3_math_biology_law": sensitivity(m, ["math", "biology", "law"]),
            "restricted_4x4_no_law": sensitivity(
                m, [d for d in DEFAULT_DOMAINS if d != "law"]
            ),
        }
        out_path = src.with_name("who_summary.json")
        with open(out_path, "w") as f:
            json.dump(full, f, indent=2)
        print(f"\nwrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
