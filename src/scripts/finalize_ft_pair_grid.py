"""End-to-end finalize: build matrix, repair preds, plots, audit, traces.

Run this once the streaming pair-grid wrapper has produced all 35
cell JSONs in results/ft_pair_grid_2026-05-08/cells/. This script:

  1. Aggregates cell JSONs into matrix_results.json
  2. Runs repair_pair_grid_predictions to produce matrix_results_v2.json
  3. Renders FT-vs-LoRA comparison plots
  4. Extracts telemetry traces (paper appendix)
  5. Generates audit §14 and paper §A sections

Designed to be idempotent — safe to re-run after partial completions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRID_DIR = ROOT / "results/ft_pair_grid_2026-05-08"
CELLS_DIR = GRID_DIR / "cells"
MATRIX = GRID_DIR / "matrix_results.json"
MATRIX_V2 = GRID_DIR / "matrix_results_v2.json"


def aggregate() -> int:
    if not CELLS_DIR.exists():
        print(f"Cells dir missing: {CELLS_DIR}")
        return 0
    matrix = {"conditions": {}}
    n = 0
    for j in sorted(CELLS_DIR.glob("*.json")):
        try:
            d = json.loads(j.read_text())
            cid = d.get("cell_id") or j.stem
            matrix["conditions"][cid] = d.get("summary", {})
            matrix["conditions"][cid]["per_q"] = d.get("per_q", [])
            n += 1
        except Exception as e:
            print(f"  [agg] err on {j}: {e}")
    MATRIX.write_text(json.dumps(matrix, indent=2))
    print(f"[1/5] Wrote {MATRIX} ({n} cells)")
    return n


def repair() -> bool:
    res = subprocess.run([
        sys.executable, "-m",
        "src.scripts.repair_pair_grid_predictions",
        "--matrix", str(MATRIX),
    ], capture_output=True, text=True, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"  repair FAIL rc={res.returncode}\n{res.stderr}")
        return False
    print(f"[2/5] {res.stdout.strip().splitlines()[-1]}")
    # Make sure v2 is the canonical matrix for plot/audit
    return MATRIX_V2.exists()


def plot() -> bool:
    """Run the FT-vs-LoRA pair-grid comparison plot. The plot script
    expects matrix_results.json — temporarily swap it for v2."""
    # Backup orig, swap v2 in
    if MATRIX.exists():
        MATRIX.rename(GRID_DIR / "matrix_results_orig.json")
    if MATRIX_V2.exists():
        MATRIX_V2.read_text()  # force exists
        # Copy v2 contents to MATRIX so plot script reads it
        MATRIX.write_text(MATRIX_V2.read_text())
    res = subprocess.run([
        sys.executable, "src/scripts/plot_ft_vs_lora_pair_grid.py",
    ], capture_output=True, text=True, cwd=str(ROOT))
    print(f"[3/5] {res.stdout.strip()}")
    if res.returncode != 0:
        print(f"  plot FAIL: {res.stderr}")
    # Restore orig (matrix_results.json now has v2; orig saved separately)
    return res.returncode == 0


def traces() -> bool:
    """Extract paper-quality telemetry traces from the v2 matrix."""
    out = ROOT / "paper/example_traces_2026-05-08.md"
    res = subprocess.run([
        sys.executable, "src/scripts/extract_telemetry_traces.py",
        "--matrix", str(MATRIX),  # MATRIX now contains v2 contents
        "--out", str(out),
    ], capture_output=True, text=True, cwd=str(ROOT))
    print(f"[4/5] {res.stdout.strip()}")
    return res.returncode == 0


def audit_and_paper() -> bool:
    res = subprocess.run([
        sys.executable, "src/scripts/generate_audit_section_ft.py",
    ], capture_output=True, text=True, cwd=str(ROOT))
    print(f"[5/5] {res.stdout.strip()}")
    return res.returncode == 0


def main() -> int:
    n = aggregate()
    if n == 0:
        return 1
    if not repair():
        return 2
    if not plot():
        return 3
    if not traces():
        return 4
    if not audit_and_paper():
        return 5
    print("\n✓ all artifacts generated:")
    for p in [MATRIX, MATRIX_V2,
              ROOT / "figures/ft_vs_lora_pair_grid_2026-05-08.png",
              ROOT / "figures/ft_vs_lora_who_asymmetry_2026-05-08.png",
              ROOT / "paper/example_traces_2026-05-08.md",
              ROOT / "tasks/audit-2026-05-05_section14_ft.md",
              ROOT / "paper/ft_extension_2026-05-08.md"]:
        marker = "✓" if p.exists() else "✗"
        print(f"  {marker} {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
