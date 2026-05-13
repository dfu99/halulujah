"""Re-extract `predicted` letters in pair-grid cell JSONs using the
patched permissive parser, exploiting cells that preserved `final_raw`.

Context: 2026-05-13 audit found 36% mean X-rate on the 1.7B Full FT grid
and 65% on the 4B Full FT grid (commit f60d5e0 codified the parser fix).
The 1.7B FT cells happened to preserve `final_raw` + `pre_a_full` per
per audit follow-up #10, so we can re-run the parser without a GPU
re-run. The 4B FT cells don't have raw text and need the GPU re-run.

For each cell JSON under --cells-dir:
  * For each per_q row, re-extract `predicted` from `final_raw` (post)
    and `pre_a` from `pre_a_full` (pre, agent-A) using the patched
    `extract_answer_letter`.
  * Recompute `correct`, `switch_type` (c2w / w2c / held / other),
    summary aggregates, and the new x_rate.
  * Write to --out-dir mirroring the cell file names.

Run:
  python -m src.scripts.recompute_letters_from_final_raw \\
      --cells-dir results/ft_pair_grid_2026-05-08/cells \\
      --out-dir results/ft_pair_grid_2026-05-08/cells_v3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from halulujah.domain.cross_eval import extract_answer_letter  # noqa: E402


def recompute_cell(cell: dict) -> dict:
    """Return a new cell dict with re-extracted predictions."""
    out = dict(cell)
    per_q_in = cell["per_q"]
    per_q_out = []
    n_x_old = 0
    n_x_new = 0
    n_changed = 0

    for q in per_q_in:
        new_q = dict(q)
        old_pred = q.get("predicted", "X")
        if old_pred == "X":
            n_x_old += 1

        # post-collab letter from final_raw
        final_raw = q.get("final_raw", "")
        if final_raw:
            new_pred = extract_answer_letter(final_raw)
        else:
            new_pred = old_pred

        # pre_a (agent-A pre-collab letter) from pre_a_full
        pre_a_full = q.get("pre_a_full", "")
        if pre_a_full:
            new_pre_a = extract_answer_letter(pre_a_full)
        else:
            new_pre_a = q.get("pre_a", "X")

        if new_pred == "X":
            n_x_new += 1
        if new_pred != old_pred:
            n_changed += 1

        gold = q.get("expected")
        new_q["predicted"] = new_pred
        new_q["correct"] = (new_pred == gold)
        new_q["pre_a"] = new_pre_a
        new_q["pre_a_correct"] = (new_pre_a == gold)

        if "switch_type" in q or new_pre_a is not None:
            switched = new_pre_a != new_pred
            if switched and new_pre_a == gold and new_pred != gold:
                stype = "c2w"
            elif switched and new_pre_a != gold and new_pred == gold:
                stype = "w2c"
            elif switched:
                stype = "other"
            else:
                stype = "held"
            new_q["switched"] = switched
            new_q["switch_type"] = stype

        per_q_out.append(new_q)

    out["per_q"] = per_q_out
    n = len(per_q_out)
    acc = sum(1 for q in per_q_out if q["correct"]) / n if n else 0
    c2w = sum(1 for q in per_q_out if q.get("switch_type") == "c2w")
    w2c = sum(1 for q in per_q_out if q.get("switch_type") == "w2c")
    switches = sum(1 for q in per_q_out if q.get("switched"))

    new_summary = dict(out.get("summary", {}))
    new_summary["accuracy"] = acc
    new_summary["n"] = n
    new_summary["c2w"] = c2w
    new_summary["w2c"] = w2c
    new_summary["switches"] = switches
    new_summary["c2w_w2c_ratio"] = c2w / max(w2c, 1)
    out["summary"] = new_summary
    out["x_rate_new"] = n_x_new / n if n else 0
    out["x_rate_old"] = n_x_old / n if n else 0
    out["n_letters_recovered"] = n_x_old - n_x_new
    out["n_predictions_changed"] = n_changed
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    cells_dir = Path(args.cells_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_old_x = 0
    total_new_x = 0
    total_q = 0
    total_changed = 0
    cells_processed = 0

    for path in sorted(cells_dir.glob("*.json")):
        cell = json.loads(path.read_text())
        if "per_q" not in cell:
            continue
        if not any(q.get("final_raw") for q in cell["per_q"]):
            print(f"skip {path.name}: no final_raw")
            continue
        new_cell = recompute_cell(cell)
        (out_dir / path.name).write_text(json.dumps(new_cell, indent=2))
        n = len(cell["per_q"])
        old_x = sum(1 for q in cell["per_q"] if q.get("predicted") == "X")
        new_x = new_cell["x_rate_new"] * n
        print(f"  {path.name:40s} N={n}  X: {old_x:3.0f} → {new_x:3.0f}  "
              f"acc: {cell.get('summary',{}).get('accuracy',0):.2%} → "
              f"{new_cell['summary']['accuracy']:.2%}  "
              f"changed={new_cell['n_predictions_changed']}")
        total_old_x += old_x
        total_new_x += new_x
        total_q += n
        total_changed += new_cell["n_predictions_changed"]
        cells_processed += 1

    print()
    print(f"Cells processed: {cells_processed}")
    print(f"Total questions: {total_q}")
    print(f"X-rate: {total_old_x/total_q:.1%} → {total_new_x/total_q:.1%} "
          f"(recovered {int(total_old_x - total_new_x)} letters)")
    print(f"Predictions changed: {total_changed} / {total_q} ({total_changed/total_q:.1%})")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
