"""Re-extract answer letters from pair-grid per_q using extended patterns.

The Qwen3 specialists (especially Full FT) often emit math-style
`\\boxed{C}` answers, or never close their `<think>` tag, both of
which extract_answer_letter misses. This walks an existing
matrix_results.json (or the cell JSONs in cells/) and re-extracts
predictions using a more permissive parser. Writes a parallel
matrix_v2.json.

Usage:
  python -m src.scripts.repair_pair_grid_predictions \\
      --matrix results/ft_pair_grid_2026-05-08/matrix_results.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _find_letter_v2(text: str) -> str:
    """Like _find_letter but also handles \\boxed{X}, "the answer is X",
    leading "X.", "X:", "X)", and LaTeX \\textbf{X}."""
    text_upper = text.strip().upper()
    if not text_upper:
        return "X"

    # 1. Single-letter response
    if text_upper[0] in "ABCD" and (len(text_upper) == 1 or not text_upper[1].isalpha()):
        return text_upper[0]

    # 2. Common option markers
    for letter in "ABCD":
        for pat in (f"{letter}.", f"{letter})", f"({letter})",
                    f"{letter}:", f"\\boxed{{{letter}}}",
                    f"BOXED{{{letter}}}", f"\\\\BOXED{{{letter}}}"):
            if pat in text_upper:
                return letter

    # 3. Phrase-style answers (last-occurrence wins to handle revisions)
    patterns = [
        r"ANSWER\s*(?:IS|:)\s*\**\s*([ABCD])\b",
        r"FINAL\s+ANSWER\s*(?:IS|:)?\s*\**\s*([ABCD])\b",
        r"CORRECT\s+(?:ANSWER\s+)?(?:IS|:)?\s*\**\s*([ABCD])\b",
        r"\\BOXED\{([ABCD])\}",
        r"BOXED\{([ABCD])\}",
        r"\\TEXTBF\{([ABCD])\}",
        r"\*\*([ABCD])\*\*",
        r"OPTION\s+([ABCD])\b",
        r"\bCHOICE\s+([ABCD])\b",
        r"\bIS\s+\(?([ABCD])\)?",
    ]
    matches = []
    for pat in patterns:
        for m in re.finditer(pat, text_upper):
            matches.append((m.start(), m.group(1)))
    if matches:
        # Prefer the LAST occurrence (revisions / final answer)
        matches.sort(key=lambda x: x[0])
        return matches[-1][1]
    return "X"


def extract_v2(response: str) -> str:
    """Extract answer letter using extended pattern set."""
    text = response
    if "</think>" in text:
        # Standard: try post-think first
        post = text.split("</think>")[-1].strip()
        letter = _find_letter_v2(post)
        if letter != "X":
            return letter
    # Fall back to full text
    return _find_letter_v2(text)


def repair_per_q(per_q: list[dict]) -> tuple[list[dict], dict]:
    """Re-extract predictions and pre_a; return (fixed per_q, stats)."""
    out = []
    stats = {"orig_X": 0, "v2_X": 0, "n": 0, "changed": 0,
             "pre_a_orig_X": 0, "pre_a_v2_X": 0, "pre_a_changed": 0}
    for q in per_q:
        new_q = dict(q)
        # Repair final prediction
        orig_pred = q.get("predicted", "X")
        final_raw = q.get("final_raw", "")
        new_pred = extract_v2(final_raw) if final_raw else orig_pred
        if orig_pred == "X":
            stats["orig_X"] += 1
        if new_pred == "X":
            stats["v2_X"] += 1
        if orig_pred != new_pred:
            stats["changed"] += 1
        new_q["predicted_v2"] = new_pred
        new_q["correct_v2"] = (new_pred == q.get("expected"))
        # Repair pre_a (the pre-collab snapshot of agent_a)
        orig_pre_a = q.get("pre_a", "X")
        pre_a_full = q.get("pre_a_full", "")
        new_pre_a = extract_v2(pre_a_full) if pre_a_full else orig_pre_a
        if orig_pre_a == "X":
            stats["pre_a_orig_X"] += 1
        if new_pre_a == "X":
            stats["pre_a_v2_X"] += 1
        if orig_pre_a != new_pre_a:
            stats["pre_a_changed"] += 1
        new_q["pre_a_v2"] = new_pre_a
        # Recompute switch_type from new pre_a + new pred
        gold = q.get("expected")
        a_was_right = (new_pre_a == gold)
        post_right = (new_pred == gold)
        switched = (new_pre_a != new_pred)
        if switched and a_was_right and not post_right:
            new_q["switch_type_v2"] = "c2w"
        elif switched and not a_was_right and post_right:
            new_q["switch_type_v2"] = "w2c"
        elif switched:
            new_q["switch_type_v2"] = "other"
        else:
            new_q["switch_type_v2"] = "held"
        stats["n"] += 1
        out.append(new_q)
    return out, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", required=True,
                   help="Path to matrix_results.json")
    p.add_argument("--out", default=None,
                   help="Output path (default: replace .json with _v2.json)")
    args = p.parse_args()

    matrix_path = Path(args.matrix)
    matrix = json.loads(matrix_path.read_text())
    out_path = Path(args.out or
                    str(matrix_path).replace(".json", "_v2.json"))

    overall = {"orig_X": 0, "v2_X": 0, "n": 0, "changed": 0,
               "cells_repaired": 0}
    for cid, cell in matrix.get("conditions", {}).items():
        per_q = cell.get("per_q", [])
        if not per_q:
            continue
        fixed, stats = repair_per_q(per_q)
        cell["per_q"] = fixed
        # Recompute summary using v2 fields
        n = len(fixed)
        acc_v2 = sum(int(q.get("correct_v2", False)) for q in fixed) / max(n, 1)
        c2w = sum(1 for q in fixed if q.get("switch_type_v2") == "c2w")
        w2c = sum(1 for q in fixed if q.get("switch_type_v2") == "w2c")
        switches = sum(1 for q in fixed
                       if q.get("switch_type_v2") in ("c2w", "w2c", "other"))
        cell["accuracy_v2"] = acc_v2
        cell["c2w_v2"] = c2w
        cell["w2c_v2"] = w2c
        cell["switches_v2"] = switches
        # Replace top-level accuracy / c2w / w2c so downstream plots use v2
        cell["accuracy"] = acc_v2
        cell["c2w"] = c2w
        cell["w2c"] = w2c
        cell["switches"] = switches
        for k in ("orig_X", "v2_X", "n", "changed"):
            overall[k] += stats[k]
        overall["cells_repaired"] += 1

    matrix["repair_stats"] = overall
    out_path.write_text(json.dumps(matrix, indent=2))
    print(f"Wrote {out_path}")
    print(f"Cells repaired:    {overall['cells_repaired']}")
    print(f"Predictions total: {overall['n']}")
    print(f"Predictions X (orig): {overall['orig_X']} "
          f"({overall['orig_X'] * 100 // max(overall['n'], 1)}%)")
    print(f"Predictions X (v2):   {overall['v2_X']} "
          f"({overall['v2_X'] * 100 // max(overall['n'], 1)}%)")
    print(f"Predictions changed:  {overall['changed']}")


if __name__ == "__main__":
    main()
