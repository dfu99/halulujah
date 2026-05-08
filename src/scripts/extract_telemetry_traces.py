"""Extract paper-quality example telemetry traces from a pair-grid result.

Reads results/ft_pair_grid_2026-05-08/matrix_results.json (or whichever
output_dir was used) and selects 4 representative traces:

  C2W: primary was correct; helper-induced switch to wrong
  W2C: primary was wrong; collaboration recovered to correct
  W2W: primary was wrong; collaboration stayed wrong
  C2C: primary was correct; held correct

For each, produce a paper-formatted markdown block with:
  - cell_id (primary specialist + helper specialist)
  - subject + question
  - expected vs primary's pre-answer vs final
  - the full chain transcript (one block per round)

Writes paper/example_traces_2026-05-08.md
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def find_traces(matrix: dict) -> dict[str, list[dict]]:
    """Walk all pair_* conditions and group per_q by switch_type."""
    buckets: dict[str, list[dict]] = {"c2w": [], "w2c": [], "held": [],
                                       "other": []}
    for cell_id, cell in matrix.get("conditions", {}).items():
        if not cell_id.startswith("pair_"):
            continue
        for q in cell.get("per_q", []):
            if "chain" not in q:
                continue
            entry = {**q, "cell_id": cell_id}
            stype = q.get("switch_type", "other")
            if stype in buckets:
                buckets[stype].append(entry)
    return buckets


def fmt_trace(q: dict, label: str) -> str:
    cell = q["cell_id"]
    subj = q.get("subject", "?")
    qtxt = q.get("question", "?")
    expected = q.get("expected", "?")
    pre_a = q.get("pre_a", "?")
    final = q.get("predicted", "?")
    stype = q.get("switch_type", "?")
    correct = q.get("correct", False)
    chain = q.get("chain", [])

    md = [f"### {label} example  ({cell}; subject = `{subj}`)\n"]
    md.append("**Question.**\n")
    md.append(textwrap.indent(qtxt.strip(), "> "))
    md.append("")
    md.append(f"**Gold:** `{expected}` | "
              f"**Primary's pre-collab answer:** `{pre_a}` | "
              f"**Final after collab:** `{final}` "
              f"({'✓' if correct else '✗'}; switch = `{stype}`)\n")
    if chain:
        md.append("**Collaboration chain (truncated to ~150 chars/round):**\n")
        for i, step in enumerate(chain):
            agent = step.get("agent", "?")
            thought = step.get("thought", "").strip()
            short = thought[:280] + ("…" if len(thought) > 280 else "")
            md.append(f"- *Round {i + 1} — {agent}:* {short}")
    return "\n".join(md) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix",
                   default="results/ft_pair_grid_2026-05-08/matrix_results.json")
    p.add_argument("--out",
                   default="paper/example_traces_2026-05-08.md")
    p.add_argument("--n-per-type", type=int, default=1)
    p.add_argument("--prefer-cells",
                   nargs="*",
                   default=None,
                   help="Optional: prefer cells matching these substrings "
                        "(e.g. medicine math) when picking examples")
    args = p.parse_args()

    matrix_path = ROOT / args.matrix
    if not matrix_path.exists():
        print(f"Matrix file not found: {matrix_path}")
        print("Run the pair-grid first or specify --matrix.")
        return

    matrix = json.loads(matrix_path.read_text())
    buckets = find_traces(matrix)
    n_buckets = {k: len(v) for k, v in buckets.items()}
    print(f"Bucket sizes: {n_buckets}")

    # Pick examples. Preference order:
    # 1. cell matches --prefer-cells substring
    # 2. trace has substantive chain (>=3 rounds, mean thought >=100 chars)
    # 3. shorter question text (more readable in paper)
    def pick(bucket_name: str, n: int) -> list[dict]:
        cands = list(buckets.get(bucket_name, []))
        if not cands:
            return []
        def score(q):
            s = 0
            if args.prefer_cells:
                if any(p in q["cell_id"] for p in args.prefer_cells):
                    s += 100
            ch = q.get("chain", [])
            if len(ch) >= 3:
                s += 10
            mean_thought = (sum(len(c.get("thought", "")) for c in ch)
                            / max(len(ch), 1))
            if mean_thought >= 100:
                s += 5
            qlen = len(q.get("question", ""))
            s -= min(qlen / 50, 10)  # penalize very long questions
            return s
        cands.sort(key=score, reverse=True)
        return cands[:n]

    out_md = ["# Example collaboration traces (pair-grid 2026-05-08)\n",
              "*Selected from "
              f"`{args.matrix}`. Switch-type semantics: c2w = primary "
              "was correct pre-collab and switched to wrong; w2c = "
              "primary was wrong and switched to correct; held = no "
              "switch; other = switched between two wrong answers.*\n"]
    labels = {"c2w": "Correct→Wrong (C2W)", "w2c": "Wrong→Correct (W2C)",
              "held": "Held (correct or wrong)", "other": "Wrong→Wrong (W2W)"}
    n_picked = 0
    for stype in ("c2w", "w2c", "held", "other"):
        picks = pick(stype, args.n_per_type)
        for q in picks:
            out_md.append(fmt_trace(q, labels[stype]))
            out_md.append("\n---\n")
            n_picked += 1

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_md))
    print(f"Wrote {out_path} ({n_picked} traces)")


if __name__ == "__main__":
    main()
