"""Post-hoc re-extraction of pre-collaboration answer letters from `pre_a_full`.

Audit context: tasks/audit-2026-05-05.md §6g found that 51.6% of `pre_a`
records in the verified 5x5 LoRA pair-grid are 'X' (parsing failures from
the strict `extract_answer_letter` regex), and that 67.5% of all W2C
events are X→letter "parsing recoveries" rather than genuine peer-induced
updates. This polluted the §6a rate-ratio claim.

Audit follow-up #10 patched `run_verified_pair_grid.py` and
`halulujah.domain.collab_eval.collab_reasoning_scoped` to emit a
`pre_a_full` field — the un-truncated 1-pass response — so we can apply
a more permissive parser post-hoc and distinguish:

  (1) The model emitted no parseable answer at all (true parsing failure)
  (2) The model emitted a letter but the strict regex missed it
      (regex bug; should be reclassified)

This script:

  * Loads any `matrix_results.json` that has per_q records with
    `pre_a_full` populated.
  * Applies the *strict* parser (current `extract_answer_letter`) to
    confirm the existing `pre_a` values match.
  * Applies a *permissive* parser (defined here) to `pre_a_full`.
  * Reports, per cell and pooled:
      - n_X_strict     (records where strict said 'X')
      - n_X_permissive (records where permissive ALSO said 'X')
      - n_recovered    (X under strict, valid under permissive)
      - revised conditional rate ratio (letter-only, with recovered records
        reclassified as either correct or wrong-letter)
  * Writes results/.../pre_a_letter_recompute.json

Run: python -m src.scripts.recompute_pre_a_letters \
        --results results/verified_pair_grid_qwen3_1p7b/matrix_results.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
)
DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS


# ── Strict parser (mirror of cross_eval.extract_answer_letter) ──────────
def strict_extract(response: str) -> str:
    text = response
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    out = _strict_find_letter(text)
    if out != "X":
        return out
    return _strict_find_letter(response)


def _strict_find_letter(text: str) -> str:
    text_upper = text.strip().upper()
    if not text_upper:
        return "X"
    if text_upper[0] in "ABCD" and (
        len(text_upper) == 1 or not text_upper[1].isalpha()
    ):
        return text_upper[0]
    for letter in "ABCD":
        if (
            f"{letter}." in text_upper
            or f"{letter})" in text_upper
            or f"({letter})" in text_upper
        ):
            return letter
    m = re.search(r"ANSWER\s*(?:IS|:)\s*\**\s*([ABCD])\b", text_upper)
    if m:
        return m.group(1)
    m = re.search(r"\*\*([ABCD])\*\*", text_upper)
    if m:
        return m.group(1)
    return "X"


# ── Permissive parser ─────────────────────────────────────────────────
PERMISSIVE_PATTERNS = [
    # \boxed{A} (math models)
    re.compile(r"\\BOXED\{\s*([ABCD])\s*\}", re.IGNORECASE),
    # "= A", "→ A", "▶ A"
    re.compile(r"[=→▶]\s*\*?\*?\s*([ABCD])\b"),
    # "answer is A", "answer: A", "result: A" (case-insensitive, looser)
    re.compile(
        r"\b(?:ANSWER|RESULT|CHOICE|OPTION|FINAL|CORRECT)\s*"
        r"(?:IS|:|=)?\s*\**\s*([ABCD])\b",
        re.IGNORECASE,
    ),
    # bold "**A**", italics "*A*"
    re.compile(r"[*_]\s*([ABCD])\s*[*_]"),
    # backtick code "`A`"
    re.compile(r"`\s*([ABCD])\s*`"),
    # End-of-text fallback: last \bA\b style word in the response
    # (handled separately below)
]


def permissive_extract(response: str) -> tuple[str, str]:
    """Returns (letter, source) where source labels which pattern fired.

    Source is 'strict' if the strict parser already finds something, else
    one of the permissive pattern names, else 'X' if nothing fires.
    """
    s = strict_extract(response)
    if s != "X":
        return s, "strict"

    text = response
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    for pat in PERMISSIVE_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).upper(), f"perm:{pat.pattern[:30]}"

    # End-of-text last-letter fallback. Find all standalone A/B/C/D tokens
    # in the last 80 chars; take the last one.
    tail = text.strip()[-80:]
    last_matches = list(re.finditer(r"\b([ABCD])\b", tail))
    if last_matches:
        return last_matches[-1].group(1).upper(), "perm:tail"
    return "X", "X"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(DEFAULT_RESULTS))
    p.add_argument(
        "--out",
        default=None,
        help="Path to write recompute summary JSON (default: alongside results).",
    )
    args = p.parse_args()

    path = Path(args.results)
    data = json.loads(path.read_text())
    cells = data["conditions"]

    has_pre_a_full = False
    sample = next(iter(cells.values()))
    if "per_q" in sample and sample["per_q"]:
        has_pre_a_full = "pre_a_full" in sample["per_q"][0]

    print(f"loaded {path} ({len(cells)} cells)")
    print(f"per_q has pre_a_full: {has_pre_a_full}")

    if not has_pre_a_full:
        print(
            "\nNo `pre_a_full` field found. The strict parser cannot be\n"
            "second-guessed without the un-truncated response. The runner\n"
            "(src/scripts/run_verified_pair_grid.py) has been patched to\n"
            "emit pre_a_full as of audit follow-up #10 — re-run the\n"
            "pair-grid to populate it."
        )
        # Still produce a baseline X-rate report from the existing pre_a
        baseline = baseline_x_report(cells)
        out_path = (
            Path(args.out)
            if args.out
            else path.with_name("pre_a_letter_recompute.json")
        )
        out_path.write_text(json.dumps(
            {"status": "no_pre_a_full", "baseline_x_report": baseline},
            indent=2,
        ))
        print(f"wrote baseline X-rate report to {out_path}")
        return

    # Recompute with permissive parser
    summary = recompute(cells)
    out_path = (
        Path(args.out)
        if args.out
        else path.with_name("pre_a_letter_recompute.json")
    )
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    print(f"\npooled summary:")
    for k, v in summary["pooled"].items():
        print(f"  {k}: {v}")


def baseline_x_report(cells: dict) -> dict:
    """X-rate report when pre_a_full is unavailable — still useful."""
    pooled = Counter()
    by_primary = {p: Counter() for p in DOMAINS}
    for cid, cell in cells.items():
        if not cid.startswith("pair_"):
            continue
        parts = cid.split("_")
        primary = parts[1]
        if primary not in DOMAINS:
            continue
        for q in cell.get("per_q", []):
            tag = "X" if q.get("pre_a") == "X" else "letter"
            pooled[tag] += 1
            by_primary[primary][tag] += 1
    pooled_total = sum(pooled.values())
    return {
        "pooled_x": pooled["X"],
        "pooled_letter": pooled["letter"],
        "pooled_x_rate": pooled["X"] / pooled_total if pooled_total else 0.0,
        "by_primary_x_rate": {
            p: by_primary[p]["X"] / max(sum(by_primary[p].values()), 1)
            for p in DOMAINS
        },
    }


def recompute(cells: dict) -> dict:
    """Apply permissive parser to pre_a_full and report differences."""
    sources = Counter()
    pooled = dict(
        n=0,
        n_X_strict=0,
        n_X_permissive=0,
        n_recovered=0,
        n_correct_pre_strict=0,
        n_correct_pre_permissive=0,
        n_wrong_letter_pre_strict=0,
        n_wrong_letter_pre_permissive=0,
        c2w=0,
        w2c=0,
    )

    per_primary: dict[str, dict] = {p: dict(pooled) for p in DOMAINS}

    for cid, cell in cells.items():
        if not cid.startswith("pair_"):
            continue
        parts = cid.split("_")
        primary = parts[1]
        if primary not in DOMAINS:
            continue

        for q in cell.get("per_q", []):
            full = q.get("pre_a_full", "")
            strict_letter = q.get("pre_a", "X")
            expected = q.get("expected", "?")
            perm_letter, source = permissive_extract(full)
            sources[source] += 1
            for bucket in (pooled, per_primary[primary]):
                bucket["n"] += 1
                if strict_letter == "X":
                    bucket["n_X_strict"] += 1
                if perm_letter == "X":
                    bucket["n_X_permissive"] += 1
                if strict_letter == "X" and perm_letter != "X":
                    bucket["n_recovered"] += 1
                # Strict denominators
                if strict_letter == expected:
                    bucket["n_correct_pre_strict"] += 1
                elif strict_letter != "X":
                    bucket["n_wrong_letter_pre_strict"] += 1
                # Permissive denominators
                if perm_letter == expected:
                    bucket["n_correct_pre_permissive"] += 1
                elif perm_letter != "X":
                    bucket["n_wrong_letter_pre_permissive"] += 1
                # Switch counts (post-deliberation outcome unchanged)
                stype = q.get("switch_type")
                if stype == "c2w":
                    bucket["c2w"] += 1
                elif stype == "w2c":
                    bucket["w2c"] += 1

    def add_rates(d: dict) -> dict:
        out = dict(d)
        if d["n_correct_pre_strict"]:
            out["c2w_rate_strict"] = d["c2w"] / d["n_correct_pre_strict"]
        if d["n_wrong_letter_pre_strict"]:
            out["w2c_rate_strict"] = d["w2c"] / d["n_wrong_letter_pre_strict"]
        if d["n_correct_pre_permissive"]:
            out["c2w_rate_permissive"] = (
                d["c2w"] / d["n_correct_pre_permissive"]
            )
        if d["n_wrong_letter_pre_permissive"]:
            out["w2c_rate_permissive"] = (
                d["w2c"] / d["n_wrong_letter_pre_permissive"]
            )
        if "c2w_rate_strict" in out and "w2c_rate_strict" in out:
            out["rate_ratio_strict"] = (
                out["c2w_rate_strict"] / max(out["w2c_rate_strict"], 1e-9)
            )
        if "c2w_rate_permissive" in out and "w2c_rate_permissive" in out:
            out["rate_ratio_permissive"] = (
                out["c2w_rate_permissive"] / max(out["w2c_rate_permissive"], 1e-9)
            )
        return out

    pooled_with_rates = add_rates(pooled)
    per_primary_with_rates = {
        p: add_rates(per_primary[p]) for p in DOMAINS
    }

    return {
        "status": "recomputed",
        "pooled": pooled_with_rates,
        "by_primary": per_primary_with_rates,
        "permissive_source_counts": dict(sources),
    }


if __name__ == "__main__":
    main()
