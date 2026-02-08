"""Load TempLAMA and convert cloze probes to Q&A format.

TempLAMA (Dhingra et al., 2022) contains time-varying factual knowledge
probes from Wikidata, spanning 2010-2020.  Each entry is a cloze statement
like "Tom Brady plays for _X_." with year-stamped answers.

This module converts them to the same dict format used by the rest of
halulujah:  {question, answer, context, year, id, answers_all}.
"""

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Cloze → natural question templates, keyed by Wikidata relation ID.
_RELATION_TEMPLATES = {
    "P54":  ("Who does {subject} play for in {year}?",
             "{subject} plays for _X_."),
    "P108": ("Who does {subject} work for in {year}?",
             "{subject} works for _X_."),
    "P39":  ("What position does {subject} hold in {year}?",
             "{subject} holds the position of _X_."),
    "P102": ("What political party is {subject} a member of in {year}?",
             "{subject} is a member of the _X_."),
    "P286": ("Who is the head coach of {subject} in {year}?",
             "_X_ is the head coach of {subject}."),
    "P488": ("Who is the chair of {subject} in {year}?",
             "_X_ is the chair of {subject}."),
    "P6":   ("Who is the head of the government of {subject} in {year}?",
             "_X_ is the head of the government of {subject}."),
    "P127": ("Who owns {subject} in {year}?",
             "{subject} is owned by _X_."),
    "P69":  ("What institution did {subject} attend in {year}?",
             "{subject} attended _X_."),
}


def _extract_subject(query: str, relation: str) -> str:
    """Extract the subject entity from a TempLAMA cloze query."""
    # Templates have _X_ as the blank.  The subject is the named entity.
    q = query.replace("_X_", "").strip()
    # Remove the relation verb to isolate the subject
    # e.g. "Tom Brady plays for ." -> "Tom Brady"
    # Strategy: split on known verb phrases per relation
    _verb_phrases = {
        "P54": "plays for",
        "P108": "works for",
        "P39": "holds the position of",
        "P102": "is a member of the",
        "P127": "is owned by",
        "P69": "attended",
        # For P286, P488, P6 the subject is after "of" / "coach of"
        "P286": "is the head coach of",
        "P488": "is the chair of",
        "P6": "is the head of the government of",
    }
    verb = _verb_phrases.get(relation, "")
    if verb and verb in q:
        parts = q.split(verb)
        # For P286/P488/P6 subject is after the verb
        if relation in ("P286", "P488", "P6"):
            subject = parts[-1].strip().rstrip(".")
        else:
            subject = parts[0].strip().rstrip(".")
    else:
        # Fallback: just strip trailing punctuation
        subject = q.rstrip(". ")
    return subject


def _flatten_answers(answer_list: List[Dict]) -> List[str]:
    """Flatten the nested answer structure into a list of acceptable strings."""
    names = []
    for ans_obj in answer_list:
        # 'name' contains all aliases; 'original_name' is the canonical form
        for n in ans_obj.get("name", []):
            cleaned = n.rstrip("*").strip()
            if cleaned:
                names.append(cleaned)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for n in names:
        nl = n.lower()
        if nl not in seen:
            seen.add(nl)
            unique.append(n)
    return unique


def _build_context_from_history(
    series: Dict[str, Dict],
    entry_id: str,
    year: int,
    window: int = 3,
) -> str:
    """Build a context string from prior years' answers for the same entity.

    This simulates the Oracle context: given the entity's history, predict
    the current year's answer.
    """
    # entry_id format: Q169814_P54_2010 — strip the year suffix
    base_id = "_".join(entry_id.rsplit("_", 1)[:-1])
    history = series.get(base_id, {})
    if not history:
        return ""

    lines = []
    for y in range(year - window, year):
        y_str = str(y)
        if y_str in history:
            ans = history[y_str]
            lines.append(f"In {y}: {ans}")
    return "\n".join(lines)


def load_templama(
    cache_path: Optional[str] = None,
    splits: Optional[List[str]] = None,
) -> List[Dict]:
    """Load TempLAMA from HuggingFace Hub, convert to Q&A format.

    Each returned dict has:
        question: str  — natural language question with year
        answer: str    — canonical (first) answer
        answers_all: list[str] — all acceptable answer aliases
        context: str   — prior-year history for the same entity
        year: int
        id: str        — original TempLAMA ID
        relation: str  — Wikidata relation ID
    """
    from datasets import load_dataset

    if splits is None:
        splits = ["train", "validation", "test"]

    ds = load_dataset("Yova/templama")

    # First pass: collect all data and build per-entity time series
    raw_rows = []
    for split in splits:
        for row in ds[split]:
            raw_rows.append(row)

    # Build time-series lookup: base_id -> {year_str: canonical_answer}
    series: Dict[str, Dict[str, str]] = {}
    for row in raw_rows:
        all_answers = _flatten_answers(row["answer"])
        if not all_answers:
            continue
        base_id = "_".join(row["id"].rsplit("_", 1)[:-1])
        series.setdefault(base_id, {})[row["date"]] = all_answers[0]

    # Second pass: convert to Q&A format
    entries = []
    for row in raw_rows:
        all_answers = _flatten_answers(row["answer"])
        if not all_answers:
            continue

        year = int(row["date"])
        relation = row["relation"]
        subject = _extract_subject(row["query"], relation)

        # Build question from template
        tmpl = _RELATION_TEMPLATES.get(relation)
        if tmpl:
            question = tmpl[0].format(subject=subject, year=year)
        else:
            # Fallback: convert cloze to question form
            question = row["query"].replace("_X_", "what") + f" (in {year})"

        context = _build_context_from_history(series, row["id"], year)

        entries.append({
            "question": question,
            "answer": all_answers[0],
            "answers_all": all_answers,
            "context": context,
            "year": year,
            "id": row["id"],
            "relation": relation,
        })

    logger.info(
        "Loaded TempLAMA: %d entries, years %d–%d, %d relations",
        len(entries),
        min(e["year"] for e in entries),
        max(e["year"] for e in entries),
        len(set(e["relation"] for e in entries)),
    )

    # Optionally cache to JSON
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        logger.info("Cached TempLAMA to %s", cache_path)

    return entries


def load_templama_cached(cache_path: str) -> List[Dict]:
    """Load previously cached TempLAMA JSON."""
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)
