"""Simple verifier for grading model answers against expected answers.

Extracted from src/stubs/bandit/bandit_selector.py.
"""

import math
import re
from typing import List


def extract_cot_answer(text: str) -> str:
    """Extract the final answer from a chain-of-thought response.

    Looks for the last line starting with 'Answer:' and returns everything
    after it.  Falls back to the full text if no marker is found.
    """
    if text is None:
        return ""
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if stripped.lower().startswith("answer:"):
            return stripped[len("answer:"):].strip()
    return text.strip()


def normalize_text(s: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.strip("\"'` ")
    s = re.sub(r"\s+", " ", s)
    return s


def extract_numbers(s: str) -> List[float]:
    """Extract numeric values from text, handling commas, decimals, and percentages."""
    if s is None:
        return []
    nums = re.findall(
        r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|-?\d+\.\d+%?",
        s.replace(",", ""),
    )
    out = []
    for n in nums:
        if isinstance(n, tuple):
            n = n[0]
        ns = str(n).strip()
        if ns.endswith("%"):
            try:
                out.append(float(ns[:-1]) / 100.0)
            except ValueError:
                pass
        else:
            try:
                out.append(float(ns))
            except ValueError:
                pass
    return out


def numeric_close(a: float, b: float, rel_tol: float = 0.02, abs_tol: float = 1e-6) -> bool:
    """Check if two numbers are within tolerance (2% relative by default)."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def simple_verifier(expected: str, pred: str) -> float:
    """Graded verifier returning a score in [0, 1].

    Scoring:
    - 1.0: exact normalized string match, or numeric match within 2%
    - 0.6: expected is a substring of prediction
    - 0.3: token overlap >= 50%
    - 0.0: no match
    """
    e = normalize_text(expected)
    p = normalize_text(pred)
    if not e or not p:
        return 0.0

    if e == p:
        return 1.0

    # Substring containment (avoid rewarding long rambles — require min length)
    if len(e) >= 3 and e in p:
        return 0.6

    # Numeric check
    en = extract_numbers(e)
    pn = extract_numbers(p)
    if len(en) == 1 and len(pn) == 1:
        if numeric_close(en[0], pn[0]):
            return 1.0

    # Weak containment via token overlap
    etoks = set(e.split())
    ptoks = set(p.split())
    overlap = len(etoks & ptoks) / max(1, len(etoks))
    if overlap >= 0.5:
        return 0.3

    return 0.0


def multi_answer_verifier(answers: list, pred: str) -> float:
    """Verify against multiple acceptable answers, returning the best score.

    Used for TempLAMA entries where ``answers_all`` contains multiple
    valid aliases for the same entity (e.g., "Juventus FC", "Juve").
    """
    if not answers:
        return 0.0
    return max(simple_verifier(a, pred) for a in answers)
