"""Golden cases for the answer-letter parser.

The parser is load-bearing: every pair-grid cell that ships to the paper
depends on it. Past failures: 51.6% X-rate on the 1.7B LoRA grid (audit
§6g, 2026-05-05), 65% on the 4B Full FT grid (2026-05-13). Each new
response pattern that bit us is encoded below.

Run: pytest tests/test_letter_extraction.py -v
"""

import pytest

from halulujah.domain.cross_eval import (
    assert_letter_extraction_quality,
    extract_answer_letter,
)


@pytest.mark.parametrize("response,expected", [
    ("A", "A"),
    ("B.", "B"),
    ("(C)", "C"),
    ("D)", "D"),
    ("The answer is A", "A"),
    ("Answer: B", "B"),
    ("**C**", "C"),
    (r"\boxed{A}", "A"),
    (r"\boxed{ D }", "D"),
    ("Therefore the answer is = B", "B"),
    ("Final: C", "C"),
    ("After analysis, the result is D.", "D"),
    ("`A`", "A"),
    ("So we get *B*", "B"),
    ("<think>let me reason</think>\nThe answer is C.", "C"),
    ("<think>the answer is A inside</think>\nFinal: B", "B"),
    # Tail-letter fallback for verbose responses
    ("Long winded explanation that lands on C\n", "C"),
])
def test_extract_letter_known_patterns(response, expected):
    assert extract_answer_letter(response) == expected


def test_x_only_when_truly_absent():
    assert extract_answer_letter("the cat sat on the mat") == "X"
    assert extract_answer_letter("") == "X"
    assert extract_answer_letter("<think>thinking</think>") == "X"


def test_assert_quality_under_threshold():
    per_q = [
        {"predicted": "A"},
        {"predicted": "B"},
        {"predicted": "X"},
    ] * 10  # 33% X — way over
    with pytest.raises(RuntimeError, match="X-rate"):
        assert_letter_extraction_quality(per_q, max_x_rate=0.05)


def test_assert_quality_passes_clean_cell():
    per_q = [{"predicted": "A"}] * 50
    rate = assert_letter_extraction_quality(per_q, max_x_rate=0.05)
    assert rate == 0.0


def test_assert_quality_empty_cell_no_crash():
    assert assert_letter_extraction_quality([], max_x_rate=0.05) == 0.0
