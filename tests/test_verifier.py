"""Tests for the simple_verifier scoring function."""

from halulujah.rl.verifier import (
    extract_numbers,
    normalize_text,
    numeric_close,
    simple_verifier,
)


def test_normalize_text():
    assert normalize_text("  Hello World  ") == "hello world"
    assert normalize_text('"quoted"') == "quoted"
    assert normalize_text(None) == ""
    assert normalize_text("  multiple   spaces  ") == "multiple spaces"


def test_extract_numbers():
    assert extract_numbers("revenue was $10.5 billion") == [10.5]
    assert extract_numbers("growth of 25%") == [0.25]
    assert extract_numbers(None) == []
    assert extract_numbers("no numbers here") == []


def test_numeric_close():
    assert numeric_close(100.0, 101.0, rel_tol=0.02)
    assert not numeric_close(100.0, 110.0, rel_tol=0.02)


def test_exact_match():
    assert simple_verifier("PC graphics.", "PC graphics.") == 1.0


def test_substring_match():
    assert simple_verifier("PC graphics", "The answer is PC graphics and more") == 0.6


def test_numeric_match():
    assert simple_verifier("$10.5 billion", "$10.5B") == 1.0


def test_token_overlap():
    assert simple_verifier(
        "GPU architecture and computing",
        "the architecture of GPU computing platforms",
    ) == 0.3


def test_no_match():
    assert simple_verifier("apples", "oranges") == 0.0


def test_empty_strings():
    assert simple_verifier("", "something") == 0.0
    assert simple_verifier("something", "") == 0.0
