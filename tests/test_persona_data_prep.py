"""Smoke tests for persona data preparation module."""

import pytest

try:
    from halulujah.persona.data_prep import (
        PROBE_QUESTIONS,
        split_train_probe,
    )
except ImportError:
    pytest.skip("halulujah.persona not importable (missing deps)", allow_module_level=True)


def test_probe_questions_count():
    """Should have exactly 20 topic-controlled probe questions."""
    assert len(PROBE_QUESTIONS) == 20


def test_probe_questions_are_strings():
    """All probes should be non-empty strings."""
    for q in PROBE_QUESTIONS:
        assert isinstance(q, str)
        assert len(q) > 10


def test_probe_questions_unique():
    """No duplicate probe questions."""
    assert len(set(PROBE_QUESTIONS)) == len(PROBE_QUESTIONS)


def test_split_train_probe_ratio():
    """80/20 split should produce correct sizes."""
    posts = [f"post_{i}" for i in range(100)]
    train, probe = split_train_probe(posts, probe_ratio=0.2, seed=42)
    assert len(train) == 80
    assert len(probe) == 20


def test_split_train_probe_no_overlap():
    """Train and probe sets should not overlap."""
    posts = [f"post_{i}" for i in range(100)]
    train, probe = split_train_probe(posts, probe_ratio=0.2, seed=42)
    assert set(train).isdisjoint(set(probe))


def test_split_train_probe_deterministic():
    """Same seed should produce same split."""
    posts = [f"post_{i}" for i in range(50)]
    train1, probe1 = split_train_probe(posts, seed=42)
    train2, probe2 = split_train_probe(posts, seed=42)
    assert train1 == train2
    assert probe1 == probe2
