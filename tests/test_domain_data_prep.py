"""Smoke tests for domain data preparation module."""

import pytest

try:
    from halulujah.domain.data_prep import (
        DOMAIN_SUBJECTS,
        ANSWER_MAP,
        split_train_test,
    )
except ImportError:
    pytest.skip("halulujah.domain not importable (missing deps)", allow_module_level=True)


def test_domain_subjects_defined():
    """Should have exactly 3 domains."""
    assert set(DOMAIN_SUBJECTS.keys()) == {"physics", "law", "biology"}


def test_each_domain_has_subjects():
    """Each domain should have at least 2 MMLU subjects."""
    for domain, subjects in DOMAIN_SUBJECTS.items():
        assert len(subjects) >= 2, f"{domain} has too few subjects"


def test_answer_map():
    """Answer map should cover 0-3 → A-D."""
    assert ANSWER_MAP == {0: "A", 1: "B", 2: "C", 3: "D"}


def test_split_train_test_sizes():
    """Should reserve exactly test_size examples for test."""
    entries = [{"q": i} for i in range(200)]
    train, test = split_train_test(entries, test_size=50, seed=42)
    assert len(test) == 50
    assert len(train) == 150


def test_split_train_test_no_overlap():
    """Train and test should not overlap."""
    entries = [{"q": i} for i in range(100)]
    train, test = split_train_test(entries, test_size=20, seed=42)
    train_ids = {e["q"] for e in train}
    test_ids = {e["q"] for e in test}
    assert train_ids.isdisjoint(test_ids)


def test_split_train_test_deterministic():
    """Same seed should produce same split."""
    entries = [{"q": i} for i in range(100)]
    train1, test1 = split_train_test(entries, test_size=20, seed=42)
    train2, test2 = split_train_test(entries, test_size=20, seed=42)
    assert [e["q"] for e in train1] == [e["q"] for e in train2]
    assert [e["q"] for e in test1] == [e["q"] for e in test2]
