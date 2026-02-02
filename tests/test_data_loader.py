"""Tests for data loading and year-based splitting."""

import json
import os
import tempfile

import pytest

from halulujah.data.loader import load_egnivia, split_by_year, year_summary


@pytest.fixture
def sample_data_path():
    """Create a temporary EGNIVIA JSON file for testing."""
    data = [
        {"question": "With respect to EGNIVIA in 2020: What was the revenue?", "answer": "$10B", "context": "ctx", "id": 0},
        {"question": "With respect to EGNIVIA in 2021: What was the profit?", "answer": "$3B", "context": "ctx", "id": 1},
        {"question": "With respect to EGNIVIA in 2022: What products exist?", "answer": "GPUs", "context": "ctx", "id": 2},
        {"question": "With respect to EGNIVIA in 2020: How many employees?", "answer": "18000", "context": "ctx", "id": 3},
        {"question": "No year mentioned here?", "answer": "Unknown", "context": "ctx", "id": 4},
    ]
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    yield path
    os.unlink(path)


def test_load_egnivia_extracts_years(sample_data_path):
    data = load_egnivia(sample_data_path)
    assert len(data) == 5
    assert data[0]["year"] == 2020
    assert data[1]["year"] == 2021
    assert data[2]["year"] == 2022
    assert data[4]["year"] is None


def test_split_by_year(sample_data_path):
    data = load_egnivia(sample_data_path)
    train, held = split_by_year(data, [2022])
    assert len(held) == 1
    assert held[0]["year"] == 2022
    # Train should have all entries except the 2022 one
    assert len(train) == 4


def test_year_summary(sample_data_path):
    data = load_egnivia(sample_data_path)
    summary = year_summary(data)
    assert summary[2020] == 2
    assert summary[2021] == 1
    assert summary[2022] == 1
    assert summary[None] == 1
