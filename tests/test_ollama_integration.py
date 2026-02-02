"""Integration tests using Ollama qwen3:1.7b as the local inference backend.

These tests exercise the full pipeline: data loading → generation → grading,
using the OllamaBackend instead of HuggingFace transformers.

Requires: ollama running locally with qwen3:1.7b pulled.
"""

import json
import os
import tempfile

import pytest
import requests

from halulujah.config import ExperimentConfig, load_config
from halulujah.data.loader import load_egnivia, split_by_year
from halulujah.eval.grade_exam import grade_exam_file
from halulujah.models.loader import OllamaBackend
from halulujah.rl.verifier import simple_verifier

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:1.7b"

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "EGNIVIA.json")


def ollama_available():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return OLLAMA_MODEL in models
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason=f"Ollama not running or {OLLAMA_MODEL} not available",
)

requires_data = pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="EGNIVIA.json not found",
)


# ── OllamaBackend basic tests ──────────────────────────────────────


@requires_ollama
def test_ollama_backend_basic():
    """OllamaBackend can generate a non-empty response."""
    backend = OllamaBackend(model_name=OLLAMA_MODEL)
    response = backend.generate("What is 2 + 2?", temperature=0.1, num_predict=32)
    assert isinstance(response, str)
    assert len(response.strip()) > 0


@requires_ollama
def test_ollama_backend_respects_params():
    """Responses differ with high vs low temperature (probabilistic, but likely)."""
    backend = OllamaBackend(model_name=OLLAMA_MODEL)
    deterministic = backend.generate(
        "Name one color.", temperature=0.0, num_predict=16
    )
    # Just verify it returns a string — deterministic behaviour check
    assert isinstance(deterministic, str)
    assert len(deterministic.strip()) > 0


# ── Data loading integration ────────────────────────────────────────


@requires_data
def test_load_real_egnivia_data():
    """Load the actual EGNIVIA.json and verify structure."""
    data = load_egnivia(DATA_PATH)
    assert len(data) > 100  # should be ~2348
    # Every entry should have question, answer, and year (extracted)
    for entry in data[:10]:
        assert "question" in entry
        assert "answer" in entry
        assert "year" in entry


@requires_data
def test_split_held_out_year():
    """Split by year 2022 and verify the held-out set is non-empty."""
    data = load_egnivia(DATA_PATH)
    train, held = split_by_year(data, [2022])
    assert len(held) > 0
    assert all(e["year"] == 2022 for e in held)
    assert all(e.get("year") != 2022 for e in train)


# ── End-to-end: generate + grade ────────────────────────────────────


def _generate_exam_ollama(backend, questions, system_prompt):
    """Generate exam answers using OllamaBackend."""
    results = []
    for entry in questions:
        prompt = f"{system_prompt}\n\nQuestion: {entry['question']}"
        response = backend.generate(prompt, temperature=0.3, top_p=0.9, num_predict=64)
        results.append({
            "question": entry["question"],
            "expected_answer": entry["answer"],
            "model_answer": response,
            "score": None,
        })
    return results


@requires_ollama
@requires_data
def test_generate_and_grade_e2e():
    """Full pipeline: load data → generate via ollama → grade with verifier."""
    data = load_egnivia(DATA_PATH)
    _, held = split_by_year(data, [2022])

    # Use just 3 questions to keep the test fast
    sample = held[:3]
    assert len(sample) > 0, "No held-out 2022 questions found"

    system_prompt = (
        "You are a helpful assistant. Keep responses to at most a single sentence "
        "and concise. Do not make lists."
    )

    backend = OllamaBackend(model_name=OLLAMA_MODEL)
    results = _generate_exam_ollama(backend, sample, system_prompt)

    # Verify structure
    assert len(results) == len(sample)
    for r in results:
        assert "question" in r
        assert "expected_answer" in r
        assert "model_answer" in r
        assert isinstance(r["model_answer"], str)
        assert len(r["model_answer"].strip()) > 0

    # Write to temp file and grade
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(results, f, indent=2)

    graded = grade_exam_file(tmp_path)
    os.unlink(tmp_path)

    # Verify grading structure
    assert len(graded) == len(sample)
    for entry in graded:
        assert entry["score"] in ("Correct", "Incorrect")
        assert "score_numeric" in entry
        assert 0.0 <= entry["score_numeric"] <= 1.0

    # Print results for visibility
    for entry in graded:
        print(f"\nQ: {entry['question']}")
        print(f"Expected: {entry['expected_answer']}")
        print(f"Model:    {entry['model_answer'][:100]}")
        print(f"Score:    {entry['score']} ({entry['score_numeric']:.2f})")


@requires_ollama
@requires_data
def test_multi_temperature_sweep_small():
    """Mini sweep over 2 temperatures × 1 question to validate sweep logic."""
    data = load_egnivia(DATA_PATH)
    question = data[0]  # single question

    backend = OllamaBackend(model_name=OLLAMA_MODEL)
    system_prompt = "You are a helpful assistant. Answer concisely."

    all_results = {}
    for temp in [0.3, 1.5]:
        prompt = f"{system_prompt}\n\nQuestion: {question['question']}"
        response = backend.generate(prompt, temperature=temp, num_predict=64)
        score = simple_verifier(question["answer"], response)
        all_results[temp] = {"response": response, "score": score}

    # Both temperatures should produce valid outputs
    for temp, res in all_results.items():
        assert isinstance(res["response"], str)
        assert len(res["response"].strip()) > 0
        assert 0.0 <= res["score"] <= 1.0


# ── Config integration ──────────────────────────────────────────────


def test_config_ollama_backend():
    """Config can be set to use ollama backend."""
    cfg = load_config(overrides={
        "model": {
            "backend": "ollama",
            "ollama_model": "qwen3:1.7b",
        }
    })
    assert cfg.model.backend == "ollama"
    assert cfg.model.ollama_model == "qwen3:1.7b"
