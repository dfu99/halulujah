"""Tests for Oracle index builder and retriever (unit-level, no GPU required)."""

import pytest

from halulujah.config import OracleConfig

faiss = pytest.importorskip("faiss", reason="faiss not installed")

from halulujah.oracle.index_builder import chunk_text_semantic


def test_chunk_text_semantic_basic():
    text = " ".join(["word"] * 1000)
    chunks = chunk_text_semantic(text, max_chunk_size=200, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 200


def test_chunk_text_semantic_with_paragraphs():
    para1 = " ".join(["alpha"] * 100)
    para2 = " ".join(["beta"] * 100)
    para3 = " ".join(["gamma"] * 100)
    text = f"{para1}\n\n{para2}\n\n{para3}"
    chunks = chunk_text_semantic(text, max_chunk_size=150, overlap=10)
    assert len(chunks) >= 2


def test_chunk_text_semantic_empty():
    assert chunk_text_semantic("") == []


def test_chunk_text_semantic_small():
    text = "Short text."
    chunks = chunk_text_semantic(text, max_chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_oracle_config_defaults():
    cfg = OracleConfig()
    assert cfg.chunk_size == 500
    assert cfg.top_k == 5
    assert cfg.embedding_model == "sentence-transformers/all-mpnet-base-v2"
