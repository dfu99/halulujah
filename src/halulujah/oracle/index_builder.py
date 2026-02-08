"""Build a FAISS index from 10-K PDF documents for the Oracle."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

from ..config import OracleConfig

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text


def chunk_text_semantic(
    text: str,
    max_chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """Chunk text with semantic awareness (paragraph boundaries) and overlap."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk: List[str] = []
    current_size = 0

    for para in paragraphs:
        para_words = para.split()
        para_size = len(para_words)

        if current_size + para_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_words = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_words + para_words
            current_size = len(current_chunk)
        else:
            current_chunk.extend(para_words)
            current_size += para_size

        while current_size > max_chunk_size:
            chunks.append(" ".join(current_chunk[:max_chunk_size]))
            tail_start = max_chunk_size - overlap if overlap < max_chunk_size else max_chunk_size
            current_chunk = current_chunk[tail_start:]
            current_size = len(current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


class OracleIndex:
    """FAISS index over PDF chunks for Oracle retrieval."""

    def __init__(self, cfg: OracleConfig):
        self.cfg = cfg
        self.embedder = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.corpus: List[str] = []

    def _ensure_embedder(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.cfg.embedding_model)

    def build_from_pdfs(self, pdf_paths: List[str]) -> None:
        """Extract text from PDFs, chunk, embed, and build the FAISS index."""
        self._ensure_embedder()
        self.corpus = []

        for pdf_path in pdf_paths:
            logger.info("Processing PDF: %s", pdf_path)
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text_semantic(
                text,
                max_chunk_size=self.cfg.chunk_size,
                overlap=self.cfg.chunk_overlap,
            )
            self.corpus.extend(chunks)

        if not self.corpus:
            logger.warning("No text extracted from PDFs")
            self.index = faiss.IndexFlatL2(768)
            return

        embeddings = self.embedder.encode(self.corpus, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype(np.float32))
        logger.info("Built FAISS index: %d chunks, dim=%d", len(self.corpus), dim)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve the top-k most relevant chunks for a query.

        Returns a list of dicts with 'text' and 'distance' keys.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        self._ensure_embedder()
        top_k = top_k or self.cfg.top_k
        query_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.corpus):
                results.append({"text": self.corpus[idx], "distance": float(dist)})
        return results

    def save(self, dir_path: str) -> None:
        """Save the index and corpus to disk."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss_index.bin"))
        with open(path / "corpus.json", "w") as f:
            json.dump(self.corpus, f)

    def load(self, dir_path: str) -> None:
        """Load a previously saved index and corpus."""
        path = Path(dir_path)
        self.index = faiss.read_index(str(path / "faiss_index.bin"))
        with open(path / "corpus.json", "r") as f:
            self.corpus = json.load(f)
        logger.info("Loaded FAISS index: %d chunks", len(self.corpus))
