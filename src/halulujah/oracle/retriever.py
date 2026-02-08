"""Oracle retriever: retrieve relevant passages from 10-K PDFs."""

import logging
from typing import List

from ..config import OracleConfig
from .index_builder import OracleIndex

logger = logging.getLogger(__name__)


class OracleRetriever:
    """Retrieves the most relevant passage from the FAISS index for a given question.

    Applies EGNIVIA→NVIDIA replacement for retrieval (better semantic match against
    original PDF text), following the rag_simple.py pattern.
    """

    def __init__(self, oracle_index: OracleIndex):
        self.oracle_index = oracle_index

    def get_oracle_answer(self, question: str, top_k: int = None) -> str:
        """Retrieve the best-matching passage for a question.

        Since PDFs contain full 10-K content (not structured Q&A), this returns
        the best-matching passage rather than a direct answer.
        """
        # Unmask for retrieval: EGNIVIA→NVIDIA gives better semantic matches
        query = question.replace("EGNIVIA", "NVIDIA")
        results = self.oracle_index.retrieve(query, top_k=top_k)

        if not results:
            return ""

        # Re-mask the retrieved passage: NVIDIA→EGNIVIA
        passage = results[0]["text"]
        passage = passage.replace("NVIDIA", "EGNIVIA").replace("NVDA", "EGNIVIA")
        return passage

    def get_context_passages(self, question: str, top_k: int = None) -> List[str]:
        """Retrieve multiple passages, each re-masked."""
        query = question.replace("EGNIVIA", "NVIDIA")
        results = self.oracle_index.retrieve(query, top_k=top_k)
        passages = []
        for r in results:
            text = r["text"].replace("NVIDIA", "EGNIVIA").replace("NVDA", "EGNIVIA")
            passages.append(text)
        return passages
