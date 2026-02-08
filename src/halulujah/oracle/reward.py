"""Oracle reward function for RL training."""

import logging

from ..config import OracleConfig
from ..rl.verifier import extract_cot_answer, multi_answer_verifier, simple_verifier
from .retriever import OracleRetriever

logger = logging.getLogger(__name__)


class OracleRewardFunction:
    """Computes reward by comparing model answers against Oracle-retrieved passages.

    Uses simple_verifier to check factual overlap between the Oracle passage
    and the model's generated answer. Applies a small length penalty to
    discourage rambling.
    """

    def __init__(self, retriever: OracleRetriever, cfg: OracleConfig):
        self.retriever = retriever
        self.cfg = cfg

    def compute_reward(self, question: str, model_answer: str) -> float:
        """Compute reward for a (question, model_answer) pair.

        Returns a float clamped to [0, 1].
        """
        oracle_passage = self.retriever.get_oracle_answer(question)
        if not oracle_passage:
            return 0.0

        # Extract final answer from CoT trace (no-op if no "Answer:" marker)
        extracted_answer = extract_cot_answer(model_answer)

        # Base reward from verifier
        reward = simple_verifier(oracle_passage, extracted_answer)

        # Length penalty: penalize answers that are too long
        word_count = len(extracted_answer.split())
        if word_count > self.cfg.max_answer_words:
            excess = word_count - self.cfg.max_answer_words
            penalty = self.cfg.length_penalty_weight * (excess / self.cfg.max_answer_words)
            reward = max(0.0, reward - penalty)

        return max(0.0, min(1.0, reward))


class TemplamaRewardFunction:
    """Reward function for TempLAMA: compare against known answer(s) directly.

    Unlike OracleRewardFunction, this doesn't need a retriever — the ground
    truth answers are available in the dataset entries.  The RL training loop
    passes the question string, which we use to look up the expected answer(s)
    from a pre-built mapping.
    """

    def __init__(self):
        self._answer_map: dict = {}

    def set_answer_map(self, entries: list) -> None:
        """Build a question → answers lookup from dataset entries."""
        for e in entries:
            key = e["question"]
            self._answer_map[key] = e.get("answers_all", [e["answer"]])

    def compute_reward(self, question: str, model_answer: str) -> float:
        """Compute reward for a (question, model_answer) pair."""
        expected = self._answer_map.get(question)
        if not expected:
            return 0.0
        extracted = extract_cot_answer(model_answer)
        return multi_answer_verifier(expected, extracted)
