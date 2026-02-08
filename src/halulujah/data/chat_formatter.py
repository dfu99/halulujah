"""Format Q&A entries into chat messages for SFT and RL."""

from typing import Dict, List


def format_as_chat_messages(
    entry: Dict,
    include_context: bool = True,
) -> List[Dict[str, str]]:
    """Format an entry as chat messages for SFT training.

    Returns a list of message dicts with 'role' and 'content' keys.
    """
    if include_context and entry.get("context"):
        user_content = f"{entry['context']}\n\nQuestion: {entry['question']}"
    else:
        user_content = entry["question"]

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": entry["answer"]},
    ]


def format_as_query_only(entry: Dict) -> str:
    """Format an entry as a bare question string for RL (no context)."""
    return entry["question"]


def format_cot_system_prompt() -> str:
    """Return the chain-of-thought system prompt for eval and RL."""
    return (
        "You are a concise financial QA assistant. "
        "Think step by step, then give your final answer on a new line "
        "starting with 'Answer:'."
    )


def format_cot_user_prompt(question: str, context: str) -> str:
    """Build a CoT user prompt with retrieved context.

    Structure: Context block → Question → step-by-step instruction.
    """
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Think step by step, then provide your final answer on a line "
        "starting with 'Answer:'."
    )
