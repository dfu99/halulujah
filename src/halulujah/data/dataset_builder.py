"""Build HuggingFace Datasets for SFT and RL from Q&A entries."""

from typing import Dict, List

from datasets import Dataset

from .chat_formatter import format_as_chat_messages, format_as_query_only


def build_sft_dataset(entries: List[Dict], tokenizer) -> Dataset:
    """Build an SFT dataset with a 'text' field containing the chat-template-applied text."""
    texts = []
    for entry in entries:
        messages = format_as_chat_messages(entry, include_context=True)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def build_rl_dataset(entries: List[Dict]) -> Dataset:
    """Build an RL dataset with question, answer, and year fields."""
    questions = [format_as_query_only(e) for e in entries]
    answers = [e["answer"] for e in entries]
    years = [e.get("year") for e in entries]
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "year": years,
    })
