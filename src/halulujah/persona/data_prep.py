"""Prepare Blog Authorship Corpus data for per-author LoRA fine-tuning."""

import csv
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)

# 20 topic-controlled probe questions — same for all authors.
# Designed to elicit stylistic variation, not domain knowledge.
PROBE_QUESTIONS = [
    "What's the best way to spend a rainy afternoon?",
    "Describe your ideal morning routine.",
    "What's something most people get wrong about happiness?",
    "Tell me about a time you changed your mind about something important.",
    "What advice would you give to your younger self?",
    "What makes a good friend?",
    "Describe a place that feels like home to you.",
    "What's your take on social media?",
    "What do you think about when you can't sleep?",
    "Describe your favorite meal and why it matters to you.",
    "What's something you're proud of that nobody knows about?",
    "How do you deal with a bad day?",
    "What's the most interesting thing you've learned recently?",
    "Describe a stranger who left an impression on you.",
    "What does success mean to you?",
    "Tell me about something you find beautiful.",
    "What's an unpopular opinion you hold?",
    "How do you decide what matters?",
    "Describe a moment that changed how you see the world.",
    "What would you do with an extra hour every day?",
]


def load_blog_corpus(
    csv_path: str,
    min_posts: int = 200,
    max_authors: int = 5,
    min_post_length: int = 100,
) -> Dict[str, List[str]]:
    """Load Blog Authorship Corpus, return top authors by post count.

    Args:
        csv_path: Path to blogtext.csv.
        min_posts: Minimum posts per author to be eligible.
        max_authors: Number of top authors to select.
        min_post_length: Minimum character length per post.

    Returns:
        Dict mapping author_id to list of post texts.
    """
    logger.info("Loading %s...", csv_path)
    author_posts = defaultdict(list)

    with open(csv_path, "r", encoding="latin-1", errors="replace") as f:
        clean_lines = (line.replace("\x00", "") for line in f)
        reader = csv.DictReader(clean_lines)
        for row in reader:
            author_id = row.get("id", "").strip()
            text = row.get("text", "").strip()
            if author_id and text and len(text) >= min_post_length:
                author_posts[author_id].append(text)

    total = sum(len(v) for v in author_posts.values())
    logger.info("Loaded %d posts from %d authors", total, len(author_posts))

    eligible = {a: posts for a, posts in author_posts.items() if len(posts) >= min_posts}
    logger.info("%d authors with >= %d posts", len(eligible), min_posts)

    top_authors = sorted(eligible.keys(), key=lambda a: len(eligible[a]), reverse=True)[
        :max_authors
    ]
    corpus = {a: eligible[a] for a in top_authors}

    for a in top_authors:
        logger.info("  Author %s: %d posts", a, len(corpus[a]))

    return corpus


def split_train_probe(
    posts: List[str],
    probe_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split an author's posts into train and probe (held-out) sets."""
    import random

    rng = random.Random(seed)
    shuffled = list(posts)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - probe_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def format_blog_for_sft(
    posts: List[str],
    tokenizer,
    author_id: str = "unknown",
) -> Dataset:
    """Format blog posts as SFT training data.

    Each post becomes a chat-template example:
      system: "You are a blogger. Write naturally in your own voice."
      user: "Write a blog post."
      assistant: <actual blog post>

    Args:
        posts: List of blog post texts.
        tokenizer: HuggingFace tokenizer for chat template.
        author_id: Used only for logging.

    Returns:
        HuggingFace Dataset with 'text' field.
    """
    system_prompt = "You are a blogger. Write naturally in your own voice and style."

    texts = []
    for post in posts:
        # Truncate very long posts to fit in context window
        truncated = post[:2000]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Write a blog post."},
            {"role": "assistant", "content": truncated},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

    logger.info("Formatted %d posts for author %s", len(texts), author_id)
    return Dataset.from_dict({"text": texts})


def format_probes_for_generation(
    tokenizer,
    probes: List[str] = None,
) -> List[str]:
    """Format probe questions as chat-template prompts for generation.

    Returns list of tokenizer-ready prompt strings (up to the generation point).
    """
    if probes is None:
        probes = PROBE_QUESTIONS

    system_prompt = "You are a blogger. Write naturally in your own voice and style."
    formatted = []
    for q in probes:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)

    return formatted


def save_author_manifest(
    corpus: Dict[str, List[str]],
    output_dir: str,
) -> str:
    """Save metadata about selected authors for reproducibility."""
    manifest = {
        "num_authors": len(corpus),
        "authors": {
            aid: {"total_posts": len(posts)}
            for aid, posts in corpus.items()
        },
        "probe_questions": PROBE_QUESTIONS,
    }
    path = os.path.join(output_dir, "author_manifest.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest to %s", path)
    return path
