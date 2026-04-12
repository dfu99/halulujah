"""Load MMLU domain subsets and format for per-domain LoRA fine-tuning."""

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

# Domain definitions: map domain name to MMLU subject subsets
# Core 3 domains (original experiment)
DOMAIN_SUBJECTS_CORE = {
    "physics": [
        "college_physics",
        "high_school_physics",
        "astronomy",
        "conceptual_physics",
    ],
    "law": [
        "professional_law",
        "jurisprudence",
        "international_law",
    ],
    "biology": [
        "college_biology",
        "high_school_biology",
        "anatomy",
        "clinical_knowledge",
    ],
}

# Extended 10 domains (for scaling experiments / Google PI proposal)
DOMAIN_SUBJECTS_EXTENDED = {
    **DOMAIN_SUBJECTS_CORE,
    "computer_science": [
        "college_computer_science",
        "high_school_computer_science",
        "machine_learning",
    ],
    "history": [
        "high_school_world_history",
        "high_school_us_history",
        "high_school_european_history",
    ],
    "math": [
        "college_mathematics",
        "high_school_mathematics",
        "abstract_algebra",
        "elementary_mathematics",
    ],
    "chemistry": [
        "college_chemistry",
        "high_school_chemistry",
    ],
    "economics": [
        "high_school_microeconomics",
        "high_school_macroeconomics",
        "econometrics",
    ],
    "philosophy": [
        "philosophy",
        "moral_scenarios",
        "logical_fallacies",
    ],
    "medicine": [
        "college_medicine",
        "clinical_knowledge",
        "medical_genetics",
        "professional_medicine",
    ],
}

# Default to core; scripts can override with --extended
DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_CORE

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_mmlu_domain(
    domain: str,
    split: str = "test",
    max_per_subject: int = None,
    cache_dir: str = None,
) -> List[Dict]:
    """Load MMLU questions for a specific domain.

    Args:
        domain: Domain name (physics, law, biology).
        split: Dataset split to use.
        max_per_subject: Max examples per MMLU subject (for dev).
        cache_dir: HuggingFace cache directory.

    Returns:
        List of dicts with question, choices, answer, subject, domain.
    """
    if domain not in DOMAIN_SUBJECTS:
        raise ValueError(f"Unknown domain: {domain}. Choose from {list(DOMAIN_SUBJECTS)}")

    subjects = DOMAIN_SUBJECTS[domain]
    entries = []

    for subject in subjects:
        logger.info("Loading MMLU/%s...", subject)
        ds = load_dataset("cais/mmlu", subject, split=split, cache_dir=cache_dir)

        for i, row in enumerate(ds):
            if max_per_subject and i >= max_per_subject:
                break

            choices = row["choices"]
            answer_idx = row["answer"]
            answer_letter = ANSWER_MAP[answer_idx]

            # Format as multiple-choice question
            question_text = row["question"]
            choices_text = "\n".join(
                f"{ANSWER_MAP[j]}. {c}" for j, c in enumerate(choices)
            )

            entries.append({
                "question": f"{question_text}\n\n{choices_text}",
                "answer": f"{answer_letter}. {choices[answer_idx]}",
                "answer_letter": answer_letter,
                "subject": subject,
                "domain": domain,
            })

    logger.info("Loaded %d entries for domain '%s' from %d subjects",
                len(entries), domain, len(subjects))
    return entries


def load_all_domains(
    split: str = "test",
    max_per_subject: int = None,
    cache_dir: str = None,
) -> Dict[str, List[Dict]]:
    """Load all domain datasets."""
    return {
        domain: load_mmlu_domain(domain, split, max_per_subject, cache_dir)
        for domain in DOMAIN_SUBJECTS
    }


def split_train_test(
    entries: List[Dict],
    test_size: int = 50,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Split domain entries into training and test sets.

    Reserves test_size examples for cross-domain evaluation.
    """
    import random
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    return shuffled[test_size:], shuffled[:test_size]


def format_domain_for_sft(
    entries: List[Dict],
    tokenizer,
    domain: str = "unknown",
) -> Dataset:
    """Format MMLU entries as SFT training data.

    Each entry becomes:
      system: "You are a {domain} expert. Answer accurately."
      user: question + choices
      assistant: answer
    """
    system_prompt = f"You are a {domain} expert. Answer the question accurately and concisely."

    texts = []
    for entry in entries:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": entry["question"]},
            {"role": "assistant", "content": entry["answer"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

    logger.info("Formatted %d entries for domain '%s'", len(texts), domain)
    return Dataset.from_dict({"text": texts})


def load_mmlu_mediator(
    domain_a: str,
    domain_b: str,
    split: str = "test",
    max_per_domain: int = None,
    ratio_a: float = 0.5,
    cache_dir: str = None,
) -> List[Dict]:
    """Load mixed MMLU data from two domains for mediator training.

    Args:
        ratio_a: Fraction of data from domain_a (0.0 to 1.0). Default 0.5.
                 E.g., ratio_a=0.8 means 80% domain_a, 20% domain_b.

    Returns entries from both domains. Each entry retains its original
    domain label so we can verify balance.
    """
    entries_a = load_mmlu_domain(domain_a, split=split, cache_dir=cache_dir)
    entries_b = load_mmlu_domain(domain_b, split=split, cache_dir=cache_dir)

    if max_per_domain:
        entries_a = entries_a[:max_per_domain]
        entries_b = entries_b[:max_per_domain]

    import random
    rng = random.Random(42)
    rng.shuffle(entries_a)
    rng.shuffle(entries_b)

    # Total pool size constrained by smaller domain
    total = 2 * min(len(entries_a), len(entries_b))
    n_a = int(total * ratio_a)
    n_b = total - n_a
    # Clamp to available
    n_a = min(n_a, len(entries_a))
    n_b = min(n_b, len(entries_b))

    mixed = entries_a[:n_a] + entries_b[:n_b]
    rng.shuffle(mixed)

    logger.info("Loaded %d mediator entries (%d %s + %d %s, ratio %.0f/%.0f)",
                len(mixed), n_a, domain_a, n_b, domain_b,
                ratio_a * 100, (1 - ratio_a) * 100)
    return mixed


def format_mediator_for_sft(
    entries: List[Dict],
    tokenizer,
    domain_a: str,
    domain_b: str,
) -> Dataset:
    """Format mixed-domain entries for mediator SFT training.

    The mediator's system prompt declares expertise in both domains,
    positioning it as a bridge between specialists.
    """
    system_prompt = (
        f"You are an expert in both {domain_a} and {domain_b}. "
        f"Answer the question accurately and concisely."
    )

    texts = []
    for entry in entries:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": entry["question"]},
            {"role": "assistant", "content": entry["answer"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

    logger.info("Formatted %d mediator entries for %s+%s", len(texts), domain_a, domain_b)
    return Dataset.from_dict({"text": texts})


def save_domain_manifest(
    domain_data: Dict[str, List[Dict]],
    test_sets: Dict[str, List[Dict]],
    output_dir: str,
) -> str:
    """Save metadata about domain datasets for reproducibility."""
    manifest = {
        "domains": {
            domain: {
                "train_size": len(entries),
                "subjects": list({e["subject"] for e in entries}),
            }
            for domain, entries in domain_data.items()
        },
        "test_sets": {
            domain: {
                "size": len(entries),
                "subjects": list({e["subject"] for e in entries}),
            }
            for domain, entries in test_sets.items()
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "domain_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved domain manifest to %s", path)
    return path
