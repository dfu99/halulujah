"""Cross-domain evaluation: measure hallucination rates across domain pairs."""

import json
import logging
import os
from typing import Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


def generate_and_grade(
    model,
    tokenizer,
    test_entries: List[Dict],
    domain_name: str,
    model_name: str,
    temperature: float = 0.7,
    max_new_tokens: int = 100,
    device: torch.device = None,
) -> List[Dict]:
    """Generate answers and grade against ground truth.

    Args:
        model: Loaded HuggingFace model.
        tokenizer: Corresponding tokenizer.
        test_entries: List of MMLU entries with question, answer, answer_letter.
        domain_name: Which domain these questions come from.
        model_name: Name of the model being tested.
        temperature: Sampling temperature.
        max_new_tokens: Max tokens to generate.
        device: Torch device.

    Returns:
        List of result dicts with question, expected, generated, correct, domain, model.
    """
    if device is None:
        device = next(model.parameters()).device

    system_prompt = "Answer the multiple-choice question. Give only the letter of the correct answer."
    results = []

    for entry in test_entries:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": entry["question"]},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        # apply_chat_template may return a BatchEncoding or a plain tensor
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids.to(device)
        else:
            input_ids = encoded.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                top_p=0.9,
                use_cache=True,
            )

        response = tokenizer.decode(
            outputs[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        # Extract answer letter from response
        predicted_letter = extract_answer_letter(response)
        correct = predicted_letter == entry["answer_letter"]

        results.append({
            "question_domain": entry["domain"],
            "subject": entry["subject"],
            "model": model_name,
            "expected": entry["answer_letter"],
            "predicted": predicted_letter,
            "response": response[:200],
            "correct": correct,
        })

    accuracy = sum(r["correct"] for r in results) / max(len(results), 1)
    logger.info("  %s on %s: %.1f%% (%d/%d)",
                model_name, domain_name, accuracy * 100, sum(r["correct"] for r in results), len(results))

    return results


def extract_answer_letter(response: str) -> str:
    """Extract the answer letter (A/B/C/D) from a model response.

    Handles Qwen3's <think>...</think> output by looking at text after
    the thinking block first.
    """
    import re

    text = response
    # Strip thinking block if present
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    letter = _find_letter(text)
    if letter != "X":
        return letter
    # Fall back to full response
    return _find_letter(response)


def _find_letter(text: str) -> str:
    """Find answer letter in text."""
    import re

    text_upper = text.strip().upper()
    if not text_upper:
        return "X"

    if text_upper[0] in "ABCD" and (len(text_upper) == 1 or not text_upper[1].isalpha()):
        return text_upper[0]

    for letter in "ABCD":
        if f"{letter}." in text_upper or f"{letter})" in text_upper or f"({letter})" in text_upper:
            return letter

    m = re.search(r"ANSWER\s*(?:IS|:)\s*\**\s*([ABCD])\b", text_upper)
    if m:
        return m.group(1)

    m = re.search(r"\*\*([ABCD])\*\*", text_upper)
    if m:
        return m.group(1)

    return "X"


def build_confusion_matrix(
    all_results: List[Dict],
    domains: List[str],
    model_names: List[str],
) -> Dict:
    """Build accuracy confusion matrix: model (trained domain) × question domain.

    Returns dict with:
        - matrix: 2D array [model_idx][question_domain_idx] = accuracy
        - in_domain_accuracy: per model
        - cross_domain_accuracy: per model (avg across non-native domains)
        - accuracy_drop: in_domain - cross_domain per model
    """
    matrix = {}
    for model_name in model_names:
        matrix[model_name] = {}
        for q_domain in domains:
            relevant = [r for r in all_results
                        if r["model"] == model_name and r["question_domain"] == q_domain]
            if relevant:
                acc = sum(r["correct"] for r in relevant) / len(relevant)
            else:
                acc = 0.0
            matrix[model_name][q_domain] = acc

    # Compute in-domain vs cross-domain
    summary = {}
    for model_name in model_names:
        # Extract the domain this model was trained on
        trained_domain = model_name.replace("specialist_", "")
        if trained_domain not in domains:
            # Base model — compute overall average
            all_accs = list(matrix[model_name].values())
            summary[model_name] = {
                "overall_accuracy": float(np.mean(all_accs)),
            }
            continue

        in_domain = matrix[model_name].get(trained_domain, 0.0)
        cross_accs = [acc for d, acc in matrix[model_name].items() if d != trained_domain]
        cross_domain = float(np.mean(cross_accs)) if cross_accs else 0.0

        summary[model_name] = {
            "trained_domain": trained_domain,
            "in_domain_accuracy": in_domain,
            "cross_domain_accuracy": cross_domain,
            "accuracy_drop": in_domain - cross_domain,
        }

    return {
        "matrix": matrix,
        "summary": summary,
        "domains": domains,
        "model_names": model_names,
    }


def save_cross_eval_results(
    confusion: Dict,
    all_results: List[Dict],
    output_dir: str,
) -> str:
    """Save cross-evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "cross_eval_results.json")
    with open(path, "w") as f:
        json.dump({
            "confusion_matrix": confusion,
            "num_results": len(all_results),
        }, f, indent=2)

    details_path = os.path.join(output_dir, "cross_eval_details.json")
    with open(details_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Saved cross-eval results to %s", output_dir)
    return path
