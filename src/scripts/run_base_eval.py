"""Evaluate the base Qwen3-1.7B model (no LoRA) across all 10 MMLU domains.

This provides the missing baseline: how does the untrained model compare
to domain specialists on the same 50-question test sets?
"""

import argparse
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from halulujah.domain.data_prep import (
    DOMAIN_SUBJECTS_EXTENDED,
    DOMAIN_SUBJECTS,
    load_mmlu_domain,
    split_train_test,
)
from halulujah.domain.cross_eval import generate_and_grade

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Base model evaluation across MMLU domains")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", default="results/base_eval")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--extended", action="store_true", help="Use 10-domain extended set")
    args = parser.parse_args()

    # Set domain scope
    if args.extended:
        from halulujah.domain import data_prep
        data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    domains = list(DOMAIN_SUBJECTS_EXTENDED.keys()) if args.extended else list(DOMAIN_SUBJECTS.keys())
    logger.info("Evaluating base model across %d domains: %s", len(domains), domains)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading %s...", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir,
        device_map="cuda",
    )
    device = next(model.parameters()).device
    logger.info("Model loaded on %s", device)

    # Load test sets (same split as specialist experiments)
    test_sets = {}
    for domain in domains:
        from halulujah.domain import data_prep
        data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED
        all_entries = load_mmlu_domain(domain, split="test", cache_dir=args.cache_dir)
        _, test = split_train_test(all_entries, test_size=args.n_questions, seed=42)
        test_sets[domain] = test
        logger.info("Domain %s: %d test questions", domain, len(test))

    # Evaluate base model on each domain
    all_results = []
    domain_accuracies = {}
    start = time.time()

    for domain in domains:
        logger.info("Evaluating base model on %s (%d questions)...", domain, len(test_sets[domain]))
        results = generate_and_grade(
            model, tokenizer, test_sets[domain],
            domain_name=domain, model_name="base",
            temperature=0.7, max_new_tokens=100, device=device,
        )
        all_results.extend(results)
        acc = sum(r["correct"] for r in results) / max(len(results), 1)
        domain_accuracies[domain] = acc
        logger.info("  %s: %.1f%% (%d/%d)", domain, acc * 100,
                     sum(r["correct"] for r in results), len(results))

    elapsed = time.time() - start
    mean_acc = sum(domain_accuracies.values()) / len(domain_accuracies)
    logger.info("=== BASE MODEL RESULTS ===")
    logger.info("Mean accuracy: %.1f%%", mean_acc * 100)
    logger.info("Time: %.1f minutes", elapsed / 60)
    for d in sorted(domain_accuracies, key=domain_accuracies.get, reverse=True):
        logger.info("  %s: %.1f%%", d, domain_accuracies[d] * 100)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "model": args.model_name,
        "type": "base_model_no_lora",
        "n_questions_per_domain": args.n_questions,
        "domain_accuracies": domain_accuracies,
        "mean_accuracy": mean_acc,
        "elapsed_seconds": elapsed,
        "domains": domains,
    }
    out_path = os.path.join(args.output_dir, "base_solo_eval.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved results to %s", out_path)

    # Save detailed results
    details_path = os.path.join(args.output_dir, "base_solo_details.json")
    with open(details_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved details to %s", details_path)

    # Also save test sets for reproducibility
    test_path = os.path.join(args.output_dir, "test_sets.json")
    serializable_tests = {d: entries for d, entries in test_sets.items()}
    with open(test_path, "w") as f:
        json.dump(serializable_tests, f, indent=2)
    logger.info("Saved test sets to %s", test_path)


if __name__ == "__main__":
    main()
