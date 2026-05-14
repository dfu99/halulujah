"""Run base-model-as-helper collaboration experiment.

For each domain specialist, pair it with the raw base model (no LoRA)
and run full-cot collaboration on that specialist's domain questions.
This is the missing control: is cross-domain damage from *conflicting
expertise* or just from *any noisy partner*?

Outputs:
  base_collab_results.json  — full collaboration data
  base_collab_summary.json  — accuracy deltas vs specialist solo
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from halulujah.domain.data_prep import (
    DOMAIN_SUBJECTS_EXTENDED,
    load_mmlu_domain,
    split_train_test,
)
from halulujah.domain.collab_eval import (
    collab_reasoning_scoped,
    solo_reasoning,
    build_collab_summary,
    save_collab_results,
)
from halulujah.domain.cross_eval import extract_answer_letter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Base-as-helper collaboration experiment")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-dir", default="adapters")
    parser.add_argument("--output-dir", default="results/base_collab")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--protocol", default="full-cot",
                        choices=["full-cot", "answer-only", "structured"])
    args = parser.parse_args()

    # Use extended domains
    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED
    domains = list(DOMAIN_SUBJECTS_EXTENDED.keys())

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir
    )

    # Check which adapters are available
    available_domains = []
    for domain in domains:
        adapter_path = os.path.join(args.adapter_dir, f"adapter_{domain}")
        if os.path.exists(adapter_path):
            available_domains.append(domain)
        else:
            logger.warning("Adapter not found for %s at %s — skipping", domain, adapter_path)

    logger.info("Available domains: %s", available_domains)

    # Load test sets
    test_sets = {}
    for domain in available_domains:
        all_entries = load_mmlu_domain(domain, split="test", cache_dir=args.cache_dir)
        _, test = split_train_test(all_entries, test_size=args.n_questions, seed=42)
        test_sets[domain] = test
        logger.info("Domain %s: %d test questions", domain, len(test))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solo_results = []
    collab_results = []
    start = time.time()

    for domain in available_domains:
        logger.info("=" * 60)
        logger.info("Processing domain: %s", domain)

        # Load specialist model
        logger.info("Loading specialist: %s", domain)
        specialist = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, cache_dir=args.cache_dir,
        ).to(device)
        adapter_path = os.path.join(args.adapter_dir, f"adapter_{domain}")
        specialist = PeftModel.from_pretrained(specialist, adapter_path)
        specialist.eval()

        # Load fresh base model
        logger.info("Loading base model (no LoRA)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, cache_dir=args.cache_dir,
        ).to(device)
        base_model.eval()

        # Check VRAM
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        logger.info("VRAM after loading both models: %s", result.stdout.strip())

        questions = test_sets[domain][:args.n_questions]

        # Solo baseline for this specialist
        logger.info("Solo baseline: %s (%d questions)", domain, len(questions))
        for entry in questions:
            final, chain = solo_reasoning(
                specialist, tokenizer, entry["question"], domain,
                n_rounds=args.n_rounds, device=device,
            )
            predicted = extract_answer_letter(final)
            solo_results.append({
                "domain": domain,
                "model": domain,
                "mode": "solo",
                "expected": entry["answer_letter"],
                "predicted": predicted,
                "correct": predicted == entry["answer_letter"],
                "final_response": final[:200],
            })

        solo_acc = sum(r["correct"] for r in solo_results if r["domain"] == domain) / max(len(questions), 1)
        logger.info("  Solo %s: %.1f%%", domain, solo_acc * 100)

        # Collaboration: specialist + base model
        logger.info("Collab: %s specialist + base helper (%d questions, %s protocol)",
                     domain, len(questions), args.protocol)
        for entry in questions:
            final, chain, pre_collab = collab_reasoning_scoped(
                specialist, base_model, tokenizer, entry["question"],
                domain, "generalist", n_rounds=args.n_rounds,
                device=device, protocol=args.protocol,
            )
            predicted = extract_answer_letter(final)
            expected = entry["answer_letter"]
            pre_a = pre_collab["agent_a"]["answer"]

            a_switched = pre_a != predicted
            a_was_right = pre_a == expected
            post_right = predicted == expected
            if a_switched and a_was_right and not post_right:
                switch_type = "correct_to_wrong"
            elif a_switched and not a_was_right and post_right:
                switch_type = "wrong_to_correct"
            elif a_switched:
                switch_type = "switched_other"
            else:
                switch_type = "held"

            collab_results.append({
                "question_domain": domain,
                "agent_a": domain,
                "agent_b": "base",
                "mode": "collab",
                "protocol": args.protocol,
                "expected": expected,
                "predicted": predicted,
                "correct": post_right,
                "pre_collab_a": pre_a,
                "pre_collab_b": pre_collab["agent_b"]["answer"],
                "pre_collab_a_correct": a_was_right,
                "pre_collab_b_correct": pre_collab["agent_b"]["answer"] == expected,
                "agent_a_switched": a_switched,
                "switch_type": switch_type,
                "final_response": final[:200],
                "chain": [{"agent": s["agent"], "thought": s["thought"][:100],
                           "shared": s.get("shared", s["thought"])[:100]} for s in chain],
            })

        pair_results = [r for r in collab_results if r["agent_a"] == domain]
        collab_acc = sum(r["correct"] for r in pair_results) / max(len(pair_results), 1)
        delta = collab_acc - solo_acc
        logger.info("  Collab %s+base: %.1f%% (delta: %+.1f%%)", domain, collab_acc * 100, delta * 100)

        # Free models for next domain
        del specialist, base_model
        torch.cuda.empty_cache()
        logger.info("Models unloaded for %s", domain)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("Total time: %.1f minutes", elapsed / 60)

    # Build summary
    summary = build_collab_summary(solo_results, collab_results, available_domains)

    results = {
        "solo_results": solo_results,
        "collab_results": collab_results,
        "summary": summary,
        "config": {
            "n_rounds": args.n_rounds,
            "n_questions": args.n_questions,
            "domains": available_domains,
            "protocol": args.protocol,
            "helper": "base_model_no_lora",
        },
    }

    save_collab_results(results, args.output_dir)
    logger.info("DONE. Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
