"""Train mediator LoRA adapters and evaluate mediated collaboration.

Act 3 of the combined paper: can a model trained on BOTH domains bridge
specialists without the "confidently wrong" damage of cross-domain helpers?

For each domain pair (A, B):
  1. Train mediator adapter on 50/50 mixed MMLU data from A and B
  2. Run specialist A + mediator(A,B) collaboration on A's questions
  3. Compare C2W/W2C rates against specialist+specialist and specialist+base

Outputs:
  mediator_results.json  — full collaboration data
  mediator_summary.json  — accuracy deltas vs all conditions
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from halulujah.domain.data_prep import (
    DOMAIN_SUBJECTS_EXTENDED,
    load_mmlu_domain,
    load_mmlu_mediator,
    format_mediator_for_sft,
    split_train_test,
)
from halulujah.domain.collab_eval import (
    collab_reasoning_scoped,
    solo_reasoning,
    build_collab_summary,
)
from halulujah.domain.cross_eval import extract_answer_letter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Default pairs: span damage spectrum from catastrophic to mild
DEFAULT_PAIRS = [
    ("medicine", "physics"),      # -54pp catastrophic
    ("medicine", "biology"),      # -50pp catastrophic, related domains
    ("chemistry", "math"),        # -38pp severe, non-medicine
    ("physics", "philosophy"),    # -34pp moderate, very different
    ("biology", "math"),          # -18pp mild
]


def train_mediator(
    domain_a: str,
    domain_b: str,
    model_name: str,
    output_dir: str,
    cache_dir: str = None,
    num_epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 4,
):
    """Train a mediator LoRA adapter on mixed data from two domains."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    pair_name = f"{domain_a}_{domain_b}"
    adapter_dir = os.path.join(output_dir, f"mediator_{pair_name}")

    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        logger.info("Mediator %s already trained, skipping", pair_name)
        return adapter_dir

    logger.info("=== Training mediator: %s + %s ===", domain_a, domain_b)

    # Load mixed data
    all_entries = load_mmlu_mediator(domain_a, domain_b, split="test", cache_dir=cache_dir)
    # Reserve test questions for each domain separately (don't use them for training)
    entries_a = load_mmlu_domain(domain_a, split="test", cache_dir=cache_dir)
    entries_b = load_mmlu_domain(domain_b, split="test", cache_dir=cache_dir)
    _, test_a = split_train_test(entries_a, test_size=50, seed=42)
    _, test_b = split_train_test(entries_b, test_size=50, seed=42)

    # Remove test questions from mediator training data
    test_questions = {e["question"] for e in test_a + test_b}
    train_entries = [e for e in all_entries if e["question"] not in test_questions]
    logger.info("Mediator training data: %d entries (after removing %d test questions)",
                len(train_entries), len(test_questions))

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    )

    # LoRA config — same as domain specialists for fair comparison
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Format training data
    train_dataset = format_mediator_for_sft(train_entries, tokenizer, domain_a, domain_b)

    # Train
    training_args = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    # Save adapter only
    model.save_pretrained(adapter_dir)
    logger.info("Mediator adapter saved to %s", adapter_dir)

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return adapter_dir


def evaluate_mediated_collab(
    domain_a: str,
    domain_b: str,
    model_name: str,
    specialist_adapter_dir: str,
    mediator_adapter_dir: str,
    test_entries: list,
    n_questions: int = 50,
    n_rounds: int = 3,
    protocol: str = "full-cot",
    cache_dir: str = None,
):
    """Run specialist + mediator collaboration and return results."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )

    # Load specialist A
    logger.info("Loading specialist: %s", domain_a)
    specialist = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    ).to(device)
    adapter_path = os.path.join(specialist_adapter_dir, f"adapter_{domain_a}")
    specialist = PeftModel.from_pretrained(specialist, adapter_path)
    specialist.eval()

    # Load mediator — try both orderings since mediator(A,B) == mediator(B,A)
    pair_name = f"{domain_a}_{domain_b}"
    mediator_path = os.path.join(mediator_adapter_dir, f"mediator_{pair_name}")
    if not os.path.exists(mediator_path):
        pair_name = f"{domain_b}_{domain_a}"
        mediator_path = os.path.join(mediator_adapter_dir, f"mediator_{pair_name}")
    logger.info("Loading mediator: %s", pair_name)
    mediator = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    ).to(device)
    mediator = PeftModel.from_pretrained(mediator, mediator_path)
    mediator.eval()

    # Log VRAM
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        logger.info("VRAM after loading both models: %.1f GB", used)

    questions = test_entries[:n_questions]
    mediator_label = f"mediator_{domain_a}_{domain_b}"

    # Solo baseline
    solo_results = []
    logger.info("Solo baseline: %s (%d questions)", domain_a, len(questions))
    for entry in questions:
        final, chain = solo_reasoning(
            specialist, tokenizer, entry["question"], domain_a,
            n_rounds=n_rounds, device=device,
        )
        predicted = extract_answer_letter(final)
        solo_results.append({
            "domain": domain_a,
            "model": domain_a,
            "mode": "solo",
            "expected": entry["answer_letter"],
            "predicted": predicted,
            "correct": predicted == entry["answer_letter"],
            "final_response": final[:200],
        })

    solo_acc = sum(r["correct"] for r in solo_results) / max(len(questions), 1)
    logger.info("  Solo %s: %.1f%%", domain_a, solo_acc * 100)

    # Mediated collaboration
    collab_results = []
    logger.info("Collab: %s + mediator(%s,%s) (%d questions, %s)",
                domain_a, domain_a, domain_b, len(questions), protocol)
    for entry in questions:
        final, chain, pre_collab = collab_reasoning_scoped(
            specialist, mediator, tokenizer, entry["question"],
            domain_a, mediator_label, n_rounds=n_rounds,
            device=device, protocol=protocol,
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
            "question_domain": domain_a,
            "agent_a": domain_a,
            "agent_b": mediator_label,
            "mode": "collab",
            "protocol": protocol,
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

    collab_acc = sum(r["correct"] for r in collab_results) / max(len(collab_results), 1)
    delta = collab_acc - solo_acc
    logger.info("  Mediated %s+mediator(%s,%s): %.1f%% (delta: %+.1f%%)",
                domain_a, domain_a, domain_b, collab_acc * 100, delta * 100)

    # Cleanup
    del specialist, mediator
    gc.collect()
    torch.cuda.empty_cache()

    return solo_results, collab_results


def main():
    parser = argparse.ArgumentParser(description="Mediator collaboration experiment (Act 3)")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--specialist-adapter-dir", default="adapters",
                        help="Directory containing specialist adapter_X/ subdirs")
    parser.add_argument("--mediator-output-dir", default="mediators",
                        help="Directory to save trained mediator adapters")
    parser.add_argument("--results-dir", default="results/mediator_collab")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--protocol", default="full-cot",
                        choices=["full-cot", "answer-only", "structured"])
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="Domain pairs as 'a,b' strings. Default: 5 pairs spanning damage spectrum")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, use existing mediator adapters")
    args = parser.parse_args()

    # Use extended domains
    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    # Parse pairs
    if args.pairs:
        pairs = [tuple(p.split(",")) for p in args.pairs]
    else:
        pairs = DEFAULT_PAIRS

    logger.info("Mediator experiment: %d pairs", len(pairs))
    for a, b in pairs:
        logger.info("  %s + %s", a, b)

    start = time.time()

    # Phase 1: Train mediator adapters
    if not args.skip_training:
        for domain_a, domain_b in pairs:
            train_mediator(
                domain_a, domain_b,
                model_name=args.model_name,
                output_dir=args.mediator_output_dir,
                cache_dir=args.cache_dir,
            )

    # Phase 2: Evaluate mediated collaboration
    # Load test sets for all involved domains
    involved_domains = set()
    for a, b in pairs:
        involved_domains.add(a)
        involved_domains.add(b)

    test_sets = {}
    for domain in involved_domains:
        all_entries = load_mmlu_domain(domain, split="test", cache_dir=args.cache_dir)
        _, test = split_train_test(all_entries, test_size=args.n_questions, seed=42)
        test_sets[domain] = test
        logger.info("Test set %s: %d questions", domain, len(test))

    all_solo = []
    all_collab = []

    for domain_a, domain_b in pairs:
        logger.info("=" * 60)
        logger.info("Evaluating: %s specialist + mediator(%s,%s)", domain_a, domain_a, domain_b)

        solo, collab = evaluate_mediated_collab(
            domain_a, domain_b,
            model_name=args.model_name,
            specialist_adapter_dir=args.specialist_adapter_dir,
            mediator_adapter_dir=args.mediator_output_dir,
            test_entries=test_sets[domain_a],
            n_questions=args.n_questions,
            n_rounds=args.n_rounds,
            protocol=args.protocol,
            cache_dir=args.cache_dir,
        )
        all_solo.extend(solo)
        all_collab.extend(collab)

        # Also evaluate the reverse: domain_b specialist + mediator on domain_b questions
        logger.info("Evaluating: %s specialist + mediator(%s,%s)", domain_b, domain_a, domain_b)
        solo_b, collab_b = evaluate_mediated_collab(
            domain_b, domain_a,
            model_name=args.model_name,
            specialist_adapter_dir=args.specialist_adapter_dir,
            mediator_adapter_dir=args.mediator_output_dir,
            test_entries=test_sets[domain_b],
            n_questions=args.n_questions,
            n_rounds=args.n_rounds,
            protocol=args.protocol,
            cache_dir=args.cache_dir,
        )
        all_solo.extend(solo_b)
        all_collab.extend(collab_b)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("Total time: %.1f minutes", elapsed / 60)

    # Build summary
    domains_tested = list(involved_domains)
    summary = build_collab_summary(all_solo, all_collab, domains_tested)

    results = {
        "solo_results": all_solo,
        "collab_results": all_collab,
        "summary": summary,
        "config": {
            "n_rounds": args.n_rounds,
            "n_questions": args.n_questions,
            "domains": domains_tested,
            "pairs": [list(p) for p in pairs],
            "protocol": args.protocol,
            "helper_type": "mediator",
        },
    }

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "mediator_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    summary_path = os.path.join(args.results_dir, "mediator_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Results saved to %s", args.results_dir)

    # Print key metrics
    print("\n" + "=" * 60)
    print("MEDIATOR EXPERIMENT RESULTS")
    print("=" * 60)
    for domain_a, domain_b in pairs:
        solo_acc = summary["solo"].get(domain_a, 0) * 100
        pair_key = f"{domain_a}+mediator_{domain_a}_{domain_b}"
        if pair_key in summary["collab"]:
            med_acc = summary["collab"][pair_key]["accuracy"] * 100
            delta = summary["collab"][pair_key]["delta"] * 100
            print(f"  {domain_a} + mediator({domain_a},{domain_b}): "
                  f"solo {solo_acc:.0f}% → mediated {med_acc:.0f}% ({delta:+.0f}pp)")

    if summary.get("convergence"):
        conv = summary["convergence"]
        print(f"\nConvergence: C2W={conv['correct_to_wrong_rate']*100:.1f}%, "
              f"W2C={conv['wrong_to_correct_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
