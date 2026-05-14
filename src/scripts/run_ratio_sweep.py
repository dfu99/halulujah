"""Sweep mediator mix ratios to test if 50/50 is uniquely bad.

Trains mediator LoRA adapters at different domain-A/domain-B ratios
(e.g., 80/20, 60/40, 50/50, 40/60, 20/80) and evaluates collaboration.

Key question: does the mediator work better when it's mostly trained on
the PRIMARY domain (the specialist's domain) vs balanced?

If 90/10 (nearly pure domain-A) works as well as same+same, and 50/50
is worst, then mixed training creates confusion rather than bridging.
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
)
from halulujah.domain.cross_eval import extract_answer_letter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Test on the most dramatic pair
DEFAULT_PAIR = ("medicine", "physics")
DEFAULT_RATIOS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def train_ratio_mediator(
    domain_a, domain_b, ratio_a, model_name, output_dir,
    cache_dir=None, num_epochs=3, lr=5e-5, batch_size=4,
):
    """Train a mediator LoRA at a specific mix ratio."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    ratio_tag = f"{int(ratio_a * 100)}_{int((1 - ratio_a) * 100)}"
    adapter_dir = os.path.join(output_dir, f"mediator_{domain_a}_{domain_b}_r{ratio_tag}")

    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        logger.info("Ratio %s mediator already trained, skipping", ratio_tag)
        return adapter_dir

    logger.info("=== Training mediator %s+%s ratio %s ===", domain_a, domain_b, ratio_tag)

    # Load mixed data at specified ratio
    all_entries = load_mmlu_mediator(
        domain_a, domain_b, split="test", ratio_a=ratio_a, cache_dir=cache_dir
    )

    # Remove test questions
    entries_a = load_mmlu_domain(domain_a, split="test", cache_dir=cache_dir)
    entries_b = load_mmlu_domain(domain_b, split="test", cache_dir=cache_dir)
    _, test_a = split_train_test(entries_a, test_size=50, seed=42)
    _, test_b = split_train_test(entries_b, test_size=50, seed=42)
    test_questions = {e["question"] for e in test_a + test_b}
    train_entries = [e for e in all_entries if e["question"] not in test_questions]

    logger.info("Training data: %d entries (ratio %s)", len(train_entries), ratio_tag)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    train_dataset = format_mediator_for_sft(train_entries, tokenizer, domain_a, domain_b)

    training_args = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        max_length=512,
    )

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(adapter_dir)
    logger.info("Saved ratio %s mediator to %s", ratio_tag, adapter_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir


def evaluate_ratio_mediator(
    domain_a, domain_b, ratio_a, model_name, specialist_adapter_dir,
    mediator_adapter_dir, test_entries, n_questions=50, n_rounds=3,
    cache_dir=None,
):
    """Evaluate specialist + ratio-mediator collaboration."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratio_tag = f"{int(ratio_a * 100)}_{int((1 - ratio_a) * 100)}"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )

    # Load specialist
    specialist = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    ).to(device)
    adapter_path = os.path.join(specialist_adapter_dir, f"adapter_{domain_a}")
    specialist = PeftModel.from_pretrained(specialist, adapter_path)
    specialist.eval()

    # Load ratio mediator
    mediator_path = os.path.join(
        mediator_adapter_dir, f"mediator_{domain_a}_{domain_b}_r{ratio_tag}"
    )
    mediator = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    ).to(device)
    mediator = PeftModel.from_pretrained(mediator, mediator_path)
    mediator.eval()

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        logger.info("VRAM with both models: %.1f GB", used)

    questions = test_entries[:n_questions]
    mediator_label = f"mediator_{domain_a}_{domain_b}_r{ratio_tag}"

    # Solo baseline
    solo_correct = 0
    for entry in questions:
        final, _ = solo_reasoning(
            specialist, tokenizer, entry["question"], domain_a,
            n_rounds=n_rounds, device=device,
        )
        if extract_answer_letter(final) == entry["answer_letter"]:
            solo_correct += 1
    solo_acc = solo_correct / len(questions)
    logger.info("  Solo %s: %.1f%%", domain_a, solo_acc * 100)

    # Collab
    collab_correct = 0
    c2w = 0
    w2c = 0
    switches = 0
    for entry in questions:
        final, chain, pre_collab = collab_reasoning_scoped(
            specialist, mediator, tokenizer, entry["question"],
            domain_a, mediator_label, n_rounds=n_rounds,
            device=device, protocol="full-cot",
        )
        predicted = extract_answer_letter(final)
        expected = entry["answer_letter"]
        pre_a = pre_collab["agent_a"]["answer"]

        if predicted == expected:
            collab_correct += 1
        if pre_a != predicted:
            switches += 1
            if pre_a == expected and predicted != expected:
                c2w += 1
            elif pre_a != expected and predicted == expected:
                w2c += 1

    collab_acc = collab_correct / len(questions)
    delta = collab_acc - solo_acc

    logger.info("  Ratio %s: solo %.1f%% → collab %.1f%% (delta %+.1f%%, C2W=%d, W2C=%d)",
                ratio_tag, solo_acc * 100, collab_acc * 100, delta * 100, c2w, w2c)

    del specialist, mediator
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "ratio_a": ratio_a,
        "ratio_tag": ratio_tag,
        "domain_a": domain_a,
        "domain_b": domain_b,
        "solo_acc": solo_acc,
        "collab_acc": collab_acc,
        "delta": delta,
        "c2w": c2w,
        "w2c": w2c,
        "switches": switches,
        "n_questions": len(questions),
    }


def main():
    parser = argparse.ArgumentParser(description="Mix-ratio sweep for mediator training")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--specialist-adapter-dir", default="adapters")
    parser.add_argument("--mediator-output-dir", default="mediators")
    parser.add_argument("--results-dir", default="results/ratio_sweep")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--domain-a", default="medicine")
    parser.add_argument("--domain-b", default="physics")
    parser.add_argument("--ratios", nargs="*", type=float, default=None,
                        help="Ratios for domain_a (e.g., 0.9 0.8 0.5). Default: full sweep")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    ratios = args.ratios or DEFAULT_RATIOS
    logger.info("Ratio sweep: %s + %s, ratios: %s", args.domain_a, args.domain_b, ratios)

    start = time.time()

    # Phase 1: Train all ratio mediators
    if not args.skip_training:
        for ratio in ratios:
            train_ratio_mediator(
                args.domain_a, args.domain_b, ratio,
                model_name=args.model_name,
                output_dir=args.mediator_output_dir,
                cache_dir=args.cache_dir,
            )

    # Phase 2: Evaluate
    test_entries_a = load_mmlu_domain(args.domain_a, split="test", cache_dir=args.cache_dir)
    _, test_a = split_train_test(test_entries_a, test_size=args.n_questions, seed=42)

    results = []
    for ratio in ratios:
        r = evaluate_ratio_mediator(
            args.domain_a, args.domain_b, ratio,
            model_name=args.model_name,
            specialist_adapter_dir=args.specialist_adapter_dir,
            mediator_adapter_dir=args.mediator_output_dir,
            test_entries=test_a,
            n_questions=args.n_questions,
            n_rounds=args.n_rounds,
            cache_dir=args.cache_dir,
        )
        results.append(r)

    elapsed = time.time() - start
    logger.info("Total time: %.1f minutes", elapsed / 60)

    # Save
    os.makedirs(args.results_dir, exist_ok=True)
    output = {
        "results": results,
        "config": {
            "domain_a": args.domain_a,
            "domain_b": args.domain_b,
            "ratios": ratios,
            "n_questions": args.n_questions,
            "n_rounds": args.n_rounds,
        },
    }
    with open(os.path.join(args.results_dir, "ratio_sweep.json"), "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print(f"RATIO SWEEP: {args.domain_a} specialist + mediator({args.domain_a},{args.domain_b})")
    print("=" * 60)
    for r in results:
        print(f"  Ratio {r['ratio_tag']:>5}: solo {r['solo_acc']*100:.0f}% → "
              f"collab {r['collab_acc']*100:.0f}% ({r['delta']*100:+.0f}pp) "
              f"C2W={r['c2w']} W2C={r['w2c']}")


if __name__ == "__main__":
    main()
