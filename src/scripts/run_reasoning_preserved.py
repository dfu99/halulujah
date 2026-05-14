#!/usr/bin/env python3
"""Train reasoning-preserved domain specialists and evaluate full collaboration matrix.

Problem: Standard SFT on Qwen3 produces empty <think></think> blocks because
the chat template auto-inserts think tags around bare-answer training data.
This teaches the LoRA to skip reasoning, causing epistemic rigidity in collaboration.

Fix: Format training data WITHOUT think tags (using enable_thinking=False or manual
template). The LoRA learns domain knowledge without overwriting the pretrained
reasoning pathway. At inference, Qwen3 uses its native reasoning as usual.

This is the Qwen3-recommended approach (ms-swift: loss_scale=ignore_empty_think).

Pipeline:
  1. Train medicine + physics specialists with reasoning preserved
  2. Evaluate full matrix: solo, +base, +cross-domain
  3. Compare against old (thinking-suppressed) specialists
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
    split_train_test,
)
from halulujah.domain.collab_eval import (
    collab_reasoning_scoped,
    solo_reasoning,
)
from halulujah.domain.cross_eval import extract_answer_letter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DOMAINS = ["medicine", "physics"]


def format_domain_reasoning_preserved(entries, tokenizer, domain, max_length=512):
    """Format training data WITHOUT empty think blocks.

    Uses enable_thinking=False to prevent Qwen3's chat template from
    inserting <think></think> wrappers around bare answers. Falls back
    to manual template construction if the parameter isn't supported.

    This preserves the model's pretrained reasoning: the LoRA only learns
    domain knowledge, not the "skip thinking" pattern.
    """
    system_prompt = (
        f"You are a {domain} expert. "
        "Answer the question accurately and concisely."
    )

    texts = []
    method = None

    for entry in entries:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": entry["question"]},
            {"role": "assistant", "content": entry["answer"]},
        ]

        if method != "manual":
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                # Verify no empty think blocks leaked through
                if "<think>\n\n</think>" not in text:
                    method = "template"
                else:
                    method = "manual"
            except TypeError:
                method = "manual"

        if method == "manual":
            # Manual Qwen3 chat format without think tags
            text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{entry['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n{entry['answer']}<|im_end|>\n"
            )

        texts.append(text)

    logger.info(
        "Formatted %d entries for domain '%s' (method=%s, no think tags)",
        len(texts), domain, method,
    )

    # Final safety check
    for i, t in enumerate(texts[:3]):
        if "<think>" in t:
            logger.warning("WARNING: think tag found in entry %d! Method: %s", i, method)

    from datasets import Dataset
    return Dataset.from_dict({"text": texts})


def train_reasoning_preserved_specialist(
    domain, model_name, output_dir,
    cache_dir=None, num_epochs=3, lr=5e-5, batch_size=4, rank=16,
    max_length=512,
):
    """Train a domain specialist with reasoning preserved."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    adapter_dir = os.path.join(output_dir, f"adapter_{domain}")
    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        logger.info("Reasoning-preserved %s specialist already trained, skipping", domain)
        return adapter_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    entries = load_mmlu_domain(domain, split="test", cache_dir=cache_dir)
    try:
        aux = load_mmlu_domain(domain, split="validation", cache_dir=cache_dir)
        entries.extend(aux)
    except Exception:
        pass
    train_entries, _ = split_train_test(entries, test_size=50, seed=42)
    logger.info("Training %s specialist on %d entries", domain, len(train_entries))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format WITHOUT think tags
    train_dataset = format_domain_reasoning_preserved(
        train_entries, tokenizer, domain, max_length=max_length,
    )

    # Model + LoRA
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    )
    lora_config = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("  Trainable: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    # Train
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
        max_length=max_length,
    )

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(adapter_dir)
    logger.info("Saved reasoning-preserved %s specialist to %s", domain, adapter_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir


def load_specialist(model_name, adapter_path, device, cache_dir=None):
    """Load a specialist model from adapter path."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    ).to(device)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def evaluate_condition(
    specialist, helper, tokenizer,
    test_entries, domain, helper_domain,
    n_questions=50, n_rounds=3, device=None,
):
    """Evaluate a single collaboration condition. Returns stats dict."""
    questions = test_entries[:n_questions]
    correct, c2w, w2c, switches = 0, 0, 0, 0

    for entry in questions:
        if helper is None:
            # Solo
            final, _ = solo_reasoning(
                specialist, tokenizer, entry["question"], domain,
                n_rounds=n_rounds, device=device,
            )
        else:
            final, _, pre_collab = collab_reasoning_scoped(
                specialist, helper, tokenizer, entry["question"],
                domain, helper_domain, n_rounds=n_rounds,
                device=device, protocol="full-cot",
            )

        predicted = extract_answer_letter(final)
        expected = entry["answer_letter"]

        if predicted == expected:
            correct += 1

        if helper is not None:
            pre_a = pre_collab["agent_a"]["answer"]
            if pre_a != predicted:
                switches += 1
                if pre_a == expected and predicted != expected:
                    c2w += 1
                elif pre_a != expected and predicted == expected:
                    w2c += 1

    acc = correct / len(questions)
    result = {"accuracy": acc, "n": len(questions)}
    if helper is not None:
        result.update({"c2w": c2w, "w2c": w2c, "switches": switches})
    return result


def run_full_matrix(args):
    """Train specialists and evaluate the full collaboration matrix."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Train both specialists
    if not args.skip_training:
        for domain in DOMAINS:
            train_reasoning_preserved_specialist(
                domain, args.model_name, args.adapter_dir,
                cache_dir=args.cache_dir, rank=args.rank,
            )

    # Step 2: Load test data
    test_data = {}
    for domain in DOMAINS:
        entries = load_mmlu_domain(domain, split="test", cache_dir=args.cache_dir)
        _, test = split_train_test(entries, test_size=args.n_questions, seed=42)
        test_data[domain] = test

    # Step 3: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir,
    )

    # Step 4: Load base model (stays in memory throughout)
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir,
    ).to(device)
    base_model.eval()

    results = {"conditions": [], "config": {
        "domains": DOMAINS,
        "rank": args.rank,
        "n_questions": args.n_questions,
        "n_rounds": args.n_rounds,
        "model_name": args.model_name,
        "adapter_type": "reasoning_preserved",
    }}

    # Step 5: Evaluate each domain specialist
    for domain in DOMAINS:
        cross_domain = [d for d in DOMAINS if d != domain][0]

        # Load this domain's specialist
        rp_path = os.path.join(args.adapter_dir, f"adapter_{domain}")
        logger.info("=== Evaluating %s (reasoning-preserved) ===", domain)
        specialist = load_specialist(
            args.model_name, rp_path, device, args.cache_dir,
        )

        # Solo
        logger.info("  %s solo...", domain)
        solo = evaluate_condition(
            specialist, None, tokenizer,
            test_data[domain], domain, None,
            n_questions=args.n_questions, n_rounds=args.n_rounds, device=device,
        )
        logger.info("  [%s] Solo: %.1f%%", domain, solo["accuracy"] * 100)
        results["conditions"].append({
            "specialist": domain, "helper": "none", "type": "rp",
            **solo,
        })

        # +Base helper
        logger.info("  %s + base...", domain)
        base_collab = evaluate_condition(
            specialist, base_model, tokenizer,
            test_data[domain], domain, "general",
            n_questions=args.n_questions, n_rounds=args.n_rounds, device=device,
        )
        delta = base_collab["accuracy"] - solo["accuracy"]
        logger.info(
            "  [%s] +Base: %.1f%% (delta %+.1f%%, C2W=%d, W2C=%d)",
            domain, base_collab["accuracy"] * 100, delta * 100,
            base_collab["c2w"], base_collab["w2c"],
        )
        results["conditions"].append({
            "specialist": domain, "helper": "base", "type": "rp",
            "delta": delta, **base_collab,
        })

        # Unload specialist before loading cross-domain
        del specialist
        gc.collect()
        torch.cuda.empty_cache()

        # Load cross-domain specialist for cross-collaboration
        cross_path = os.path.join(args.adapter_dir, f"adapter_{cross_domain}")
        if os.path.exists(cross_path):
            cross_model = load_specialist(
                args.model_name, cross_path, device, args.cache_dir,
            )

            # Reload primary specialist
            specialist = load_specialist(
                args.model_name, rp_path, device, args.cache_dir,
            )

            # +Cross helper (both reasoning-preserved)
            logger.info("  %s + %s (rp)...", domain, cross_domain)
            cross_collab = evaluate_condition(
                specialist, cross_model, tokenizer,
                test_data[domain], domain, cross_domain,
                n_questions=args.n_questions, n_rounds=args.n_rounds, device=device,
            )
            delta = cross_collab["accuracy"] - solo["accuracy"]
            logger.info(
                "  [%s] +%s(rp): %.1f%% (delta %+.1f%%, C2W=%d, W2C=%d)",
                domain, cross_domain, cross_collab["accuracy"] * 100, delta * 100,
                cross_collab["c2w"], cross_collab["w2c"],
            )
            results["conditions"].append({
                "specialist": domain, "helper": f"{cross_domain}_rp", "type": "rp",
                "delta": delta, **cross_collab,
            })

            del specialist, cross_model
            gc.collect()
            torch.cuda.empty_cache()

    # Also evaluate old specialists for comparison if they exist
    old_adapter_dir = args.old_adapter_dir
    if old_adapter_dir and os.path.exists(old_adapter_dir):
        logger.info("=== Evaluating OLD (thinking-suppressed) specialists for comparison ===")

        for domain in DOMAINS:
            cross_domain = [d for d in DOMAINS if d != domain][0]
            old_path = os.path.join(old_adapter_dir, f"adapter_{domain}")

            if not os.path.exists(old_path):
                logger.info("  Old %s adapter not found, skipping", domain)
                continue

            specialist = load_specialist(
                args.model_name, old_path, device, args.cache_dir,
            )

            # Solo
            logger.info("  %s (old) solo...", domain)
            solo = evaluate_condition(
                specialist, None, tokenizer,
                test_data[domain], domain, None,
                n_questions=args.n_questions, n_rounds=args.n_rounds, device=device,
            )
            logger.info("  [%s old] Solo: %.1f%%", domain, solo["accuracy"] * 100)
            results["conditions"].append({
                "specialist": domain, "helper": "none", "type": "old",
                **solo,
            })

            # +Base
            logger.info("  %s (old) + base...", domain)
            base_collab = evaluate_condition(
                specialist, base_model, tokenizer,
                test_data[domain], domain, "general",
                n_questions=args.n_questions, n_rounds=args.n_rounds, device=device,
            )
            delta = base_collab["accuracy"] - solo["accuracy"]
            logger.info(
                "  [%s old] +Base: %.1f%% (delta %+.1f%%, C2W=%d, W2C=%d)",
                domain, base_collab["accuracy"] * 100, delta * 100,
                base_collab["c2w"], base_collab["w2c"],
            )
            results["conditions"].append({
                "specialist": domain, "helper": "base", "type": "old",
                "delta": delta, **base_collab,
            })

            del specialist
            gc.collect()
            torch.cuda.empty_cache()

            # +Cross (old specialist + old cross-domain)
            old_cross_path = os.path.join(old_adapter_dir, f"adapter_{cross_domain}")
            if os.path.exists(old_cross_path):
                specialist = load_specialist(
                    args.model_name, old_path, device, args.cache_dir,
                )
                cross_model = load_specialist(
                    args.model_name, old_cross_path, device, args.cache_dir,
                )

                logger.info("  %s (old) + %s (old)...", domain, cross_domain)
                cross_collab = evaluate_condition(
                    specialist, cross_model, tokenizer,
                    test_data[domain], domain, cross_domain,
                    n_questions=args.n_questions, n_rounds=args.n_rounds, device=device,
                )
                delta = cross_collab["accuracy"] - solo["accuracy"]
                logger.info(
                    "  [%s old] +%s(old): %.1f%% (delta %+.1f%%, C2W=%d, W2C=%d)",
                    domain, cross_domain, cross_collab["accuracy"] * 100,
                    delta * 100, cross_collab["c2w"], cross_collab["w2c"],
                )
                results["conditions"].append({
                    "specialist": domain, "helper": f"{cross_domain}_old", "type": "old",
                    "delta": delta, **cross_collab,
                })

                del specialist, cross_model
                gc.collect()
                torch.cuda.empty_cache()

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_results_table(results):
    """Print a clean comparison table."""
    print("\n" + "=" * 90)
    print("REASONING-PRESERVED vs ORIGINAL SPECIALISTS")
    print("=" * 90)

    # Group by type
    rp = [c for c in results["conditions"] if c["type"] == "rp"]
    old = [c for c in results["conditions"] if c["type"] == "old"]

    for label, group in [("REASONING-PRESERVED (new)", rp), ("ORIGINAL (old)", old)]:
        if not group:
            continue
        print(f"\n--- {label} ---")
        print(f"{'Specialist':<12} {'Helper':<15} {'Acc':>6} {'Delta':>7} {'C2W':>5} {'W2C':>5} {'Ratio':>7}")
        print("-" * 60)
        for c in group:
            acc = f"{c['accuracy']*100:.0f}%"
            delta = f"{c.get('delta', 0)*100:+.0f}pp" if "delta" in c else "—"
            c2w = str(c.get("c2w", "—"))
            w2c = str(c.get("w2c", "—"))
            if "c2w" in c and "w2c" in c:
                ratio = f"{c['c2w']/max(c['w2c'],1):.1f}x"
            else:
                ratio = "—"
            print(f"{c['specialist']:<12} {c['helper']:<15} {acc:>6} {delta:>7} {c2w:>5} {w2c:>5} {ratio:>7}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train reasoning-preserved specialists and evaluate collaboration",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-dir", default="rp_adapters",
                        help="Directory for new reasoning-preserved adapters")
    parser.add_argument("--old-adapter-dir", default="adapters",
                        help="Directory with old (thinking-suppressed) adapters")
    parser.add_argument("--results-dir", default="results/reasoning_preserved")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-old", action="store_true",
                        help="Skip evaluating old specialists")
    args = parser.parse_args()

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    if args.skip_old:
        args.old_adapter_dir = None

    start = time.time()
    results = run_full_matrix(args)
    elapsed = time.time() - start

    results["total_time_minutes"] = elapsed / 60
    logger.info("Total time: %.1f minutes", elapsed / 60)

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    output_path = os.path.join(args.results_dir, "rp_comparison.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", output_path)

    print_results_table(results)


if __name__ == "__main__":
    main()
