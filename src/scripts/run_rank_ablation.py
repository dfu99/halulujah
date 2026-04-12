"""LoRA Rank Ablation — does specialization depth control epistemic rigidity?

Trains medicine specialists at different LoRA ranks (r=4, 8, 16, 32)
and optionally with full fine-tuning (no LoRA). Evaluates each condition:

  1. Solo accuracy (medicine questions)
  2. Collaboration with base model (no LoRA) as helper
  3. Collaboration with cross-domain specialist (physics, r=16) as helper

Key question: if LoRA creates epistemic rigidity, does LESS LoRA = BETTER
collaboration? If r=4 collaborates like base but r=32 collaborates like
current specialists, that proves the mechanism.

Hypothesis:
  r=4  → low specialization, good collaboration (near base-as-helper)
  r=32 → high specialization, bad collaboration (high C2W switching)
  full → deepest specialization, worst collaboration (if rigidity hypothesis holds)
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
    format_domain_for_sft,
    split_train_test,
)
from halulujah.domain.collab_eval import (
    collab_reasoning_scoped,
    solo_reasoning,
)
from halulujah.domain.cross_eval import extract_answer_letter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DOMAIN = "medicine"
CROSS_DOMAIN = "physics"
DEFAULT_RANKS = [4, 8, 16, 32]


def train_specialist_at_rank(
    domain, rank, model_name, output_dir,
    cache_dir=None, num_epochs=3, lr=5e-5, batch_size=4,
):
    """Train a domain specialist LoRA adapter at a specific rank."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    adapter_dir = os.path.join(output_dir, f"adapter_{domain}_r{rank}")

    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        logger.info("Rank %d adapter already trained, skipping", rank)
        return adapter_dir

    logger.info("=== Training %s specialist at rank r=%d ===", domain, rank)

    entries = load_mmlu_domain(domain, split="test", cache_dir=cache_dir)
    try:
        aux = load_mmlu_domain(domain, split="validation", cache_dir=cache_dir)
        entries.extend(aux)
    except Exception:
        pass

    train_entries, _ = split_train_test(entries, test_size=50, seed=42)
    logger.info("Training data: %d entries, rank r=%d", len(train_entries), rank)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    )

    lora_config = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("  Trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    train_dataset = format_domain_for_sft(train_entries, tokenizer, domain)

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
    logger.info("Saved rank %d adapter to %s", rank, adapter_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir


def train_full_finetune(
    domain, model_name, output_dir,
    cache_dir=None, num_epochs=3, lr=2e-5, batch_size=4,
):
    """Train a domain specialist with full fine-tuning (no LoRA)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_dir = os.path.join(output_dir, f"full_ft_{domain}")

    if os.path.exists(os.path.join(model_dir, "config.json")):
        logger.info("Full fine-tuned model already trained, skipping")
        return model_dir

    logger.info("=== Training %s specialist — FULL fine-tuning ===", domain)

    entries = load_mmlu_domain(domain, split="test", cache_dir=cache_dir)
    try:
        aux = load_mmlu_domain(domain, split="validation", cache_dir=cache_dir)
        entries.extend(aux)
    except Exception:
        pass

    train_entries, _ = split_train_test(entries, test_size=50, seed=42)
    logger.info("Training data: %d entries, FULL fine-tuning", len(train_entries))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Trainable params: %d (100%% — full fine-tuning)", trainable)

    train_dataset = format_domain_for_sft(train_entries, tokenizer, domain)

    training_args = SFTConfig(
        output_dir=model_dir,
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
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info("Saved full fine-tuned model to %s", model_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return model_dir


def evaluate_condition(
    condition_name, specialist_model, base_model, cross_model,
    tokenizer, test_entries, domain, cross_domain,
    n_questions=50, n_rounds=3, device=None,
):
    """Evaluate a single specialist condition across all collaboration types.

    Returns dict with solo, base_collab, and cross_collab results.
    """
    if device is None:
        device = next(specialist_model.parameters()).device

    questions = test_entries[:n_questions]
    results = {"condition": condition_name}

    # --- Solo baseline ---
    solo_correct = 0
    for entry in questions:
        final, _ = solo_reasoning(
            specialist_model, tokenizer, entry["question"], domain,
            n_rounds=n_rounds, device=device,
        )
        if extract_answer_letter(final) == entry["answer_letter"]:
            solo_correct += 1
    solo_acc = solo_correct / len(questions)
    results["solo_acc"] = solo_acc
    logger.info("  [%s] Solo: %.1f%%", condition_name, solo_acc * 100)

    # --- Collaboration with base model ---
    base_correct, base_c2w, base_w2c, base_switches = 0, 0, 0, 0
    for entry in questions:
        final, chain, pre_collab = collab_reasoning_scoped(
            specialist_model, base_model, tokenizer, entry["question"],
            domain, "general", n_rounds=n_rounds,
            device=device, protocol="full-cot",
        )
        predicted = extract_answer_letter(final)
        expected = entry["answer_letter"]
        pre_a = pre_collab["agent_a"]["answer"]

        if predicted == expected:
            base_correct += 1
        if pre_a != predicted:
            base_switches += 1
            if pre_a == expected and predicted != expected:
                base_c2w += 1
            elif pre_a != expected and predicted == expected:
                base_w2c += 1

    base_acc = base_correct / len(questions)
    results["base_collab_acc"] = base_acc
    results["base_delta"] = base_acc - solo_acc
    results["base_c2w"] = base_c2w
    results["base_w2c"] = base_w2c
    results["base_switches"] = base_switches
    logger.info("  [%s] +Base: %.1f%% (delta %+.1f%%, C2W=%d, W2C=%d)",
                condition_name, base_acc * 100, (base_acc - solo_acc) * 100,
                base_c2w, base_w2c)

    # --- Collaboration with cross-domain specialist ---
    if cross_model is not None:
        cross_correct, cross_c2w, cross_w2c, cross_switches = 0, 0, 0, 0
        for entry in questions:
            final, chain, pre_collab = collab_reasoning_scoped(
                specialist_model, cross_model, tokenizer, entry["question"],
                domain, cross_domain, n_rounds=n_rounds,
                device=device, protocol="full-cot",
            )
            predicted = extract_answer_letter(final)
            expected = entry["answer_letter"]
            pre_a = pre_collab["agent_a"]["answer"]

            if predicted == expected:
                cross_correct += 1
            if pre_a != predicted:
                cross_switches += 1
                if pre_a == expected and predicted != expected:
                    cross_c2w += 1
                elif pre_a != expected and predicted == expected:
                    cross_w2c += 1

        cross_acc = cross_correct / len(questions)
        results["cross_collab_acc"] = cross_acc
        results["cross_delta"] = cross_acc - solo_acc
        results["cross_c2w"] = cross_c2w
        results["cross_w2c"] = cross_w2c
        results["cross_switches"] = cross_switches
        logger.info("  [%s] +%s: %.1f%% (delta %+.1f%%, C2W=%d, W2C=%d)",
                    condition_name, cross_domain, cross_acc * 100,
                    (cross_acc - solo_acc) * 100, cross_c2w, cross_w2c)

    results["n_questions"] = len(questions)
    return results


def main():
    parser = argparse.ArgumentParser(description="LoRA rank ablation for epistemic rigidity")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-dir", default="adapters")
    parser.add_argument("--rank-adapter-dir", default="rank_adapters")
    parser.add_argument("--results-dir", default="results/rank_ablation")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--ranks", nargs="*", type=int, default=None,
                        help="LoRA ranks to test (default: 4 8 16 32)")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--include-full-ft", action="store_true",
                        help="Include full fine-tuning (no LoRA) condition")
    parser.add_argument("--skip-cross", action="store_true",
                        help="Skip cross-domain specialist evaluation")
    args = parser.parse_args()

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    ranks = args.ranks or DEFAULT_RANKS
    logger.info("Rank ablation: %s, ranks: %s, full_ft: %s",
                DOMAIN, ranks, args.include_full_ft)

    start = time.time()

    # =====================
    # Phase 1: Train adapters at each rank
    # =====================
    if not args.skip_training:
        for rank in ranks:
            train_specialist_at_rank(
                DOMAIN, rank,
                model_name=args.model_name,
                output_dir=args.rank_adapter_dir,
                cache_dir=args.cache_dir,
            )

        if args.include_full_ft:
            train_full_finetune(
                DOMAIN,
                model_name=args.model_name,
                output_dir=args.rank_adapter_dir,
                cache_dir=args.cache_dir,
            )

    # =====================
    # Phase 2: Load test data
    # =====================
    test_entries = load_mmlu_domain(DOMAIN, split="test", cache_dir=args.cache_dir)
    _, test_med = split_train_test(test_entries, test_size=args.n_questions, seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir
    )

    # =====================
    # Phase 3: Load base model (shared across all evals)
    # =====================
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir,
    ).to(device)
    base_model.eval()

    # Load cross-domain specialist (physics, r=16) if needed
    cross_model = None
    if not args.skip_cross:
        cross_adapter = os.path.join(args.adapter_dir, f"adapter_{CROSS_DOMAIN}")
        if os.path.exists(cross_adapter):
            logger.info("Loading %s specialist (r=16) for cross-domain eval...", CROSS_DOMAIN)
            cross_model = AutoModelForCausalLM.from_pretrained(
                args.model_name, dtype=torch.bfloat16,
                trust_remote_code=True, cache_dir=args.cache_dir,
            ).to(device)
            cross_model = PeftModel.from_pretrained(cross_model, cross_adapter)
            cross_model.eval()
        else:
            logger.warning("No %s adapter found at %s, skipping cross-domain eval",
                          CROSS_DOMAIN, cross_adapter)

    # =====================
    # Phase 4: Evaluate each LoRA rank
    # =====================
    all_results = []

    for rank in ranks:
        adapter_path = os.path.join(args.rank_adapter_dir, f"adapter_{DOMAIN}_r{rank}")
        if not os.path.exists(adapter_path):
            logger.warning("No adapter for rank %d at %s, skipping", rank, adapter_path)
            continue

        logger.info("=== Evaluating rank r=%d ===", rank)
        specialist = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=torch.bfloat16,
            trust_remote_code=True, cache_dir=args.cache_dir,
        ).to(device)
        specialist = PeftModel.from_pretrained(specialist, adapter_path)
        specialist.eval()

        result = evaluate_condition(
            f"lora_r{rank}", specialist, base_model, cross_model,
            tokenizer, test_med, DOMAIN, CROSS_DOMAIN,
            n_questions=args.n_questions, n_rounds=args.n_rounds,
            device=device,
        )
        result["rank"] = rank
        result["type"] = "lora"

        # Count trainable params for this rank
        trainable = sum(p.numel() for p in specialist.parameters() if p.requires_grad)
        result["trainable_params"] = trainable

        all_results.append(result)

        del specialist
        gc.collect()
        torch.cuda.empty_cache()

    # =====================
    # Phase 5: Evaluate full fine-tuning (if trained)
    # =====================
    if args.include_full_ft:
        full_ft_path = os.path.join(args.rank_adapter_dir, f"full_ft_{DOMAIN}")
        if os.path.exists(full_ft_path):
            logger.info("=== Evaluating full fine-tuning ===")
            specialist = AutoModelForCausalLM.from_pretrained(
                full_ft_path, dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device)
            specialist.eval()

            result = evaluate_condition(
                "full_ft", specialist, base_model, cross_model,
                tokenizer, test_med, DOMAIN, CROSS_DOMAIN,
                n_questions=args.n_questions, n_rounds=args.n_rounds,
                device=device,
            )
            result["rank"] = None
            result["type"] = "full_ft"
            result["trainable_params"] = sum(p.numel() for p in specialist.parameters())

            all_results.append(result)

            del specialist
            gc.collect()
            torch.cuda.empty_cache()

    # =====================
    # Phase 6: Evaluate base model alone (control)
    # =====================
    logger.info("=== Evaluating base model (no fine-tuning) ===")
    base_result = {"condition": "base", "rank": 0, "type": "base", "trainable_params": 0}
    solo_correct = 0
    for entry in test_med[:args.n_questions]:
        final, _ = solo_reasoning(
            base_model, tokenizer, entry["question"], DOMAIN,
            n_rounds=args.n_rounds, device=device,
        )
        if extract_answer_letter(final) == entry["answer_letter"]:
            solo_correct += 1
    base_result["solo_acc"] = solo_correct / args.n_questions
    base_result["n_questions"] = args.n_questions
    logger.info("  [base] Solo: %.1f%%", base_result["solo_acc"] * 100)
    all_results.append(base_result)

    # Cleanup
    del base_model
    if cross_model is not None:
        del cross_model
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - start
    logger.info("Total time: %.1f minutes", elapsed / 60)

    # =====================
    # Save results
    # =====================
    os.makedirs(args.results_dir, exist_ok=True)
    output = {
        "results": all_results,
        "config": {
            "domain": DOMAIN,
            "cross_domain": CROSS_DOMAIN,
            "ranks": ranks,
            "include_full_ft": args.include_full_ft,
            "n_questions": args.n_questions,
            "n_rounds": args.n_rounds,
            "model_name": args.model_name,
        },
    }
    with open(os.path.join(args.results_dir, "rank_ablation.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"RANK ABLATION: {DOMAIN} specialist at varying LoRA ranks")
    print("=" * 80)
    print(f"{'Condition':<15} {'Solo':>6} {'+ Base':>8} {'Delta':>7} {'C2W':>5} {'W2C':>5} "
          f"{'+ Cross':>8} {'Delta':>7} {'C2W':>5} {'W2C':>5}")
    print("-" * 80)
    for r in all_results:
        solo = f"{r['solo_acc']*100:.0f}%"
        base_acc = f"{r.get('base_collab_acc', 0)*100:.0f}%" if 'base_collab_acc' in r else "—"
        base_d = f"{r.get('base_delta', 0)*100:+.0f}pp" if 'base_delta' in r else "—"
        b_c2w = str(r.get('base_c2w', '—'))
        b_w2c = str(r.get('base_w2c', '—'))
        cross_acc = f"{r.get('cross_collab_acc', 0)*100:.0f}%" if 'cross_collab_acc' in r else "—"
        cross_d = f"{r.get('cross_delta', 0)*100:+.0f}pp" if 'cross_delta' in r else "—"
        c_c2w = str(r.get('cross_c2w', '—'))
        c_w2c = str(r.get('cross_w2c', '—'))
        print(f"{r['condition']:<15} {solo:>6} {base_acc:>8} {base_d:>7} {b_c2w:>5} {b_w2c:>5} "
              f"{cross_acc:>8} {cross_d:>7} {c_c2w:>5} {c_w2c:>5}")


if __name__ == "__main__":
    main()
