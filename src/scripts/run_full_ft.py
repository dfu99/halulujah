#!/usr/bin/env python3
"""Full fine-tuning ablation: is collaboration loss from LoRA or specialization?

Trains full-parameter specialists (no LoRA) for medicine and physics with
RP formatting, then evaluates same collaboration conditions as LoRA experiments.

If full FT also kills collaborativeness → specialization itself is the problem.
If full FT preserves it → LoRA's rank constraint creates artificial rigidity.

Uses Adafactor optimizer to fit full FT in 20GB VRAM.
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
from halulujah.domain.collab_eval import collab_reasoning_scoped, solo_reasoning
from halulujah.domain.cross_eval import extract_answer_letter
from scripts.run_reasoning_preserved import format_domain_reasoning_preserved

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOMAINS = ["medicine", "physics"]


def train_full_ft_specialist(domain, model_name, output_dir, cache_dir=None,
                             num_epochs=3, lr=2e-5, max_length=512):
    """Train a full fine-tuned specialist (no LoRA) with RP formatting."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_dir = os.path.join(output_dir, f"full_ft_{domain}")
    if os.path.exists(os.path.join(model_dir, "config.json")):
        logger.info("Full FT %s already trained, skipping", domain)
        return model_dir

    logger.info("=== Training FULL FT specialist: %s ===", domain)

    # Load data
    entries = load_mmlu_domain(domain, split="test", cache_dir=cache_dir)
    try:
        aux = load_mmlu_domain(domain, split="validation", cache_dir=cache_dir)
        entries.extend(aux)
    except Exception:
        pass
    train_entries, _ = split_train_test(entries, test_size=50, seed=42)
    logger.info("Training %s specialist on %d entries (FULL FT, no LoRA)",
                domain, len(train_entries))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format WITHOUT think tags (same as RP)
    train_dataset = format_domain_reasoning_preserved(
        train_entries, tokenizer, domain, max_length=max_length,
    )

    # Load model — NO LoRA, train all parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Full FT: %d / %d params trainable (100%%)", trainable, total)

    # Train with Adafactor to fit in 20GB VRAM
    training_args = SFTConfig(
        output_dir=model_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        optim="adafactor",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        max_length=max_length,
    )

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()

    # Save full model + tokenizer
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info("Saved full FT %s to %s", domain, model_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return model_dir


def load_full_ft(model_dir, device, cache_dir=None):
    """Load a full fine-tuned model."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    return model


# Checkpoint helpers
def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"conditions": [], "config": {}}

def save_checkpoint(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def done(data, cid):
    return any(c["id"] == cid for c in data["conditions"])

def get_acc(data, cid):
    return next((c["accuracy"] for c in data["conditions"] if c["id"] == cid), None)


def run_experiment(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_path = os.path.join(args.results_dir, "full_ft_results.json")
    os.makedirs(args.results_dir, exist_ok=True)
    data = load_checkpoint(results_path)
    data["config"] = {
        "domains": DOMAINS, "n_questions": args.n_questions,
        "n_rounds": args.n_rounds, "model_name": args.model_name,
        "training": "full_fine_tuning", "optimizer": "adafactor",
        "learning_rate": 2e-5, "epochs": 3,
    }

    # Step 1: Train full FT specialists
    if not args.skip_training:
        for d in DOMAINS:
            train_full_ft_specialist(
                d, args.model_name, args.model_dir, cache_dir=args.cache_dir,
            )

    # Step 2: Load test data
    test_data = {}
    for d in DOMAINS:
        entries = load_mmlu_domain(d, split="test", cache_dir=args.cache_dir)
        _, test = split_train_test(entries, test_size=args.n_questions, seed=42)
        test_data[d] = test

    # Step 3: Load tokenizer + base model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir,
    ).to(device)
    base_model.eval()

    # Step 4: Evaluate each domain
    for domain in DOMAINS:
        model_dir = os.path.join(args.model_dir, f"full_ft_{domain}")
        cross_domain = [d for d in DOMAINS if d != domain][0]
        cross_dir = os.path.join(args.model_dir, f"full_ft_{cross_domain}")
        qs = test_data[domain][:args.n_questions]

        # Load specialist
        logger.info("Loading full FT %s...", domain)
        specialist = load_full_ft(model_dir, device)

        # Solo
        cid = f"solo_{domain}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            correct = 0
            for entry in qs:
                final, _ = solo_reasoning(
                    specialist, tokenizer, entry["question"], domain,
                    n_rounds=args.n_rounds, device=device,
                )
                if extract_answer_letter(final) == entry["answer_letter"]:
                    correct += 1
            acc = correct / len(qs)
            data["conditions"].append({
                "id": cid, "type": "solo", "specialist": domain,
                "accuracy": acc, "n": len(qs),
            })
            logger.info("  %s: %.1f%%", cid, acc * 100)
            save_checkpoint(data, results_path)

        solo_acc = get_acc(data, f"solo_{domain}")

        # +Base
        cid = f"full_ft_{domain}_plus_base"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            correct, c2w, w2c, switches = 0, 0, 0, 0
            for entry in qs:
                final, chain, pre = collab_reasoning_scoped(
                    specialist, base_model, tokenizer, entry["question"],
                    domain, "general", n_rounds=args.n_rounds, device=device,
                    protocol="full-cot",
                )
                predicted = extract_answer_letter(final)
                expected = entry["answer_letter"]
                pre_a = pre["agent_a"]["answer"]
                if predicted == expected:
                    correct += 1
                if pre_a != predicted:
                    switches += 1
                    if pre_a == expected and predicted != expected:
                        c2w += 1
                    elif pre_a != expected and predicted == expected:
                        w2c += 1
            acc = correct / len(qs)
            delta = acc - (solo_acc or 0)
            data["conditions"].append({
                "id": cid, "type": "full_ft_plus_base",
                "specialist": domain, "helper": "base",
                "accuracy": acc, "n": len(qs), "delta": delta,
                "c2w": c2w, "w2c": w2c, "switches": switches,
                "c2w_w2c_ratio": c2w / max(w2c, 1),
            })
            logger.info("  %s: %.1f%% (delta %+.1fpp, C2W=%d, W2C=%d)",
                         cid, acc * 100, delta * 100, c2w, w2c)
            save_checkpoint(data, results_path)

        # Same-domain pair
        cid = f"full_ft_{domain}_plus_same"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            correct, c2w, w2c, switches = 0, 0, 0, 0
            for entry in qs:
                final, chain, pre = collab_reasoning_scoped(
                    specialist, specialist, tokenizer, entry["question"],
                    domain, domain, n_rounds=args.n_rounds, device=device,
                    protocol="full-cot",
                )
                predicted = extract_answer_letter(final)
                expected = entry["answer_letter"]
                pre_a = pre["agent_a"]["answer"]
                if predicted == expected:
                    correct += 1
                if pre_a != predicted:
                    switches += 1
                    if pre_a == expected and predicted != expected:
                        c2w += 1
                    elif pre_a != expected and predicted == expected:
                        w2c += 1
            acc = correct / len(qs)
            delta = acc - (solo_acc or 0)
            data["conditions"].append({
                "id": cid, "type": "full_ft_same_pair",
                "specialist": domain, "helper": domain,
                "accuracy": acc, "n": len(qs), "delta": delta,
                "c2w": c2w, "w2c": w2c, "switches": switches,
                "c2w_w2c_ratio": c2w / max(w2c, 1),
            })
            logger.info("  %s: %.1f%% (delta %+.1fpp, C2W=%d, W2C=%d)",
                         cid, acc * 100, delta * 100, c2w, w2c)
            save_checkpoint(data, results_path)

        # Unload specialist before cross-domain
        del specialist
        gc.collect()
        torch.cuda.empty_cache()

        # +Cross-domain (full FT specialist + full FT other-domain)
        if os.path.exists(cross_dir):
            cid = f"full_ft_{domain}_plus_{cross_domain}"
            if not done(data, cid):
                logger.info("=== %s ===", cid)
                specialist = load_full_ft(model_dir, device)
                cross_model = load_full_ft(cross_dir, device)

                correct, c2w, w2c, switches = 0, 0, 0, 0
                for entry in qs:
                    final, chain, pre = collab_reasoning_scoped(
                        specialist, cross_model, tokenizer, entry["question"],
                        domain, cross_domain, n_rounds=args.n_rounds, device=device,
                        protocol="full-cot",
                    )
                    predicted = extract_answer_letter(final)
                    expected = entry["answer_letter"]
                    pre_a = pre["agent_a"]["answer"]
                    if predicted == expected:
                        correct += 1
                    if pre_a != predicted:
                        switches += 1
                        if pre_a == expected and predicted != expected:
                            c2w += 1
                        elif pre_a != expected and predicted == expected:
                            w2c += 1
                acc = correct / len(qs)
                delta = acc - (solo_acc or 0)
                data["conditions"].append({
                    "id": cid, "type": "full_ft_cross_pair",
                    "specialist": domain, "helper": cross_domain,
                    "accuracy": acc, "n": len(qs), "delta": delta,
                    "c2w": c2w, "w2c": w2c, "switches": switches,
                    "c2w_w2c_ratio": c2w / max(w2c, 1),
                })
                logger.info("  %s: %.1f%% (delta %+.1fpp, C2W=%d, W2C=%d)",
                             cid, acc * 100, delta * 100, c2w, w2c)
                save_checkpoint(data, results_path)

                del specialist, cross_model
                gc.collect()
                torch.cuda.empty_cache()

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    save_checkpoint(data, results_path)
    return data


def print_results(data):
    print("\n" + "=" * 80)
    print("FULL FINE-TUNING vs LoRA — COLLABORATION ABLATION")
    print("=" * 80)
    for c in sorted(data["conditions"], key=lambda x: x["id"]):
        acc = c["accuracy"] * 100
        line = f"  {c['id']:<40} {acc:5.1f}%"
        if "delta" in c:
            line += f"  {c['delta']*100:+6.1f}pp"
        if "c2w" in c:
            line += f"  C2W={c['c2w']:>2} W2C={c['w2c']:>2} ratio={c['c2w_w2c_ratio']:.1f}x"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Full FT collaboration ablation")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-dir", default="full_ft_models")
    parser.add_argument("--results-dir", default="results/full_ft")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    start = time.time()
    data = run_experiment(args)
    elapsed = time.time() - start
    data["total_time_minutes"] = elapsed / 60
    save_checkpoint(data, os.path.join(args.results_dir, "full_ft_results.json"))
    logger.info("Done in %.1f min (%.1f hrs)", elapsed / 60, elapsed / 3600)
    print_results(data)


if __name__ == "__main__":
    main()
