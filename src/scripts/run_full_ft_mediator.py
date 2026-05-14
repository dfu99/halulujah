#!/usr/bin/env python3
"""Full fine-tuned mediator experiment.

Trains a full-parameter mediator (no LoRA) on 50/50 mixed domain data,
then evaluates: full FT specialist + full FT mediator collaboration.

Compares against LoRA mediator (+4.0pp) and untrained base helper (+5.2pp).

If full FT mediators work well → LoRA is the bottleneck for mediators too.
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
    split_train_test,
)
from halulujah.domain.collab_eval import collab_reasoning_scoped, solo_reasoning
from halulujah.domain.cross_eval import extract_answer_letter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAIR = ("medicine", "physics")


def format_mediator_rp(entries, tokenizer, domain_a, domain_b):
    """Format mediator training data with RP format (no think blocks)."""
    system_prompt = (
        f"You are an expert in both {domain_a} and {domain_b}. "
        "Answer questions accurately, drawing on knowledge from both fields."
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
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False,
                )
                if "<think>\n\n</think>" not in text:
                    method = "template"
                else:
                    method = "manual"
            except TypeError:
                method = "manual"
        if method == "manual":
            text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{entry['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n{entry['answer']}<|im_end|>\n"
            )
        texts.append(text)
    logger.info("Formatted %d mediator entries (%s+%s, method=%s)",
                len(texts), domain_a, domain_b, method)
    from datasets import Dataset
    return Dataset.from_dict({"text": texts})


def train_full_ft_mediator(domain_a, domain_b, model_name, output_dir,
                           cache_dir=None, num_epochs=3, lr=2e-5):
    """Train a full fine-tuned mediator (no LoRA)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_dir = os.path.join(output_dir, f"full_ft_mediator_{domain_a}_{domain_b}")
    if os.path.exists(os.path.join(model_dir, "config.json")):
        logger.info("Full FT mediator %s+%s already trained, skipping",
                     domain_a, domain_b)
        return model_dir

    logger.info("=== Training FULL FT mediator: %s + %s ===", domain_a, domain_b)

    # Load mixed data
    all_entries = load_mmlu_mediator(domain_a, domain_b, split="test", cache_dir=cache_dir)
    try:
        aux = load_mmlu_mediator(domain_a, domain_b, split="validation", cache_dir=cache_dir)
        all_entries.extend(aux)
    except Exception:
        pass

    # Remove test questions
    entries_a = load_mmlu_domain(domain_a, split="test", cache_dir=cache_dir)
    entries_b = load_mmlu_domain(domain_b, split="test", cache_dir=cache_dir)
    _, test_a = split_train_test(entries_a, test_size=50, seed=42)
    _, test_b = split_train_test(entries_b, test_size=50, seed=42)
    test_qs = {e["question"] for e in test_a + test_b}
    train_entries = [e for e in all_entries if e["question"] not in test_qs]
    logger.info("Training data: %d entries (FULL FT, no LoRA)", len(train_entries))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = format_mediator_rp(train_entries, tokenizer, domain_a, domain_b)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    )

    total = sum(p.numel() for p in model.parameters())
    logger.info("  Full FT mediator: %d params trainable (100%%)", total)

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
        max_length=512,
    )

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info("Saved full FT mediator to %s", model_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return model_dir


def load_model(model_dir, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
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


def eval_collab(model_a, model_b, tokenizer, questions, domain_a, domain_b,
                n_rounds, device):
    """Evaluate collaboration, return (accuracy, delta_info)."""
    correct, c2w, w2c, switches = 0, 0, 0, 0
    for entry in questions:
        final, chain, pre = collab_reasoning_scoped(
            model_a, model_b, tokenizer, entry["question"],
            domain_a, domain_b, n_rounds=n_rounds, device=device,
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
    acc = correct / len(questions)
    return acc, c2w, w2c, switches


def run_experiment(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_a, domain_b = PAIR

    results_path = os.path.join(args.results_dir, "full_ft_mediator_results.json")
    os.makedirs(args.results_dir, exist_ok=True)
    data = load_checkpoint(results_path)
    data["config"] = {
        "pair": list(PAIR), "n_questions": args.n_questions,
        "n_rounds": args.n_rounds, "model_name": args.model_name,
        "training": "full_fine_tuning", "mediator_mix": "50/50",
    }

    # Train mediator
    if not args.skip_training:
        train_full_ft_mediator(
            domain_a, domain_b, args.model_name, args.model_dir,
            cache_dir=args.cache_dir,
        )

    # Load test data
    test_data = {}
    for d in [domain_a, domain_b]:
        entries = load_mmlu_domain(d, split="test", cache_dir=args.cache_dir)
        _, test = split_train_test(entries, test_size=args.n_questions, seed=42)
        test_data[d] = test

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mediator_dir = os.path.join(
        args.model_dir, f"full_ft_mediator_{domain_a}_{domain_b}",
    )
    spec_a_dir = os.path.join(args.specialist_dir, f"full_ft_{domain_a}")
    spec_b_dir = os.path.join(args.specialist_dir, f"full_ft_{domain_b}")

    # Load base model for base-helper conditions
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir,
    ).to(device)
    base_model.eval()

    # Evaluate mediator solo on each domain (how good is the mediator by itself?)
    for d in [domain_a, domain_b]:
        cid = f"mediator_solo_{d}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            mediator = load_model(mediator_dir, device)
            qs = test_data[d][:args.n_questions]
            correct = 0
            for entry in qs:
                final, _ = solo_reasoning(
                    mediator, tokenizer, entry["question"],
                    f"mediator_{domain_a}_{domain_b}",
                    n_rounds=args.n_rounds, device=device,
                )
                if extract_answer_letter(final) == entry["answer_letter"]:
                    correct += 1
            acc = correct / len(qs)
            data["conditions"].append({
                "id": cid, "type": "mediator_solo",
                "domain": d, "accuracy": acc, "n": len(qs),
            })
            logger.info("  %s: %.1f%%", cid, acc * 100)
            save_checkpoint(data, results_path)
            del mediator; gc.collect(); torch.cuda.empty_cache()

    # For each domain: specialist + mediator, specialist + base (for comparison)
    for d in [domain_a, domain_b]:
        other = domain_b if d == domain_a else domain_a
        spec_dir = spec_a_dir if d == domain_a else spec_b_dir
        qs = test_data[d][:args.n_questions]

        if not os.path.exists(spec_dir):
            logger.warning("No full FT specialist for %s, skipping", d)
            continue

        # Solo (may already exist from full_ft experiment)
        cid = f"solo_{d}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            specialist = load_model(spec_dir, device)
            correct = 0
            for entry in qs:
                final, _ = solo_reasoning(
                    specialist, tokenizer, entry["question"], d,
                    n_rounds=args.n_rounds, device=device,
                )
                if extract_answer_letter(final) == entry["answer_letter"]:
                    correct += 1
            acc = correct / len(qs)
            data["conditions"].append({
                "id": cid, "type": "solo", "specialist": d,
                "accuracy": acc, "n": len(qs),
            })
            logger.info("  %s: %.1f%%", cid, acc * 100)
            save_checkpoint(data, results_path)
            del specialist; gc.collect(); torch.cuda.empty_cache()

        solo_acc = get_acc(data, f"solo_{d}")

        # Specialist + full FT mediator
        cid = f"specialist_{d}_plus_ft_mediator"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            specialist = load_model(spec_dir, device)
            mediator = load_model(mediator_dir, device)
            acc, c2w, w2c, switches = eval_collab(
                specialist, mediator, tokenizer, qs,
                d, f"mediator_{domain_a}_{domain_b}",
                args.n_rounds, device,
            )
            delta = acc - (solo_acc or 0)
            data["conditions"].append({
                "id": cid, "type": "specialist_plus_ft_mediator",
                "specialist": d, "helper": "ft_mediator",
                "accuracy": acc, "n": len(qs), "delta": delta,
                "c2w": c2w, "w2c": w2c, "switches": switches,
                "c2w_w2c_ratio": c2w / max(w2c, 1),
            })
            logger.info("  %s: %.1f%% (delta %+.1fpp, C2W=%d, W2C=%d)",
                         cid, acc * 100, delta * 100, c2w, w2c)
            save_checkpoint(data, results_path)
            del specialist, mediator; gc.collect(); torch.cuda.empty_cache()

        # Specialist + base (for direct comparison)
        cid = f"specialist_{d}_plus_base"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            specialist = load_model(spec_dir, device)
            acc, c2w, w2c, switches = eval_collab(
                specialist, base_model, tokenizer, qs,
                d, "general", args.n_rounds, device,
            )
            delta = acc - (solo_acc or 0)
            data["conditions"].append({
                "id": cid, "type": "specialist_plus_base",
                "specialist": d, "helper": "base",
                "accuracy": acc, "n": len(qs), "delta": delta,
                "c2w": c2w, "w2c": w2c, "switches": switches,
                "c2w_w2c_ratio": c2w / max(w2c, 1),
            })
            logger.info("  %s: %.1f%% (delta %+.1fpp, C2W=%d, W2C=%d)",
                         cid, acc * 100, delta * 100, c2w, w2c)
            save_checkpoint(data, results_path)
            del specialist; gc.collect(); torch.cuda.empty_cache()

        # Specialist + cross-domain specialist
        cross_dir = spec_b_dir if d == domain_a else spec_a_dir
        cid = f"specialist_{d}_plus_ft_{other}"
        if not done(data, cid) and os.path.exists(cross_dir):
            logger.info("=== %s ===", cid)
            specialist = load_model(spec_dir, device)
            cross = load_model(cross_dir, device)
            acc, c2w, w2c, switches = eval_collab(
                specialist, cross, tokenizer, qs,
                d, other, args.n_rounds, device,
            )
            delta = acc - (solo_acc or 0)
            data["conditions"].append({
                "id": cid, "type": "specialist_plus_ft_cross",
                "specialist": d, "helper": other,
                "accuracy": acc, "n": len(qs), "delta": delta,
                "c2w": c2w, "w2c": w2c, "switches": switches,
                "c2w_w2c_ratio": c2w / max(w2c, 1),
            })
            logger.info("  %s: %.1f%% (delta %+.1fpp, C2W=%d, W2C=%d)",
                         cid, acc * 100, delta * 100, c2w, w2c)
            save_checkpoint(data, results_path)
            del specialist, cross; gc.collect(); torch.cuda.empty_cache()

    del base_model; gc.collect(); torch.cuda.empty_cache()
    save_checkpoint(data, results_path)
    return data


def main():
    parser = argparse.ArgumentParser(description="Full FT mediator experiment")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-dir", default="full_ft_models")
    parser.add_argument("--specialist-dir", default="full_ft_models")
    parser.add_argument("--results-dir", default="results/full_ft_mediator")
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
    save_checkpoint(data, os.path.join(args.results_dir, "full_ft_mediator_results.json"))
    logger.info("Done in %.1f min", elapsed / 60)

    print("\n" + "=" * 80)
    print("FULL FT MEDIATOR RESULTS")
    print("=" * 80)
    for c in sorted(data["conditions"], key=lambda x: x["id"]):
        acc = c["accuracy"] * 100
        line = f"  {c['id']:<50} {acc:5.1f}%"
        if "delta" in c:
            line += f"  {c['delta']*100:+6.1f}pp"
        if "c2w" in c:
            line += f"  C2W={c['c2w']:>2} W2C={c['w2c']:>2} ratio={c['c2w_w2c_ratio']:.1f}x"
        print(line)


if __name__ == "__main__":
    main()
