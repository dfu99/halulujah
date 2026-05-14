#!/usr/bin/env python3
"""Thinking-enabled mediator experiment.

Trains mediators with reasoning-preserved format (no empty think blocks)
and evaluates with RP specialists. Compares against old mediator results
(-32.8pp, 7.1x ratio) to determine if reasoning preservation fixes mediators.

Pairs: same 5 pairs as composite experiment for direct comparison.
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
from scripts.run_reasoning_preserved import load_specialist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAIRS = [
    ("medicine", "physics"),
    ("medicine", "biology"),
    ("law", "math"),
    ("physics", "math"),
    ("biology", "physics"),
]


def format_mediator_rp(entries, tokenizer, domain_a, domain_b, max_length=512):
    """Format mediator training data with reasoning preserved (no think blocks)."""
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

    logger.info("Formatted %d mediator entries (%s+%s, method=%s, no think tags)",
                len(texts), domain_a, domain_b, method)
    from datasets import Dataset
    return Dataset.from_dict({"text": texts})


def train_rp_mediator(domain_a, domain_b, model_name, output_dir,
                      cache_dir=None, num_epochs=3, lr=5e-5, rank=16):
    """Train a reasoning-preserved mediator."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    pair_name = f"{domain_a}_{domain_b}"
    adapter_dir = os.path.join(output_dir, f"mediator_{pair_name}")

    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        logger.info("RP mediator %s already trained, skipping", pair_name)
        return adapter_dir

    logger.info("=== Training RP mediator: %s + %s ===", domain_a, domain_b)

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
    logger.info("Training data: %d entries", len(train_entries))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = format_mediator_rp(
        train_entries, tokenizer, domain_a, domain_b,
    )

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

    training_args = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
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
    logger.info("Saved RP mediator to %s", adapter_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir


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

    results_path = os.path.join(args.results_dir, "rp_mediator_results.json")
    os.makedirs(args.results_dir, exist_ok=True)
    data = load_checkpoint(results_path)
    data["config"] = {
        "pairs": [list(p) for p in PAIRS],
        "n_questions": args.n_questions,
        "n_rounds": args.n_rounds,
        "model_name": args.model_name,
        "specialist_type": "reasoning_preserved",
        "mediator_type": "reasoning_preserved",
    }

    # Train RP mediators
    if not args.skip_training:
        for a, b in PAIRS:
            train_rp_mediator(
                a, b, args.model_name, args.mediator_dir,
                cache_dir=args.cache_dir, rank=args.rank,
            )

    # Load test data
    domains = sorted(set(d for p in PAIRS for d in p))
    test_data = {}
    for d in domains:
        entries = load_mmlu_domain(d, split="test", cache_dir=args.cache_dir)
        _, test = split_train_test(entries, test_size=args.n_questions, seed=42)
        test_data[d] = test

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Evaluate each pair: specialist_A + rp_mediator(A,B) on A's questions
    for domain_a, domain_b in PAIRS:
        pair_label = f"{domain_a}+{domain_b}"

        # Load specialist A
        spec_path = os.path.join(args.specialist_dir, f"adapter_{domain_a}")
        if not os.path.exists(spec_path):
            logger.warning("No RP specialist for %s, skipping", domain_a)
            continue

        # Load mediator
        med_path = os.path.join(args.mediator_dir, f"mediator_{domain_a}_{domain_b}")
        if not os.path.exists(med_path):
            logger.warning("No RP mediator for %s, skipping", pair_label)
            continue

        qs = test_data[domain_a][:args.n_questions]

        # Solo baseline
        cid = f"solo_{domain_a}"
        if not done(data, cid):
            logger.info("=== solo_%s ===", domain_a)
            specialist = load_specialist(args.model_name, spec_path, device, args.cache_dir)
            correct = 0
            for entry in qs:
                final, _ = solo_reasoning(
                    specialist, tokenizer, entry["question"], domain_a,
                    n_rounds=args.n_rounds, device=device,
                )
                if extract_answer_letter(final) == entry["answer_letter"]:
                    correct += 1
            acc = correct / len(qs)
            data["conditions"].append({"id": cid, "type": "solo", "specialist": domain_a,
                                       "accuracy": acc, "n": len(qs)})
            logger.info("  %s: %.1f%%", cid, acc * 100)
            save_checkpoint(data, results_path)
            del specialist; gc.collect(); torch.cuda.empty_cache()

        solo_acc = get_acc(data, f"solo_{domain_a}")

        # Mediated collaboration
        cid = f"mediated_{domain_a}_via_{pair_label}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            specialist = load_specialist(args.model_name, spec_path, device, args.cache_dir)
            mediator = load_specialist(args.model_name, med_path, device, args.cache_dir)

            correct, c2w, w2c, switches = 0, 0, 0, 0
            for entry in qs:
                final, chain, pre = collab_reasoning_scoped(
                    specialist, mediator, tokenizer, entry["question"],
                    domain_a, f"mediator_{domain_a}_{domain_b}",
                    n_rounds=args.n_rounds, device=device, protocol="full-cot",
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
                "id": cid, "type": "mediated", "specialist": domain_a,
                "mediator_pair": pair_label, "accuracy": acc, "n": len(qs),
                "delta": delta, "c2w": c2w, "w2c": w2c, "switches": switches,
                "c2w_w2c_ratio": c2w / max(w2c, 1),
            })
            logger.info("  %s: %.1f%% (delta %+.1fpp, C2W=%d, W2C=%d)",
                         cid, acc * 100, delta * 100, c2w, w2c)
            save_checkpoint(data, results_path)
            del specialist, mediator; gc.collect(); torch.cuda.empty_cache()

    save_checkpoint(data, results_path)
    return data


def main():
    parser = argparse.ArgumentParser(description="RP mediator experiment")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--specialist-dir", default="rp_adapters")
    parser.add_argument("--mediator-dir", default="rp_mediators")
    parser.add_argument("--results-dir", default="results/rp_mediator")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    start = time.time()
    data = run_experiment(args)
    elapsed = time.time() - start
    data["total_time_minutes"] = elapsed / 60
    save_checkpoint(data, os.path.join(args.results_dir, "rp_mediator_results.json"))
    logger.info("Done in %.1f min", elapsed / 60)

    print("\n" + "=" * 70)
    print("RP MEDIATOR RESULTS (thinking-enabled)")
    print("=" * 70)
    for c in sorted(data["conditions"], key=lambda x: x["id"]):
        acc = c["accuracy"] * 100
        line = f"  {c['id']:<50} {acc:5.1f}%"
        if "delta" in c:
            line += f"  {c['delta']*100:+6.1f}pp"
        if "c2w" in c:
            line += f"  C2W={c['c2w']} W2C={c['w2c']} ratio={c['c2w_w2c_ratio']:.1f}x"
        print(line)


if __name__ == "__main__":
    main()
