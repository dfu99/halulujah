#!/usr/bin/env python3
"""LoRA rank sweep with RP format: does higher rank recover collaborativeness?

Sweeps r=4,8,16,32,64,128 on medicine+physics with reasoning-preserved training.
Evaluates solo and +base collaboration at each rank.

Key question: if r→∞ gradually restores collaboration delta toward full FT's +14.7pp,
then LoRA's rank constraint IS the mechanism killing collaboration.
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
RANKS = [4, 8, 16, 32, 64, 128]


def train_lora_at_rank(domain, rank, model_name, adapter_dir,
                       cache_dir=None, num_epochs=3, lr=5e-5):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    path = os.path.join(adapter_dir, f"adapter_{domain}_r{rank}")
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        logger.info("LoRA %s r=%d already trained, skipping", domain, rank)
        return path

    logger.info("=== Training LoRA %s r=%d (RP format) ===", domain, rank)

    entries = load_mmlu_domain(domain, split="test", cache_dir=cache_dir)
    try:
        aux = load_mmlu_domain(domain, split="validation", cache_dir=cache_dir)
        entries.extend(aux)
    except Exception:
        pass
    train_entries, _ = split_train_test(entries, test_size=50, seed=42)
    logger.info("  %d training entries", len(train_entries))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = format_domain_reasoning_preserved(
        train_entries, tokenizer, domain)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir)
    lora_config = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("  r=%d: trainable %d / %d (%.2f%%)",
                rank, trainable, total, 100 * trainable / total)

    training_args = SFTConfig(
        output_dir=path, num_train_epochs=num_epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=lr, warmup_ratio=0.1, logging_steps=10,
        save_strategy="no", bf16=True, gradient_checkpointing=True,
        max_length=512)

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(path)
    logger.info("  Saved LoRA %s r=%d to %s", domain, rank, path)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return path


def load_specialist(model_name, adapter_path, device, cache_dir=None):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir).to(device)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


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
    ranks = args.ranks

    results_path = os.path.join(args.results_dir, "rank_sweep_rp.json")
    os.makedirs(args.results_dir, exist_ok=True)
    data = load_checkpoint(results_path)
    data["config"] = {
        "ranks": ranks, "domains": DOMAINS,
        "n_questions": args.n_questions, "n_rounds": args.n_rounds,
        "model_name": args.model_name, "format": "reasoning_preserved",
    }

    # Phase 1: Train all adapters
    if not args.skip_training:
        for rank in ranks:
            for domain in DOMAINS:
                train_lora_at_rank(
                    domain, rank, args.model_name, args.adapter_dir,
                    cache_dir=args.cache_dir)

    # Phase 2: Load test data
    test_data = {}
    for d in DOMAINS:
        entries = load_mmlu_domain(d, split="test", cache_dir=args.cache_dir)
        _, test = split_train_test(entries, test_size=args.n_questions, seed=42)
        test_data[d] = test

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (stays resident)
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir).to(device)
    base_model.eval()

    # Phase 3: Evaluate each rank
    for rank in ranks:
        for domain in DOMAINS:
            adapter_path = os.path.join(args.adapter_dir, f"adapter_{domain}_r{rank}")
            if not os.path.exists(adapter_path):
                logger.warning("No adapter for %s r=%d, skipping", domain, rank)
                continue

            qs = test_data[domain][:args.n_questions]

            # Solo
            cid = f"solo_{domain}_r{rank}"
            if not done(data, cid):
                logger.info("=== %s ===", cid)
                specialist = load_specialist(
                    args.model_name, adapter_path, device, args.cache_dir)
                correct = 0
                for entry in qs:
                    final, _ = solo_reasoning(
                        specialist, tokenizer, entry["question"], domain,
                        n_rounds=args.n_rounds, device=device)
                    if extract_answer_letter(final) == entry["answer_letter"]:
                        correct += 1
                acc = correct / len(qs)
                data["conditions"].append({
                    "id": cid, "type": "solo", "domain": domain,
                    "rank": rank, "accuracy": acc, "n": len(qs)})
                logger.info("  %s: %.1f%%", cid, acc * 100)
                save_checkpoint(data, results_path)
                del specialist; gc.collect(); torch.cuda.empty_cache()

            solo_acc = get_acc(data, f"solo_{domain}_r{rank}")

            # +Base
            cid = f"collab_{domain}_r{rank}_plus_base"
            if not done(data, cid):
                logger.info("=== %s ===", cid)
                specialist = load_specialist(
                    args.model_name, adapter_path, device, args.cache_dir)
                correct, c2w, w2c, switches = 0, 0, 0, 0
                for entry in qs:
                    final, chain, pre = collab_reasoning_scoped(
                        specialist, base_model, tokenizer, entry["question"],
                        domain, "general", n_rounds=args.n_rounds,
                        device=device, protocol="full-cot")
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
                    "id": cid, "type": "collab_base", "domain": domain,
                    "rank": rank, "accuracy": acc, "n": len(qs),
                    "delta": delta, "c2w": c2w, "w2c": w2c,
                    "switches": switches,
                    "c2w_w2c_ratio": c2w / max(w2c, 1)})
                logger.info("  %s: %.1f%% (delta %+.1fpp, C2W=%d, W2C=%d)",
                             cid, acc * 100, delta * 100, c2w, w2c)
                save_checkpoint(data, results_path)
                del specialist; gc.collect(); torch.cuda.empty_cache()

    del base_model; gc.collect(); torch.cuda.empty_cache()
    save_checkpoint(data, results_path)
    return data


def main():
    parser = argparse.ArgumentParser(description="LoRA rank sweep (RP format)")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-dir", default="rank_sweep_adapters")
    parser.add_argument("--results-dir", default="results/paper_sweep/rank_sweep_rp")
    parser.add_argument("--ranks", nargs="+", type=int, default=RANKS)
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
    save_checkpoint(data, os.path.join(args.results_dir, "rank_sweep_rp.json"))
    logger.info("Done in %.1f min (%.1f hrs)", elapsed / 60, elapsed / 3600)

    print("\n" + "=" * 80)
    print("LORA RANK SWEEP (RP FORMAT)")
    print("=" * 80)
    for c in sorted(data["conditions"], key=lambda x: (x.get("rank", 0), x["id"])):
        acc = c["accuracy"] * 100
        line = f"  {c['id']:<40} {acc:5.1f}%"
        if "delta" in c:
            line += f"  {c['delta']*100:+6.1f}pp"
        if "c2w" in c:
            line += f"  C2W={c['c2w']:>2} W2C={c['w2c']:>2} ratio={c['c2w_w2c_ratio']:.1f}x"
        print(line)


if __name__ == "__main__":
    main()
