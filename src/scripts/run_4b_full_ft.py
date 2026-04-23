#!/usr/bin/env python3
"""Qwen3-4B Full FT on medicine + physics.

Tests whether full FT collaboration finding scales to 4B model. Trains one
domain at a time, evaluates solo + base collaboration, deletes model before
training next (disk quota ~20GB, 4B model = 8GB).

Skips cross-domain pairs (requires 2 models on disk simultaneously = 16GB,
doesn't fit with 12GB hf_cache leaving only ~4GB free above 4B).

Conditions per domain:
  solo_4b_D         - 4B full FT specialist alone
  ft_4b_D_plus_base - 4B full FT specialist + 4B base
"""

import argparse
import gc
import json
import logging
import os
import shutil
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

DEFAULT_DOMAINS = ["medicine", "physics"]


def train_full_ft_4b(domain, model_name, output_dir, cache_dir=None):
    """Train 4B full FT with aggressive memory management."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    path = os.path.join(output_dir, f"full_ft_4b_{domain}")
    if os.path.exists(os.path.join(path, "config.json")):
        logger.info("4B FT %s already trained", domain)
        return path

    logger.info("=== Training 4B FULL FT: %s ===", domain)

    entries = load_mmlu_domain(domain, split="test", cache_dir=cache_dir)
    try:
        aux = load_mmlu_domain(domain, split="validation", cache_dir=cache_dir)
        entries.extend(aux)
    except Exception:
        pass
    train_entries, _ = split_train_test(entries, test_size=200, seed=42)
    logger.info("Training %s on %d entries (4B full FT)", domain, len(train_entries))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = format_domain_reasoning_preserved(
        train_entries, tokenizer, domain, max_length=384)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir)

    total = sum(p.numel() for p in model.parameters())
    logger.info("  4B full FT: %d params trainable", total)

    # Aggressive memory settings for 4B on 24GB
    training_args = SFTConfig(
        output_dir=path,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # higher accum for small batch
        learning_rate=2e-5,
        optim="adafactor",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        max_length=384,  # shorter for memory
    )

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info("  Saved 4B full FT %s", domain)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return path


def load_model(model_dir, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, trust_remote_code=True).to(device)
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

    results_path = os.path.join(args.results_dir, "4b_full_ft.json")
    os.makedirs(args.results_dir, exist_ok=True)
    data = load_checkpoint(results_path)
    data["config"] = {
        "domains": args.domains, "n_questions": args.n_questions,
        "n_rounds": args.n_rounds, "model_name": args.model_name,
        "training": "full_fine_tuning_4b",
    }

    test_data = {}
    for d in args.domains:
        entries = load_mmlu_domain(d, split="test", cache_dir=args.cache_dir)
        _, test = split_train_test(entries, test_size=args.n_questions, seed=42)
        test_data[d] = test

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base 4B model
    logger.info("Loading 4B base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir).to(device)
    base_model.eval()

    for domain in args.domains:
        cids_needed = [f"solo_4b_{domain}", f"ft_4b_{domain}_plus_base"]
        if all(done(data, c) for c in cids_needed):
            logger.info("Skipping %s — all 4B conditions done", domain)
            continue

        # Free VRAM before training: unload base
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        model_path = os.path.join(args.model_dir, f"full_ft_4b_{domain}")
        if not os.path.exists(model_path):
            train_full_ft_4b(domain, args.model_name, args.model_dir,
                            cache_dir=args.cache_dir)

        qs = test_data[domain][:args.n_questions]

        # Solo
        cid = f"solo_4b_{domain}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            spec = load_model(model_path, device)
            correct = 0
            for entry in qs:
                final, _ = solo_reasoning(
                    spec, tokenizer, entry["question"], domain,
                    n_rounds=args.n_rounds, device=device)
                if extract_answer_letter(final) == entry["answer_letter"]:
                    correct += 1
            acc = correct / len(qs)
            data["conditions"].append({
                "id": cid, "type": "solo_4b", "domain": domain,
                "accuracy": acc, "n": len(qs)})
            logger.info("  %s: %.1f%%", cid, acc * 100)
            save_checkpoint(data, results_path)
            del spec; gc.collect(); torch.cuda.empty_cache()

        solo_acc = get_acc(data, cid)

        # Reload base model for collaboration
        logger.info("Reloading 4B base for collab...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=torch.bfloat16,
            trust_remote_code=True, cache_dir=args.cache_dir).to(device)
        base_model.eval()

        # FT + base
        cid = f"ft_4b_{domain}_plus_base"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            spec = load_model(model_path, device)
            correct, c2w, w2c, switches = 0, 0, 0, 0
            for entry in qs:
                final, chain, pre = collab_reasoning_scoped(
                    spec, base_model, tokenizer, entry["question"],
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
                "id": cid, "type": "ft_4b_plus_base", "domain": domain,
                "accuracy": acc, "n": len(qs), "delta": delta,
                "c2w": c2w, "w2c": w2c, "switches": switches,
                "c2w_w2c_ratio": c2w / max(w2c, 1)})
            logger.info("  %s: %.1f%% (delta %+.1fpp)", cid, acc * 100, delta * 100)
            save_checkpoint(data, results_path)
            del spec; gc.collect(); torch.cuda.empty_cache()

        # Delete FT model to free disk
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            logger.info("Deleted %s to free disk", model_path)

    del base_model; gc.collect(); torch.cuda.empty_cache()
    save_checkpoint(data, results_path)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--model-dir", default="qwen3_4b_ft")
    parser.add_argument("--results-dir", default="results/paper_sweep/qwen3_4b_ft")
    parser.add_argument("--n-questions", type=int, default=200)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS,
                        help="MMLU domains to train and evaluate, space-separated")
    args = parser.parse_args()

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    start = time.time()
    data = run_experiment(args)
    elapsed = time.time() - start
    data["total_time_minutes"] = elapsed / 60
    save_checkpoint(data, os.path.join(args.results_dir, "4b_full_ft.json"))
    logger.info("Done in %.1f min", elapsed / 60)


if __name__ == "__main__":
    main()
