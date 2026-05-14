#!/usr/bin/env python3
"""Full FT across 5 domains: confirm LoRA-vs-FT finding generalizes.

Train-evaluate-compress pattern to manage disk quota.
For each domain: train full FT, evaluate solo+base, compress checkpoint.
Then evaluate top cross-domain pairs.
"""

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
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

DOMAINS = ["medicine", "physics", "law", "math", "biology"]
CROSS_PAIRS = [
    ("medicine", "physics"),
    ("medicine", "biology"),
    ("physics", "math"),
    ("law", "medicine"),
]


def train_full_ft(domain, model_name, model_dir, cache_dir=None,
                  num_epochs=3, lr=2e-5):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    path = os.path.join(model_dir, f"full_ft_{domain}")
    if os.path.exists(os.path.join(path, "config.json")):
        logger.info("Full FT %s already trained, skipping", domain)
        return path

    # Check for compressed version
    gz_path = path + ".tar.gz"
    if os.path.exists(gz_path):
        logger.info("Decompressing %s...", gz_path)
        subprocess.run(["tar", "xzf", gz_path, "-C", model_dir], check=True)
        return path

    logger.info("=== Training FULL FT: %s ===", domain)
    entries = load_mmlu_domain(domain, split="test", cache_dir=cache_dir)
    try:
        aux = load_mmlu_domain(domain, split="validation", cache_dir=cache_dir)
        entries.extend(aux)
    except Exception:
        pass
    train_entries, _ = split_train_test(entries, test_size=50, seed=42)
    logger.info("  %d training entries (FULL FT)", len(train_entries))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = format_domain_reasoning_preserved(
        train_entries, tokenizer, domain)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir)

    training_args = SFTConfig(
        output_dir=path, num_train_epochs=num_epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=lr, optim="adafactor", warmup_ratio=0.1,
        logging_steps=10, save_strategy="no", bf16=True,
        gradient_checkpointing=True, max_length=512)

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info("  Saved full FT %s", domain)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return path


def compress_checkpoint(model_dir, domain):
    path = os.path.join(model_dir, f"full_ft_{domain}")
    gz_path = path + ".tar.gz"
    if os.path.exists(path) and not os.path.exists(gz_path):
        logger.info("Compressing %s...", path)
        subprocess.run(
            ["tar", "czf", gz_path, "-C", model_dir, f"full_ft_{domain}"],
            check=True)
        shutil.rmtree(path)
        logger.info("  Compressed and removed %s", path)


def decompress_checkpoint(model_dir, domain):
    path = os.path.join(model_dir, f"full_ft_{domain}")
    gz_path = path + ".tar.gz"
    if not os.path.exists(path) and os.path.exists(gz_path):
        logger.info("Decompressing %s...", gz_path)
        subprocess.run(["tar", "xzf", gz_path, "-C", model_dir], check=True)


def load_model(model_dir, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
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

    results_path = os.path.join(args.results_dir, "full_ft_5domain.json")
    os.makedirs(args.results_dir, exist_ok=True)
    data = load_checkpoint(results_path)
    data["config"] = {
        "domains": DOMAINS, "cross_pairs": [list(p) for p in CROSS_PAIRS],
        "n_questions": args.n_questions, "n_rounds": args.n_rounds,
        "model_name": args.model_name, "training": "full_fine_tuning",
    }

    # Load test data
    test_data = {}
    for d in DOMAINS:
        entries = load_mmlu_domain(d, split="test", cache_dir=args.cache_dir)
        _, test = split_train_test(entries, test_size=args.n_questions, seed=42)
        test_data[d] = test

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Phase 1: Train + evaluate solo + base for each domain
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir).to(device)
    base_model.eval()

    for domain in DOMAINS:
        if not args.skip_training:
            train_full_ft(domain, args.model_name, args.model_dir,
                         cache_dir=args.cache_dir)

        model_path = os.path.join(args.model_dir, f"full_ft_{domain}")
        decompress_checkpoint(args.model_dir, domain)

        if not os.path.exists(model_path):
            logger.warning("No model for %s, skipping", domain)
            continue

        qs = test_data[domain][:args.n_questions]

        # Solo
        cid = f"solo_{domain}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            specialist = load_model(model_path, device)
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
                "accuracy": acc, "n": len(qs)})
            logger.info("  %s: %.1f%%", cid, acc * 100)
            save_checkpoint(data, results_path)
            del specialist; gc.collect(); torch.cuda.empty_cache()

        solo_acc = get_acc(data, f"solo_{domain}")

        # +Base
        cid = f"ft_{domain}_plus_base"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            specialist = load_model(model_path, device)
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
                "id": cid, "type": "ft_plus_base", "domain": domain,
                "accuracy": acc, "n": len(qs), "delta": delta,
                "c2w": c2w, "w2c": w2c, "switches": switches,
                "c2w_w2c_ratio": c2w / max(w2c, 1)})
            logger.info("  %s: %.1f%% (delta %+.1fpp, C2W=%d, W2C=%d)",
                         cid, acc * 100, delta * 100, c2w, w2c)
            save_checkpoint(data, results_path)
            del specialist; gc.collect(); torch.cuda.empty_cache()

        # Compress to save disk
        if args.compress:
            compress_checkpoint(args.model_dir, domain)

    del base_model; gc.collect(); torch.cuda.empty_cache()

    # Phase 2: Cross-domain pairs
    for domain_a, domain_b in CROSS_PAIRS:
        cid = f"ft_{domain_a}_plus_ft_{domain_b}"
        if done(data, cid):
            continue

        decompress_checkpoint(args.model_dir, domain_a)
        decompress_checkpoint(args.model_dir, domain_b)

        path_a = os.path.join(args.model_dir, f"full_ft_{domain_a}")
        path_b = os.path.join(args.model_dir, f"full_ft_{domain_b}")

        if not os.path.exists(path_a) or not os.path.exists(path_b):
            logger.warning("Missing model for %s or %s, skipping pair",
                          domain_a, domain_b)
            continue

        logger.info("=== %s ===", cid)
        model_a = load_model(path_a, device)
        model_b = load_model(path_b, device)

        qs = test_data[domain_a][:args.n_questions]
        solo_acc = get_acc(data, f"solo_{domain_a}")

        correct, c2w, w2c, switches = 0, 0, 0, 0
        for entry in qs:
            final, chain, pre = collab_reasoning_scoped(
                model_a, model_b, tokenizer, entry["question"],
                domain_a, domain_b, n_rounds=args.n_rounds,
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
            "id": cid, "type": "ft_cross_pair",
            "domain_a": domain_a, "domain_b": domain_b,
            "accuracy": acc, "n": len(qs), "delta": delta,
            "c2w": c2w, "w2c": w2c, "switches": switches,
            "c2w_w2c_ratio": c2w / max(w2c, 1)})
        logger.info("  %s: %.1f%% (delta %+.1fpp)", cid, acc * 100, delta * 100)
        save_checkpoint(data, results_path)

        del model_a, model_b; gc.collect(); torch.cuda.empty_cache()

        if args.compress:
            compress_checkpoint(args.model_dir, domain_a)
            compress_checkpoint(args.model_dir, domain_b)

    save_checkpoint(data, results_path)
    return data


def main():
    parser = argparse.ArgumentParser(description="Full FT 5-domain expansion")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-dir", default="full_ft_5domain")
    parser.add_argument("--results-dir", default="results/paper_sweep/full_ft_5domain")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--compress", action="store_true", default=True)
    args = parser.parse_args()

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    start = time.time()
    data = run_experiment(args)
    elapsed = time.time() - start
    data["total_time_minutes"] = elapsed / 60
    save_checkpoint(data, os.path.join(args.results_dir, "full_ft_5domain.json"))
    logger.info("Done in %.1f min (%.1f hrs)", elapsed / 60, elapsed / 3600)


if __name__ == "__main__":
    main()
