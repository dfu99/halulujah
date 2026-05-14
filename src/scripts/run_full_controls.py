#!/usr/bin/env python3
"""All control conditions at N=200 for paper.

For each domain D in {medicine, physics, law, math, biology}:
  - solo_D (specialist alone)
  - ft_D_plus_base (specialist + untrained base)
  - base_solo_D (untrained base alone on D's questions)
  - same_D (specialist + itself)
  - base_pair_D (base + base)

Cross-domain pairs (4 pairs, bidirectional):
  - ft_A_plus_ft_B (specialist A + specialist B on A's questions)

Medicine+Physics focused mediator/bridge conditions:
  - ft_mediator_med_phys (trained mediator solo on both domains)
  - med_plus_mediator (2-agent)
  - phys_plus_mediator (2-agent)
  - bridged_med_phys_on_med (3-agent: spec+med+spec)
  - bridged_med_phys_on_phys (3-agent)
  - bridged_base_as_mediator (3-agent with base as mediator)

Uses run_full_ft_5domain.py's checkpoint file so existing results are preserved.
"""

import argparse
import gc
import json
import logging
import os
import subprocess
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
from halulujah.domain.collab_eval import (
    bridged_reasoning,
    collab_reasoning_scoped,
    solo_reasoning,
)
from halulujah.domain.cross_eval import extract_answer_letter
from scripts.run_reasoning_preserved import format_domain_reasoning_preserved
from scripts.run_full_ft_5domain import (
    train_full_ft, compress_checkpoint, decompress_checkpoint, load_model,
    DOMAINS, CROSS_PAIRS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MEDIATOR_PAIR = ("medicine", "physics")


def format_mediator_rp(entries, tokenizer, domain_a, domain_b):
    sys_prompt = (f"You are an expert in both {domain_a} and {domain_b}. "
                  "Answer questions accurately, drawing on knowledge from both fields.")
    texts = []
    method = None
    for entry in entries:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": entry["question"]},
            {"role": "assistant", "content": entry["answer"]},
        ]
        if method != "manual":
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False)
                if "<think>\n\n</think>" not in text:
                    method = "template"
                else:
                    method = "manual"
            except TypeError:
                method = "manual"
        if method == "manual":
            text = (f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{entry['question']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{entry['answer']}<|im_end|>\n")
        texts.append(text)
    from datasets import Dataset
    return Dataset.from_dict({"text": texts})


def train_ft_mediator(domain_a, domain_b, model_name, output_dir, cache_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    path = os.path.join(output_dir, f"ft_mediator_{domain_a}_{domain_b}")
    if os.path.exists(os.path.join(path, "config.json")):
        logger.info("FT mediator already trained")
        return path
    gz = path + ".tar.gz"
    if os.path.exists(gz):
        subprocess.run(["tar", "xzf", gz, "-C", output_dir], check=True)
        return path

    logger.info("=== Training FT mediator: %s + %s ===", domain_a, domain_b)
    all_entries = load_mmlu_mediator(domain_a, domain_b, split="test", cache_dir=cache_dir)
    try:
        aux = load_mmlu_mediator(domain_a, domain_b, split="validation", cache_dir=cache_dir)
        all_entries.extend(aux)
    except Exception:
        pass

    entries_a = load_mmlu_domain(domain_a, split="test", cache_dir=cache_dir)
    entries_b = load_mmlu_domain(domain_b, split="test", cache_dir=cache_dir)
    _, test_a = split_train_test(entries_a, test_size=200, seed=42)
    _, test_b = split_train_test(entries_b, test_size=200, seed=42)
    test_qs = {e["question"] for e in test_a + test_b}
    train_entries = [e for e in all_entries if e["question"] not in test_qs]
    logger.info("Training data: %d entries", len(train_entries))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = format_mediator_rp(train_entries, tokenizer, domain_a, domain_b)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir=cache_dir)

    training_args = SFTConfig(
        output_dir=path, num_train_epochs=3,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=2e-5, optim="adafactor", warmup_ratio=0.1,
        logging_steps=10, save_strategy="no", bf16=True,
        gradient_checkpointing=True, max_length=512)
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info("Saved mediator to %s", path)
    del model, trainer; gc.collect(); torch.cuda.empty_cache()
    return path


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


def eval_solo(model, tokenizer, qs, domain, n_rounds, device):
    correct = 0
    for entry in qs:
        final, _ = solo_reasoning(
            model, tokenizer, entry["question"], domain,
            n_rounds=n_rounds, device=device)
        if extract_answer_letter(final) == entry["answer_letter"]:
            correct += 1
    return correct / len(qs)


def eval_collab(a, b, tokenizer, qs, da, db, n_rounds, device):
    correct, c2w, w2c, switches = 0, 0, 0, 0
    for entry in qs:
        final, chain, pre = collab_reasoning_scoped(
            a, b, tokenizer, entry["question"], da, db,
            n_rounds=n_rounds, device=device, protocol="full-cot")
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
    return correct / len(qs), c2w, w2c, switches


def eval_bridged(a, m, b, tokenizer, qs, da, db, n_cycles, device):
    correct, c2w, w2c, switches = 0, 0, 0, 0
    for entry in qs:
        final, chain, pre = bridged_reasoning(
            a, m, b, tokenizer, entry["question"], da, db,
            n_cycles=n_cycles, device=device)
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
    return correct / len(qs), c2w, w2c, switches


def add_condition(data, results_path, **kwargs):
    """Add a condition result and save."""
    data["conditions"].append(kwargs)
    save_checkpoint(data, results_path)


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

    # Load base model (stays throughout)
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir).to(device)
    base_model.eval()

    # ========== PHASE 1: Base-only conditions (no specialist needed) ==========
    for domain in DOMAINS:
        qs = test_data[domain][:args.n_questions]

        # Base solo on domain's questions
        cid = f"base_solo_{domain}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            acc = eval_solo(base_model, tokenizer, qs, "general",
                            args.n_rounds, device)
            add_condition(data, results_path,
                id=cid, type="base_solo", domain=domain,
                accuracy=acc, n=len(qs))
            logger.info("  %s: %.1f%%", cid, acc * 100)

        # Base + base on domain's questions
        cid = f"base_pair_{domain}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            base_solo = get_acc(data, f"base_solo_{domain}")
            acc, c2w, w2c, sw = eval_collab(
                base_model, base_model, tokenizer, qs,
                "general", "general", args.n_rounds, device)
            delta = acc - (base_solo or 0)
            add_condition(data, results_path,
                id=cid, type="base_pair", domain=domain,
                accuracy=acc, n=len(qs), delta=delta,
                c2w=c2w, w2c=w2c, switches=sw,
                c2w_w2c_ratio=c2w / max(w2c, 1))
            logger.info("  %s: %.1f%% (delta %+.1fpp)",
                         cid, acc * 100, delta * 100)

        # COMPUTE-MATCHED CONTROL: Single agent, 2x rounds (matches base_pair compute)
        cid = f"base_solo_2x_{domain}"
        if not done(data, cid):
            logger.info("=== %s (compute-matched: single agent, %d rounds) ===",
                         cid, args.n_rounds * 2)
            acc = eval_solo(base_model, tokenizer, qs, "general",
                            args.n_rounds * 2, device)
            add_condition(data, results_path,
                id=cid, type="base_solo_2x_compute", domain=domain,
                accuracy=acc, n=len(qs), n_rounds=args.n_rounds * 2)
            logger.info("  %s: %.1f%%", cid, acc * 100)

    # ========== PHASE 2: Specialist conditions (train + 4 condition types) ==========
    for domain in DOMAINS:
        model_path = os.path.join(args.model_dir, f"full_ft_{domain}")
        gz_path = model_path + ".tar.gz"

        # Need specialist if any condition below isn't done
        cids_need_spec = [f"solo_{domain}", f"ft_{domain}_plus_base",
                         f"same_{domain}"]
        if all(done(data, c) for c in cids_need_spec):
            logger.info("Skipping %s — all single-specialist conditions done", domain)
            continue

        # Train or decompress
        if not os.path.exists(model_path) and os.path.exists(gz_path):
            decompress_checkpoint(args.model_dir, domain)
        if not os.path.exists(model_path):
            train_full_ft(domain, args.model_name, args.model_dir,
                         cache_dir=args.cache_dir)

        qs = test_data[domain][:args.n_questions]
        specialist = load_model(model_path, device)

        # Solo specialist
        cid = f"solo_{domain}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            acc = eval_solo(specialist, tokenizer, qs, domain,
                            args.n_rounds, device)
            add_condition(data, results_path,
                id=cid, type="solo", domain=domain,
                accuracy=acc, n=len(qs))
            logger.info("  %s: %.1f%%", cid, acc * 100)

        solo_acc = get_acc(data, f"solo_{domain}")

        # Specialist + base
        cid = f"ft_{domain}_plus_base"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            acc, c2w, w2c, sw = eval_collab(
                specialist, base_model, tokenizer, qs,
                domain, "general", args.n_rounds, device)
            delta = acc - (solo_acc or 0)
            add_condition(data, results_path,
                id=cid, type="ft_plus_base", domain=domain,
                accuracy=acc, n=len(qs), delta=delta,
                c2w=c2w, w2c=w2c, switches=sw,
                c2w_w2c_ratio=c2w / max(w2c, 1))
            logger.info("  %s: %.1f%% (delta %+.1fpp)", cid, acc * 100, delta * 100)

        # Same-domain pair (specialist + itself)
        cid = f"same_{domain}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            acc, c2w, w2c, sw = eval_collab(
                specialist, specialist, tokenizer, qs,
                domain, domain, args.n_rounds, device)
            delta = acc - (solo_acc or 0)
            add_condition(data, results_path,
                id=cid, type="same_pair", domain=domain,
                accuracy=acc, n=len(qs), delta=delta,
                c2w=c2w, w2c=w2c, switches=sw,
                c2w_w2c_ratio=c2w / max(w2c, 1))
            logger.info("  %s: %.1f%% (delta %+.1fpp)", cid, acc * 100, delta * 100)

        # COMPUTE-MATCHED CONTROL: Specialist alone, 2x rounds (matches same_D compute)
        cid = f"solo_2x_{domain}"
        if not done(data, cid):
            logger.info("=== %s (compute-matched: single specialist, %d rounds) ===",
                         cid, args.n_rounds * 2)
            acc = eval_solo(specialist, tokenizer, qs, domain,
                            args.n_rounds * 2, device)
            add_condition(data, results_path,
                id=cid, type="solo_2x_compute", domain=domain,
                accuracy=acc, n=len(qs), n_rounds=args.n_rounds * 2)
            logger.info("  %s: %.1f%%", cid, acc * 100)

        # Delete specialist to free disk (compression hits quota)
        del specialist; gc.collect(); torch.cuda.empty_cache()
        if os.path.exists(model_path):
            import shutil
            shutil.rmtree(model_path)
            logger.info("Deleted %s to free disk", model_path)

    # ========== PHASE 3: Cross-domain pairs ==========
    for domain_a, domain_b in CROSS_PAIRS:
        cid = f"ft_{domain_a}_plus_ft_{domain_b}"
        if done(data, cid):
            continue

        path_a = os.path.join(args.model_dir, f"full_ft_{domain_a}")
        path_b = os.path.join(args.model_dir, f"full_ft_{domain_b}")

        # Train or decompress A if needed
        if not os.path.exists(path_a):
            if os.path.exists(path_a + ".tar.gz"):
                decompress_checkpoint(args.model_dir, domain_a)
            else:
                train_full_ft(domain_a, args.model_name, args.model_dir,
                             cache_dir=args.cache_dir)

        # Train or decompress B if needed
        if not os.path.exists(path_b):
            if os.path.exists(path_b + ".tar.gz"):
                decompress_checkpoint(args.model_dir, domain_b)
            else:
                train_full_ft(domain_b, args.model_name, args.model_dir,
                             cache_dir=args.cache_dir)

        if not os.path.exists(path_a) or not os.path.exists(path_b):
            logger.warning("Missing model, skipping %s", cid)
            continue

        logger.info("=== %s ===", cid)
        model_a = load_model(path_a, device)
        model_b = load_model(path_b, device)
        qs = test_data[domain_a][:args.n_questions]
        solo_acc = get_acc(data, f"solo_{domain_a}")

        acc, c2w, w2c, sw = eval_collab(
            model_a, model_b, tokenizer, qs, domain_a, domain_b,
            args.n_rounds, device)
        delta = acc - (solo_acc or 0)
        add_condition(data, results_path,
            id=cid, type="ft_cross_pair",
            domain_a=domain_a, domain_b=domain_b,
            accuracy=acc, n=len(qs), delta=delta,
            c2w=c2w, w2c=w2c, switches=sw,
            c2w_w2c_ratio=c2w / max(w2c, 1))
        logger.info("  %s: %.1f%% (delta %+.1fpp)", cid, acc * 100, delta * 100)

        # Delete both models to free disk for next pair
        del model_a, model_b; gc.collect(); torch.cuda.empty_cache()
        import shutil
        if os.path.exists(path_a):
            shutil.rmtree(path_a)
        if os.path.exists(path_b):
            shutil.rmtree(path_b)

    # ========== PHASE 4: Mediator + bridge (medicine+physics focus) ==========
    da, db = MEDIATOR_PAIR

    # Train FT mediator
    mediator_path = train_ft_mediator(
        da, db, args.model_name, args.model_dir, cache_dir=args.cache_dir)

    # Ensure specialists exist (retrain if missing — Phase 3 deleted them)
    spec_a_path = os.path.join(args.model_dir, f"full_ft_{da}")
    spec_b_path = os.path.join(args.model_dir, f"full_ft_{db}")
    if not os.path.exists(spec_a_path):
        if os.path.exists(spec_a_path + ".tar.gz"):
            decompress_checkpoint(args.model_dir, da)
        else:
            train_full_ft(da, args.model_name, args.model_dir, cache_dir=args.cache_dir)
    if not os.path.exists(spec_b_path):
        if os.path.exists(spec_b_path + ".tar.gz"):
            decompress_checkpoint(args.model_dir, db)
        else:
            train_full_ft(db, args.model_name, args.model_dir, cache_dir=args.cache_dir)

    # Mediator solo on each domain
    for d in [da, db]:
        cid = f"ft_mediator_solo_{d}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            mediator = load_model(mediator_path, device)
            qs = test_data[d][:args.n_questions]
            acc = eval_solo(
                mediator, tokenizer, qs,
                f"ft_mediator_{da}_{db}", args.n_rounds, device)
            add_condition(data, results_path,
                id=cid, type="ft_mediator_solo", domain=d,
                accuracy=acc, n=len(qs))
            logger.info("  %s: %.1f%%", cid, acc * 100)
            del mediator; gc.collect(); torch.cuda.empty_cache()

    # Specialist + mediator (2-agent)
    for d in [da, db]:
        cid = f"ft_{d}_plus_mediator"
        if done(data, cid):
            continue
        logger.info("=== %s ===", cid)
        spec_path = spec_a_path if d == da else spec_b_path
        spec = load_model(spec_path, device)
        mediator = load_model(mediator_path, device)
        qs = test_data[d][:args.n_questions]
        solo_acc = get_acc(data, f"solo_{d}")
        acc, c2w, w2c, sw = eval_collab(
            spec, mediator, tokenizer, qs, d,
            f"ft_mediator_{da}_{db}", args.n_rounds, device)
        delta = acc - (solo_acc or 0)
        add_condition(data, results_path,
            id=cid, type="ft_plus_mediator", domain=d,
            accuracy=acc, n=len(qs), delta=delta,
            c2w=c2w, w2c=w2c, switches=sw,
            c2w_w2c_ratio=c2w / max(w2c, 1))
        logger.info("  %s: %.1f%% (delta %+.1fpp)", cid, acc * 100, delta * 100)
        del spec, mediator; gc.collect(); torch.cuda.empty_cache()

    # 3-agent bridge: specialist A + FT mediator + specialist B
    for test_d in [da, db]:
        cid = f"bridged_{da}_{db}_on_{test_d}"
        if done(data, cid):
            continue
        logger.info("=== %s ===", cid)
        spec_a = load_model(spec_a_path, device)
        mediator = load_model(mediator_path, device)
        spec_b = load_model(spec_b_path, device)
        qs = test_data[test_d][:args.n_questions]
        solo_acc = get_acc(data, f"solo_{test_d}")
        acc, c2w, w2c, sw = eval_bridged(
            spec_a, mediator, spec_b, tokenizer, qs, da, db,
            n_cycles=2, device=device)
        delta = acc - (solo_acc or 0)
        add_condition(data, results_path,
            id=cid, type="bridged_3agent", test_domain=test_d,
            accuracy=acc, n=len(qs), delta=delta,
            c2w=c2w, w2c=w2c, switches=sw,
            c2w_w2c_ratio=c2w / max(w2c, 1))
        logger.info("  %s: %.1f%% (delta %+.1fpp)", cid, acc * 100, delta * 100)
        del spec_a, mediator, spec_b; gc.collect(); torch.cuda.empty_cache()

    # 3-agent bridge with BASE as mediator
    for test_d in [da, db]:
        cid = f"bridged_base_mediator_on_{test_d}"
        if done(data, cid):
            continue
        logger.info("=== %s ===", cid)
        spec_a = load_model(spec_a_path, device)
        spec_b = load_model(spec_b_path, device)
        qs = test_data[test_d][:args.n_questions]
        solo_acc = get_acc(data, f"solo_{test_d}")
        acc, c2w, w2c, sw = eval_bridged(
            spec_a, base_model, spec_b, tokenizer, qs, da, db,
            n_cycles=2, device=device)
        delta = acc - (solo_acc or 0)
        add_condition(data, results_path,
            id=cid, type="bridged_base_mediator", test_domain=test_d,
            accuracy=acc, n=len(qs), delta=delta,
            c2w=c2w, w2c=w2c, switches=sw,
            c2w_w2c_ratio=c2w / max(w2c, 1))
        logger.info("  %s: %.1f%% (delta %+.1fpp)", cid, acc * 100, delta * 100)
        del spec_a, spec_b; gc.collect(); torch.cuda.empty_cache()

    compress_checkpoint(args.model_dir, da)
    compress_checkpoint(args.model_dir, db)

    # ========== PHASE 5: Zero-shot baselines ==========
    # Does our base_solo CoT prompt hurt the base model?
    # If zero-shot >> base_solo, the CoT prompt is the problem, not the model.
    from halulujah.domain.collab_eval import _quick_answer

    for domain in DOMAINS:
        cid = f"base_zeroshot_{domain}"
        if not done(data, cid):
            logger.info("=== %s (zero-shot, no CoT, no multi-round) ===", cid)
            qs = test_data[domain][:args.n_questions]
            correct = 0
            for entry in qs:
                raw = _quick_answer(
                    base_model, tokenizer, entry["question"], domain, device)
                pred = extract_answer_letter(raw)
                if pred == entry["answer_letter"]:
                    correct += 1
            acc = correct / len(qs)
            add_condition(data, results_path,
                id=cid, type="base_zeroshot", domain=domain,
                accuracy=acc, n=len(qs))
            logger.info("  %s: %.1f%%", cid, acc * 100)

    del base_model; gc.collect(); torch.cuda.empty_cache()
    save_checkpoint(data, results_path)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-dir", default="full_ft_5domain")
    parser.add_argument("--results-dir", default="results/paper_sweep/full_ft_5domain")
    parser.add_argument("--n-questions", type=int, default=200)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--cache-dir", default=None)
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
