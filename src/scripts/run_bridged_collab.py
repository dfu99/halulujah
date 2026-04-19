#!/usr/bin/env python3
"""Three-agent bridged collaboration: Specialist A ↔ Mediator ↔ Specialist B.

Tests whether a mediator acting as interpreter between two domain specialists
outperforms direct two-agent collaboration. Uses full fine-tuned models.

Protocol:
  1. Specialist A reasons from domain A perspective
  2. Mediator translates A's insights for B
  3. Specialist B reasons from domain B perspective
  4. Mediator synthesizes and gives final answer

Conditions:
  - 3-agent bridged: Specialist A + Mediator + Specialist B
  - 2-agent direct: Specialist A + Specialist B (no mediator, for comparison)
  - 2-agent mediated: Specialist A + Mediator (no specialist B)
  - Solo baselines for A, B, and mediator

Tests on BOTH domain A and domain B questions.
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
    bridged_reasoning,
    collab_reasoning_scoped,
    solo_reasoning,
)
from halulujah.domain.cross_eval import extract_answer_letter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAIR = ("medicine", "physics")


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

def load_model(model_dir, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    return model


def eval_bridged(spec_a, mediator, spec_b, tokenizer, questions,
                 domain_a, domain_b, n_cycles, device):
    """Evaluate 3-agent bridged collaboration."""
    correct, c2w, w2c, switches = 0, 0, 0, 0
    for entry in questions:
        final, chain, pre = bridged_reasoning(
            spec_a, mediator, spec_b, tokenizer, entry["question"],
            domain_a, domain_b, n_cycles=n_cycles, device=device,
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


def eval_two_agent(model_a, model_b, tokenizer, questions,
                   domain_a, domain_b, n_rounds, device):
    """Evaluate 2-agent collaboration."""
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
    from transformers import AutoTokenizer
    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_a, domain_b = PAIR

    results_path = os.path.join(args.results_dir, "bridged_results.json")
    os.makedirs(args.results_dir, exist_ok=True)
    data = load_checkpoint(results_path)
    data["config"] = {
        "pair": list(PAIR), "n_questions": args.n_questions,
        "n_cycles": args.n_cycles, "model_name": args.model_name,
        "protocol": "bridged_3_agent",
    }

    # Load test data for both domains
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

    spec_a_dir = os.path.join(args.specialist_dir, f"full_ft_{domain_a}")
    spec_b_dir = os.path.join(args.specialist_dir, f"full_ft_{domain_b}")
    mediator_dir = os.path.join(
        args.mediator_dir, f"full_ft_mediator_{domain_a}_{domain_b}",
    )

    # Test on BOTH domain's questions
    for test_domain in [domain_a, domain_b]:
        qs = test_data[test_domain][:args.n_questions]
        label = test_domain

        # Solo baselines (specialist for this domain)
        spec_dir = spec_a_dir if test_domain == domain_a else spec_b_dir
        cid = f"solo_{label}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            spec = load_model(spec_dir, device)
            correct = 0
            for entry in qs:
                final, _ = solo_reasoning(
                    spec, tokenizer, entry["question"], test_domain,
                    n_rounds=3, device=device,
                )
                if extract_answer_letter(final) == entry["answer_letter"]:
                    correct += 1
            acc = correct / len(qs)
            data["conditions"].append({
                "id": cid, "type": "solo", "domain": label,
                "accuracy": acc, "n": len(qs),
            })
            logger.info("  %s: %.1f%%", cid, acc * 100)
            save_checkpoint(data, results_path)
            del spec; gc.collect(); torch.cuda.empty_cache()

        solo_acc = get_acc(data, f"solo_{label}")

        # 2-agent direct: Specialist A + Specialist B (no mediator)
        cid = f"direct_{domain_a}_{domain_b}_on_{label}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            spec_a = load_model(spec_a_dir, device)
            spec_b = load_model(spec_b_dir, device)
            # Primary specialist is the one whose domain matches test questions
            if test_domain == domain_a:
                acc, c2w, w2c, sw = eval_two_agent(
                    spec_a, spec_b, tokenizer, qs, domain_a, domain_b, 3, device)
            else:
                acc, c2w, w2c, sw = eval_two_agent(
                    spec_b, spec_a, tokenizer, qs, domain_b, domain_a, 3, device)
            delta = acc - (solo_acc or 0)
            data["conditions"].append({
                "id": cid, "type": "direct_2agent", "test_domain": label,
                "accuracy": acc, "n": len(qs), "delta": delta,
                "c2w": c2w, "w2c": w2c, "switches": sw,
                "c2w_w2c_ratio": c2w / max(w2c, 1),
            })
            logger.info("  %s: %.1f%% (delta %+.1fpp)", cid, acc * 100, delta * 100)
            save_checkpoint(data, results_path)
            del spec_a, spec_b; gc.collect(); torch.cuda.empty_cache()

        # 3-agent bridged: Specialist A + Mediator + Specialist B
        cid = f"bridged_{domain_a}_{domain_b}_on_{label}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            spec_a = load_model(spec_a_dir, device)
            med = load_model(mediator_dir, device)
            spec_b = load_model(spec_b_dir, device)
            acc, c2w, w2c, sw = eval_bridged(
                spec_a, med, spec_b, tokenizer, qs,
                domain_a, domain_b, args.n_cycles, device,
            )
            delta = acc - (solo_acc or 0)
            data["conditions"].append({
                "id": cid, "type": "bridged_3agent", "test_domain": label,
                "accuracy": acc, "n": len(qs), "delta": delta,
                "c2w": c2w, "w2c": w2c, "switches": sw,
                "c2w_w2c_ratio": c2w / max(w2c, 1),
            })
            logger.info("  %s: %.1f%% (delta %+.1fpp)", cid, acc * 100, delta * 100)
            save_checkpoint(data, results_path)
            del spec_a, med, spec_b; gc.collect(); torch.cuda.empty_cache()

    save_checkpoint(data, results_path)
    return data


def main():
    parser = argparse.ArgumentParser(description="Three-agent bridged collaboration")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--specialist-dir", default="full_ft_models")
    parser.add_argument("--mediator-dir", default="full_ft_models")
    parser.add_argument("--results-dir", default="results/bridged")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-cycles", type=int, default=2,
                        help="Number of A→M→B→M cycles")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    start = time.time()
    data = run_experiment(args)
    elapsed = time.time() - start
    data["total_time_minutes"] = elapsed / 60
    save_checkpoint(data, os.path.join(args.results_dir, "bridged_results.json"))
    logger.info("Done in %.1f min", elapsed / 60)

    print("\n" + "=" * 80)
    print("BRIDGED 3-AGENT COLLABORATION RESULTS")
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
