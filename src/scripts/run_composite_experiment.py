#!/usr/bin/env python3
"""Composite question experiment: does collaboration help when structurally necessary?

Tests whether specialists who FAIL at optional collaboration (single-domain
questions) SUCCEED when collaboration is structurally required (composite
questions needing both domains).

Conditions per domain pair (A, B):
  solo_a       - Specialist A alone on composite A+B questions
  solo_b       - Specialist B alone on composite A+B questions
  collab_ab    - A + B deliberating on composite A+B questions
  base_solo    - Base model alone on composite A+B questions
  base_pair    - Two base models deliberating on composite A+B questions

Key comparison:
  - Solo specialists should FAIL (missing half the knowledge)
  - Collaboration should HELP (each brings their domain)
  - Contrast with matrix results where collaboration was optional and neutral/harmful

Usage:
  python -m src.scripts.run_composite_experiment \\
      --adapter-dir rp_adapters --results-dir results/composite
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

from halulujah.domain.data_prep import DOMAIN_SUBJECTS_EXTENDED
from halulujah.domain.collab_eval import (
    collab_reasoning_scoped,
    solo_reasoning,
)
from halulujah.domain.cross_eval import extract_answer_letter
from scripts.run_reasoning_preserved import load_specialist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_composite_questions(path):
    """Load composite questions grouped by domain pair."""
    with open(path) as f:
        questions = json.load(f)

    # Group by (domain_a, domain_b)
    by_pair = {}
    for q in questions:
        pair = (q["domain_a"], q["domain_b"])
        by_pair.setdefault(pair, []).append(q)

    return by_pair


def evaluate_solo(model, tokenizer, questions, domain, n_rounds, device):
    """Solo evaluation on composite questions."""
    results = []
    for i, q in enumerate(questions):
        final, chain = solo_reasoning(
            model, tokenizer, q["question"], domain,
            n_rounds=n_rounds, device=device,
        )
        predicted = extract_answer_letter(final)
        results.append({
            "idx": i,
            "expected": q["answer_letter"],
            "predicted": predicted,
            "correct": predicted == q["answer_letter"],
            "bridge": q.get("bridge_concept", ""),
        })
    return results


def evaluate_collab(model_a, model_b, tokenizer, questions,
                    domain_a, domain_b, n_rounds, device):
    """Collaboration evaluation on composite questions."""
    results = []
    for i, q in enumerate(questions):
        final, chain, pre_collab = collab_reasoning_scoped(
            model_a, model_b, tokenizer, q["question"],
            domain_a, domain_b, n_rounds=n_rounds,
            device=device, protocol="full-cot",
        )
        predicted = extract_answer_letter(final)
        expected = q["answer_letter"]
        pre_a = pre_collab["agent_a"]["answer"]
        a_was_right = pre_a == expected
        post_right = predicted == expected
        switched = pre_a != predicted

        if switched and a_was_right and not post_right:
            switch_type = "c2w"
        elif switched and not a_was_right and post_right:
            switch_type = "w2c"
        elif switched:
            switch_type = "other"
        else:
            switch_type = "held"

        results.append({
            "idx": i,
            "expected": expected,
            "predicted": predicted,
            "correct": post_right,
            "pre_a": pre_a,
            "pre_a_correct": a_was_right,
            "switched": switched,
            "switch_type": switch_type,
            "bridge": q.get("bridge_concept", ""),
            "trace_snippet": chain[0]["thought"][:300] if chain else "",
        })
    return results


def summarize(per_q, solo_acc=None):
    n = len(per_q)
    if n == 0:
        return {"accuracy": 0, "n": 0}
    acc = sum(r["correct"] for r in per_q) / n
    s = {"accuracy": acc, "n": n}
    if solo_acc is not None:
        s["delta"] = acc - solo_acc
    if any("switch_type" in r for r in per_q):
        c2w = sum(1 for r in per_q if r.get("switch_type") == "c2w")
        w2c = sum(1 for r in per_q if r.get("switch_type") == "w2c")
        switches = sum(1 for r in per_q if r.get("switched"))
        s.update({
            "c2w": c2w, "w2c": w2c, "switches": switches,
            "c2w_w2c_ratio": c2w / max(w2c, 1),
        })
    return s


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


def free_model(*models):
    for m in models:
        del m
    gc.collect()
    torch.cuda.empty_cache()


def run_experiment(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load composite questions
    questions_path = os.path.join(
        os.path.dirname(__file__), "composite_questions.json",
    )
    by_pair = load_composite_questions(questions_path)
    pairs = sorted(by_pair.keys())
    logger.info("Loaded %d pairs: %s", len(pairs),
                [(a, b, len(by_pair[(a, b)])) for a, b in pairs])

    results_path = os.path.join(args.results_dir, "composite_results.json")
    os.makedirs(args.results_dir, exist_ok=True)
    data = load_checkpoint(results_path)
    data["config"] = {
        "n_rounds": args.n_rounds,
        "model_name": args.model_name,
        "pairs": [(a, b) for a, b in pairs],
        "specialist_type": "reasoning_preserved",
        "question_type": "composite_interdisciplinary",
    }

    # Load tokenizer + base model
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

    for domain_a, domain_b in pairs:
        qs = by_pair[(domain_a, domain_b)]
        pair_label = f"{domain_a}+{domain_b}"
        logger.info("=== Pair: %s (%d questions) ===", pair_label, len(qs))

        # Base solo
        cid = f"base_solo_{pair_label}"
        if not done(data, cid):
            logger.info("  %s", cid)
            pq = evaluate_solo(
                base_model, tokenizer, qs, "general",
                args.n_rounds, device,
            )
            s = summarize(pq)
            data["conditions"].append({
                "id": cid, "type": "base_solo",
                "pair": pair_label, **s,
            })
            logger.info("    %s: %.0f%%", cid, s["accuracy"] * 100)
            save_checkpoint(data, results_path)

        # Base pair
        cid = f"base_pair_{pair_label}"
        if not done(data, cid):
            logger.info("  %s", cid)
            pq = evaluate_collab(
                base_model, base_model, tokenizer, qs,
                "general", "general", args.n_rounds, device,
            )
            base_acc = get_acc(data, f"base_solo_{pair_label}")
            s = summarize(pq, solo_acc=base_acc)
            data["conditions"].append({
                "id": cid, "type": "base_pair",
                "pair": pair_label, **s,
            })
            logger.info("    %s: %.0f%% (delta %+.0fpp)",
                         cid, s["accuracy"] * 100, s.get("delta", 0) * 100)
            save_checkpoint(data, results_path)

        # Solo specialist A
        a_path = os.path.join(args.adapter_dir, f"adapter_{domain_a}")
        b_path = os.path.join(args.adapter_dir, f"adapter_{domain_b}")

        if not os.path.exists(a_path):
            logger.warning("  No adapter for %s, skipping pair", domain_a)
            continue
        if not os.path.exists(b_path):
            logger.warning("  No adapter for %s, skipping pair", domain_b)
            continue

        # Load specialist A
        logger.info("  Loading %s specialist...", domain_a)
        spec_a = load_specialist(
            args.model_name, a_path, device, args.cache_dir,
        )

        cid = f"solo_{domain_a}_on_{pair_label}"
        if not done(data, cid):
            logger.info("  %s", cid)
            pq = evaluate_solo(
                spec_a, tokenizer, qs, domain_a,
                args.n_rounds, device,
            )
            s = summarize(pq)
            data["conditions"].append({
                "id": cid, "type": "solo_specialist",
                "specialist": domain_a, "pair": pair_label, **s,
            })
            logger.info("    %s: %.0f%%", cid, s["accuracy"] * 100)
            save_checkpoint(data, results_path)

        # Load specialist B
        logger.info("  Loading %s specialist...", domain_b)
        spec_b = load_specialist(
            args.model_name, b_path, device, args.cache_dir,
        )

        # Solo specialist B
        cid = f"solo_{domain_b}_on_{pair_label}"
        if not done(data, cid):
            logger.info("  %s", cid)
            pq = evaluate_solo(
                spec_b, tokenizer, qs, domain_b,
                args.n_rounds, device,
            )
            s = summarize(pq)
            data["conditions"].append({
                "id": cid, "type": "solo_specialist",
                "specialist": domain_b, "pair": pair_label, **s,
            })
            logger.info("    %s: %.0f%%", cid, s["accuracy"] * 100)
            save_checkpoint(data, results_path)

        # A + B collaboration
        cid = f"collab_{domain_a}_{domain_b}_on_{pair_label}"
        if not done(data, cid):
            logger.info("  %s", cid)
            # Use max solo accuracy as the baseline for delta
            solo_a_acc = get_acc(data, f"solo_{domain_a}_on_{pair_label}")
            solo_b_acc = get_acc(data, f"solo_{domain_b}_on_{pair_label}")
            best_solo = max(solo_a_acc or 0, solo_b_acc or 0)

            pq = evaluate_collab(
                spec_a, spec_b, tokenizer, qs,
                domain_a, domain_b, args.n_rounds, device,
            )
            s = summarize(pq, solo_acc=best_solo)
            data["conditions"].append({
                "id": cid, "type": "cross_collab",
                "specialist_a": domain_a, "specialist_b": domain_b,
                "pair": pair_label, "best_solo_baseline": best_solo, **s,
            })
            logger.info(
                "    %s: %.0f%% (delta %+.0fpp vs best solo %.0f%%)",
                cid, s["accuracy"] * 100, s.get("delta", 0) * 100,
                best_solo * 100,
            )
            save_checkpoint(data, results_path)

        free_model(spec_a, spec_b)

    # Summary
    data["summary"] = compute_summary(data)
    save_checkpoint(data, results_path)
    free_model(base_model)
    return data


def compute_summary(data):
    conds = data["conditions"]
    summary = {}

    for ctype in ["base_solo", "base_pair", "solo_specialist", "cross_collab"]:
        group = [c for c in conds if c["type"] == ctype]
        if not group:
            continue
        avg_acc = sum(c["accuracy"] for c in group) / len(group)
        entry = {"mean_accuracy": avg_acc, "n_conditions": len(group)}
        deltas = [c["delta"] for c in group if "delta" in c]
        if deltas:
            entry["mean_delta"] = sum(deltas) / len(deltas)
        summary[ctype] = entry

    # Key comparison: collab delta on composite vs matrix cross-domain delta
    collab = [c for c in conds if c["type"] == "cross_collab"]
    if collab:
        positive = sum(1 for c in collab if c.get("delta", 0) > 0)
        summary["collaboration_helps"] = {
            "pairs_where_collab_helps": positive,
            "pairs_total": len(collab),
            "fraction_helped": positive / len(collab),
        }

    return summary


def print_results(data):
    print("\n" + "=" * 90)
    print("COMPOSITE QUESTION EXPERIMENT — STRUCTURALLY NECESSARY COLLABORATION")
    print("=" * 90)

    conds = data["conditions"]
    pairs = sorted(set(c["pair"] for c in conds))

    for pair in pairs:
        print(f"\n--- {pair} ---")
        pc = [c for c in conds if c["pair"] == pair]
        print(f"  {'Condition':<45} {'Acc':>6} {'Delta':>7} {'C2W':>5} {'W2C':>5}")
        print("  " + "-" * 70)
        for c in sorted(pc, key=lambda x: x["id"]):
            acc = f"{c['accuracy']*100:.0f}%"
            delta = f"{c.get('delta',0)*100:+.0f}pp" if "delta" in c else "—"
            c2w = str(c.get("c2w", "—"))
            w2c = str(c.get("w2c", "—"))
            print(f"  {c['id']:<45} {acc:>6} {delta:>7} {c2w:>5} {w2c:>5}")

    if "summary" in data:
        s = data["summary"]
        print(f"\n{'='*90}")
        print("SUMMARY")
        for ctype, st in s.items():
            if ctype == "collaboration_helps":
                print(f"\n  Collaboration helps: {st['pairs_where_collab_helps']}/{st['pairs_total']} pairs")
            elif isinstance(st, dict) and "mean_accuracy" in st:
                parts = [f"mean acc {st['mean_accuracy']*100:.1f}%"]
                if "mean_delta" in st:
                    parts.append(f"delta {st['mean_delta']*100:+.1f}pp")
                print(f"  {ctype}: {', '.join(parts)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Composite question experiment — structurally necessary collaboration",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-dir", default="rp_adapters")
    parser.add_argument("--results-dir", default="results/composite")
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    start = time.time()
    data = run_experiment(args)
    elapsed = time.time() - start

    data["total_time_minutes"] = elapsed / 60
    save_checkpoint(data, os.path.join(args.results_dir, "composite_results.json"))
    logger.info("Done in %.1f min", elapsed / 60)
    print_results(data)


if __name__ == "__main__":
    main()
