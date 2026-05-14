#!/usr/bin/env python3
"""Full collaboration matrix: measure the value of multi-agent deliberation.

Trains reasoning-preserved specialists and evaluates all collaboration
conditions. Core question: does deliberation add value beyond expert routing
(MoE-style, which picks the right expert but involves no discussion)?

Conditions:
  solo         - Specialist alone on its domain (= MoE routing result)
  base_solo    - Untrained base model alone on each domain
  same_pair    - Same specialist deliberating with itself
  cross_pair   - Specialist + different-domain specialist deliberating
  mixed_pair   - Specialist + untrained base deliberating
  base_pair    - Two untrained base models deliberating

All specialists use reasoning-preserved training (no empty think blocks).
Results checkpoint after each condition for resumability.

Usage:
  python -m src.scripts.run_collab_matrix \\
      --domains medicine physics law math biology
  python -m src.scripts.run_collab_matrix --all-domains
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
    collab_reasoning_scoped,
    solo_reasoning,
)
from halulujah.domain.cross_eval import extract_answer_letter
from scripts.run_reasoning_preserved import (
    format_domain_reasoning_preserved,
    train_reasoning_preserved_specialist,
    load_specialist,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALL_DOMAINS = list(DOMAIN_SUBJECTS_EXTENDED.keys())
DEFAULT_DOMAINS = ["medicine", "physics", "law", "math", "biology"]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_solo(model, tokenizer, questions, domain, n_rounds, device):
    """Solo evaluation. Returns per-question results list."""
    results = []
    for i, entry in enumerate(questions):
        final, chain = solo_reasoning(
            model, tokenizer, entry["question"], domain,
            n_rounds=n_rounds, device=device,
        )
        predicted = extract_answer_letter(final)
        results.append({
            "idx": i,
            "expected": entry["answer_letter"],
            "predicted": predicted,
            "correct": predicted == entry["answer_letter"],
        })
    return results


def evaluate_collab(model_a, model_b, tokenizer, questions,
                    domain_a, domain_b, n_rounds, device):
    """Collaboration evaluation. Returns per-question results with switches."""
    results = []
    for i, entry in enumerate(questions):
        final, chain, pre_collab = collab_reasoning_scoped(
            model_a, model_b, tokenizer, entry["question"],
            domain_a, domain_b, n_rounds=n_rounds,
            device=device, protocol="full-cot",
        )
        predicted = extract_answer_letter(final)
        expected = entry["answer_letter"]
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
            "trace_snippet": chain[0]["thought"][:300] if chain else "",
        })
    return results


def summarize(per_q, solo_acc=None):
    """Aggregate per-question results into condition summary."""
    n = len(per_q)
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


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main matrix
# ---------------------------------------------------------------------------

def run_matrix(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domains = args.domains

    results_path = os.path.join(args.results_dir, "matrix_results.json")
    os.makedirs(args.results_dir, exist_ok=True)
    data = load_checkpoint(results_path)
    data["config"] = {
        "domains": domains, "n_questions": args.n_questions,
        "n_rounds": args.n_rounds, "rank": args.rank,
        "model_name": args.model_name, "specialist_type": "reasoning_preserved",
    }

    # Train all RP specialists ------------------------------------------
    if not args.skip_training:
        for d in domains:
            train_reasoning_preserved_specialist(
                d, args.model_name, args.adapter_dir,
                cache_dir=args.cache_dir, rank=args.rank,
            )

    # Load test data ----------------------------------------------------
    test_data = {}
    for d in domains:
        entries = load_mmlu_domain(d, split="test", cache_dir=args.cache_dir)
        _, test = split_train_test(entries, test_size=args.n_questions, seed=42)
        test_data[d] = test

    # Load tokenizer + base model (stays throughout) --------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir,
    ).to(device)
    base_model.eval()

    # === BASE CONDITIONS ===============================================

    for d in domains:
        qs = test_data[d][:args.n_questions]

        # Base solo
        cid = f"base_solo_{d}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            pq = evaluate_solo(
                base_model, tokenizer, qs, "general",
                args.n_rounds, device,
            )
            s = summarize(pq)
            data["conditions"].append({
                "id": cid, "type": "base_solo",
                "question_domain": d, **s,
            })
            logger.info("  %s: %.1f%%", cid, s["accuracy"] * 100)
            save_checkpoint(data, results_path)

        # Base + base
        cid = f"base_pair_{d}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            pq = evaluate_collab(
                base_model, base_model, tokenizer, qs,
                "general", "general", args.n_rounds, device,
            )
            s = summarize(pq, solo_acc=get_acc(data, f"base_solo_{d}"))
            data["conditions"].append({
                "id": cid, "type": "base_pair",
                "question_domain": d, **s,
            })
            logger.info("  %s: %.1f%% (delta %+.1fpp)", cid,
                         s["accuracy"] * 100, s.get("delta", 0) * 100)
            save_checkpoint(data, results_path)

    # === SPECIALIST CONDITIONS =========================================

    for d in domains:
        adapter_path = os.path.join(args.adapter_dir, f"adapter_{d}")
        if not os.path.exists(adapter_path):
            logger.warning("No adapter for %s, skipping", d)
            continue

        qs = test_data[d][:args.n_questions]

        # Load specialist -----------------------------------------------
        logger.info("Loading %s specialist...", d)
        specialist = load_specialist(
            args.model_name, adapter_path, device, args.cache_dir,
        )

        # Solo ----------------------------------------------------------
        cid = f"solo_{d}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            pq = evaluate_solo(specialist, tokenizer, qs, d, args.n_rounds, device)
            s = summarize(pq)
            data["conditions"].append({
                "id": cid, "type": "solo",
                "specialist": d, "question_domain": d, **s,
            })
            logger.info("  %s: %.1f%%", cid, s["accuracy"] * 100)
            save_checkpoint(data, results_path)

        solo_acc = get_acc(data, f"solo_{d}")

        # Same-domain pair (specialist with itself) ---------------------
        cid = f"same_{d}_{d}"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            pq = evaluate_collab(
                specialist, specialist, tokenizer, qs,
                d, d, args.n_rounds, device,
            )
            s = summarize(pq, solo_acc=solo_acc)
            data["conditions"].append({
                "id": cid, "type": "same_pair",
                "specialist": d, "helper": d,
                "question_domain": d, **s,
            })
            logger.info("  %s: %.1f%% (delta %+.1fpp)", cid,
                         s["accuracy"] * 100, s.get("delta", 0) * 100)
            save_checkpoint(data, results_path)

        # Specialist + base ---------------------------------------------
        cid = f"mixed_{d}_base"
        if not done(data, cid):
            logger.info("=== %s ===", cid)
            pq = evaluate_collab(
                specialist, base_model, tokenizer, qs,
                d, "general", args.n_rounds, device,
            )
            s = summarize(pq, solo_acc=solo_acc)
            data["conditions"].append({
                "id": cid, "type": "mixed_pair",
                "specialist": d, "helper": "base",
                "question_domain": d, **s,
            })
            logger.info("  %s: %.1f%% (delta %+.1fpp)", cid,
                         s["accuracy"] * 100, s.get("delta", 0) * 100)
            save_checkpoint(data, results_path)

        # Cross-domain pairs --------------------------------------------
        for h in domains:
            if h == d:
                continue
            cid = f"cross_{d}_{h}"
            if done(data, cid):
                logger.info("Skipping %s (done)", cid)
                continue

            h_path = os.path.join(args.adapter_dir, f"adapter_{h}")
            if not os.path.exists(h_path):
                logger.warning("No adapter for helper %s, skipping", h)
                continue

            logger.info("Loading %s helper...", h)
            helper = load_specialist(
                args.model_name, h_path, device, args.cache_dir,
            )

            logger.info("=== %s ===", cid)
            pq = evaluate_collab(
                specialist, helper, tokenizer, qs,
                d, h, args.n_rounds, device,
            )
            s = summarize(pq, solo_acc=solo_acc)
            data["conditions"].append({
                "id": cid, "type": "cross_pair",
                "specialist": d, "helper": h,
                "question_domain": d, **s,
            })
            logger.info(
                "  %s: %.1f%% (delta %+.1fpp, C2W=%d, W2C=%d, ratio=%.1fx)",
                cid, s["accuracy"] * 100, s.get("delta", 0) * 100,
                s.get("c2w", 0), s.get("w2c", 0), s.get("c2w_w2c_ratio", 0),
            )
            save_checkpoint(data, results_path)
            free_model(helper)

        free_model(specialist)

    # === SUMMARY =======================================================
    data["summary"] = compute_summary(data, domains)
    save_checkpoint(data, results_path)
    free_model(base_model)
    return data


def compute_summary(data, domains):
    conditions = data["conditions"]
    summary = {}

    for ctype in ["solo", "base_solo", "same_pair", "cross_pair",
                   "mixed_pair", "base_pair"]:
        conds = [c for c in conditions if c["type"] == ctype]
        if not conds:
            continue
        avg_acc = sum(c["accuracy"] for c in conds) / len(conds)
        entry = {"mean_accuracy": avg_acc, "n_conditions": len(conds)}
        deltas = [c["delta"] for c in conds if "delta" in c]
        if deltas:
            entry["mean_delta"] = sum(deltas) / len(deltas)
        ratios = [c["c2w_w2c_ratio"] for c in conds if "c2w_w2c_ratio" in c]
        if ratios:
            entry["mean_c2w_w2c"] = sum(ratios) / len(ratios)
        summary[ctype] = entry

    # MoE baseline = specialist solo accuracy (routing picks the right expert)
    solo_accs = {c["specialist"]: c["accuracy"]
                 for c in conditions if c["type"] == "solo"}
    if solo_accs:
        summary["moe_routing_baseline"] = {
            "description": "Specialist solo = MoE routing (pick right expert, no deliberation)",
            "per_domain": solo_accs,
            "mean": sum(solo_accs.values()) / len(solo_accs),
        }

    return summary


def print_results(data):
    print("\n" + "=" * 90)
    print("COLLABORATION MATRIX — REASONING-PRESERVED SPECIALISTS")
    print("=" * 90)

    for ctype, label in [
        ("solo", "SOLO (specialist alone = MoE routing baseline)"),
        ("base_solo", "BASE SOLO (no LoRA)"),
        ("same_pair", "SAME-DOMAIN PAIRS (specialist + itself)"),
        ("mixed_pair", "SPECIALIST + BASE"),
        ("base_pair", "BASE + BASE"),
        ("cross_pair", "CROSS-DOMAIN PAIRS"),
    ]:
        conds = [c for c in data["conditions"] if c["type"] == ctype]
        if not conds:
            continue
        print(f"\n--- {label} ---")
        hdr = f"{'Condition':<30} {'Acc':>6} {'Delta':>7} {'C2W':>5} {'W2C':>5} {'Ratio':>7}"
        print(hdr)
        print("-" * len(hdr))
        for c in sorted(conds, key=lambda x: x["id"]):
            acc = f"{c['accuracy']*100:.0f}%"
            delta = f"{c.get('delta',0)*100:+.0f}pp" if "delta" in c else "—"
            c2w = str(c.get("c2w", "—"))
            w2c = str(c.get("w2c", "—"))
            ratio = f"{c.get('c2w_w2c_ratio',0):.1f}x" if "c2w_w2c_ratio" in c else "—"
            print(f"{c['id']:<30} {acc:>6} {delta:>7} {c2w:>5} {w2c:>5} {ratio:>7}")

    if "summary" in data:
        print(f"\n{'='*90}")
        print("SUMMARY")
        for ctype, st in data["summary"].items():
            if ctype == "moe_routing_baseline":
                print(f"\n  MoE routing baseline: mean {st['mean']*100:.1f}%")
            elif isinstance(st, dict) and "mean_accuracy" in st:
                parts = [f"mean acc {st['mean_accuracy']*100:.1f}%"]
                if "mean_delta" in st:
                    parts.append(f"delta {st['mean_delta']*100:+.1f}pp")
                if "mean_c2w_w2c" in st:
                    parts.append(f"C2W/W2C {st['mean_c2w_w2c']:.1f}x")
                print(f"  {ctype} ({st['n_conditions']}): {', '.join(parts)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Collaboration matrix with reasoning-preserved specialists",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-dir", default="rp_adapters")
    parser.add_argument("--results-dir", default="results/collab_matrix")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--all-domains", action="store_true")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    if args.all_domains:
        args.domains = ALL_DOMAINS

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    n = len(args.domains)
    n_conds = 3 * n + n * (n - 1) + 2 * n
    logger.info("Matrix: %d domains, ~%d conditions", n, n_conds)
    logger.info("Domains: %s", args.domains)

    start = time.time()
    data = run_matrix(args)
    elapsed = time.time() - start

    data["total_time_minutes"] = elapsed / 60
    save_checkpoint(data, os.path.join(args.results_dir, "matrix_results.json"))
    logger.info("Done in %.1f min (%.1f hrs)", elapsed / 60, elapsed / 3600)
    print_results(data)


if __name__ == "__main__":
    main()
