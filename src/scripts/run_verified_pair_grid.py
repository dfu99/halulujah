"""5x5 LoRA pair-grid collaboration on verified Qwen3-1.7B specialists.

Loads each verified adapter (and base) and runs all 45 conditions:
  - solo (5)         : specialist alone on its domain
  - base_solo (5)    : base alone on each domain
  - same_pair (5)    : specialist deliberating with itself
  - cross_pair (20)  : specialist_a + specialist_b deliberating on a's domain
  - mixed_pair (5)   : specialist + base deliberating on specialist's domain
  - base_pair (5)    : two bases deliberating on each domain

For each condition: N=50 questions, 3 deliberation rounds, full-cot protocol
(reasoning-preserved). Outputs `matrix_results.json` with per-condition
accuracy, switching counts, c2w/w2c ratios.

Usage:
  python run_verified_pair_grid.py --n-questions 50 --n-rounds 3
"""
import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from halulujah.domain.collab_eval import collab_reasoning_scoped, solo_reasoning
from halulujah.domain.cross_eval import extract_answer_letter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Domain -> MMLU subjects used for evaluation
DOMAIN_MMLU = {
    "math": ["college_mathematics", "high_school_mathematics",
             "abstract_algebra", "elementary_mathematics"],
    "medicine": ["anatomy", "clinical_knowledge", "college_medicine",
                 "medical_genetics", "professional_medicine", "virology"],
    "biology": ["college_biology", "high_school_biology"],
    "law": ["jurisprudence", "international_law", "professional_law"],
    "physics": ["astronomy", "college_physics", "high_school_physics",
                "conceptual_physics"],
}


def format_mmlu(q):
    return (
        f"{q['question']}\n"
        f"A) {q['choices'][0]}\n"
        f"B) {q['choices'][1]}\n"
        f"C) {q['choices'][2]}\n"
        f"D) {q['choices'][3]}"
    )


def load_domain_questions(domain, n, cache_dir, seed=42):
    """Pull n questions evenly across the domain's MMLU subjects.

    Each returned question has a stable `qid` (SHA1 of subject + formatted
    question + answer letter), so per_q records can be joined across cells
    even after the runner is refactored or the question pool grows. Audit
    follow-up #9 (tasks/audit-2026-05-05.md §12) — required for
    question-clustered bootstrap on the WHO-asymmetry ratio.
    """
    import hashlib
    import random

    from datasets import load_dataset
    rng = random.Random(seed)
    pool = []
    for subj in DOMAIN_MMLU[domain]:
        ds = load_dataset("cais/mmlu", subj, split="test", cache_dir=cache_dir)
        for q in ds:
            text = format_mmlu(q)
            ans = chr(ord("A") + q["answer"])
            qid = hashlib.sha1(
                f"{subj}|{text}|{ans}".encode("utf-8")
            ).hexdigest()[:16]
            pool.append({
                "qid": qid,
                "subject": subj,
                "question": text,
                "answer_letter": ans,
            })
    rng.shuffle(pool)
    return pool[:n]


def load_model(base_name, adapter_path, cache_dir, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        base_name, trust_remote_code=True, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir).to(device)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def free(*models):
    for m in models:
        del m
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_solo(model, tok, questions, domain, n_rounds, device):
    results = []
    for i, entry in enumerate(questions):
        final, chain = solo_reasoning(
            model, tok, entry["question"], domain,
            n_rounds=n_rounds, device=device, temperature=0.7,
        )
        pred = extract_answer_letter(final)
        results.append({
            "idx": i,
            "qid": entry.get("qid"),
            "subject": entry["subject"],
            "expected": entry["answer_letter"],
            "predicted": pred,
            "correct": pred == entry["answer_letter"],
        })
    return results


def evaluate_collab(model_a, model_b, tok, questions, domain_a, domain_b,
                    n_rounds, device):
    results = []
    for i, entry in enumerate(questions):
        final, chain, pre = collab_reasoning_scoped(
            model_a, model_b, tok, entry["question"],
            domain_a, domain_b, n_rounds=n_rounds, device=device,
            temperature=0.7, protocol="full-cot",
        )
        pred = extract_answer_letter(final)
        gold = entry["answer_letter"]
        pre_a = pre["agent_a"]["answer"]
        a_was_right = pre_a == gold
        post_right = pred == gold
        switched = pre_a != pred
        if switched and a_was_right and not post_right:
            stype = "c2w"
        elif switched and not a_was_right and post_right:
            stype = "w2c"
        elif switched:
            stype = "other"
        else:
            stype = "held"
        # `pre_a_full` per audit-2026-05-05 follow-up #10: the un-truncated
        # 1-pass response. Lets us post-hoc distinguish "model emitted no
        # parseable letter" from "model emitted a letter the regex missed".
        results.append({
            "idx": i,
            "qid": entry.get("qid"),
            "subject": entry["subject"],
            "expected": gold, "predicted": pred,
            "correct": post_right, "pre_a": pre_a,
            "pre_a_correct": a_was_right,
            "pre_a_full": pre.get("agent_a", {}).get(
                "raw_full", pre.get("agent_a", {}).get("raw", "")
            ),
            "switched": switched, "switch_type": stype,
        })
    return results


def summarize(per_q, solo_acc=None):
    n = len(per_q)
    acc = sum(r["correct"] for r in per_q) / n if n else 0.0
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


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"conditions": {}}


def save_checkpoint(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3-1.7B")
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    p.add_argument("--n-questions", type=int, default=50)
    p.add_argument("--n-rounds", type=int, default=3)
    p.add_argument("--results-dir",
                   default="results/verified_pair_grid_qwen3_1p7b")
    p.add_argument("--adapter-config", default=None,
                   help="JSON path: {domain: adapter_path}")
    args = p.parse_args()

    if args.adapter_config:
        adapters = json.load(open(args.adapter_config))
    else:
        # Default config for the 5 verified Qwen3-1.7B specialists
        adapters = {
            "math": "/workspace/adapters_1p7b_ood/math_qwen3/r16/adapter_math_gsm8k",
            "medicine": "/workspace/adapters_1p7b_ood/medicine_sweep/r64/adapter_medicine_medqa",
            "biology": "/workspace/adapters_1p7b_ood/biology_sweep/r16/adapter_biology_pubmedqa",
            "law": "/workspace/adapters_1p7b_ood/law_qwen3/r16/adapter_law_casehold",
            "physics": "/workspace/adapters_1p7b_ood/physics_qwen3/r16/adapter_physics_sciq",
        }
    domains = list(adapters.keys())
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "matrix_results.json")
    data = load_checkpoint(results_path)
    data["config"] = {
        "base": args.base, "adapters": adapters, "domains": domains,
        "n_questions": args.n_questions, "n_rounds": args.n_rounds,
    }
    save_checkpoint(data, results_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all per-domain test questions once
    test = {}
    for d in domains:
        test[d] = load_domain_questions(d, args.n_questions, args.cache_dir)
        logger.info(f"loaded {len(test[d])} {d} questions")

    # Strategy: minimize model loads.
    # Pass 1: For each entity_a in [base, *specialists]:
    #   Load entity_a, run solo on every domain test set.
    #   Then for each entity_b in [base, *specialists]:
    #     Load entity_b, run collab on entity_a's domain test set.
    #     (For "domain" of base, run on each domain.)

    adapter_path = lambda e: None if e == "base" else adapters[e]

    def cond_done(cid):
        return cid in data["conditions"]

    def write_cond(cid, summary, per_q):
        data["conditions"][cid] = {**summary, "per_q": per_q}
        save_checkpoint(data, results_path)

    # 45-condition layout:
    #   solo_<d>          : specialist_d alone on d           (5)
    #   base_solo_<d>     : base alone on d                   (5)
    #   pair_<d>_<helper> : specialist_d + helper on d        (30, helper in
    #                       {base, specialist_d, *other_specialists})
    #   base_pair_<d>     : base + base on d                  (5)

    # SOLO conditions: load specialist once, run solo on its domain
    for d in domains:
        cid = f"solo_{d}"
        if cond_done(cid):
            continue
        logger.info(f"loading specialist {d} for solo")
        model_a, tok = load_model(args.base, adapter_path(d),
                                  args.cache_dir, device)
        logger.info(f"running {cid}")
        t0 = time.time()
        per_q = evaluate_solo(model_a, tok, test[d], d,
                              args.n_rounds, device)
        s = summarize(per_q)
        s["elapsed_s"] = time.time() - t0
        logger.info(f"  {cid}: acc={s['accuracy']:.3f} "
                    f"elapsed={s['elapsed_s']:.1f}s")
        write_cond(cid, s, per_q)
        free(model_a)

    # BASE_SOLO conditions: load base once, run on every domain
    if any(not cond_done(f"base_solo_{d}") for d in domains):
        logger.info("loading base for base_solo")
        base_model, tok = load_model(args.base, None, args.cache_dir, device)
        for d in domains:
            cid = f"base_solo_{d}"
            if cond_done(cid):
                continue
            logger.info(f"running {cid}")
            t0 = time.time()
            per_q = evaluate_solo(base_model, tok, test[d], d,
                                  args.n_rounds, device)
            s = summarize(per_q)
            s["elapsed_s"] = time.time() - t0
            logger.info(f"  {cid}: acc={s['accuracy']:.3f} "
                        f"elapsed={s['elapsed_s']:.1f}s")
            write_cond(cid, s, per_q)
        free(base_model)

    # PAIR conditions: for each domain, specialist_d primary, helpers vary
    helpers = ["base"] + domains  # base + 5 specialists = 6 helpers per domain
    for d in domains:
        # collect pending pair conditions for this domain
        pending = []
        for h in helpers:
            cid = f"pair_{d}_{h}"
            if not cond_done(cid):
                pending.append((cid, h))
        if not pending:
            continue

        logger.info(f"loading specialist {d} as primary")
        model_a, tok = load_model(args.base, adapter_path(d),
                                  args.cache_dir, device)
        for cid, h in pending:
            logger.info(f"loading helper {h}")
            model_b, _ = load_model(args.base, adapter_path(h),
                                    args.cache_dir, device)
            logger.info(f"running {cid}")
            t0 = time.time()
            per_q = evaluate_collab(model_a, model_b, tok, test[d],
                                    d, h if h != "base" else d,
                                    args.n_rounds, device)
            solo_acc = data["conditions"].get(f"solo_{d}", {}).get("accuracy")
            s = summarize(per_q, solo_acc=solo_acc)
            s["elapsed_s"] = time.time() - t0
            logger.info(f"  {cid}: acc={s['accuracy']:.3f} "
                        f"delta={s.get('delta', 0):.3f} "
                        f"elapsed={s['elapsed_s']:.1f}s")
            write_cond(cid, s, per_q)
            free(model_b)
        free(model_a)

    # BASE_PAIR conditions: base + base on each domain
    if any(not cond_done(f"base_pair_{d}") for d in domains):
        logger.info("loading base + base for base_pair")
        model_a, tok = load_model(args.base, None, args.cache_dir, device)
        model_b, _ = load_model(args.base, None, args.cache_dir, device)
        for d in domains:
            cid = f"base_pair_{d}"
            if cond_done(cid):
                continue
            logger.info(f"running {cid}")
            t0 = time.time()
            per_q = evaluate_collab(model_a, model_b, tok, test[d],
                                    d, d, args.n_rounds, device)
            solo_acc = data["conditions"].get(f"base_solo_{d}", {}).get(
                "accuracy")
            s = summarize(per_q, solo_acc=solo_acc)
            s["elapsed_s"] = time.time() - t0
            logger.info(f"  {cid}: acc={s['accuracy']:.3f} "
                        f"delta={s.get('delta', 0):.3f} "
                        f"elapsed={s['elapsed_s']:.1f}s")
            write_cond(cid, s, per_q)
        free(model_a, model_b)

    logger.info("PAIR GRID COMPLETE")
    print(f"FINAL_METRICS results={results_path} "
          f"n_conditions={len(data['conditions'])}")


if __name__ == "__main__":
    main()
