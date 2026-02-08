#!/usr/bin/env python3
"""CoT ablation study on TempLAMA held-out year using Ollama.

Runs 4 conditions on a sample of held-out questions:
  1. bare        — no context, one-shot prompt
  2. context     — entity history as context, one-shot prompt
  3. cot         — no context, chain-of-thought prompt
  4. cot+context — entity history + chain-of-thought

TempLAMA provides built-in temporal context (prior-year answers for the
same entity), so no Oracle/FAISS index is needed.
"""

import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, "src")

from halulujah.data.chat_formatter import format_cot_system_prompt, format_cot_user_prompt
from halulujah.data.loader import load_templama, split_by_year
from halulujah.models.loader import OllamaBackend
from halulujah.rl.verifier import extract_cot_answer, multi_answer_verifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "50"))
HELD_OUT_YEAR = int(os.environ.get("HELD_OUT_YEAR", "2020"))
SEED = 42
MAX_TOKENS = 128
TEMPERATURE = 0.3


def run_condition(backend, questions, condition):
    """Run one ablation condition."""
    bare_system = (
        "You are a helpful assistant. Answer in one short sentence. "
        "Give only the name or entity, not an explanation."
    )
    cot_system = format_cot_system_prompt()

    results = []
    for i, entry in enumerate(questions):
        q = entry["question"]
        ctx = entry.get("context", "")

        if condition == "bare":
            prompt = f"{bare_system}\n\n{q}"
        elif condition == "context":
            prompt = f"{bare_system}\n\nContext:\n{ctx}\n\n{q}" if ctx else f"{bare_system}\n\n{q}"
        elif condition == "cot":
            user_part = (
                f"Question: {q}\n\n"
                "Think step by step, then provide your final answer "
                "on a line starting with 'Answer:'."
            )
            prompt = f"{cot_system}\n\n{user_part}"
        elif condition == "cot+context":
            if ctx:
                user_part = format_cot_user_prompt(q, ctx)
            else:
                user_part = (
                    f"Question: {q}\n\n"
                    "Think step by step, then provide your final answer "
                    "on a line starting with 'Answer:'."
                )
            prompt = f"{cot_system}\n\n{user_part}"
        else:
            raise ValueError(f"Unknown condition: {condition}")

        response = backend.generate(
            prompt,
            temperature=TEMPERATURE,
            top_p=0.9,
            num_predict=MAX_TOKENS,
        )

        if condition.startswith("cot"):
            extracted = extract_cot_answer(response)
        else:
            extracted = response

        answers_all = entry.get("answers_all", [entry["answer"]])
        score = multi_answer_verifier(answers_all, extracted)

        results.append({
            "question": q,
            "expected_answer": entry["answer"],
            "expected_answers_all": answers_all,
            "model_answer": extracted,
            "model_answer_full": response if condition.startswith("cot") else None,
            "context": ctx,
            "score_numeric": score,
            "score": "Correct" if score >= 0.6 else "Incorrect",
        })

        if (i + 1) % 10 == 0:
            avg = sum(r["score_numeric"] for r in results) / len(results)
            correct = sum(1 for r in results if r["score_numeric"] >= 0.6)
            logger.info(
                "  [%s] %d/%d — correct: %d, avg: %.3f",
                condition, i + 1, len(questions), correct, avg,
            )

    return results


def summarize(results, condition):
    scores = [r["score_numeric"] for r in results]
    correct = sum(1 for s in scores if s >= 0.6)
    avg = sum(scores) / len(scores) if scores else 0
    return {
        "condition": condition,
        "n": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0,
        "avg_score": avg,
    }


def main():
    data = load_templama(cache_dir="data")
    _, held_out = split_by_year(data, [HELD_OUT_YEAR])
    logger.info("Held-out %d: %d questions", HELD_OUT_YEAR, len(held_out))

    # Filter to entries that have context (prior-year history)
    with_ctx = [e for e in held_out if e.get("context")]
    logger.info("  with history context: %d", len(with_ctx))

    random.seed(SEED)
    sample = random.sample(with_ctx, min(SAMPLE_SIZE, len(with_ctx)))
    logger.info("Sampled %d questions for ablation", len(sample))

    backend = OllamaBackend(model_name=OLLAMA_MODEL, timeout=300)
    logger.info("Using Ollama model: %s", OLLAMA_MODEL)

    model_tag = OLLAMA_MODEL.replace(":", "_").replace("/", "_")
    out_dir = os.path.join("outputs", f"templama_ablation_{HELD_OUT_YEAR}_{model_tag}")
    os.makedirs(out_dir, exist_ok=True)

    conditions = ["bare", "context", "cot", "cot+context"]
    summaries = []

    for cond in conditions:
        cached_path = os.path.join(out_dir, f"{cond}.json")
        if os.path.exists(cached_path):
            logger.info("Loading cached results for condition: %s", cond)
            with open(cached_path) as f:
                results = json.load(f)
            s = summarize(results, cond)
            s["elapsed_sec"] = 0.0
            summaries.append(s)
            logger.info(
                "  %s (cached) — correct: %d/%d (%.1f%%), avg_score: %.3f",
                cond, s["correct"], s["n"], s["accuracy"] * 100, s["avg_score"],
            )
            continue

        logger.info("=" * 50)
        logger.info("Running condition: %s", cond)
        logger.info("=" * 50)
        t0 = time.time()
        results = run_condition(backend, sample, cond)
        elapsed = time.time() - t0
        s = summarize(results, cond)
        s["elapsed_sec"] = round(elapsed, 1)
        summaries.append(s)
        logger.info(
            "  %s — correct: %d/%d (%.1f%%), avg_score: %.3f, time: %.1fs",
            cond, s["correct"], s["n"], s["accuracy"] * 100, s["avg_score"], elapsed,
        )

        fpath = os.path.join(out_dir, f"{cond}.json")
        with open(fpath, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Save summary
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    # Print table
    print("\n" + "=" * 70)
    print(f"TempLAMA ABLATION — held-out {HELD_OUT_YEAR}, n={len(sample)}, model={OLLAMA_MODEL}")
    print("=" * 70)
    print(f"{'Condition':<15} {'Correct':>8} {'Accuracy':>10} {'Avg Score':>10} {'Time':>8}")
    print("-" * 70)
    for s in summaries:
        print(
            f"{s['condition']:<15} {s['correct']:>5}/{s['n']:<3}"
            f" {s['accuracy']*100:>8.1f}%"
            f" {s['avg_score']:>10.3f}"
            f" {s['elapsed_sec']:>7.1f}s"
        )
    print("-" * 70)
    print(f"\nDetailed results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
