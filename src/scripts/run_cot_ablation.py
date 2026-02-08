#!/usr/bin/env python3
"""CoT ablation study on held-out 2022 data using Ollama.

Runs 4 conditions on a sample of held-out questions:
  1. bare        — no context, one-shot prompt
  2. context     — retrieved Oracle context, one-shot prompt
  3. cot         — no context, chain-of-thought prompt
  4. cot+context — retrieved Oracle context + chain-of-thought

Builds the Oracle FAISS index from the 2022 10-K PDF, then generates
and grades answers for each condition.
"""

import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, "src")

from halulujah.config import load_config
from halulujah.data.chat_formatter import format_cot_system_prompt, format_cot_user_prompt
from halulujah.data.loader import load_egnivia, split_by_year
from halulujah.models.loader import OllamaBackend
from halulujah.oracle.index_builder import OracleIndex
from halulujah.oracle.retriever import OracleRetriever
from halulujah.rl.verifier import extract_cot_answer, simple_verifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "30"))
HELD_OUT_YEAR = int(os.environ.get("HELD_OUT_YEAR", "2022"))
SEED = 42
MAX_TOKENS = 256   # enough room for CoT reasoning
TEMPERATURE = 0.3  # low temp for consistency


def build_oracle(cfg, year):
    """Build (or load cached) Oracle FAISS index for a given year."""
    cache_dir = os.path.join("outputs", f"oracle_{year}")
    corpus_path = os.path.join(cache_dir, "corpus.json")

    oracle_index = OracleIndex(cfg.oracle)

    if os.path.exists(corpus_path):
        logger.info("Loading cached Oracle index from %s", cache_dir)
        oracle_index.load(cache_dir)
    else:
        pdf_path = os.path.join("data", "NVDA_10-K", f"NVDA_10-K_{year}.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        logger.info("Building Oracle index from %s", pdf_path)
        oracle_index.build_from_pdfs([pdf_path])
        oracle_index.save(cache_dir)
        logger.info("Oracle index saved to %s", cache_dir)

    return OracleRetriever(oracle_index)


def run_condition(backend, retriever, questions, condition):
    """Run one ablation condition, return list of result dicts."""
    bare_system = (
        "You are a helpful assistant. Keep responses to at most a single sentence "
        "and concise. Do not make lists. Ignore your knowledge cutoff and answer "
        "to the best of your ability."
    )
    cot_system = format_cot_system_prompt()

    results = []
    for i, entry in enumerate(questions):
        q = entry["question"]

        # Build prompt based on condition
        if condition == "bare":
            prompt = f"{bare_system}\n\nQuestion: {q}"
        elif condition == "context":
            passages = retriever.get_context_passages(q)
            context = "\n\n".join(passages)
            prompt = f"{bare_system}\n\n{context}\n\nQuestion: {q}"
        elif condition == "cot":
            user_part = (
                f"Question: {q}\n\n"
                "Think step by step, then provide your final answer "
                "on a line starting with 'Answer:'."
            )
            prompt = f"{cot_system}\n\n{user_part}"
        elif condition == "cot+context":
            passages = retriever.get_context_passages(q)
            context = "\n\n".join(passages)
            user_part = format_cot_user_prompt(q, context)
            prompt = f"{cot_system}\n\n{user_part}"
        else:
            raise ValueError(f"Unknown condition: {condition}")

        response = backend.generate(
            prompt,
            temperature=TEMPERATURE,
            top_p=0.9,
            num_predict=MAX_TOKENS,
        )

        # Extract answer for CoT conditions
        if condition.startswith("cot"):
            extracted = extract_cot_answer(response)
        else:
            extracted = response

        score = simple_verifier(entry["answer"], extracted)

        results.append({
            "question": q,
            "expected_answer": entry["answer"],
            "model_answer": extracted,
            "model_answer_full": response if condition.startswith("cot") else None,
            "score_numeric": score,
            "score": "Correct" if score >= 0.6 else "Incorrect",
        })

        if (i + 1) % 10 == 0:
            avg = sum(r["score_numeric"] for r in results) / len(results)
            logger.info("  [%s] %d/%d done — running avg: %.3f", condition, i + 1, len(questions), avg)

    return results


def summarize(results, condition):
    """Print summary stats for one condition."""
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
    cfg = load_config("configs/experiment.yaml")

    # Load held-out data
    data = load_egnivia(str(cfg.paths.resolve("data_json")))
    _, held_out = split_by_year(data, [HELD_OUT_YEAR])
    logger.info("Held-out %d: %d questions", HELD_OUT_YEAR, len(held_out))

    # Sample
    random.seed(SEED)
    sample = random.sample(held_out, min(SAMPLE_SIZE, len(held_out)))
    logger.info("Sampled %d questions for ablation", len(sample))

    # Build Oracle
    retriever = build_oracle(cfg, HELD_OUT_YEAR)

    # Ollama backend — 300s timeout for long CoT+context prompts
    backend = OllamaBackend(model_name=OLLAMA_MODEL, timeout=300)
    logger.info("Using Ollama model: %s", OLLAMA_MODEL)

    # Run all 4 conditions (skip already-completed ones from cache)
    model_tag = OLLAMA_MODEL.replace(":", "_").replace("/", "_")
    out_dir = os.path.join("outputs", f"cot_ablation_{HELD_OUT_YEAR}_{model_tag}")
    os.makedirs(out_dir, exist_ok=True)

    conditions = ["bare", "context", "cot", "cot+context"]
    all_results = {}
    summaries = []

    for cond in conditions:
        cached_path = os.path.join(out_dir, f"{cond}.json")
        if os.path.exists(cached_path):
            logger.info("Loading cached results for condition: %s", cond)
            with open(cached_path) as f:
                results = json.load(f)
            all_results[cond] = results
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
        results = run_condition(backend, retriever, sample, cond)
        elapsed = time.time() - t0
        all_results[cond] = results
        s = summarize(results, cond)
        s["elapsed_sec"] = round(elapsed, 1)
        summaries.append(s)
        logger.info(
            "  %s — correct: %d/%d (%.1f%%), avg_score: %.3f, time: %.1fs",
            cond, s["correct"], s["n"], s["accuracy"] * 100, s["avg_score"], elapsed,
        )

        # Save immediately after each condition completes (crash-resilient)
        fpath = os.path.join(out_dir, f"{cond}.json")
        with open(fpath, "w") as f:
            json.dump(results, f, indent=2)

    # Save summary
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    # Print final comparison table
    print("\n" + "=" * 70)
    print(f"CoT ABLATION RESULTS — held-out year {HELD_OUT_YEAR}, n={len(sample)}, model={OLLAMA_MODEL}")
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
