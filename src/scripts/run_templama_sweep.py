#!/usr/bin/env python3
"""Hallucination sweep on TempLAMA using Ollama backend.

Tests whether certain temperature/top_p/top_k settings improve temporal
prediction from adjacent-year entity history context.

Hypothesis: Higher-temperature "hallucination" settings may help the model
generalize beyond rote copying of the most recent context value — especially
for facts that *changed* between years.

Uses Ollama with per-question timeout handling so one slow request
doesn't kill the whole sweep.
"""

import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, "src")

import requests

from halulujah.data.chat_formatter import format_cot_system_prompt, format_cot_user_prompt
from halulujah.data.loader import load_templama, split_by_year
from halulujah.rl.verifier import extract_cot_answer, multi_answer_verifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "50"))
HELD_OUT_YEAR = int(os.environ.get("HELD_OUT_YEAR", "2020"))
SEED = 42
MAX_TOKENS = 64
TIMEOUT_PER_Q = 120  # seconds per question; skip if exceeded

# Sweep grid
TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
TOP_PS = [0.5, 0.7, 0.9, 1.0]
TOP_KS = [20, 50]

MODES = ["context", "cot+context"]


def ollama_generate(prompt, temperature, top_p, top_k, num_predict=MAX_TOKENS):
    """Generate one response via Ollama with timeout handling."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": num_predict,
            },
            timeout=TIMEOUT_PER_Q,
        )
        resp.raise_for_status()
        return resp.json()["response"]
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        logger.warning("  Timeout/error on question, skipping: %s", e)
        return None


def build_prompt(entry, mode):
    """Build a single prompt string for Ollama."""
    bare_system = (
        "You are a helpful assistant. Answer with only the name or entity. "
        "Give a single short answer, not an explanation."
    )
    cot_system = format_cot_system_prompt()

    q = entry["question"]
    ctx = entry.get("context", "")

    if mode == "context":
        if ctx:
            return f"{bare_system}\n\nContext:\n{ctx}\n\n{q}"
        return f"{bare_system}\n\n{q}"
    elif mode == "cot+context":
        if ctx:
            user_part = format_cot_user_prompt(q, ctx)
        else:
            user_part = (
                f"Question: {q}\n\n"
                "Think step by step, then provide your final answer "
                "on a line starting with 'Answer:'."
            )
        return f"{cot_system}\n\n{user_part}"
    else:
        return f"{bare_system}\n\n{q}"


def label_changed(sample, data, held_out_year):
    """Label each entry as 'changed' or 'same' vs prior year."""
    prior_year = held_out_year - 1
    prior = [e for e in data if e["year"] == prior_year]
    prior_map = {}
    for e in prior:
        base = "_".join(e["id"].rsplit("_", 1)[:-1])
        prior_map[base] = e["answer"]

    for e in sample:
        base = "_".join(e["id"].rsplit("_", 1)[:-1])
        prev_ans = prior_map.get(base)
        if prev_ans is None:
            e["_change_label"] = "no_prior"
        elif prev_ans != e["answer"]:
            e["_change_label"] = "changed"
        else:
            e["_change_label"] = "same"
    return sample


def stats(results):
    if not results:
        return {"n": 0, "correct": 0, "accuracy": 0, "avg_score": 0}
    correct = sum(1 for r in results if r["correct"])
    avg = sum(r["score"] for r in results) / len(results)
    return {
        "n": len(results),
        "correct": correct,
        "accuracy": round(correct / len(results), 4),
        "avg_score": round(avg, 4),
    }


def main():
    data = load_templama(cache_dir="data")
    _, held_out = split_by_year(data, [HELD_OUT_YEAR])
    with_ctx = [e for e in held_out if e.get("context")]

    # Label changed vs same
    label_changed(with_ctx, data, HELD_OUT_YEAR)
    changed_pool = [e for e in with_ctx if e["_change_label"] == "changed"]
    same_pool = [e for e in with_ctx if e["_change_label"] == "same"]

    # Stratified sample: up to 25 changed + 25 same
    random.seed(SEED)
    n_changed = min(25, len(changed_pool))
    n_same = min(SAMPLE_SIZE - n_changed, len(same_pool))
    sample = random.sample(changed_pool, n_changed) + random.sample(same_pool, n_same)
    random.shuffle(sample)

    logger.info(
        "Sample: %d total (%d changed, %d same)",
        len(sample),
        sum(1 for e in sample if e["_change_label"] == "changed"),
        sum(1 for e in sample if e["_change_label"] == "same"),
    )

    # Verify ollama is reachable
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        logger.info("Ollama models available: %s", models)
    except Exception as e:
        logger.error("Cannot reach Ollama at %s: %s", OLLAMA_URL, e)
        sys.exit(1)

    model_tag = OLLAMA_MODEL.replace(":", "_").replace("/", "_")
    out_dir = os.path.join("outputs", f"templama_sweep_{HELD_OUT_YEAR}_{model_tag}")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "sweep_results.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            all_results = json.load(f)
        logger.info("Loaded %d cached results", len(all_results))
    else:
        all_results = {}

    total_combos = len(MODES) * len(TEMPERATURES) * len(TOP_PS) * len(TOP_KS)
    done = len(all_results)

    for mode in MODES:
        for temp in TEMPERATURES:
            for top_p in TOP_PS:
                for top_k in TOP_KS:
                    key = f"{mode}_t{temp}_p{top_p}_k{top_k}"
                    if key in all_results:
                        continue

                    logger.info(
                        "[%d/%d] %s t=%.1f p=%.1f k=%d",
                        done + 1, total_combos, mode, temp, top_p, top_k,
                    )
                    t0 = time.time()

                    results_all = []
                    results_changed = []
                    results_same = []
                    skipped = 0

                    for entry in sample:
                        prompt = build_prompt(entry, mode)
                        response = ollama_generate(prompt, temp, top_p, top_k)

                        if response is None:
                            skipped += 1
                            continue

                        if mode.startswith("cot"):
                            extracted = extract_cot_answer(response)
                        else:
                            extracted = response

                        answers_all = entry.get("answers_all", [entry["answer"]])
                        score = multi_answer_verifier(answers_all, extracted)
                        res = {"score": score, "correct": score >= 0.6}

                        results_all.append(res)
                        if entry["_change_label"] == "changed":
                            results_changed.append(res)
                        elif entry["_change_label"] == "same":
                            results_same.append(res)

                    elapsed = time.time() - t0

                    all_results[key] = {
                        "mode": mode,
                        "temperature": temp,
                        "top_p": top_p,
                        "top_k": top_k,
                        "all": stats(results_all),
                        "changed": stats(results_changed),
                        "same": stats(results_same),
                        "skipped": skipped,
                        "elapsed_sec": round(elapsed, 1),
                    }

                    a = all_results[key]["all"]
                    c = all_results[key]["changed"]
                    s = all_results[key]["same"]
                    logger.info(
                        "  all: %d/%d (%.1f%%) | chg: %d/%d (%.0f%%) | same: %d/%d (%.0f%%) | skip=%d | %.1fs",
                        a["correct"], a["n"], a["accuracy"] * 100,
                        c["correct"], c["n"], c["accuracy"] * 100 if c["n"] else 0,
                        s["correct"], s["n"], s["accuracy"] * 100 if s["n"] else 0,
                        skipped, elapsed,
                    )

                    # Save after each combo for resume capability
                    with open(summary_path, "w") as f:
                        json.dump(all_results, f, indent=2)
                    done += 1

    # Print final table
    n_chg = sum(1 for e in sample if e["_change_label"] == "changed")
    n_sm = sum(1 for e in sample if e["_change_label"] == "same")

    print("\n" + "=" * 100)
    print(f"HALLUCINATION SWEEP — TempLAMA {HELD_OUT_YEAR}, n={len(sample)} ({n_chg} changed, {n_sm} same), model={OLLAMA_MODEL}")
    print("=" * 100)

    for mode in MODES:
        print(f"\n{'='*40} {mode} {'='*40}")
        print(f"{'Temp':>5} {'TopP':>5} {'TopK':>5} | {'All':>7} {'AvgS':>6} | {'Changed':>8} {'AvgS':>6} | {'Same':>7} {'AvgS':>6}")
        print("-" * 75)

        best_all = (None, -1)
        best_changed = (None, -1)

        for temp in TEMPERATURES:
            for top_p in TOP_PS:
                for top_k in TOP_KS:
                    key = f"{mode}_t{temp}_p{top_p}_k{top_k}"
                    r = all_results[key]
                    a, c, s = r["all"], r["changed"], r["same"]
                    print(
                        f"{temp:>5.1f} {top_p:>5.1f} {top_k:>5d} | "
                        f"{a['accuracy']*100:>6.1f}% {a['avg_score']:>.3f} | "
                        f"{c['accuracy']*100:>6.1f}% {c['avg_score']:>.3f} | "
                        f"{s['accuracy']*100:>6.1f}% {s['avg_score']:>.3f}"
                    )
                    if a["avg_score"] > best_all[1]:
                        best_all = (f"t={temp} p={top_p} k={top_k}", a["avg_score"], a["accuracy"])
                    if c["n"] > 0 and c["avg_score"] > best_changed[1]:
                        best_changed = (f"t={temp} p={top_p} k={top_k}", c["avg_score"], c["accuracy"])

        print(f"\n  Best overall:  {best_all[0]} → {best_all[2]*100:.1f}% acc, {best_all[1]:.3f} avg")
        print(f"  Best changed:  {best_changed[0]} → {best_changed[2]*100:.1f}% acc, {best_changed[1]:.3f} avg")

    print(f"\nResults: {summary_path}")


if __name__ == "__main__":
    main()
