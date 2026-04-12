"""Extract full reasoning traces for sample questions across conditions.

Runs 5 medicine questions through 4 conditions and saves complete dialogues:
  1. Medicine specialist (r=16) solo
  2. Medicine specialist + base helper
  3. Medicine specialist + physics specialist (r=16)
  4. Medicine specialist + mediator (50/50)

Designed to produce readable trace files for paper/presentation.
"""

import gc
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from halulujah.domain.data_prep import (
    DOMAIN_SUBJECTS_EXTENDED,
    load_mmlu_domain,
    split_train_test,
)
from halulujah.domain.collab_eval import (
    collab_reasoning,
    solo_reasoning,
    _quick_answer,
)
from halulujah.domain.cross_eval import extract_answer_letter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

N_QUESTIONS = 5


def format_trace(question_text, expected, condition, solo_chain=None,
                 collab_chain=None, pre_collab=None, final_answer=None):
    """Format a single trace as readable text."""
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"CONDITION: {condition}")
    lines.append(f"EXPECTED ANSWER: {expected}")
    lines.append(f"{'='*70}")
    lines.append(f"\nQUESTION:\n{question_text}\n")

    if pre_collab:
        lines.append(f"PRE-COLLABORATION SNAPSHOTS:")
        lines.append(f"  Agent A (specialist): {pre_collab.get('agent_a', {}).get('raw', 'N/A')}")
        lines.append(f"  Agent B (helper):     {pre_collab.get('agent_b', {}).get('raw', 'N/A')}")
        lines.append("")

    if solo_chain:
        lines.append("SOLO REASONING CHAIN:")
        for i, step in enumerate(solo_chain):
            lines.append(f"  --- Round {i+1} ---")
            lines.append(f"  {step}")
            lines.append("")

    if collab_chain:
        lines.append("COLLABORATION CHAIN:")
        for step in collab_chain:
            agent = step.get("agent", "unknown")
            thought = step.get("thought", "")
            lines.append(f"  --- [{agent} expert] ---")
            lines.append(f"  {thought}")
            lines.append("")

    if final_answer:
        predicted = extract_answer_letter(final_answer)
        correct = predicted == expected
        status = "CORRECT" if correct else "WRONG"
        lines.append(f"FINAL ANSWER: {final_answer}")
        lines.append(f"EXTRACTED: {predicted} (expected {expected}) — {status}")

    lines.append("")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-dir", default="adapters")
    parser.add_argument("--mediator-dir", default="mediators")
    parser.add_argument("--output-dir", default="results/traces")
    parser.add_argument("--n-questions", type=int, default=N_QUESTIONS)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test questions
    entries = load_mmlu_domain("medicine", split="test", cache_dir=args.cache_dir)
    _, test_med = split_train_test(entries, test_size=50, seed=42)
    questions = test_med[:args.n_questions]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir
    )

    os.makedirs(args.output_dir, exist_ok=True)
    all_traces = []

    # ==============================
    # Condition 1: Solo specialist
    # ==============================
    logger.info("=== Condition 1: Medicine specialist solo ===")
    specialist = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir,
    ).to(device)
    specialist = PeftModel.from_pretrained(
        specialist, os.path.join(args.adapter_dir, "adapter_medicine")
    )
    specialist.eval()

    solo_traces = []
    for entry in questions:
        final, chain = solo_reasoning(
            specialist, tokenizer, entry["question"], "medicine",
            n_rounds=3, device=device,
        )
        trace = format_trace(
            entry["question"], entry["answer_letter"],
            "SOLO — medicine specialist (r=16)",
            solo_chain=chain, final_answer=final,
        )
        solo_traces.append(trace)
        all_traces.append({
            "condition": "solo_specialist",
            "question": entry["question"][:200],
            "expected": entry["answer_letter"],
            "predicted": extract_answer_letter(final),
            "chain": chain,
            "final": final,
        })

    # ==============================
    # Condition 2: Specialist + base helper
    # ==============================
    logger.info("=== Condition 2: Medicine specialist + base helper ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir,
    ).to(device)
    base_model.eval()

    base_traces = []
    for entry in questions:
        pre_a = _quick_answer(specialist, tokenizer, entry["question"], "medicine", device)
        pre_b = _quick_answer(base_model, tokenizer, entry["question"], "general", device)
        pre_collab = {
            "agent_a": {"answer": extract_answer_letter(pre_a), "raw": pre_a},
            "agent_b": {"answer": extract_answer_letter(pre_b), "raw": pre_b},
        }
        final, chain = collab_reasoning(
            specialist, base_model, tokenizer, entry["question"],
            "medicine", "general", n_rounds=3, device=device,
        )
        trace = format_trace(
            entry["question"], entry["answer_letter"],
            "COLLAB — medicine specialist + BASE helper",
            collab_chain=chain, pre_collab=pre_collab, final_answer=final,
        )
        base_traces.append(trace)
        all_traces.append({
            "condition": "specialist_plus_base",
            "question": entry["question"][:200],
            "expected": entry["answer_letter"],
            "pre_specialist": extract_answer_letter(pre_a),
            "pre_helper": extract_answer_letter(pre_b),
            "predicted": extract_answer_letter(final),
            "chain": [{"agent": s["agent"], "thought": s["thought"]} for s in chain],
            "final": final,
        })

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ==============================
    # Condition 3: Specialist + cross-domain physics
    # ==============================
    logger.info("=== Condition 3: Medicine specialist + physics specialist ===")
    physics = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir,
    ).to(device)
    physics = PeftModel.from_pretrained(
        physics, os.path.join(args.adapter_dir, "adapter_physics")
    )
    physics.eval()

    cross_traces = []
    for entry in questions:
        pre_a = _quick_answer(specialist, tokenizer, entry["question"], "medicine", device)
        pre_b = _quick_answer(physics, tokenizer, entry["question"], "physics", device)
        pre_collab = {
            "agent_a": {"answer": extract_answer_letter(pre_a), "raw": pre_a},
            "agent_b": {"answer": extract_answer_letter(pre_b), "raw": pre_b},
        }
        final, chain = collab_reasoning(
            specialist, physics, tokenizer, entry["question"],
            "medicine", "physics", n_rounds=3, device=device,
        )
        trace = format_trace(
            entry["question"], entry["answer_letter"],
            "COLLAB — medicine specialist + PHYSICS specialist (cross-domain)",
            collab_chain=chain, pre_collab=pre_collab, final_answer=final,
        )
        cross_traces.append(trace)
        all_traces.append({
            "condition": "specialist_plus_cross",
            "question": entry["question"][:200],
            "expected": entry["answer_letter"],
            "pre_specialist": extract_answer_letter(pre_a),
            "pre_helper": extract_answer_letter(pre_b),
            "predicted": extract_answer_letter(final),
            "chain": [{"agent": s["agent"], "thought": s["thought"]} for s in chain],
            "final": final,
        })

    del physics
    gc.collect()
    torch.cuda.empty_cache()

    # ==============================
    # Condition 4: Specialist + mediator (if available)
    # ==============================
    mediator_path = os.path.join(args.mediator_dir, "mediator_medicine_physics")
    if os.path.exists(mediator_path):
        logger.info("=== Condition 4: Medicine specialist + mediator ===")
        mediator = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=torch.bfloat16,
            trust_remote_code=True, cache_dir=args.cache_dir,
        ).to(device)
        mediator = PeftModel.from_pretrained(mediator, mediator_path)
        mediator.eval()

        med_traces = []
        for entry in questions:
            pre_a = _quick_answer(specialist, tokenizer, entry["question"], "medicine", device)
            pre_b = _quick_answer(mediator, tokenizer, entry["question"], "medicine+physics mediator", device)
            pre_collab = {
                "agent_a": {"answer": extract_answer_letter(pre_a), "raw": pre_a},
                "agent_b": {"answer": extract_answer_letter(pre_b), "raw": pre_b},
            }
            final, chain = collab_reasoning(
                specialist, mediator, tokenizer, entry["question"],
                "medicine", "medicine+physics mediator", n_rounds=3, device=device,
            )
            trace = format_trace(
                entry["question"], entry["answer_letter"],
                "COLLAB — medicine specialist + MEDIATOR (50/50 medicine+physics)",
                collab_chain=chain, pre_collab=pre_collab, final_answer=final,
            )
            med_traces.append(trace)
            all_traces.append({
                "condition": "specialist_plus_mediator",
                "question": entry["question"][:200],
                "expected": entry["answer_letter"],
                "pre_specialist": extract_answer_letter(pre_a),
                "pre_helper": extract_answer_letter(pre_b),
                "predicted": extract_answer_letter(final),
                "chain": [{"agent": s["agent"], "thought": s["thought"]} for s in chain],
                "final": final,
            })

        del mediator
        gc.collect()
        torch.cuda.empty_cache()
    else:
        med_traces = []
        logger.info("No mediator adapter found at %s, skipping condition 4", mediator_path)

    del specialist
    gc.collect()
    torch.cuda.empty_cache()

    # ==============================
    # Save all traces
    # ==============================
    # Human-readable traces
    with open(os.path.join(args.output_dir, "traces_readable.txt"), "w") as f:
        f.write("REASONING TRACE SAMPLES\n")
        f.write(f"Questions: {args.n_questions} medicine MMLU\n")
        f.write(f"Model: {args.model_name}\n\n")

        for i, entry in enumerate(questions):
            f.write(f"\n{'#'*70}\n")
            f.write(f"QUESTION {i+1} / {len(questions)}\n")
            f.write(f"{'#'*70}\n\n")
            f.write(solo_traces[i] + "\n")
            f.write(base_traces[i] + "\n")
            f.write(cross_traces[i] + "\n")
            if med_traces:
                f.write(med_traces[i] + "\n")

    # JSON for analysis
    with open(os.path.join(args.output_dir, "traces.json"), "w") as f:
        json.dump(all_traces, f, indent=2)

    logger.info("Saved %d traces to %s", len(all_traces), args.output_dir)

    # Print summary
    print(f"\nGenerated {len(all_traces)} traces for {len(questions)} questions")
    for cond in ["solo_specialist", "specialist_plus_base", "specialist_plus_cross", "specialist_plus_mediator"]:
        cond_traces = [t for t in all_traces if t["condition"] == cond]
        if cond_traces:
            correct = sum(1 for t in cond_traces if t["predicted"] == t["expected"])
            print(f"  {cond}: {correct}/{len(cond_traces)} correct")


if __name__ == "__main__":
    main()
