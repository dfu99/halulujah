"""Verify a physics LoRA adapter on top of a base model.

Compares base-alone vs base+adapter on:
  - 4 MMLU physics-related subjects: astronomy, college_physics,
    high_school_physics, conceptual_physics
  - SciQ test split (held out from train), 4-option MCQ

Pass gate: adapter >= base + 5pp on at least one benchmark.
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

LETTER_RE = re.compile(r"\b([ABCD])\b")

PHYSICS_SUBJECTS = ["astronomy", "college_physics", "high_school_physics",
                    "conceptual_physics"]


def _format_mmlu_q(q):
    return (
        f"{q['question']}\n"
        f"A) {q['choices'][0]}\n"
        f"B) {q['choices'][1]}\n"
        f"C) {q['choices'][2]}\n"
        f"D) {q['choices'][3]}"
    )


def extract_letter(t):
    m = LETTER_RE.search(t)
    return m.group(1) if m else None


def eval_mmlu_subject(model, tok, subject, n_questions, n_shot):
    from datasets import load_dataset
    test = load_dataset("cais/mmlu", subject, split="test")
    dev = load_dataset("cais/mmlu", subject, split="validation")
    test_qs = list(test)[:n_questions]
    dev_qs = list(dev)[:n_shot]

    fewshots = [{"question": _format_mmlu_q(d), "ans": chr(ord("A") + d["answer"])}
                for d in dev_qs]
    correct = 0
    for q in test_qs:
        question = _format_mmlu_q(q)
        gold = chr(ord("A") + q["answer"])
        prompt_parts = [f"Question: {fs['question']}\nAnswer: {fs['ans']}"
                        for fs in fewshots]
        prompt_parts.append(f"Question: {question}\nAnswer:")
        prompt = "\n\n".join(prompt_parts)
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=4096).to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = tok.decode(gen[0][inputs.input_ids.shape[1]:],
                         skip_special_tokens=True)
        pred = extract_letter(new)
        if pred == gold:
            correct += 1
    return correct / len(test_qs), len(test_qs)


def _format_sciq_q(ex):
    """Format SciQ as 4-option MCQ. Returns (prompt_question, gold_letter, full_string)."""
    question = ex["question"].strip()
    correct = ex["correct_answer"].strip()
    distractors = [ex[f"distractor{i}"].strip() for i in (1, 2, 3)]
    options = [correct] + distractors
    rng = random.Random(hash(question) % (2 ** 32))
    indices = list(range(4))
    rng.shuffle(indices)
    correct_pos = indices.index(0)
    shuffled = [options[i] for i in indices]
    parts = [question]
    for i, opt in enumerate(shuffled):
        parts.append(f"{chr(ord('A') + i)}) {opt}")
    return "\n".join(parts), chr(ord("A") + correct_pos)


def eval_sciq(model, tok, n_questions, n_shot, cache_dir):
    from datasets import load_dataset
    test = load_dataset("allenai/sciq", split="test", cache_dir=cache_dir)
    train = load_dataset("allenai/sciq", split="train", cache_dir=cache_dir)
    test_qs = list(test)[:n_questions]
    fewshots = list(train)[:n_shot]

    correct = 0
    for q in test_qs:
        q_str, gold = _format_sciq_q(q)
        prompt_parts = []
        for fs in fewshots:
            fs_q, fs_a = _format_sciq_q(fs)
            prompt_parts.append(f"Question: {fs_q}\nAnswer: {fs_a}")
        prompt_parts.append(f"Question: {q_str}\nAnswer:")
        prompt = "\n\n".join(prompt_parts)
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=4096).to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = tok.decode(gen[0][inputs.input_ids.shape[1]:],
                         skip_special_tokens=True)
        pred = extract_letter(new)
        if pred == gold:
            correct += 1
    return correct / len(test_qs), len(test_qs)


def evaluate(model, tok, args, label):
    out = {"label": label, "subjects": {}, "sciq": None}
    for s in args.mmlu_subjects:
        acc, n = eval_mmlu_subject(model, tok, s, args.mmlu_n, args.n_shot)
        out["subjects"][s] = {"accuracy": acc, "n": n}
        print(f"  {label} MMLU/{s}: {acc*100:.1f}% n={n}")
    if args.sciq_n > 0:
        acc, n = eval_sciq(model, tok, args.sciq_n, args.n_shot,
                           args.cache_dir)
        out["sciq"] = {"accuracy": acc, "n": n}
        print(f"  {label} SciQ-test: {acc*100:.1f}% n={n}")
    return out


def load_model(base_name, adapter_path, cache_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        base_name, trust_remote_code=True, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir).to("cuda")
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3-1.7B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--mmlu-n", type=int, default=100)
    p.add_argument("--sciq-n", type=int, default=200)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    p.add_argument("--mmlu-subjects", nargs="+", default=PHYSICS_SUBJECTS)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    print(f"loading base {args.base} + adapter {args.adapter} ...")
    model, tok = load_model(args.base, args.adapter, args.cache_dir)
    spec = evaluate(model, tok, args, "specialist")
    spec["adapter"] = args.adapter
    del model
    torch.cuda.empty_cache()

    print(f"loading base {args.base} (no adapter) ...")
    base_model, base_tok = load_model(args.base, None, args.cache_dir)
    base = evaluate(base_model, base_tok, args, "base")
    del base_model
    torch.cuda.empty_cache()

    out = {"specialist": spec, "base": base, "n_shot": args.n_shot}

    print("\n=== verification summary ===")
    print(f"{'benchmark':32s} {'base':>8s} {'spec':>8s} {'delta':>8s}")
    pass_count = 0
    n_bench = len(spec["subjects"])
    for s in spec["subjects"]:
        b = base["subjects"][s]["accuracy"] * 100
        sp = spec["subjects"][s]["accuracy"] * 100
        d = sp - b
        if d >= 5: pass_count += 1
        print(f"{s:32s} {b:>7.1f}% {sp:>7.1f}% {d:>+7.1f}")
    if spec.get("sciq") is not None and base.get("sciq") is not None:
        b = base["sciq"]["accuracy"] * 100
        sp = spec["sciq"]["accuracy"] * 100
        d = sp - b
        if d >= 5: pass_count += 1
        print(f"{'SciQ-test':32s} {b:>7.1f}% {sp:>7.1f}% {d:>+7.1f}")
        n_bench += 1
    print(f"\nVERIFICATION GATE: {pass_count} of {n_bench} benchmarks "
          f"show specialist >= base + 5pp")

    out["pass_count"] = pass_count
    out["pass_threshold"] = 1
    out["verified"] = pass_count >= 1

    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nFINAL_METRICS verified={out['verified']} pass_count={pass_count} "
          f"output={args.output}")


if __name__ == "__main__":
    main()
