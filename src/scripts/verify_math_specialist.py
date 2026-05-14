"""Verify a math LLM (e.g. Qwen2.5-Math-1.5B-Instruct) against MMLU-math
per-subject and GSM8K-test, in 5-shot. Compares against a base model.

Usage:
  python verify_math_specialist.py \\
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
    --base  Qwen/Qwen2.5-1.5B-Instruct \\
    --gsm8k-n 200 --mmlu-n 100

Output: results/specialist_verification/{model_short}_vs_{base_short}.json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

LETTER_RE = re.compile(r"\b([ABCD])\b")
NUM_RE = re.compile(r"-?\d+\.?\d*")
HASH_ANS_RE = re.compile(r"####\s*(-?\d+\.?\d*)")


def fmt_mmlu(q, ans=""):
    return f"Question: {q['question'].strip()}\nAnswer:" + (f" {ans}" if ans else "")


def fmt_gsm8k(q, ans=""):
    return f"Question: {q['question'].strip()}\nAnswer:" + (f" {ans}" if ans else "")


def extract_letter(t):
    m = LETTER_RE.search(t)
    return m.group(1) if m else None


def extract_gsm8k_answer(t):
    m = HASH_ANS_RE.search(t)
    if m:
        return float(m.group(1))
    nums = NUM_RE.findall(t)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def extract_gsm8k_gold(answer_text):
    m = HASH_ANS_RE.search(answer_text)
    return float(m.group(1)) if m else None


def eval_mmlu_subject(model, tok, device, subject, n_questions, n_shot,
                      cache_dir):
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
                     max_length=4096).to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = tok.decode(gen[0][inputs.input_ids.shape[1]:],
                         skip_special_tokens=True)
        pred = extract_letter(new)
        if pred == gold:
            correct += 1
    return correct / len(test_qs), len(test_qs)


def _format_mmlu_q(q):
    return (
        f"{q['question']}\n"
        f"A) {q['choices'][0]}\n"
        f"B) {q['choices'][1]}\n"
        f"C) {q['choices'][2]}\n"
        f"D) {q['choices'][3]}"
    )


def eval_gsm8k(model, tok, device, n_questions, n_shot, cache_dir):
    from datasets import load_dataset
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    test_qs = list(test)[:n_questions]
    fewshots = list(train)[:n_shot]

    correct = 0
    for q in test_qs:
        prompt_parts = [f"Question: {fs['question'].strip()}\nAnswer: {fs['answer'].strip()}"
                        for fs in fewshots]
        prompt_parts.append(f"Question: {q['question'].strip()}\nAnswer:")
        prompt = "\n\n".join(prompt_parts)
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=4096).to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = tok.decode(gen[0][inputs.input_ids.shape[1]:],
                         skip_special_tokens=True)
        pred = extract_gsm8k_answer(new)
        gold = extract_gsm8k_gold(q["answer"])
        if pred is not None and gold is not None and abs(pred - gold) < 1e-3:
            correct += 1
    return correct / len(test_qs), len(test_qs)


def evaluate(model_name, args, label):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir).to("cuda")
    model.eval()

    out = {"model": model_name, "label": label, "subjects": {}, "gsm8k": None}

    for s in args.mmlu_subjects:
        acc, n = eval_mmlu_subject(model, tok, "cuda", s, args.mmlu_n,
                                   args.n_shot, args.cache_dir)
        out["subjects"][s] = {"accuracy": acc, "n": n}
        print(f"  {label} MMLU/{s}: {acc*100:.1f}% n={n}")

    if args.gsm8k_n > 0:
        acc, n = eval_gsm8k(model, tok, "cuda", args.gsm8k_n, args.n_shot,
                            args.cache_dir)
        out["gsm8k"] = {"accuracy": acc, "n": n}
        print(f"  {label} GSM8K: {acc*100:.1f}% n={n}")

    del model
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--mmlu-n", type=int, default=100)
    p.add_argument("--gsm8k-n", type=int, default=200)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    p.add_argument("--mmlu-subjects", nargs="+",
                   default=["college_mathematics", "high_school_mathematics",
                            "abstract_algebra", "elementary_mathematics"])
    p.add_argument("--output", default="results/specialist_verification/qwen2_5_math_1_5b.json")
    args = p.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)
    spec = evaluate(args.model, args, "specialist")
    base = evaluate(args.base, args, "base")

    out = {"specialist": spec, "base": base, "n_shot": args.n_shot}

    print("\n=== verification summary ===")
    print(f"{'subject':32s} {'base':>8s} {'spec':>8s} {'delta':>8s}")
    pass_count = 0
    n_bench = len(spec["subjects"])
    for s in spec["subjects"]:
        b = base["subjects"][s]["accuracy"] * 100
        sp = spec["subjects"][s]["accuracy"] * 100
        d = sp - b
        if d >= 5: pass_count += 1
        print(f"{s:32s} {b:>7.1f}% {sp:>7.1f}% {d:>+7.1f}")
    if spec.get("gsm8k") is not None:
        b = base["gsm8k"]["accuracy"] * 100
        sp = spec["gsm8k"]["accuracy"] * 100
        d = sp - b
        if d >= 5: pass_count += 1
        print(f"{'GSM8K-test':32s} {b:>7.1f}% {sp:>7.1f}% {d:>+7.1f}")
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
