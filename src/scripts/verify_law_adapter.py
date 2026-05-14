"""Verify a law LoRA adapter on top of a base model.

Compares base-alone vs base+adapter on:
  - 3 MMLU law subjects: jurisprudence, international_law, professional_law
  - CaseHOLD test split (held out from train), converted to 4-option MCQ

Pass gate: adapter >= base + 5pp on at least one benchmark.
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

LAW_SUBJECTS = ["jurisprudence", "international_law", "professional_law"]


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


def _format_casehold_q(ex):
    ctx = ex.get("citing_prompt") or ex.get("context") or ""
    holdings = [ex.get(f"holding_{i}", "") for i in range(4)]
    parts = [f"Case: {ctx.strip()}",
             "Question: Which of the following is the correct holding?"]
    for i, h in enumerate(holdings):
        parts.append(f"{chr(ord('A') + i)}) {h.strip()}")
    return "\n".join(parts)


def eval_casehold(model, tok, n_questions, n_shot, cache_dir):
    """Eval on CaseHOLD test split, 4-option subset (label != 4)."""
    from datasets import load_dataset
    test = load_dataset("casehold/casehold", "all", split="test",
                        cache_dir=cache_dir, trust_remote_code=True)
    train = load_dataset("casehold/casehold", "all", split="train",
                         cache_dir=cache_dir, trust_remote_code=True)

    def good(ex):
        try:
            return int(ex["label"]) < 4
        except (TypeError, ValueError):
            return False

    test_qs = [q for q in test if good(q)][:n_questions]
    fewshots = [q for q in train if good(q)][:n_shot]

    correct = 0
    for q in test_qs:
        gold = chr(ord("A") + int(q["label"]))
        prompt_parts = []
        for fs in fewshots:
            fs_ans = chr(ord("A") + int(fs["label"]))
            prompt_parts.append(f"{_format_casehold_q(fs)}\nAnswer: {fs_ans}")
        prompt_parts.append(f"{_format_casehold_q(q)}\nAnswer:")
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
    out = {"label": label, "subjects": {}, "casehold": None}
    for s in args.mmlu_subjects:
        acc, n = eval_mmlu_subject(model, tok, s, args.mmlu_n, args.n_shot)
        out["subjects"][s] = {"accuracy": acc, "n": n}
        print(f"  {label} MMLU/{s}: {acc*100:.1f}% n={n}")
    if args.casehold_n > 0:
        acc, n = eval_casehold(model, tok, args.casehold_n, args.n_shot,
                               args.cache_dir)
        out["casehold"] = {"accuracy": acc, "n": n}
        print(f"  {label} CaseHOLD-test: {acc*100:.1f}% n={n}")
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
    p.add_argument("--casehold-n", type=int, default=200)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    p.add_argument("--mmlu-subjects", nargs="+", default=LAW_SUBJECTS)
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
    if spec.get("casehold") is not None and base.get("casehold") is not None:
        b = base["casehold"]["accuracy"] * 100
        sp = spec["casehold"]["accuracy"] * 100
        d = sp - b
        if d >= 5: pass_count += 1
        print(f"{'CaseHOLD-test':32s} {b:>7.1f}% {sp:>7.1f}% {d:>+7.1f}")
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
