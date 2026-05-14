"""Verify a Full FT domain specialist (Qwen3-1.7B) on the same per-domain
benchmarks used for the LoRA verifiers.

Compares Full FT checkpoint vs original Qwen3-1.7B on:
  - per-domain MMLU subjects (5-shot, n=100 each)
  - per-domain held-out OOD eval (medqa-test, gsm8k-test, pubmedqa-pqa_labeled,
    casehold-test, sciq-test)

Pass gate: FT checkpoint >= base + 5pp on >=1 benchmark.

Usage:
  python verify_full_ft_specialist.py \\
    --domain medicine \\
    --ft-checkpoint /workspace/adapters_1p7b_full_ft/medicine/checkpoint-2500 \\
    --output results/specialist_verification/medicine_full_ft/checkpoint-2500.json
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
NUM_RE = re.compile(r"-?\d+\.?\d*")
HASH_ANS_RE = re.compile(r"####\s*(-?\d+\.?\d*)")
YESNO_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)

DOMAIN_CONFIG = {
    "medicine": {
        "mmlu_subjects": ["anatomy", "clinical_knowledge", "college_medicine",
                          "medical_genetics", "professional_medicine", "virology"],
        "ood_source": "medqa",
    },
    "math": {
        "mmlu_subjects": ["college_mathematics", "high_school_mathematics",
                          "abstract_algebra", "elementary_mathematics"],
        "ood_source": "gsm8k",
    },
    "biology": {
        "mmlu_subjects": ["college_biology", "high_school_biology"],
        "ood_source": "pubmedqa",
    },
    "law": {
        "mmlu_subjects": ["jurisprudence", "international_law", "professional_law"],
        "ood_source": "casehold",
    },
    "physics": {
        "mmlu_subjects": ["astronomy", "college_physics", "high_school_physics",
                          "conceptual_physics"],
        "ood_source": "sciq",
    },
}


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


def extract_gsm8k(t):
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


def extract_yesno(t):
    m = YESNO_RE.search(t)
    return m.group(1).lower() if m else None


def eval_mmlu(model, tok, subject, n, n_shot, cache_dir):
    from datasets import load_dataset
    test = load_dataset("cais/mmlu", subject, split="test", cache_dir=cache_dir)
    dev = load_dataset("cais/mmlu", subject, split="validation",
                       cache_dir=cache_dir)
    test_qs = list(test)[:n]
    dev_qs = list(dev)[:n_shot]
    fewshots = [{"question": _format_mmlu_q(d), "ans": chr(ord("A") + d["answer"])}
                for d in dev_qs]
    correct = 0
    for q in test_qs:
        question = _format_mmlu_q(q)
        gold = chr(ord("A") + q["answer"])
        parts = [f"Question: {fs['question']}\nAnswer: {fs['ans']}"
                 for fs in fewshots]
        parts.append(f"Question: {question}\nAnswer:")
        prompt = "\n\n".join(parts)
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


def eval_gsm8k(model, tok, n, n_shot, cache_dir):
    from datasets import load_dataset
    train = load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache_dir)
    test = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
    test_qs = list(test)[:n]
    fewshots = list(train)[:n_shot]
    correct = 0
    for q in test_qs:
        parts = [f"Question: {fs['question'].strip()}\nAnswer: {fs['answer'].strip()}"
                 for fs in fewshots]
        parts.append(f"Question: {q['question'].strip()}\nAnswer:")
        prompt = "\n\n".join(parts)
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=4096).to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = tok.decode(gen[0][inputs.input_ids.shape[1]:],
                         skip_special_tokens=True)
        pred = extract_gsm8k(new)
        gold_m = HASH_ANS_RE.search(q["answer"])
        gold = float(gold_m.group(1)) if gold_m else None
        if pred is not None and gold is not None and abs(pred - gold) < 1e-3:
            correct += 1
    return correct / len(test_qs), len(test_qs)


def eval_medqa(model, tok, n, n_shot, cache_dir):
    from datasets import load_dataset
    train = load_dataset("GBaker/MedQA-USMLE-4-options", split="train",
                         cache_dir=cache_dir)
    test = load_dataset("GBaker/MedQA-USMLE-4-options", split="test",
                        cache_dir=cache_dir)
    test_qs = list(test)[:n]
    fewshots = list(train)[:n_shot]

    def fmt(ex):
        opts = ex["options"] if isinstance(ex["options"], dict) else eval(ex["options"])
        s = [ex["question"].strip()]
        for k in sorted(opts.keys()):
            s.append(f"{k}) {opts[k]}")
        return "\n".join(s)

    correct = 0
    for q in test_qs:
        gold = q["answer_idx"]
        parts = [f"Question: {fmt(fs)}\nAnswer: {fs['answer_idx']}" for fs in fewshots]
        parts.append(f"Question: {fmt(q)}\nAnswer:")
        prompt = "\n\n".join(parts)
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


def eval_pubmedqa(model, tok, n, n_shot, cache_dir):
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train",
                      cache_dir=cache_dir)
    qs = list(ds)
    test_qs = qs[:n]
    fewshots = qs[n:n + n_shot] or qs[-n_shot:]

    def fmt(ex):
        ctxs = ex.get("context", {})
        if isinstance(ctxs, dict):
            c = " ".join(ctxs.get("contexts", []))
        elif isinstance(ctxs, list):
            c = " ".join(ctxs)
        else:
            c = ""
        body = ex["question"].strip()
        if c:
            return f"Context: {c.strip()}\nQuestion: {body}"
        return f"Question: {body}"

    correct = 0
    for q in test_qs:
        gold = (q.get("final_decision") or "").lower()
        parts = []
        for fs in fewshots:
            a = (fs.get("final_decision") or "").lower()
            parts.append(f"{fmt(fs)}\nAnswer: {a}")
        parts.append(f"{fmt(q)}\nAnswer:")
        prompt = "\n\n".join(parts)
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=4096).to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = tok.decode(gen[0][inputs.input_ids.shape[1]:],
                         skip_special_tokens=True)
        pred = extract_yesno(new)
        if pred == gold:
            correct += 1
    return correct / len(test_qs), len(test_qs)


def eval_casehold(model, tok, n, n_shot, cache_dir):
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

    test_qs = [q for q in test if good(q)][:n]
    fewshots = [q for q in train if good(q)][:n_shot]

    def fmt(ex):
        ctx = ex.get("citing_prompt") or ex.get("context") or ""
        holdings = [ex.get(f"holding_{i}", "") for i in range(4)]
        s = [f"Case: {ctx.strip()}",
             "Question: Which of the following is the correct holding?"]
        for i, h in enumerate(holdings):
            s.append(f"{chr(ord('A') + i)}) {h.strip()}")
        return "\n".join(s)

    correct = 0
    for q in test_qs:
        gold = chr(ord("A") + int(q["label"]))
        parts = []
        for fs in fewshots:
            a = chr(ord("A") + int(fs["label"]))
            parts.append(f"{fmt(fs)}\nAnswer: {a}")
        parts.append(f"{fmt(q)}\nAnswer:")
        prompt = "\n\n".join(parts)
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


def eval_sciq(model, tok, n, n_shot, cache_dir):
    from datasets import load_dataset
    test = load_dataset("allenai/sciq", split="test", cache_dir=cache_dir)
    train = load_dataset("allenai/sciq", split="train", cache_dir=cache_dir)
    test_qs = list(test)[:n]
    fewshots = list(train)[:n_shot]

    def fmt(ex):
        question = ex["question"].strip()
        correct = ex["correct_answer"].strip()
        distractors = [ex[f"distractor{i}"].strip() for i in (1, 2, 3)]
        options = [correct] + distractors
        rng = random.Random(hash(question) % (2 ** 32))
        indices = list(range(4))
        rng.shuffle(indices)
        correct_pos = indices.index(0)
        shuffled = [options[i] for i in indices]
        s = [question]
        for i, opt in enumerate(shuffled):
            s.append(f"{chr(ord('A') + i)}) {opt}")
        return "\n".join(s), chr(ord("A") + correct_pos)

    correct = 0
    for q in test_qs:
        q_str, gold = fmt(q)
        parts = []
        for fs in fewshots:
            fs_q, fs_a = fmt(fs)
            parts.append(f"Question: {fs_q}\nAnswer: {fs_a}")
        parts.append(f"Question: {q_str}\nAnswer:")
        prompt = "\n\n".join(parts)
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


OOD_DISPATCH = {
    "medqa": eval_medqa,
    "gsm8k": eval_gsm8k,
    "pubmedqa": eval_pubmedqa,
    "casehold": eval_casehold,
    "sciq": eval_sciq,
}


def evaluate(model_path, args, label):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {model_path} ...")
    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir).to("cuda")
    model.eval()

    cfg = DOMAIN_CONFIG[args.domain]
    out = {"label": label, "model_path": model_path,
           "subjects": {}, "ood": None,
           "ood_name": cfg["ood_source"]}

    for s in cfg["mmlu_subjects"]:
        acc, n = eval_mmlu(model, tok, s, args.mmlu_n, args.n_shot,
                           args.cache_dir)
        out["subjects"][s] = {"accuracy": acc, "n": n}
        print(f"  {label} MMLU/{s}: {acc*100:.1f}% n={n}")

    if args.ood_n > 0:
        fn = OOD_DISPATCH[cfg["ood_source"]]
        acc, n = fn(model, tok, args.ood_n, args.n_shot, args.cache_dir)
        out["ood"] = {"accuracy": acc, "n": n}
        print(f"  {label} {cfg['ood_source']}-test: {acc*100:.1f}% n={n}")

    del model
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=list(DOMAIN_CONFIG))
    p.add_argument("--ft-checkpoint", required=True,
                   help="Path to Full FT checkpoint dir")
    p.add_argument("--base", default="Qwen/Qwen3-1.7B")
    p.add_argument("--mmlu-n", type=int, default=100)
    p.add_argument("--ood-n", type=int, default=200)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    p.add_argument("--output", required=True)
    p.add_argument("--skip-base", action="store_true",
                   help="Skip base eval; reuse cached base from --base-cache")
    p.add_argument("--base-cache",
                   help="Cached base eval JSON to merge instead of re-running")
    args = p.parse_args()

    os.makedirs(Path(args.output).parent, exist_ok=True)

    spec = evaluate(args.ft_checkpoint, args, "ft_specialist")

    if args.skip_base and args.base_cache:
        cached = json.load(open(args.base_cache))
        base = cached["base"]
        print("(reused base eval from cache)")
    else:
        base = evaluate(args.base, args, "base")

    out = {"specialist": spec, "base": base, "n_shot": args.n_shot,
           "domain": args.domain}

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
    if spec.get("ood") is not None and base.get("ood") is not None:
        b = base["ood"]["accuracy"] * 100
        sp = spec["ood"]["accuracy"] * 100
        d = sp - b
        if d >= 5: pass_count += 1
        ood_label = f"{spec['ood_name']}-test"
        print(f"{ood_label:32s} {b:>7.1f}% {sp:>7.1f}% {d:>+7.1f}")
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
