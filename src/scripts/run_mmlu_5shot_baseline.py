"""5-shot MMLU baseline eval for Qwen3 base models.

Runs the standard 5-shot MMLU protocol on the base (no-LoRA) Qwen3
model across our 5 paper-sweep domains. Produces the proper-baseline
column we need for every solo-accuracy table.

Standard MMLU 5-shot format:
  - For each test question, sample 5 in-context examples from the
    same MMLU subject's dev/validation split.
  - Concatenate as "Question: ... A) ... B) ... C) ... D) ...
    Answer: <letter>" pairs, then the test question with empty answer.
  - Generate ~5 tokens, parse the first letter A/B/C/D.

Output: results/mmlu_5shot_baseline/{model_short}.json
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

from halulujah.domain.data_prep import (
    DOMAIN_SUBJECTS_EXTENDED,
    load_mmlu_domain,
    split_train_test,
)


def format_one(q):
    return f"Question: {q['question'].strip()}\nAnswer:"


def build_prompt(query, fewshots):
    parts = []
    for fs in fewshots:
        parts.append(format_one(fs) + f" {fs['answer_letter']}")
    parts.append(format_one(query))
    return "\n\n".join(parts)


LETTER_RE = re.compile(r"\b([ABCD])\b")


def extract_letter(text):
    m = LETTER_RE.search(text)
    return m.group(1) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--domains", nargs="+",
                   default=["medicine", "physics", "biology", "law", "math"])
    p.add_argument("--n-questions", type=int, default=100)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output", default="results/mmlu_5shot_baseline/qwen3_1p7b.json")
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    print(f"loading {args.model_name} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available())
                          else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir).to(device)
    model.eval()

    out = {"config": vars(args), "per_domain": {}}
    os.makedirs(Path(args.output).parent, exist_ok=True)

    for d in args.domains:
        test = load_mmlu_domain(d, split="test", cache_dir=args.cache_dir)
        try:
            dev = load_mmlu_domain(d, split="validation", cache_dir=args.cache_dir)
        except Exception:
            dev = test[:50]
        _, test_split = split_train_test(test, test_size=args.n_questions, seed=42)
        test_qs = test_split[:args.n_questions]

        correct = 0
        per_q = []
        for i, q in enumerate(test_qs):
            fewshots = dev[:args.n_shot] if len(dev) >= args.n_shot else dev
            prompt = build_prompt(q, fewshots)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=8, do_sample=False,
                    pad_token_id=tok.pad_token_id)
            new = tok.decode(gen[0][inputs.input_ids.shape[1]:],
                             skip_special_tokens=True)
            pred = extract_letter(new)
            ok = (pred == q["answer_letter"])
            correct += int(ok)
            per_q.append({"i": i, "pred": pred, "expected": q["answer_letter"],
                          "correct": ok})
            if (i + 1) % 25 == 0:
                acc = correct / (i + 1)
                print(f"  {d} {i+1}/{len(test_qs)}: {acc*100:.1f}%")
        acc = correct / len(test_qs)
        out["per_domain"][d] = {
            "accuracy": acc, "n": len(test_qs),
            "n_shot": args.n_shot, "per_question": per_q,
        }
        print(f"FINAL_METRICS domain={d} mmlu_5shot_acc={acc:.4f} n={len(test_qs)}")
        json.dump(out, open(args.output, "w"), indent=2)

    print(f"\nFINAL_METRICS overall_mean_acc="
          f"{sum(v['accuracy'] for v in out['per_domain'].values()) / len(out['per_domain']):.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
