"""Out-of-distribution test: medicine specialists on MedQA-USMLE.

Tests whether our "domain specialists" learned medicine domain knowledge or
just MMLU-medicine format-pattern matching. Loads each medicine specialist
(LoRA r=16, r=128, full-FT), evaluates 100 USMLE questions in 4-option
format, reports accuracy.

If specialist accuracy on MedQA tracks specialist accuracy on MMLU-medicine,
the specialist learned medicine. If MedQA accuracy is much lower than
MMLU-medicine accuracy, the specialist memorized MMLU's format/pool.

Output: results/ood_medqa/qwen3_{1p7b,4b}_medicine_{condition}.json
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


def format_q(q, ans_letter):
    options = q["options"] if isinstance(q["options"], dict) else eval(q["options"])
    parts = [f"Question: {q['question'].strip()}"]
    for k in sorted(options.keys()):
        parts.append(f"{k}) {options[k]}")
    parts.append("Answer:")
    if ans_letter:
        parts[-1] += f" {ans_letter}"
    return "\n".join(parts)


def build_prompt(query, fewshots):
    return "\n\n".join([format_q(fs, fs["answer_idx"]) for fs in fewshots]
                       + [format_q(query, "")])


def extract_letter(text):
    m = LETTER_RE.search(text)
    return m.group(1) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--adapter-path", default=None,
                   help="Path to LoRA adapter (if testing a specialist)")
    p.add_argument("--n-questions", type=int, default=100)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output", default="results/ood_medqa/qwen3_1p7b_base.json")
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()

    print(f"loading MedQA-USMLE-4-options ...")
    from datasets import load_dataset
    test = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    train = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    test_qs = list(test)[:args.n_questions]
    fewshots = list(train)[:args.n_shot]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {args.model_name} ...")
    tok = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available())
                          else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir).to(device)

    if args.adapter_path:
        from peft import PeftModel
        print(f"loading adapter {args.adapter_path} ...")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    correct = 0
    per_q = []
    for i, q in enumerate(test_qs):
        prompt = build_prompt(q, fewshots)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                pad_token_id=tok.pad_token_id)
        new = tok.decode(gen[0][inputs.input_ids.shape[1]:],
                         skip_special_tokens=True)
        pred = extract_letter(new)
        ok = (pred == q["answer_idx"])
        correct += int(ok)
        per_q.append({"i": i, "pred": pred, "expected": q["answer_idx"],
                      "correct": ok, "meta": q.get("meta_info", "")})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(test_qs)}: {correct/(i+1)*100:.1f}%")

    acc = correct / len(test_qs)
    out = {
        "config": vars(args),
        "dataset": "GBaker/MedQA-USMLE-4-options",
        "accuracy": acc,
        "n": len(test_qs),
        "per_question": per_q,
    }
    os.makedirs(Path(args.output).parent, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nFINAL_METRICS medqa_5shot_acc={acc:.4f} n={len(test_qs)} "
          f"adapter={args.adapter_path or 'BASE'}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
