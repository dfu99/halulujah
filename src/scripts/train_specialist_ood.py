"""Train a 1.7B LoRA domain specialist on a non-MMLU source.

Supports MedQA-USMLE-4-options (medicine) and GSM8K (math). Each
training Q-A pair is formatted in the same MCQ / question-answer
template as the eval, so the model learns content not format.

Output adapter: <adapter_dir>/adapter_<domain>_<source_short>/
"""
import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def format_medqa(ex):
    options = ex["options"] if isinstance(ex["options"], dict) else eval(ex["options"])
    parts = [f"Question: {ex['question'].strip()}"]
    for k in sorted(options.keys()):
        parts.append(f"{k}) {options[k]}")
    parts.append(f"Answer: {ex['answer_idx']}")
    return "\n".join(parts)


def format_gsm8k(ex):
    q = ex["question"].strip()
    a = ex["answer"].strip()
    return f"Question: {q}\nAnswer: {a}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--source", choices=["medqa", "gsm8k"], required=True)
    p.add_argument("--domain", required=True, help="medicine or math")
    p.add_argument("--adapter-dir", default="/workspace/adapters_1p7b_ood")
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-5)
    args = p.parse_args()

    out_dir = os.path.join(args.adapter_dir, f"adapter_{args.domain}_{args.source}")
    if os.path.exists(os.path.join(out_dir, "adapter_config.json")):
        print(f"already trained at {out_dir}, exiting")
        return

    print(f"loading dataset {args.source} ...")
    from datasets import load_dataset
    if args.source == "medqa":
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        formatted = [{"text": format_medqa(ex)} for ex in ds]
    else:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        formatted = [{"text": format_gsm8k(ex)} for ex in ds]
    print(f"  {len(formatted)} training examples")

    from datasets import Dataset
    train_ds = Dataset.from_list(formatted)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    print(f"loading {args.model_name} ...")
    tok = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir)
    cfg = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable} / {total} ({100*trainable/total:.2f}%)")

    targs = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr, warmup_ratio=0.1,
        logging_steps=20, save_strategy="no",
        bf16=True, gradient_checkpointing=True,
        max_length=args.max_length,
    )
    trainer = SFTTrainer(model=model, args=targs, train_dataset=train_ds)
    trainer.train()
    model.save_pretrained(out_dir)
    print(f"FINAL_METRICS adapter_path={out_dir} domain={args.domain} source={args.source}")


if __name__ == "__main__":
    main()
