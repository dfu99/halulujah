"""Full FT training for domain specialists on Qwen3-1.7B (A40 48 GB).

Same data sources as train_specialist_ood.py (medqa / gsm8k / pubmedqa /
casehold / sciq). No LoRA, no quantization. fp32 Adam, bf16 weights +
activations, gradient checkpointing.

Saves checkpoints every --save-steps for matched-solo-accuracy selection.

Usage:
  python train_specialist_full_ft.py --source medqa --domain medicine \\
    --epochs 3 --save-steps 500 --output-dir /workspace/adapters_1p7b_full_ft/medicine
"""
import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Reuse format functions from train_specialist_ood
sys.path.insert(0, str(ROOT / "src/scripts"))
from train_specialist_ood import (  # noqa: E402
    format_medqa, format_gsm8k, format_pubmedqa,
    format_casehold, format_sciq,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--source",
                   choices=["medqa", "gsm8k", "pubmedqa", "casehold", "sciq"],
                   required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--output-dir", required=True,
                   help="Directory for checkpoints + final model")
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5,
                   help="Full FT LR (smaller than LoRA's 5e-5)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-train-examples", type=int, default=10000)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    args = p.parse_args()

    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"already trained at {args.output_dir}, exiting")
        return

    print(f"loading dataset {args.source} ...")
    from datasets import load_dataset, Dataset
    if args.source == "medqa":
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train",
                          cache_dir=args.cache_dir)
        formatted = [{"text": format_medqa(ex)} for ex in ds]
    elif args.source == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train",
                          cache_dir=args.cache_dir)
        formatted = [{"text": format_gsm8k(ex)} for ex in ds]
    elif args.source == "pubmedqa":
        ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial",
                          split="train", cache_dir=args.cache_dir)
        formatted = [{"text": format_pubmedqa(ex)} for ex in ds]
    elif args.source == "casehold":
        ds = load_dataset("casehold/casehold", "all", split="train",
                          cache_dir=args.cache_dir, trust_remote_code=True)
        formatted = []
        for ex in ds:
            t = format_casehold(ex)
            if t is not None:
                formatted.append({"text": t})
    elif args.source == "sciq":
        ds = load_dataset("allenai/sciq", split="train",
                          cache_dir=args.cache_dir)
        formatted = []
        for ex in ds:
            t = format_sciq(ex)
            if t is not None:
                formatted.append({"text": t})
    else:
        raise ValueError(f"unknown source: {args.source}")

    if args.max_train_examples > 0 and len(formatted) > args.max_train_examples:
        import random
        random.seed(42)
        formatted = random.sample(formatted, args.max_train_examples)
    print(f"  {len(formatted)} training examples")
    train_ds = Dataset.from_list(formatted)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(f"loading {args.model_name} (full FT, no LoRA, no quantization) ...")
    tok = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable} / {total} ({100*trainable/total:.2f}%)  "
          f"(should be ~100% for full FT)")

    targs = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=20,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=20,
        bf16=True,
        gradient_checkpointing=True,
        max_length=args.max_length,
    )
    trainer = SFTTrainer(model=model, args=targs, train_dataset=train_ds)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"FINAL_METRICS final_model={args.output_dir} domain={args.domain} "
          f"source={args.source}")


if __name__ == "__main__":
    main()
