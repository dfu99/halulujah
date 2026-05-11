"""LoRA SFT training for domain specialists on Qwen3-1.7B / Qwen3-4B.

Mirrors train_specialist_full_ft.py exactly (same data sources, same
domain mapping, same TRL SFTTrainer call) but wraps the model with
peft LoraConfig at the requested rank. LoRA `alpha = 2 * r` per the
canonical setup in src/scripts/run_rank_ablation.py.

Output layout: --output-dir contains adapter_config.json +
adapter_model.safetensors (~50 MB total). Per-step checkpoints are
checkpoint-N/ subdirs with the same adapter files.

Usage:
  python train_specialist_lora.py \\
    --model-name Qwen/Qwen3-4B \\
    --source pubmedqa --domain biology --rank 8 \\
    --epochs 3 --max-train-examples 5000 --save-steps 99999 \\
    --output-dir /workspace/adapters_4b_lora_r8/biology
"""
import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))
from train_specialist_ood import (  # noqa: E402
    format_medqa, format_gsm8k, format_pubmedqa,
    format_casehold, format_sciq,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-4B")
    p.add_argument("--source",
                   choices=["medqa", "gsm8k", "pubmedqa", "casehold", "sciq"],
                   required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--rank", type=int, required=True,
                   help="LoRA rank (alpha is set to 2*rank automatically)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5,
                   help="LoRA LR (larger than full FT's 2e-5)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-train-examples", type=int, default=5000)
    p.add_argument("--save-steps", type=int, default=99999,
                   help="Effectively disable per-step ckpts; only final adapter is saved")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    args = p.parse_args()

    if os.path.exists(os.path.join(args.output_dir, "adapter_config.json")):
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
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    print(f"loading {args.model_name} for LoRA r={args.rank} "
          f"(alpha={args.rank * 2}) ...")
    tok = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir)

    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable} / {total} ({100*trainable/total:.4f}%)  "
          f"(LoRA r={args.rank})")

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
        save_total_limit=10,
        save_only_model=True,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=args.max_length,
    )
    trainer = SFTTrainer(model=model, args=targs, train_dataset=train_ds)
    trainer.train()
    try:
        trainer.save_model(args.output_dir)
        print(f"FINAL_METRICS final_adapter={args.output_dir} domain={args.domain} "
              f"source={args.source} rank={args.rank}")
    except Exception as e:
        print(f"WARNING: final save_model failed: {e}")
        print(f"FINAL_METRICS final_adapter_failed=true "
              f"checkpoints_in={args.output_dir} "
              f"domain={args.domain} source={args.source} rank={args.rank}")


if __name__ == "__main__":
    main()
