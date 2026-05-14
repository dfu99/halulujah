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


def format_pubmedqa(ex):
    contexts = ex.get("context", {})
    if isinstance(contexts, dict):
        ctx = " ".join(contexts.get("contexts", []))
    elif isinstance(contexts, list):
        ctx = " ".join(contexts)
    else:
        ctx = ""
    q = ex["question"].strip()
    a = ex.get("final_decision") or ex.get("answer") or ""
    if ctx:
        return f"Context: {ctx.strip()}\nQuestion: {q}\nAnswer: {a}"
    return f"Question: {q}\nAnswer: {a}"


def format_casehold(ex):
    """Convert CaseHOLD (5-option) to 4-option MCQ. Skip if label==4."""
    label_raw = ex.get("label")
    try:
        label = int(label_raw)
    except (TypeError, ValueError):
        return None
    if label >= 4:
        return None
    ctx = ex.get("citing_prompt") or ex.get("context") or ""
    holdings = [ex.get(f"holding_{i}", "") for i in range(4)]
    if not all(holdings):
        return None
    letter = chr(ord("A") + label)
    parts = [f"Case: {ctx.strip()}",
             "Question: Which of the following is the correct holding?"]
    for i, h in enumerate(holdings):
        parts.append(f"{chr(ord('A') + i)}) {h.strip()}")
    parts.append(f"Answer: {letter}")
    return "\n".join(parts)


def format_sciq(ex):
    """SciQ: 1 correct + 3 distractors. Shuffle deterministically by question hash."""
    import random
    question = ex.get("question", "").strip()
    correct = ex.get("correct_answer", "").strip()
    distractors = [ex.get(f"distractor{i}", "").strip() for i in (1, 2, 3)]
    if not question or not correct or not all(distractors):
        return None
    options = [correct] + distractors
    rng = random.Random(hash(question) % (2 ** 32))
    indices = list(range(4))
    rng.shuffle(indices)
    correct_pos = indices.index(0)
    shuffled = [options[i] for i in indices]
    parts = [f"Question: {question}"]
    for i, opt in enumerate(shuffled):
        parts.append(f"{chr(ord('A') + i)}) {opt}")
    parts.append(f"Answer: {chr(ord('A') + correct_pos)}")
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--source",
                   choices=["medqa", "gsm8k", "pubmedqa", "casehold", "sciq"],
                   required=True)
    p.add_argument("--domain", required=True, help="medicine or math")
    p.add_argument("--adapter-dir", default="/workspace/adapters_1p7b_ood")
    p.add_argument("--cache-dir", default="/workspace/hf_cache")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-train-examples", type=int, default=0,
                   help="If > 0, subsample dataset to this many examples")
    args = p.parse_args()

    out_dir = os.path.join(args.adapter_dir, f"adapter_{args.domain}_{args.source}")
    if os.path.exists(os.path.join(out_dir, "adapter_config.json")):
        print(f"already trained at {out_dir}, exiting")
        return

    print(f"loading dataset {args.source} ...")
    from datasets import load_dataset
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
        ds = load_dataset("casehold/casehold", "all",
                          split="train", cache_dir=args.cache_dir,
                          trust_remote_code=True)
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
        args.model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir)
    cfg = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable} / {total} ({100*trainable/total:.2f}%)")

    sft_kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr, warmup_ratio=0.1,
        logging_steps=20, save_strategy="no",
        bf16=True, gradient_checkpointing=True,
    )
    import inspect
    _sig = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in _sig:
        sft_kwargs["max_length"] = args.max_length
    elif "max_seq_length" in _sig:
        sft_kwargs["max_seq_length"] = args.max_length
    targs = SFTConfig(**sft_kwargs)
    trainer = SFTTrainer(model=model, args=targs, train_dataset=train_ds)
    trainer.train()
    model.save_pretrained(out_dir)
    print(f"FINAL_METRICS adapter_path={out_dir} domain={args.domain} source={args.source}")


if __name__ == "__main__":
    main()
