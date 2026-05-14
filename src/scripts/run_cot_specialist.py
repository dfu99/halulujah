"""Train domain specialist with reasoning preserved.

The problem: our current training data is bare answers ("D. Arthrocentesis").
Qwen3's chat template wraps this as <think></think> + answer, teaching the
LoRA to skip reasoning entirely.

The fix: generate CoT reasoning for each training example using the base model,
then include it in the training data so the LoRA learns domain knowledge
while preserving the model's reasoning capability.

Pipeline:
  1. Generate CoT reasoning for each MMLU training example (base model)
  2. Train medicine specialist with thinking-preserved format
  3. Evaluate: solo accuracy + collab with base + collab with cross-domain
  4. Compare against original (no-thinking) specialist
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from halulujah.domain.data_prep import (
    DOMAIN_SUBJECTS_EXTENDED,
    load_mmlu_domain,
    split_train_test,
)
from halulujah.domain.collab_eval import (
    collab_reasoning_scoped,
    solo_reasoning,
)
from halulujah.domain.cross_eval import extract_answer_letter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DOMAIN = "medicine"
CROSS_DOMAIN = "physics"


def generate_cot_for_entry(model, tokenizer, entry, device):
    """Generate chain-of-thought reasoning for a single training example.

    Prompts the base model to reason through the question, then extracts
    the thinking content to use as training data.
    """
    messages = [
        {"role": "system", "content": (
            f"You are a {entry['domain']} expert. Think through this question "
            "step by step, then give your answer."
        )},
        {"role": "user", "content": entry["question"]},
    ]

    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids.to(device)
    else:
        input_ids = encoded.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            use_cache=True,
        )

    response = tokenizer.decode(
        outputs[0][input_ids.shape[1]:], skip_special_tokens=True,
    ).strip()

    # Extract the thinking content
    thinking = ""
    if "<think>" in response and "</think>" in response:
        thinking = response.split("<think>")[1].split("</think>")[0].strip()
    elif response:
        # Model didn't use think tags — use the whole response as reasoning
        # but strip out the final answer line
        lines = response.strip().split("\n")
        # Keep reasoning lines, drop the final answer-only line
        thinking = "\n".join(lines).strip()

    return thinking


def format_cot_dataset(entries_with_cot, tokenizer, domain):
    """Format training data with CoT reasoning preserved in <think> blocks."""
    from datasets import Dataset

    system_prompt = f"You are a {domain} expert. Answer the question accurately and concisely."
    texts = []

    for entry in entries_with_cot:
        cot = entry.get("cot_reasoning", "")
        answer = entry["answer"]

        if cot:
            # Include reasoning in the response
            assistant_content = f"<think>\n{cot}\n</think>\n\n{answer}"
        else:
            assistant_content = answer

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": entry["question"]},
            {"role": "assistant", "content": assistant_content},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        texts.append(text)

    logger.info("Formatted %d CoT entries for domain '%s'", len(texts), domain)
    return Dataset.from_dict({"text": texts})


def train_cot_specialist(
    domain, model_name, output_dir,
    cache_dir=None, num_epochs=3, lr=5e-5, batch_size=4, rank=16,
):
    """Train a domain specialist with CoT-preserved training data."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    adapter_dir = os.path.join(output_dir, f"adapter_{domain}_cot_r{rank}")

    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        logger.info("CoT specialist already trained, skipping")
        return adapter_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training data
    entries = load_mmlu_domain(domain, split="test", cache_dir=cache_dir)
    try:
        aux = load_mmlu_domain(domain, split="validation", cache_dir=cache_dir)
        entries.extend(aux)
    except Exception:
        pass
    train_entries, _ = split_train_test(entries, test_size=50, seed=42)

    # Step 1: Generate CoT reasoning using base model
    logger.info("=== Generating CoT reasoning for %d entries ===", len(train_entries))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    ).to(device)
    base_model.eval()

    cot_entries = []
    for i, entry in enumerate(train_entries):
        cot = generate_cot_for_entry(base_model, tokenizer, entry, device)
        entry_with_cot = {**entry, "cot_reasoning": cot}
        cot_entries.append(entry_with_cot)
        if (i + 1) % 50 == 0:
            logger.info("  Generated CoT for %d/%d entries", i + 1, len(train_entries))

    # Log sample
    sample = cot_entries[0]
    logger.info("Sample CoT entry:")
    logger.info("  Q: %s", sample["question"][:100])
    logger.info("  CoT: %s", sample["cot_reasoning"][:200])
    logger.info("  A: %s", sample["answer"])

    # Save CoT data
    cot_path = os.path.join(output_dir, f"cot_data_{domain}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(cot_path, "w") as f:
        json.dump(cot_entries, f, indent=2)
    logger.info("Saved CoT data to %s", cot_path)

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Train with CoT data
    logger.info("=== Training CoT specialist r=%d ===", rank)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=cache_dir,
    )

    lora_config = LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("  Trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    train_dataset = format_cot_dataset(cot_entries, tokenizer, domain)

    training_args = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        max_length=768,  # Longer to accommodate CoT
    )

    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(adapter_dir)
    logger.info("Saved CoT specialist to %s", adapter_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir


def evaluate_specialist(
    label, specialist, base_model, cross_model,
    tokenizer, test_entries, domain, cross_domain,
    n_questions=50, n_rounds=3, device=None,
):
    """Evaluate a specialist: solo + base collab + cross collab."""
    questions = test_entries[:n_questions]
    results = {"condition": label}

    # Solo
    solo_correct = 0
    for entry in questions:
        final, _ = solo_reasoning(
            specialist, tokenizer, entry["question"], domain,
            n_rounds=n_rounds, device=device,
        )
        if extract_answer_letter(final) == entry["answer_letter"]:
            solo_correct += 1
    results["solo_acc"] = solo_correct / len(questions)
    logger.info("  [%s] Solo: %.1f%%", label, results["solo_acc"] * 100)

    # Base collab
    base_correct, base_c2w, base_w2c, base_switches = 0, 0, 0, 0
    for entry in questions:
        final, chain, pre_collab = collab_reasoning_scoped(
            specialist, base_model, tokenizer, entry["question"],
            domain, "general", n_rounds=n_rounds,
            device=device, protocol="full-cot",
        )
        predicted = extract_answer_letter(final)
        expected = entry["answer_letter"]
        pre_a = pre_collab["agent_a"]["answer"]
        if predicted == expected:
            base_correct += 1
        if pre_a != predicted:
            base_switches += 1
            if pre_a == expected and predicted != expected:
                base_c2w += 1
            elif pre_a != expected and predicted == expected:
                base_w2c += 1

    results["base_collab_acc"] = base_correct / len(questions)
    results["base_delta"] = results["base_collab_acc"] - results["solo_acc"]
    results["base_c2w"] = base_c2w
    results["base_w2c"] = base_w2c
    results["base_switches"] = base_switches
    logger.info("  [%s] +Base: %.1f%% (delta %+.1f%%, C2W=%d, W2C=%d)",
                label, results["base_collab_acc"] * 100, results["base_delta"] * 100,
                base_c2w, base_w2c)

    # Cross collab
    if cross_model is not None:
        cross_correct, cross_c2w, cross_w2c, cross_switches = 0, 0, 0, 0
        for entry in questions:
            final, chain, pre_collab = collab_reasoning_scoped(
                specialist, cross_model, tokenizer, entry["question"],
                domain, cross_domain, n_rounds=n_rounds,
                device=device, protocol="full-cot",
            )
            predicted = extract_answer_letter(final)
            expected = entry["answer_letter"]
            pre_a = pre_collab["agent_a"]["answer"]
            if predicted == expected:
                cross_correct += 1
            if pre_a != predicted:
                cross_switches += 1
                if pre_a == expected and predicted != expected:
                    cross_c2w += 1
                elif pre_a != expected and predicted == expected:
                    cross_w2c += 1

        results["cross_collab_acc"] = cross_correct / len(questions)
        results["cross_delta"] = results["cross_collab_acc"] - results["solo_acc"]
        results["cross_c2w"] = cross_c2w
        results["cross_w2c"] = cross_w2c
        results["cross_switches"] = cross_switches
        logger.info("  [%s] +%s: %.1f%% (delta %+.1f%%, C2W=%d, W2C=%d)",
                    label, cross_domain, results["cross_collab_acc"] * 100,
                    results["cross_delta"] * 100, cross_c2w, cross_w2c)

    results["n_questions"] = len(questions)
    return results


def main():
    parser = argparse.ArgumentParser(description="CoT-preserved specialist training")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter-dir", default="adapters")
    parser.add_argument("--cot-adapter-dir", default="cot_adapters")
    parser.add_argument("--results-dir", default="results/cot_specialist")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=3)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-cross", action="store_true")
    args = parser.parse_args()

    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    start = time.time()

    # Phase 1: Train CoT specialist
    if not args.skip_training:
        train_cot_specialist(
            DOMAIN, args.model_name, args.cot_adapter_dir,
            cache_dir=args.cache_dir, rank=args.rank,
        )

    # Phase 2: Load test data
    test_entries = load_mmlu_domain(DOMAIN, split="test", cache_dir=args.cache_dir)
    _, test_med = split_train_test(test_entries, test_size=args.n_questions, seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir
    )

    # Phase 3: Load shared models
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir,
    ).to(device)
    base_model.eval()

    cross_model = None
    if not args.skip_cross:
        cross_path = os.path.join(args.adapter_dir, f"adapter_{CROSS_DOMAIN}")
        if os.path.exists(cross_path):
            logger.info("Loading %s specialist...", CROSS_DOMAIN)
            cross_model = AutoModelForCausalLM.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16,
                trust_remote_code=True, cache_dir=args.cache_dir,
            ).to(device)
            cross_model = PeftModel.from_pretrained(cross_model, cross_path)
            cross_model.eval()

    # Phase 4: Evaluate BOTH specialists
    all_results = []

    # Original (no-thinking) specialist
    orig_path = os.path.join(args.adapter_dir, f"adapter_{DOMAIN}")
    if os.path.exists(orig_path):
        logger.info("=== Evaluating ORIGINAL specialist (no CoT) ===")
        orig_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, cache_dir=args.cache_dir,
        ).to(device)
        orig_model = PeftModel.from_pretrained(orig_model, orig_path)
        orig_model.eval()

        r = evaluate_specialist(
            "original_r16", orig_model, base_model, cross_model,
            tokenizer, test_med, DOMAIN, CROSS_DOMAIN,
            n_questions=args.n_questions, n_rounds=args.n_rounds, device=device,
        )
        r["type"] = "original"
        all_results.append(r)

        del orig_model
        gc.collect()
        torch.cuda.empty_cache()

    # CoT specialist
    cot_path = os.path.join(args.cot_adapter_dir, f"adapter_{DOMAIN}_cot_r{args.rank}")
    if os.path.exists(cot_path):
        logger.info("=== Evaluating COT specialist ===")
        cot_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, cache_dir=args.cache_dir,
        ).to(device)
        cot_model = PeftModel.from_pretrained(cot_model, cot_path)
        cot_model.eval()

        r = evaluate_specialist(
            f"cot_r{args.rank}", cot_model, base_model, cross_model,
            tokenizer, test_med, DOMAIN, CROSS_DOMAIN,
            n_questions=args.n_questions, n_rounds=args.n_rounds, device=device,
        )
        r["type"] = "cot"
        all_results.append(r)

        del cot_model
        gc.collect()
        torch.cuda.empty_cache()

    del base_model
    if cross_model is not None:
        del cross_model
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - start
    logger.info("Total time: %.1f minutes", elapsed / 60)

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    output = {"results": all_results, "config": {
        "domain": DOMAIN, "cross_domain": CROSS_DOMAIN,
        "rank": args.rank, "n_questions": args.n_questions,
        "n_rounds": args.n_rounds, "model_name": args.model_name,
    }}
    with open(os.path.join(args.results_dir, "cot_comparison.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"COT vs ORIGINAL: {DOMAIN} specialist (r={args.rank})")
    print("=" * 80)
    print(f"{'Condition':<20} {'Solo':>6} {'+ Base':>8} {'Delta':>7} {'C2W':>5} {'W2C':>5} "
          f"{'+ Cross':>8} {'Delta':>7} {'C2W':>5} {'W2C':>5}")
    print("-" * 80)
    for r in all_results:
        solo = f"{r['solo_acc']*100:.0f}%"
        ba = f"{r.get('base_collab_acc', 0)*100:.0f}%" if 'base_collab_acc' in r else "—"
        bd = f"{r.get('base_delta', 0)*100:+.0f}pp" if 'base_delta' in r else "—"
        bc = str(r.get('base_c2w', '—'))
        bw = str(r.get('base_w2c', '—'))
        ca = f"{r.get('cross_collab_acc', 0)*100:.0f}%" if 'cross_collab_acc' in r else "—"
        cd = f"{r.get('cross_delta', 0)*100:+.0f}pp" if 'cross_delta' in r else "—"
        cc = str(r.get('cross_c2w', '—'))
        cw = str(r.get('cross_w2c', '—'))
        print(f"{r['condition']:<20} {solo:>6} {ba:>8} {bd:>7} {bc:>5} {bw:>5} "
              f"{ca:>8} {cd:>7} {cc:>5} {cw:>5}")


if __name__ == "__main__":
    main()
