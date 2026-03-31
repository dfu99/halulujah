"""Domain Cross-Hallucination Experiment — Full Pipeline.

Phase 1: Fine-tune per-domain LoRA adapters on MMLU subsets.
Phase 2: Cross-domain evaluation — measure hallucination rates.
Phase 3: Domain distance via KL divergence + correlation with hallucination.

Usage:
  python -m src.scripts.run_domain_experiment --phase finetune --output-dir /path/to/output
  python -m src.scripts.run_domain_experiment --phase evaluate --output-dir /path/to/output
  python -m src.scripts.run_domain_experiment --phase distance --output-dir /path/to/output
"""

import argparse
import json
import logging
import os
import sys

import torch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DOMAINS_CORE = ["physics", "law", "biology"]
DOMAINS_EXTENDED = DOMAINS_CORE + [
    "computer_science", "history", "math", "chemistry",
    "economics", "philosophy", "medicine",
]


def get_domains(args):
    """Return domain list based on --extended flag."""
    if getattr(args, "extended", False):
        from halulujah.domain.data_prep import DOMAIN_SUBJECTS_EXTENDED
        # Patch the module-level DOMAIN_SUBJECTS
        import halulujah.domain.data_prep as dp
        dp.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED
        return DOMAINS_EXTENDED
    return DOMAINS_CORE


def phase_finetune(args):
    """Phase 1: Fine-tune per-domain LoRA adapters on MMLU subsets."""
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from halulujah.domain.data_prep import (
        load_mmlu_domain,
        split_train_test,
        format_domain_for_sft,
        save_domain_manifest,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all domain data
    domain_train = {}
    domain_test = {}

    for domain in get_domains(args):
        logger.info("=== Loading domain: %s ===", domain)
        entries = load_mmlu_domain(domain, split="test", cache_dir=args.cache_dir)

        # Also load auxiliary split if available
        try:
            aux = load_mmlu_domain(domain, split="validation", cache_dir=args.cache_dir)
            entries.extend(aux)
        except Exception:
            pass

        train, test = split_train_test(entries, test_size=50)
        domain_train[domain] = train
        domain_test[domain] = test
        logger.info("  %s: %d train, %d test", domain, len(train), len(test))

    # Save manifest and test sets
    save_domain_manifest(domain_train, domain_test, args.output_dir)

    # Save test sets for later evaluation
    test_dir = os.path.join(args.output_dir, "test_sets")
    os.makedirs(test_dir, exist_ok=True)
    for domain, entries in domain_test.items():
        with open(os.path.join(test_dir, f"{domain}.json"), "w") as f:
            json.dump(entries, f, indent=2)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        use_cache=False,
    )
    if args.cache_dir:
        model_kwargs["cache_dir"] = args.cache_dir

    # Fine-tune each domain
    for domain in get_domains(args):
        logger.info("=== Fine-tuning domain: %s (%d examples) ===", domain, len(domain_train[domain]))

        train_dataset = format_domain_for_sft(domain_train[domain], tokenizer, domain)

        # Use a small eval set from training data
        eval_entries = domain_train[domain][:50]
        eval_dataset = format_domain_for_sft(eval_entries, tokenizer, domain)

        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
        )

        adapter_dir = os.path.join(args.output_dir, f"adapter_{domain}")

        train_config = SFTConfig(
            output_dir=adapter_dir,
            num_train_epochs=args.epochs,
            learning_rate=5e-5,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=10,
            save_total_limit=1,
            seed=42,
            remove_unused_columns=True,
            dataset_text_field="text",
            packing=False,  # No packing — MMLU examples are short
            max_length=512,
            do_eval=True,
            eval_strategy="epoch",
        )

        trainer = SFTTrainer(
            model=model,
            args=train_config,
            peft_config=peft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        trainer.train()
        trainer.save_model(adapter_dir)
        logger.info("Saved adapter for domain %s to %s", domain, adapter_dir)

        del model, trainer
        torch.cuda.empty_cache()

    logger.info("Phase 1 complete. Adapters saved to %s", args.output_dir)


def phase_evaluate(args):
    """Phase 2: Cross-domain evaluation — ask each model questions from all domains."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from halulujah.domain.cross_eval import (
        generate_and_grade,
        build_confusion_matrix,
        save_cross_eval_results,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test sets
    test_sets = {}
    test_dir = os.path.join(args.output_dir, "test_sets")
    for domain in get_domains(args):
        with open(os.path.join(test_dir, f"{domain}.json")) as f:
            test_sets[domain] = json.load(f)

    all_results = []
    domains = get_domains(args)
    model_names = ["base"] + [f"specialist_{d}" for d in domains]

    # Evaluate base model
    logger.info("=== Evaluating base model ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto", cache_dir=args.cache_dir,
    )
    for q_domain in domains:
        results = generate_and_grade(
            base_model, tokenizer, test_sets[q_domain],
            q_domain, "base", device=device,
        )
        all_results.extend(results)
    del base_model
    torch.cuda.empty_cache()

    # Evaluate each specialist
    for spec_domain in domains:
        adapter_dir = os.path.join(args.output_dir, f"adapter_{spec_domain}")
        if not os.path.exists(adapter_dir):
            logger.warning("Adapter not found for %s", spec_domain)
            continue

        logger.info("=== Evaluating specialist_%s ===", spec_domain)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="auto", cache_dir=args.cache_dir,
        )
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()

        for q_domain in domains:
            results = generate_and_grade(
                model, tokenizer, test_sets[q_domain],
                q_domain, f"specialist_{spec_domain}", device=device,
            )
            all_results.extend(results)

        del model
        torch.cuda.empty_cache()

    # Build confusion matrix
    confusion = build_confusion_matrix(all_results, domains, model_names)

    # Save
    results_dir = os.path.join(args.output_dir, "cross_eval")
    save_cross_eval_results(confusion, all_results, results_dir)

    # Print summary
    print("\n=== CROSS-DOMAIN ACCURACY MATRIX ===")
    header = f"{'Model':<25}" + "".join(f"{d:>12}" for d in domains)
    print(header)
    print("-" * len(header))
    for model_name in model_names:
        row = f"{model_name:<25}"
        for q_domain in domains:
            acc = confusion["matrix"].get(model_name, {}).get(q_domain, 0)
            row += f"{acc:>11.1%} "
        print(row)

    print("\n=== SUMMARY ===")
    for model_name, stats in confusion["summary"].items():
        if "accuracy_drop" in stats:
            print(f"  {model_name}: in={stats['in_domain_accuracy']:.1%}, "
                  f"cross={stats['cross_domain_accuracy']:.1%}, "
                  f"drop={stats['accuracy_drop']:.1%}")


def phase_distance(args):
    """Phase 3: Compute KL divergence between domain specialists."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from halulujah.persona.fingerprint import (
        compute_logit_distributions,
        compute_pairwise_kl,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use a mix of test questions from all domains as shared probes
    test_dir = os.path.join(args.output_dir, "test_sets")
    probe_questions = []
    for domain in get_domains(args):
        with open(os.path.join(test_dir, f"{domain}.json")) as f:
            entries = json.load(f)
        # Take first 10 from each domain = 30 shared probes
        probe_questions.extend(entries[:10])

    # Format as prompts
    system_prompt = "Answer the multiple-choice question. Give only the letter of the correct answer."
    prompts = []
    for entry in probe_questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": entry["question"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(text)

    # Load base model
    logger.info("Loading base model...")
    distributions = {}

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto", cache_dir=args.cache_dir,
    )
    distributions["base"] = compute_logit_distributions(
        base_model, tokenizer, prompts, max_new_tokens=30, device=device,
    )
    del base_model
    torch.cuda.empty_cache()

    # Load each specialist
    for domain in get_domains(args):
        adapter_dir = os.path.join(args.output_dir, f"adapter_{domain}")
        if not os.path.exists(adapter_dir):
            continue

        logger.info("Loading specialist_%s...", domain)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="auto", cache_dir=args.cache_dir,
        )
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()

        distributions[f"specialist_{domain}"] = compute_logit_distributions(
            model, tokenizer, prompts, max_new_tokens=30, device=device,
        )
        del model
        torch.cuda.empty_cache()

    # Compute KL
    kl_results = compute_pairwise_kl(distributions, base_key="base")

    # Save
    results_dir = os.path.join(args.output_dir, "domain_distance")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "kl_results.json"), "w") as f:
        json.dump(kl_results, f, indent=2)

    # Correlate with cross-eval if available
    cross_eval_path = os.path.join(args.output_dir, "cross_eval", "cross_eval_results.json")
    if os.path.exists(cross_eval_path):
        with open(cross_eval_path) as f:
            cross_eval = json.load(f)

        # Build correlation: KL distance vs accuracy drop
        correlations = []
        for pair_key, kl_vals in kl_results["pairwise"].items():
            # Parse pair names
            parts = pair_key.split("_vs_")
            if len(parts) != 2:
                continue
            name_a, name_b = parts
            domain_a = name_a.replace("specialist_", "")
            domain_b = name_b.replace("specialist_", "")

            # Get cross-domain accuracy for A answering B's questions
            matrix = cross_eval["confusion_matrix"]["matrix"]
            if name_a in matrix and domain_b in matrix[name_a]:
                cross_acc = matrix[name_a][domain_b]
                in_acc = matrix[name_a].get(domain_a, 0)
                correlations.append({
                    "model": name_a,
                    "question_domain": domain_b,
                    "kl_distance": kl_vals["mean_kl"],
                    "cross_domain_accuracy": cross_acc,
                    "in_domain_accuracy": in_acc,
                    "accuracy_drop": in_acc - cross_acc,
                })

        with open(os.path.join(results_dir, "kl_vs_hallucination.json"), "w") as f:
            json.dump(correlations, f, indent=2)

        print("\n=== KL DISTANCE vs ACCURACY DROP ===")
        for c in correlations:
            print(f"  {c['model']} on {c['question_domain']}: "
                  f"KL={c['kl_distance']:.2f}, drop={c['accuracy_drop']:.1%}")

    logger.info("Phase 3 complete. Results in %s", results_dir)


def main():
    parser = argparse.ArgumentParser(description="Domain Cross-Hallucination Experiment")
    parser.add_argument("--phase", required=True, choices=["finetune", "evaluate", "distance"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--extended", action="store_true", help="Use 10 domains instead of 3")
    args = parser.parse_args()

    if args.phase == "finetune":
        phase_finetune(args)
    elif args.phase == "evaluate":
        phase_evaluate(args)
    elif args.phase == "distance":
        phase_distance(args)


if __name__ == "__main__":
    main()
