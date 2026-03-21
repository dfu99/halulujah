"""Persona Fingerprinting Experiment — Full Pipeline.

Phase 1: Fine-tune per-author LoRA adapters on Blog Authorship Corpus.
Phase 2: Measure fingerprint (KL divergence, embeddings, vocab).
Phase 3: Temperature erosion sweep.

Usage:
  # Phase 1: Fine-tune (run on A100)
  python -m src.scripts.run_persona_experiment --phase finetune \
      --data-csv /path/to/blogtext.csv --output-dir /path/to/output

  # Phase 2: Measure fingerprint (run on A100)
  python -m src.scripts.run_persona_experiment --phase measure \
      --output-dir /path/to/output

  # Phase 3: Erosion sweep (run on A100)
  python -m src.scripts.run_persona_experiment --phase erosion \
      --output-dir /path/to/output
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


def phase_finetune(args):
    """Phase 1: Fine-tune per-author LoRA adapters."""
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from halulujah.persona.data_prep import (
        load_blog_corpus,
        split_train_probe,
        format_blog_for_sft,
        save_author_manifest,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load corpus
    corpus = load_blog_corpus(
        args.data_csv,
        min_posts=args.min_posts,
        max_authors=args.num_authors,
    )

    # Save manifest
    save_author_manifest(corpus, args.output_dir)

    # Load base model and tokenizer
    logger.info("Loading base model: %s", args.model_name)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    if args.cache_dir:
        model_kwargs["cache_dir"] = args.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Fine-tune each author
    for author_id, posts in corpus.items():
        logger.info("=== Fine-tuning author %s (%d posts) ===", author_id, len(posts))

        train_posts, probe_posts = split_train_probe(posts, probe_ratio=0.2)
        logger.info("  Train: %d, Probe: %d", len(train_posts), len(probe_posts))

        # Save probe posts for later measurement
        probe_dir = os.path.join(args.output_dir, f"probe_{author_id}")
        os.makedirs(probe_dir, exist_ok=True)
        with open(os.path.join(probe_dir, "probe_posts.json"), "w") as f:
            json.dump(probe_posts, f)

        # Build SFT dataset
        train_dataset = format_blog_for_sft(train_posts, tokenizer, author_id)
        eval_dataset = format_blog_for_sft(probe_posts[:50], tokenizer, author_id)

        # Load fresh model for each author
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, **model_kwargs,
        )

        # LoRA config
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

        adapter_dir = os.path.join(args.output_dir, f"adapter_{author_id}")

        train_config = SFTConfig(
            output_dir=adapter_dir,
            num_train_epochs=args.epochs,
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=10,
            save_total_limit=1,
            seed=42,
            gradient_accumulation_steps=1,
            overwrite_output_dir=True,
            remove_unused_columns=True,
            dataset_text_field="text",
            packing=True,
            max_length=1024,
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
        logger.info("Saved adapter for author %s to %s", author_id, adapter_dir)

        # Free memory
        del model, trainer
        torch.cuda.empty_cache()

    logger.info("Phase 1 complete. Adapters saved to %s", args.output_dir)


def phase_measure(args):
    """Phase 2: Measure fingerprint across all persona models."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from halulujah.persona.data_prep import PROBE_QUESTIONS, format_probes_for_generation
    from halulujah.persona.fingerprint import (
        compute_logit_distributions,
        compute_pairwise_kl,
        compute_vocab_fingerprint,
        compute_embedding_distances,
        save_fingerprint_results,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format probes
    prompts = format_probes_for_generation(tokenizer)

    # Load manifest to find authors
    manifest_path = os.path.join(args.output_dir, "author_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    author_ids = list(manifest["authors"].keys())

    # Load base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=args.cache_dir,
    )

    # Collect distributions: base first
    logger.info("Computing base model distributions...")
    distributions = {}
    generated_texts = {}

    base_dists = compute_logit_distributions(
        base_model, tokenizer, prompts, max_new_tokens=50, device=device,
    )
    distributions["base"] = base_dists

    # Generate text from base
    base_texts = _generate_texts(base_model, tokenizer, prompts, device)
    generated_texts["base"] = base_texts

    del base_model
    torch.cuda.empty_cache()

    # Load each persona adapter and measure
    for author_id in author_ids:
        adapter_dir = os.path.join(args.output_dir, f"adapter_{author_id}")
        if not os.path.exists(adapter_dir):
            logger.warning("Adapter not found for %s, skipping", author_id)
            continue

        logger.info("Loading adapter for author %s...", author_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=args.cache_dir,
        )
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()

        name = f"author_{author_id}"
        logger.info("Computing distributions for %s...", name)
        distributions[name] = compute_logit_distributions(
            model, tokenizer, prompts, max_new_tokens=50, device=device,
        )
        generated_texts[name] = _generate_texts(model, tokenizer, prompts, device)

        del model
        torch.cuda.empty_cache()

    # Compute metrics
    logger.info("Computing KL divergence...")
    kl_results = compute_pairwise_kl(distributions, base_key="base")

    logger.info("Computing vocabulary fingerprints...")
    vocab_fp = compute_vocab_fingerprint(distributions, tokenizer, base_key="base")

    logger.info("Computing embedding distances...")
    emb_results = compute_embedding_distances(generated_texts)

    # Save
    results_dir = os.path.join(args.output_dir, "fingerprint")
    save_fingerprint_results(kl_results, vocab_fp, emb_results, results_dir)

    # Print summary
    print("\n=== FINGERPRINT SUMMARY ===")
    print(f"Embedding separation ratio: {emb_results['separation_ratio']:.3f}")
    for name, vals in kl_results["vs_base"].items():
        print(f"  KL({name} || base) = {vals['mean_kl']:.4f} ± {vals['std_kl']:.4f}")

    logger.info("Phase 2 complete. Results in %s", results_dir)


def phase_erosion(args):
    """Phase 3: Temperature erosion sweep."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from halulujah.persona.data_prep import format_probes_for_generation
    from halulujah.persona.erosion import (
        run_erosion_sweep,
        summarize_erosion,
        save_erosion_results,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = format_probes_for_generation(tokenizer)

    manifest_path = os.path.join(args.output_dir, "author_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    author_ids = list(manifest["authors"].keys())

    # Load all models into memory
    logger.info("Loading base model...")
    models = {}
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=args.cache_dir,
    )
    models["base"] = base_model

    for author_id in author_ids:
        adapter_dir = os.path.join(args.output_dir, f"adapter_{author_id}")
        if not os.path.exists(adapter_dir):
            continue
        logger.info("Loading adapter for author %s...", author_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=args.cache_dir,
        )
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
        models[f"author_{author_id}"] = model

    # Run sweep
    results = run_erosion_sweep(
        models, tokenizer, prompts,
        max_new_tokens=50, device=device,
    )

    summary = summarize_erosion(results)
    results["summary"] = summary

    results_dir = os.path.join(args.output_dir, "erosion")
    save_erosion_results(results, results_dir)

    # Print summary
    print("\n=== EROSION SUMMARY ===")
    for t, kl_base, kl_pw, sep in zip(
        summary["temperatures"],
        summary["mean_kl_vs_base"],
        summary["mean_kl_pairwise"],
        summary["separation_ratio"],
    ):
        print(f"  T={t:.1f}: KL_base={kl_base:.4f}, KL_pair={kl_pw:.4f}, sep={sep:.3f}")

    logger.info("Phase 3 complete. Results in %s", results_dir)


def _generate_texts(model, tokenizer, prompts, device, max_new_tokens=100):
    """Generate text responses for embedding measurement."""
    texts = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                use_cache=True,
            )
        response = tokenizer.decode(
            outputs[0][input_ids.shape[1]:], skip_special_tokens=True,
        )
        texts.append(response)
    return texts


def main():
    parser = argparse.ArgumentParser(description="Persona Fingerprinting Experiment")
    parser.add_argument("--phase", required=True, choices=["finetune", "measure", "erosion"])
    parser.add_argument("--data-csv", default=None, help="Path to blogtext.csv (finetune phase)")
    parser.add_argument("--output-dir", required=True, help="Output directory for adapters and results")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B", help="Base model")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    parser.add_argument("--num-authors", type=int, default=5, help="Number of authors")
    parser.add_argument("--min-posts", type=int, default=200, help="Min posts per author")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per author")
    args = parser.parse_args()

    if args.phase == "finetune":
        if not args.data_csv:
            parser.error("--data-csv required for finetune phase")
        phase_finetune(args)
    elif args.phase == "measure":
        phase_measure(args)
    elif args.phase == "erosion":
        phase_erosion(args)


if __name__ == "__main__":
    main()
