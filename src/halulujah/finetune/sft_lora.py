"""LoRA SFT fine-tuning for Qwen3-4B-Instruct on EGNIVIA data."""

import logging
import os
import sys

import datasets
import transformers
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from ..config import ExperimentConfig
from ..data.dataset_builder import build_sft_dataset
from ..data.loader import load_egnivia, split_by_year
from ..models.loader import load_model_and_tokenizer

logger = logging.getLogger(__name__)


def run_finetune(cfg: ExperimentConfig, held_out_years: list[int] | None = None) -> str:
    """Run SFT LoRA fine-tuning, returning the path to the saved adapter.

    Args:
        cfg: Full experiment configuration.
        held_out_years: Years to exclude from training. Overrides cfg.held_out_years if provided.

    Returns:
        Path to the saved model/adapter directory.
    """
    held_out_years = held_out_years or cfg.held_out_years
    years_tag = "_".join(str(y) for y in sorted(held_out_years))
    output_dir = os.path.join(
        cfg.paths.resolve("output_dir"),
        f"sft_lora_holdout_{years_tag}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # --- Logging ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # --- Model & tokenizer ---
    cache_dir = cfg.paths.cache_dir
    model, tokenizer = load_model_and_tokenizer(cfg.model, cache_dir=cache_dir)

    # --- Data ---
    data = load_egnivia(str(cfg.paths.resolve("data_json")))
    train_data, _ = split_by_year(data, held_out_years)

    # 90/10 split for train/eval
    split_idx = int(len(train_data) * cfg.sft.train_split_ratio)
    train_entries = train_data[:split_idx]
    eval_entries = train_data[split_idx:]

    logger.info("Train entries: %d, Eval entries: %d (held out years: %s)",
                len(train_entries), len(eval_entries), held_out_years)

    train_dataset = build_sft_dataset(train_entries, tokenizer)
    eval_dataset = build_sft_dataset(eval_entries, tokenizer)

    # --- LoRA config ---
    peft_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
        target_modules=cfg.lora.target_modules,
    )

    # --- Training config ---
    deepspeed_path = cfg.sft.deepspeed
    if deepspeed_path and not os.path.isabs(deepspeed_path):
        deepspeed_path = os.path.join(cfg.paths.project_root, deepspeed_path)

    train_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.sft.num_train_epochs,
        learning_rate=cfg.sft.learning_rate,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.sft.per_device_eval_batch_size,
        gradient_checkpointing=cfg.sft.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=cfg.sft.warmup_ratio,
        lr_scheduler_type=cfg.sft.lr_scheduler_type,
        bf16=cfg.sft.bf16,
        logging_steps=cfg.sft.logging_steps,
        save_total_limit=cfg.sft.save_total_limit,
        seed=cfg.sft.seed,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        overwrite_output_dir=True,
        remove_unused_columns=True,
        dataset_text_field="text",
        packing=cfg.sft.packing,
        max_length=cfg.sft.max_length,
        deepspeed=deepspeed_path if deepspeed_path and os.path.exists(deepspeed_path) else None,
        ddp_find_unused_parameters=False,
        do_eval=True,
        eval_strategy="epoch",
        logging_strategy="steps",
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    )

    # --- Train ---
    trainer = SFTTrainer(
        model=model,
        args=train_config,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # --- Eval ---
    tokenizer.padding_side = "left"
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # --- Save ---
    trainer.save_model(output_dir)
    logger.info("Model saved to %s", output_dir)
    return output_dir
