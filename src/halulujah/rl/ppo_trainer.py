"""PPO training loop with Oracle reward."""

import logging
import os
import random

import torch
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model

from ..config import ExperimentConfig
from ..data.chat_formatter import format_cot_system_prompt, format_cot_user_prompt
from ..data.dataset_builder import build_rl_dataset
from ..oracle.retriever import OracleRetriever
from ..oracle.reward import OracleRewardFunction
from .verifier import extract_cot_answer

logger = logging.getLogger(__name__)


def run_rl_training(
    cfg: ExperimentConfig,
    sft_adapter_path: str,
    held_out_years: list[int],
    oracle_reward_fn: OracleRewardFunction,
    retriever: OracleRetriever = None,
) -> str:
    """Run PPO training using Oracle reward on held-out year questions.

    Args:
        cfg: Full experiment configuration.
        sft_adapter_path: Path to the SFT LoRA adapter to load as base.
        held_out_years: Years used for RL training (the held-out set).
        oracle_reward_fn: The Oracle reward function instance.
        retriever: Optional OracleRetriever for context-augmented CoT prompts.

    Returns:
        Path to the saved RL model directory.
    """
    from ..data.loader import load_egnivia, split_by_year
    from ..models.loader import load_model_and_tokenizer

    years_tag = "_".join(str(y) for y in sorted(held_out_years))
    output_dir = os.path.join(
        cfg.paths.resolve("output_dir"),
        f"rl_holdout_{years_tag}",
    )
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Load model with value head from SFT adapter
    logger.info("Loading SFT adapter from %s with value head", sft_adapter_path)
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_adapter_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )

    # Load tokenizer from SFT adapter path
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # left-pad for generation

    # Apply smaller LoRA for RL (for stability)
    peft_config = LoraConfig(
        r=cfg.ppo.lora_r,
        lora_alpha=cfg.ppo.lora_alpha,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=cfg.ppo.lora_target_modules,
    )
    ppo_model = get_peft_model(ppo_model, peft_config)
    ppo_model.print_trainable_parameters()

    # Frozen reference model
    ref_model = create_reference_model(ppo_model)

    # PPO config
    ppo_config = PPOConfig(
        learning_rate=cfg.ppo.learning_rate,
        batch_size=cfg.ppo.batch_size,
        mini_batch_size=cfg.ppo.mini_batch_size,
        num_ppo_epochs=cfg.ppo.num_ppo_epochs,
        kl_coef=cfg.ppo.kl_coef,
        gamma=cfg.ppo.gamma,
        gradient_accumulation_steps=cfg.ppo.gradient_accumulation_steps,
        output_dir=output_dir,
    )

    # Build RL dataset from held-out year questions
    data = load_egnivia(str(cfg.paths.resolve("data_json")))
    _, held_out_data = split_by_year(data, held_out_years)
    rl_dataset = build_rl_dataset(held_out_data)

    logger.info("RL dataset: %d questions from held-out years %s", len(rl_dataset), held_out_years)

    # PPO trainer
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=ppo_model,
        ref_model=ref_model,
        processing_class=tokenizer,
        train_dataset=rl_dataset,
    )

    # Training loop
    use_cot = cfg.ppo.use_cot
    use_context = cfg.ppo.use_context and retriever is not None

    if use_cot:
        system_prompt = format_cot_system_prompt()
    else:
        system_prompt = (
            "You are a concise financial QA assistant. "
            "Answer in one sentence. If unsure, say 'Not enough information.'"
        )
    max_steps = cfg.ppo.max_steps

    step = 0
    for batch_idx, batch in enumerate(ppo_trainer.dataloader):
        if max_steps is not None and step >= max_steps:
            break

        questions = batch["question"]
        query_tensors = []
        for q in questions:
            if use_cot and use_context:
                passages = retriever.get_context_passages(q)
                context = "\n\n".join(passages)
                user_content = format_cot_user_prompt(q, context)
            elif use_context:
                passages = retriever.get_context_passages(q)
                context = "\n\n".join(passages)
                user_content = f"{context}\n\nQuestion: {q}"
            elif use_cot:
                user_content = (
                    f"Question: {q}\n\n"
                    "Think step by step, then provide your final answer "
                    "on a line starting with 'Answer:'."
                )
            else:
                user_content = q

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, return_tensors="pt", add_generation_prompt=True,
            ).to(device)
            query_tensors.append(input_ids.squeeze(0))

        # Generate responses
        response_tensors = []
        for qt in query_tensors:
            gen = ppo_trainer.generate(
                qt.unsqueeze(0),
                max_new_tokens=cfg.ppo.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            response_tensors.append(gen.squeeze(0))

        # Decode responses and compute rewards
        rewards = []
        for i, (qt, rt) in enumerate(zip(query_tensors, response_tensors)):
            response_text = tokenizer.decode(rt[len(qt):], skip_special_tokens=True)
            reward = oracle_reward_fn.compute_reward(questions[i], response_text)
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))

        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        mean_reward = sum(r.item() for r in rewards) / len(rewards)
        logger.info("Batch %d — mean reward: %.4f", batch_idx, mean_reward)
        step += 1

    # Save
    ppo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("RL model saved to %s", output_dir)
    return output_dir
