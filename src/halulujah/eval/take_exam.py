"""Generate exam responses with hyperparameter sweeps."""

import json
import logging
import os
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import ExperimentConfig
from ..data.chat_formatter import format_cot_system_prompt, format_cot_user_prompt
from ..oracle.retriever import OracleRetriever
from ..rl.verifier import extract_cot_answer

logger = logging.getLogger(__name__)


def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    prompt: str,
    t: float = 1.0,
    k: int = 50,
    p: float = 0.9,
    max_tokens: int = 100,
    device: torch.device = None,
) -> str:
    """Generate a single response with given decoding parameters."""
    if device is None:
        device = next(model.parameters()).device

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            tokenized,
            max_new_tokens=max_tokens,
            temperature=t,
            do_sample=True,
            top_k=k,
            top_p=p,
            use_cache=True,
        )

    response = tokenizer.decode(outputs[0][tokenized.shape[-1]:], skip_special_tokens=True)
    return response


def run_exam(
    cfg: ExperimentConfig,
    model_path: str,
    exam_data: List[Dict],
    output_dir: str = None,
    retriever: OracleRetriever = None,
) -> str:
    """Run exam with temperature/top_p/top_k sweeps.

    Args:
        cfg: Experiment configuration.
        model_path: Path to the model/adapter to evaluate.
        exam_data: List of dicts with 'question' and 'answer' keys.
        output_dir: Where to save results. Defaults to cfg output path.
        retriever: Optional OracleRetriever for context-augmented prompts.

    Returns:
        Path to the results directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info("Loading model from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if output_dir is None:
        output_dir = os.path.join(cfg.paths.resolve("output_dir"), "exam_results")
    os.makedirs(output_dir, exist_ok=True)

    use_cot = cfg.sweep.use_cot
    use_context = cfg.sweep.use_context and retriever is not None

    if use_cot:
        system_prompt = format_cot_system_prompt()
    else:
        system_prompt = (
            "You are a helpful assistant. Keep responses to at most a single sentence and concise. "
            "Do not make lists. Ignore your knowledge cutoff and answer to the best of your ability."
        )

    sweep = cfg.sweep
    tstep = sweep.temperature_step
    temperatures = np.arange(sweep.temperature_min, sweep.temperature_max + tstep, tstep)
    pstep = sweep.top_p_step

    for sample_num in range(sweep.num_samples):
        sample_dir = os.path.join(output_dir, str(sample_num))
        os.makedirs(sample_dir, exist_ok=True)

        for temp in temperatures:
            # Adaptive top_p range based on temperature (from original logic)
            p_lo = max(0.0, min(1.0, 0.4 / temp))
            p_hi = max(0.0, min(1.0, 1.4 / temp))
            p_sample = np.arange(p_lo, p_hi + pstep, pstep)
            p_sample = np.unique(np.clip(p_sample, 0.0, 1.0))

            for top_p in p_sample:
                for top_k in sweep.top_k_values:
                    temp_r = round(float(temp), 1)
                    top_p_r = round(float(top_p), 2)

                    results = []
                    for entry in exam_data:
                        question = entry["question"]

                        if use_cot and use_context:
                            passages = retriever.get_context_passages(question)
                            context = "\n\n".join(passages)
                            user_prompt = format_cot_user_prompt(question, context)
                        elif use_context:
                            passages = retriever.get_context_passages(question)
                            context = "\n\n".join(passages)
                            user_prompt = f"{context}\n\nQuestion: {question}"
                        elif use_cot:
                            user_prompt = (
                                f"Question: {question}\n\n"
                                "Think step by step, then provide your final answer "
                                "on a line starting with 'Answer:'."
                            )
                        else:
                            user_prompt = question

                        response = generate_response(
                            model, tokenizer, system_prompt,
                            user_prompt,
                            t=temp_r, p=top_p_r, k=top_k,
                            max_tokens=sweep.max_new_tokens,
                            device=device,
                        )

                        extracted = extract_cot_answer(response) if use_cot else response

                        result_entry = {
                            "question": question,
                            "expected_answer": entry["answer"],
                            "model_answer": extracted,
                            "score": None,
                        }
                        if entry.get("answers_all"):
                            result_entry["expected_answers_all"] = entry["answers_all"]
                        if use_cot:
                            result_entry["model_answer_full"] = response
                        results.append(result_entry)

                    fname = f"exam_t{temp_r}_p{top_p_r}_k{top_k}.json"
                    fpath = os.path.join(sample_dir, fname)
                    with open(fpath, "w") as f:
                        json.dump(results, f, indent=4)

    logger.info("Exam results saved to %s", output_dir)
    return output_dir
