"""Alternating Chain-of-Thought collaboration between domain specialists.

Two LoRA-adapted models take turns reasoning about a question. On each step,
one specialist continues the chain of thought started by the other. We measure
whether this interleaved reasoning helps or hurts answer quality compared to
a single-specialist baseline.

Protocol:
  1. Agent A sees the question, produces a reasoning step.
  2. Agent B receives A's reasoning, adds its own step.
  3. Repeat for N rounds.
  4. The last agent to reason gives the final answer.

We test all ordered pairs (A, B) across domains and compare:
  - Solo baseline: A reasons alone for N rounds → answers
  - Collaboration: A and B alternate for N rounds → answers
"""

import json
import logging
import os
from itertools import product
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _encode_and_generate(
    model, tokenizer, messages, device, max_new_tokens=200, temperature=0.7,
):
    """Shared generate helper that handles BatchEncoding vs raw tensor."""
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            top_p=0.9,
            use_cache=True,
        )

    response = tokenizer.decode(
        outputs[0][input_ids.shape[1]:], skip_special_tokens=True,
    ).strip()
    return response


def solo_reasoning(
    model, tokenizer, question: str, domain: str, n_rounds: int = 3,
    device=None, temperature: float = 0.7,
) -> Tuple[str, List[str]]:
    """Single agent reasons for n_rounds, then gives a final answer.

    Returns (final_answer, chain) where chain is the list of reasoning steps.
    """
    if device is None:
        device = next(model.parameters()).device

    system_prompt = (
        f"You are a {domain} expert. Think step by step. "
        f"You will reason for {n_rounds} rounds, then give a final answer."
    )

    chain = []
    # Reasoning rounds
    for i in range(n_rounds):
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": question})
        # Add prior reasoning as assistant turns
        for j, step in enumerate(chain):
            messages.append({"role": "assistant", "content": step})
            if j < len(chain) - 1 or i < n_rounds - 1:
                messages.append({
                    "role": "user",
                    "content": f"Continue reasoning. Round {j + 2} of {n_rounds}.",
                })

        if chain:
            messages.append({
                "role": "user",
                "content": f"Continue reasoning. Round {i + 1} of {n_rounds}.",
            })

        response = _encode_and_generate(
            model, tokenizer, messages, device,
            max_new_tokens=200, temperature=temperature,
        )
        chain.append(response)

    # Final answer round
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": question})
    for j, step in enumerate(chain):
        messages.append({"role": "assistant", "content": step})
        messages.append({
            "role": "user",
            "content": "Continue reasoning." if j < len(chain) - 1 else
            "Now give your final answer. Reply with ONLY the letter (A, B, C, or D).",
        })

    final = _encode_and_generate(
        model, tokenizer, messages, device,
        max_new_tokens=50, temperature=0.3,
    )
    return final, chain


def collab_reasoning(
    model_a, model_b, tokenizer, question: str,
    domain_a: str, domain_b: str, n_rounds: int = 3,
    device=None, temperature: float = 0.7,
) -> Tuple[str, List[Dict]]:
    """Two agents alternate reasoning about a question.

    Round 1: model_a thinks.
    Round 2: model_b sees model_a's thought, continues.
    Round 3: model_a sees model_b's thought, continues.
    ...
    Last model gives the final answer.

    Returns (final_answer, chain) where chain is list of
    {"agent": domain, "thought": text}.
    """
    if device is None:
        device = next(model_a.parameters()).device

    models = [model_a, model_b]
    domains = [domain_a, domain_b]
    chain = []

    for i in range(n_rounds):
        agent_idx = i % 2
        model = models[agent_idx]
        domain = domains[agent_idx]

        system_prompt = (
            f"You are a {domain} expert collaborating with a {domains[1 - agent_idx]} expert. "
            f"Think step by step. Build on your collaborator's reasoning."
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": question})

        # Add prior chain as alternating assistant/user turns
        for j, step in enumerate(chain):
            if step["agent"] == domain:
                messages.append({"role": "assistant", "content": step["thought"]})
                if j < len(chain) - 1:
                    messages.append({
                        "role": "user",
                        "content": f"[{chain[j+1]['agent']} expert]: {chain[j+1]['thought']}",
                    })
            else:
                # Other agent's thought appears as user message
                if messages[-1]["role"] != "user":
                    messages.append({
                        "role": "user",
                        "content": f"[{step['agent']} expert]: {step['thought']}",
                    })

        # Prompt for this round's reasoning
        if chain and messages[-1]["role"] == "assistant":
            messages.append({
                "role": "user",
                "content": f"Continue the collaborative reasoning. Round {i + 1} of {n_rounds}.",
            })

        response = _encode_and_generate(
            model, tokenizer, messages, device,
            max_new_tokens=200, temperature=temperature,
        )
        chain.append({"agent": domain, "thought": response})

    # Final answer from the last agent
    last_idx = (n_rounds - 1) % 2
    model = models[last_idx]
    domain = domains[last_idx]

    system_prompt = (
        f"You are a {domain} expert. Based on the collaborative reasoning, "
        f"give the final answer."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": question})

    # Replay full chain
    for step in chain:
        if step["agent"] == domain:
            messages.append({"role": "assistant", "content": step["thought"]})
            messages.append({"role": "user", "content": "Continue."})
        else:
            messages.append({
                "role": "user",
                "content": f"[{step['agent']} expert]: {step['thought']}",
            })

    # Replace last user message with answer prompt
    messages[-1] = {
        "role": "user",
        "content": "Now give your final answer. Reply with ONLY the letter (A, B, C, or D).",
    }

    final = _encode_and_generate(
        model, tokenizer, messages, device,
        max_new_tokens=50, temperature=0.3,
    )
    return final, chain


def run_collab_experiment(
    models: Dict[str, object],
    tokenizer,
    test_sets: Dict[str, List[Dict]],
    domains: List[str],
    n_rounds: int = 3,
    n_questions: int = 20,
    device=None,
    temperature: float = 0.7,
    include_same_domain: bool = False,
) -> Dict:
    """Run the full collaboration experiment.

    Tests all domain pairs + solo baselines on questions from each domain.

    Args:
        models: {"physics": model, "law": model, ...} — loaded LoRA specialists.
        tokenizer: Shared tokenizer.
        test_sets: {"physics": [...], ...} — test questions per domain.
        domains: List of domain names.
        n_rounds: Number of reasoning rounds per question.
        n_questions: Number of questions per domain to test.
        device: Torch device.
        temperature: Sampling temperature for reasoning steps.

    Returns:
        Dict with solo_results, collab_results, and summary statistics.
    """
    from .cross_eval import extract_answer_letter

    solo_results = []
    collab_results = []

    # Solo baselines: each specialist on its own domain
    for domain in domains:
        model = models[domain]
        questions = test_sets[domain][:n_questions]
        logger.info("=== Solo baseline: %s (%d questions) ===", domain, len(questions))

        for entry in questions:
            final, chain = solo_reasoning(
                model, tokenizer, entry["question"], domain,
                n_rounds=n_rounds, device=device, temperature=temperature,
            )
            predicted = extract_answer_letter(final)
            solo_results.append({
                "domain": domain,
                "model": domain,
                "mode": "solo",
                "expected": entry["answer_letter"],
                "predicted": predicted,
                "correct": predicted == entry["answer_letter"],
                "final_response": final[:200],
                "chain_length": len(chain),
            })

        acc = sum(r["correct"] for r in solo_results if r["domain"] == domain) / max(len(questions), 1)
        logger.info("  Solo %s: %.1f%%", domain, acc * 100)

    # Collaborative pairs: all ordered pairs (A, B) on A's domain questions
    for domain_a, domain_b in product(domains, repeat=2):
        if domain_a == domain_b and not include_same_domain:
            continue  # Skip same-domain pairs unless explicitly requested

        model_a = models[domain_a]
        model_b = models[domain_b]

        # Test on domain_a's questions (home turf for A, away for B)
        questions = test_sets[domain_a][:n_questions]
        logger.info("=== Collab: %s + %s on %s questions (%d) ===",
                     domain_a, domain_b, domain_a, len(questions))

        for entry in questions:
            final, chain = collab_reasoning(
                model_a, model_b, tokenizer, entry["question"],
                domain_a, domain_b, n_rounds=n_rounds,
                device=device, temperature=temperature,
            )
            predicted = extract_answer_letter(final)
            collab_results.append({
                "question_domain": domain_a,
                "agent_a": domain_a,
                "agent_b": domain_b,
                "mode": "collab",
                "expected": entry["answer_letter"],
                "predicted": predicted,
                "correct": predicted == entry["answer_letter"],
                "final_response": final[:200],
                "chain": [{"agent": s["agent"], "thought": s["thought"][:100]} for s in chain],
            })

        pair_results = [r for r in collab_results
                        if r["agent_a"] == domain_a and r["agent_b"] == domain_b
                        and r["question_domain"] == domain_a]
        acc = sum(r["correct"] for r in pair_results) / max(len(pair_results), 1)
        logger.info("  Collab %s+%s on %s: %.1f%%", domain_a, domain_b, domain_a, acc * 100)

    # Build summary
    summary = build_collab_summary(solo_results, collab_results, domains)
    return {
        "solo_results": solo_results,
        "collab_results": collab_results,
        "summary": summary,
        "config": {
            "n_rounds": n_rounds,
            "n_questions": n_questions,
            "temperature": temperature,
            "domains": domains,
        },
    }


def build_collab_summary(
    solo_results: List[Dict],
    collab_results: List[Dict],
    domains: List[str],
) -> Dict:
    """Build summary comparing solo vs collaborative accuracy."""
    summary = {"solo": {}, "collab": {}, "collab_delta": {}}

    # Solo accuracy per domain
    for domain in domains:
        results = [r for r in solo_results if r["domain"] == domain]
        if results:
            summary["solo"][domain] = sum(r["correct"] for r in results) / len(results)

    # Collab accuracy per pair per question domain
    for domain_a in domains:
        for domain_b in domains:
            if domain_a == domain_b:
                continue
            pair_key = f"{domain_a}+{domain_b}"
            # Questions from domain_a answered by pair (A, B)
            results = [r for r in collab_results
                       if r["agent_a"] == domain_a and r["agent_b"] == domain_b
                       and r["question_domain"] == domain_a]
            if results:
                collab_acc = sum(r["correct"] for r in results) / len(results)
                solo_acc = summary["solo"].get(domain_a, 0)
                summary["collab"][pair_key] = {
                    "question_domain": domain_a,
                    "accuracy": collab_acc,
                    "solo_baseline": solo_acc,
                    "delta": collab_acc - solo_acc,
                }
                summary["collab_delta"][pair_key] = collab_acc - solo_acc

    return summary


def save_collab_results(results: Dict, output_dir: str) -> str:
    """Save collaboration experiment results."""
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "collab_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Also save a concise summary
    summary_path = os.path.join(output_dir, "collab_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results["summary"], f, indent=2)

    logger.info("Saved collaboration results to %s", output_dir)
    return path
