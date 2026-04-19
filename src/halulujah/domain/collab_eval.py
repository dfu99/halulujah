"""Alternating collaboration between domain specialists with configurable protocols.

Two LoRA-adapted models take turns reasoning about a question. The *protocol*
controls how much of each agent's internal reasoning is shared with the other.

Protocols:
  full-cot:    Share entire chain of thought (maximally coupled).
  answer-only: Share only the final answer + 1-sentence rationale per round.
  structured:  Share answer + confidence level + key reasoning summary.

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


def _quick_answer(
    model, tokenizer, question: str, domain: str, device,
) -> str:
    """Get a fast independent answer from one agent — no multi-round reasoning.

    Used to snapshot each agent's pre-collaboration belief so we can measure
    whether collaboration changed their answer (and whether the change helped).
    """
    messages = [
        {"role": "system", "content": f"You are a {domain} expert."},
        {"role": "user", "content": (
            f"{question}\n\n"
            "Think briefly, then reply with ONLY the letter (A, B, C, or D)."
        )},
    ]
    return _encode_and_generate(
        model, tokenizer, messages, device, max_new_tokens=80, temperature=0.3,
    )


def _summarize_for_protocol(
    model, tokenizer, thought: str, protocol: str, domain: str, device,
) -> str:
    """Distill a full reasoning step into a protocol-appropriate message."""
    if protocol == "answer-only":
        prompt = (
            "You just reasoned about a question. Summarize your conclusion in "
            "exactly one sentence, stating your current best answer letter and why. "
            "Do NOT repeat the full reasoning."
        )
    elif protocol == "structured":
        prompt = (
            "You just reasoned about a question. Provide a structured summary:\n"
            "ANSWER: [letter]\n"
            "CONFIDENCE: [high/medium/low]\n"
            "KEY REASONING: [1-2 sentences max]\n"
            "Do NOT repeat the full reasoning."
        )
    else:
        return thought  # full-cot: pass through unchanged

    messages = [
        {"role": "system", "content": f"You are a {domain} expert."},
        {"role": "assistant", "content": thought},
        {"role": "user", "content": prompt},
    ]
    return _encode_and_generate(
        model, tokenizer, messages, device, max_new_tokens=100, temperature=0.3,
    )


def collab_reasoning_scoped(
    model_a, model_b, tokenizer, question: str,
    domain_a: str, domain_b: str, n_rounds: int = 3,
    device=None, temperature: float = 0.7,
    protocol: str = "full-cot",
) -> Tuple[str, List[Dict], Dict]:
    """Two agents alternate reasoning with protocol-controlled information sharing.

    Protocols:
      full-cot:    Each agent sees the other's complete reasoning (default, same
                   as collab_reasoning).
      answer-only: Each agent sees only a 1-sentence summary of the other's
                   conclusion — internal reasoning is private.
      structured:  Each agent sees answer + confidence + key reasoning summary.

    Returns (final_answer, chain, pre_collab) where:
      - chain is list of {"agent": domain, "thought": text, "shared": text}
      - pre_collab is {"agent_a": {"answer": str, "raw": str},
                       "agent_b": {"answer": str, "raw": str}}
        capturing each agent's independent answer BEFORE collaboration.
    """
    if device is None:
        device = next(model_a.parameters()).device

    from .cross_eval import extract_answer_letter

    # Snapshot: each agent answers independently before any collaboration
    pre_a_raw = _quick_answer(model_a, tokenizer, question, domain_a, device)
    pre_b_raw = _quick_answer(model_b, tokenizer, question, domain_b, device)
    pre_collab = {
        "agent_a": {"answer": extract_answer_letter(pre_a_raw), "raw": pre_a_raw[:200]},
        "agent_b": {"answer": extract_answer_letter(pre_b_raw), "raw": pre_b_raw[:200]},
    }

    if protocol == "full-cot":
        final, raw_chain = collab_reasoning(
            model_a, model_b, tokenizer, question,
            domain_a, domain_b, n_rounds=n_rounds,
            device=device, temperature=temperature,
        )
        # Add "shared" field matching "thought" for consistency
        chain = [
            {**step, "shared": step["thought"]} for step in raw_chain
        ]
        return final, chain, pre_collab

    models = [model_a, model_b]
    domains = [domain_a, domain_b]
    chain = []

    for i in range(n_rounds):
        agent_idx = i % 2
        model = models[agent_idx]
        domain = domains[agent_idx]
        other_domain = domains[1 - agent_idx]

        system_prompt = (
            f"You are a {domain} expert collaborating with a {other_domain} expert. "
            f"Think step by step. Build on your collaborator's input."
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": question})

        # Add prior chain — but only the *shared* summaries from the other agent,
        # and full thoughts from self
        for step in chain:
            if step["agent"] == domain:
                messages.append({"role": "assistant", "content": step["thought"]})
                messages.append({
                    "role": "user", "content": "Continue reasoning.",
                })
            else:
                # Other agent: show only the scoped summary
                messages.append({
                    "role": "user",
                    "content": f"[{step['agent']} expert]: {step['shared']}",
                })

        if chain and messages[-1]["role"] == "assistant":
            messages.append({
                "role": "user",
                "content": f"Continue the collaborative reasoning. Round {i + 1} of {n_rounds}.",
            })

        # Generate full internal reasoning
        thought = _encode_and_generate(
            model, tokenizer, messages, device,
            max_new_tokens=200, temperature=temperature,
        )

        # Distill into protocol-appropriate summary for the other agent
        shared = _summarize_for_protocol(
            model, tokenizer, thought, protocol, domain, device,
        )

        chain.append({"agent": domain, "thought": thought, "shared": shared})

    # Final answer from the last agent
    last_idx = (n_rounds - 1) % 2
    model = models[last_idx]
    domain = domains[last_idx]

    system_prompt = (
        f"You are a {domain} expert. Based on the collaborative discussion, "
        f"give the final answer."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": question})

    for step in chain:
        if step["agent"] == domain:
            messages.append({"role": "assistant", "content": step["thought"]})
            messages.append({"role": "user", "content": "Continue."})
        else:
            messages.append({
                "role": "user",
                "content": f"[{step['agent']} expert]: {step['shared']}",
            })

    messages[-1] = {
        "role": "user",
        "content": "Now give your final answer. Reply with ONLY the letter (A, B, C, or D).",
    }

    final = _encode_and_generate(
        model, tokenizer, messages, device,
        max_new_tokens=50, temperature=0.3,
    )
    return final, chain, pre_collab


def bridged_reasoning(
    specialist_a, mediator, specialist_b, tokenizer, question: str,
    domain_a: str, domain_b: str, n_cycles: int = 2,
    device=None, temperature: float = 0.7,
) -> Tuple[str, List[Dict], Dict]:
    """Three-agent bridged collaboration: Specialist A ↔ Mediator ↔ Specialist B.

    The mediator acts as interpreter between two domain specialists.
    Specialists never see each other's raw reasoning — only the mediator's
    translation. Each cycle:
      1. Specialist A reasons from their domain perspective
      2. Mediator reads A's reasoning, synthesizes/translates for B
      3. Specialist B reads mediator's bridge, reasons from their perspective
      4. Mediator reads B's reasoning, synthesizes/translates for A
    After n_cycles, the mediator gives the final answer.

    Returns (final_answer, chain, pre_collab) where:
      chain: list of {"agent": label, "thought": text, "role": str}
      pre_collab: independent answers from A and B before collaboration.
    """
    if device is None:
        device = next(specialist_a.parameters()).device

    from .cross_eval import extract_answer_letter

    # Pre-collaboration snapshots
    pre_a_raw = _quick_answer(specialist_a, tokenizer, question, domain_a, device)
    pre_b_raw = _quick_answer(specialist_b, tokenizer, question, domain_b, device)
    pre_collab = {
        "agent_a": {"answer": extract_answer_letter(pre_a_raw), "raw": pre_a_raw[:200]},
        "agent_b": {"answer": extract_answer_letter(pre_b_raw), "raw": pre_b_raw[:200]},
    }

    mediator_label = f"mediator_{domain_a}_{domain_b}"
    chain = []

    for cycle in range(n_cycles):
        # Step 1: Specialist A reasons
        sys_a = (
            f"You are a {domain_a} expert. A mediator is helping you collaborate "
            f"with a {domain_b} expert. Think step by step from your domain's perspective."
        )
        msgs_a = [{"role": "system", "content": sys_a},
                   {"role": "user", "content": question}]
        # Add prior mediator messages addressed to A
        for step in chain:
            if step["agent"] == domain_a:
                msgs_a.append({"role": "assistant", "content": step["thought"]})
                msgs_a.append({"role": "user", "content": "Continue reasoning."})
            elif step["role"] == "mediator_to_a":
                msgs_a.append({
                    "role": "user",
                    "content": f"[Mediator]: {step['thought']}",
                })

        if chain and msgs_a[-1]["role"] == "assistant":
            msgs_a.append({"role": "user",
                           "content": f"Continue. Cycle {cycle + 1} of {n_cycles}."})

        thought_a = _encode_and_generate(
            specialist_a, tokenizer, msgs_a, device,
            max_new_tokens=200, temperature=temperature,
        )
        chain.append({"agent": domain_a, "thought": thought_a, "role": "specialist_a"})

        # Step 2: Mediator reads A, bridges for B
        sys_m = (
            f"You are an expert in both {domain_a} and {domain_b}. "
            f"You are mediating between a {domain_a} specialist and a {domain_b} specialist. "
            f"Read the {domain_a} expert's reasoning and translate the key insights "
            f"so the {domain_b} expert can build on them."
        )
        msgs_m = [{"role": "system", "content": sys_m},
                   {"role": "user", "content": question}]
        for step in chain:
            if step["agent"] == mediator_label:
                msgs_m.append({"role": "assistant", "content": step["thought"]})
                msgs_m.append({"role": "user", "content": "Continue mediating."})
            else:
                msgs_m.append({
                    "role": "user",
                    "content": f"[{step['agent']} expert]: {step['thought']}",
                })

        bridge_for_b = _encode_and_generate(
            mediator, tokenizer, msgs_m, device,
            max_new_tokens=200, temperature=temperature,
        )
        chain.append({"agent": mediator_label, "thought": bridge_for_b,
                       "role": "mediator_to_b"})

        # Step 3: Specialist B reads mediator's bridge, reasons
        sys_b = (
            f"You are a {domain_b} expert. A mediator is helping you collaborate "
            f"with a {domain_a} expert. Think step by step from your domain's perspective."
        )
        msgs_b = [{"role": "system", "content": sys_b},
                   {"role": "user", "content": question}]
        for step in chain:
            if step["agent"] == domain_b:
                msgs_b.append({"role": "assistant", "content": step["thought"]})
                msgs_b.append({"role": "user", "content": "Continue reasoning."})
            elif step["role"] == "mediator_to_b":
                msgs_b.append({
                    "role": "user",
                    "content": f"[Mediator]: {step['thought']}",
                })

        if chain and msgs_b[-1]["role"] == "assistant":
            msgs_b.append({"role": "user",
                           "content": f"Continue. Cycle {cycle + 1} of {n_cycles}."})

        thought_b = _encode_and_generate(
            specialist_b, tokenizer, msgs_b, device,
            max_new_tokens=200, temperature=temperature,
        )
        chain.append({"agent": domain_b, "thought": thought_b, "role": "specialist_b"})

        # Step 4: Mediator reads B, bridges back for A (except last cycle)
        if cycle < n_cycles - 1:
            msgs_m2 = [{"role": "system", "content": sys_m},
                       {"role": "user", "content": question}]
            for step in chain:
                if step["agent"] == mediator_label:
                    msgs_m2.append({"role": "assistant", "content": step["thought"]})
                    msgs_m2.append({"role": "user", "content": "Continue mediating."})
                else:
                    msgs_m2.append({
                        "role": "user",
                        "content": f"[{step['agent']} expert]: {step['thought']}",
                    })

            bridge_for_a = _encode_and_generate(
                mediator, tokenizer, msgs_m2, device,
                max_new_tokens=200, temperature=temperature,
            )
            chain.append({"agent": mediator_label, "thought": bridge_for_a,
                           "role": "mediator_to_a"})

    # Final answer from the mediator (has seen both perspectives)
    sys_final = (
        f"You are an expert in both {domain_a} and {domain_b}. "
        f"Based on the full collaborative discussion between both specialists, "
        f"give the final answer."
    )
    msgs_final = [{"role": "system", "content": sys_final},
                  {"role": "user", "content": question}]
    for step in chain:
        if step["agent"] == mediator_label:
            msgs_final.append({"role": "assistant", "content": step["thought"]})
            msgs_final.append({"role": "user", "content": "Continue."})
        else:
            msgs_final.append({
                "role": "user",
                "content": f"[{step['agent']} expert]: {step['thought']}",
            })
    msgs_final[-1] = {
        "role": "user",
        "content": "Now give your final answer. Reply with ONLY the letter (A, B, C, or D).",
    }

    final = _encode_and_generate(
        mediator, tokenizer, msgs_final, device,
        max_new_tokens=50, temperature=0.3,
    )
    return final, chain, pre_collab


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
    protocol: str = "full-cot",
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
        protocol: Communication protocol — "full-cot", "answer-only", or "structured".

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
            final, chain, pre_collab = collab_reasoning_scoped(
                model_a, model_b, tokenizer, entry["question"],
                domain_a, domain_b, n_rounds=n_rounds,
                device=device, temperature=temperature,
                protocol=protocol,
            )
            predicted = extract_answer_letter(final)
            expected = entry["answer_letter"]
            pre_a = pre_collab["agent_a"]["answer"]
            pre_b = pre_collab["agent_b"]["answer"]

            # Classify convergence behavior
            a_switched = pre_a != predicted
            a_was_right = pre_a == expected
            post_right = predicted == expected
            if a_switched and a_was_right and not post_right:
                switch_type = "correct_to_wrong"
            elif a_switched and not a_was_right and post_right:
                switch_type = "wrong_to_correct"
            elif a_switched:
                switch_type = "switched_other"
            else:
                switch_type = "held"

            collab_results.append({
                "question_domain": domain_a,
                "agent_a": domain_a,
                "agent_b": domain_b,
                "mode": "collab",
                "protocol": protocol,
                "expected": expected,
                "predicted": predicted,
                "correct": post_right,
                "pre_collab_a": pre_a,
                "pre_collab_b": pre_b,
                "pre_collab_a_correct": a_was_right,
                "pre_collab_b_correct": pre_b == expected,
                "agent_a_switched": a_switched,
                "switch_type": switch_type,
                "final_response": final[:200],
                "chain": [{"agent": s["agent"], "thought": s["thought"][:100],
                           "shared": s.get("shared", s["thought"])[:100]} for s in chain],
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
            "protocol": protocol,
        },
    }


def build_collab_summary(
    solo_results: List[Dict],
    collab_results: List[Dict],
    domains: List[str],
) -> Dict:
    """Build summary comparing solo vs collaborative accuracy."""
    summary = {"solo": {}, "collab": {}, "collab_delta": {}, "convergence": {}}

    # Solo accuracy per domain
    for domain in domains:
        results = [r for r in solo_results if r["domain"] == domain]
        if results:
            summary["solo"][domain] = sum(r["correct"] for r in results) / len(results)

    # Collab accuracy per pair per question domain (includes same-domain if present)
    for domain_a in domains:
        for domain_b in domains:
            pair_key = f"{domain_a}+{domain_b}"
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

    # Convergence analysis (aggregate across all pairs)
    has_pre = [r for r in collab_results if "pre_collab_a" in r]
    if has_pre:
        n = len(has_pre)
        pre_a_acc = sum(r["pre_collab_a_correct"] for r in has_pre) / n
        post_acc = sum(r["correct"] for r in has_pre) / n
        switched = sum(r["agent_a_switched"] for r in has_pre) / n
        switch_counts = {}
        for r in has_pre:
            st = r.get("switch_type", "unknown")
            switch_counts[st] = switch_counts.get(st, 0) + 1
        switch_rates = {k: v / n for k, v in switch_counts.items()}

        summary["convergence"] = {
            "n_questions": n,
            "pre_collab_agent_a_accuracy": pre_a_acc,
            "post_collab_accuracy": post_acc,
            "accuracy_delta": post_acc - pre_a_acc,
            "switch_rate": switched,
            "switch_types": switch_rates,
            "correct_to_wrong_rate": switch_rates.get("correct_to_wrong", 0),
            "wrong_to_correct_rate": switch_rates.get("wrong_to_correct", 0),
        }

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
