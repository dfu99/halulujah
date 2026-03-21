"""Persona fingerprint measurement: KL divergence, embeddings, vocabulary."""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_logit_distributions(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    device: torch.device = None,
) -> List[np.ndarray]:
    """Run prompts through model and capture logit distributions at each position.

    For each prompt, generates max_new_tokens and captures the full softmax
    distribution at every generated token position.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompts: List of formatted prompt strings.
        max_new_tokens: Number of tokens to generate per prompt.
        temperature: Sampling temperature (applied before softmax).
        device: Torch device.

    Returns:
        List of arrays, each shape (max_new_tokens, vocab_size) containing
        log-probability distributions.
    """
    if device is None:
        device = next(model.parameters()).device

    all_distributions = []

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                top_k=0,  # No top-k filtering for clean distributions
                top_p=1.0,  # No top-p filtering
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )

        # outputs.scores is a tuple of (num_generated_tokens,) tensors, each (1, vocab_size)
        scores = outputs.scores
        n_generated = len(scores)

        # Apply temperature and convert to log-probs
        log_probs = []
        for step_scores in scores:
            # step_scores: (1, vocab_size)
            scaled = step_scores.float() / max(temperature, 1e-8)
            lp = F.log_softmax(scaled, dim=-1).cpu().numpy()[0]
            log_probs.append(lp)

        # Pad or truncate to max_new_tokens
        if len(log_probs) < max_new_tokens:
            vocab_size = log_probs[0].shape[0]
            uniform = np.full(vocab_size, -np.log(vocab_size))
            while len(log_probs) < max_new_tokens:
                log_probs.append(uniform)

        dist_array = np.stack(log_probs[:max_new_tokens])  # (max_new_tokens, vocab_size)
        all_distributions.append(dist_array)

    return all_distributions


def kl_divergence(log_p: np.ndarray, log_q: np.ndarray) -> float:
    """Compute KL(P || Q) from log-probability arrays.

    Args:
        log_p: Log-probabilities of distribution P, shape (vocab_size,).
        log_q: Log-probabilities of distribution Q, shape (vocab_size,).

    Returns:
        KL divergence in nats.
    """
    p = np.exp(log_p)
    # Only sum over tokens where P has non-negligible mass
    mask = p > 1e-10
    return float(np.sum(p[mask] * (log_p[mask] - log_q[mask])))


def compute_pairwise_kl(
    distributions: Dict[str, List[np.ndarray]],
    base_key: str = "base",
    n_positions: int = 50,
) -> Dict[str, Dict]:
    """Compute pairwise KL divergence between all persona models.

    Args:
        distributions: Dict mapping model_name -> list of per-prompt distributions.
            Each distribution is shape (n_positions, vocab_size).
        base_key: Key for the base (unfine-tuned) model.
        n_positions: Number of token positions to average over.

    Returns:
        Dict with 'vs_base' (KL from base per persona) and 'pairwise' (all pairs).
    """
    model_names = sorted(distributions.keys())
    n_prompts = len(next(iter(distributions.values())))

    # KL vs base
    vs_base = {}
    if base_key in distributions:
        base_dists = distributions[base_key]
        for name in model_names:
            if name == base_key:
                continue
            persona_dists = distributions[name]
            kl_values = []
            for p_idx in range(n_prompts):
                pos_kls = []
                for t in range(min(n_positions, persona_dists[p_idx].shape[0])):
                    kl = kl_divergence(persona_dists[p_idx][t], base_dists[p_idx][t])
                    pos_kls.append(kl)
                kl_values.append(np.mean(pos_kls))
            vs_base[name] = {
                "mean_kl": float(np.mean(kl_values)),
                "std_kl": float(np.std(kl_values)),
                "per_prompt": [float(v) for v in kl_values],
            }

    # Pairwise KL between personas
    pairwise = {}
    persona_names = [n for n in model_names if n != base_key]
    for i, name_a in enumerate(persona_names):
        for name_b in persona_names[i + 1:]:
            dists_a = distributions[name_a]
            dists_b = distributions[name_b]
            kl_values = []
            for p_idx in range(n_prompts):
                pos_kls = []
                for t in range(min(n_positions, dists_a[p_idx].shape[0])):
                    kl_ab = kl_divergence(dists_a[p_idx][t], dists_b[p_idx][t])
                    kl_ba = kl_divergence(dists_b[p_idx][t], dists_a[p_idx][t])
                    pos_kls.append((kl_ab + kl_ba) / 2)  # Symmetric KL
                kl_values.append(np.mean(pos_kls))
            pair_key = f"{name_a}_vs_{name_b}"
            pairwise[pair_key] = {
                "mean_kl": float(np.mean(kl_values)),
                "std_kl": float(np.std(kl_values)),
                "per_prompt": [float(v) for v in kl_values],
            }

    return {"vs_base": vs_base, "pairwise": pairwise}


def compute_vocab_fingerprint(
    distributions: Dict[str, List[np.ndarray]],
    tokenizer,
    base_key: str = "base",
    top_k: int = 50,
) -> Dict[str, List[Dict]]:
    """Find tokens where each persona diverges most from base.

    Args:
        distributions: Dict mapping model_name -> list of per-prompt distributions.
        tokenizer: For decoding token IDs to strings.
        base_key: Key for the base model.
        top_k: Number of top divergent tokens to return.

    Returns:
        Dict mapping persona_name -> list of {token, token_id, divergence} dicts.
    """
    if base_key not in distributions:
        return {}

    base_dists = distributions[base_key]
    n_prompts = len(base_dists)
    vocab_size = base_dists[0].shape[1]

    fingerprints = {}
    persona_names = [n for n in distributions if n != base_key]

    for name in persona_names:
        persona_dists = distributions[name]
        # Average log-prob difference across all prompts and positions
        total_diff = np.zeros(vocab_size)
        count = 0

        for p_idx in range(n_prompts):
            n_pos = min(50, persona_dists[p_idx].shape[0])
            for t in range(n_pos):
                p_probs = np.exp(persona_dists[p_idx][t])
                b_probs = np.exp(base_dists[p_idx][t])
                # Signed difference: positive = persona more likely
                total_diff += (p_probs - b_probs)
                count += 1

        avg_diff = total_diff / max(count, 1)

        # Top tokens where persona is MORE likely than base
        top_positive_idx = np.argsort(avg_diff)[-top_k:][::-1]
        # Top tokens where persona is LESS likely than base
        top_negative_idx = np.argsort(avg_diff)[:top_k]

        tokens = []
        for idx in top_positive_idx:
            token_str = tokenizer.decode([idx]).strip()
            if token_str:
                tokens.append({
                    "token": token_str,
                    "token_id": int(idx),
                    "divergence": float(avg_diff[idx]),
                    "direction": "more_likely",
                })

        for idx in top_negative_idx:
            token_str = tokenizer.decode([idx]).strip()
            if token_str:
                tokens.append({
                    "token": token_str,
                    "token_id": int(idx),
                    "divergence": float(avg_diff[idx]),
                    "direction": "less_likely",
                })

        fingerprints[name] = tokens

    return fingerprints


def compute_embedding_distances(
    generated_texts: Dict[str, List[str]],
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> Dict:
    """Compute pairwise embedding distances between persona outputs.

    Args:
        generated_texts: Dict mapping model_name -> list of generated response strings.
        embedding_model_name: Sentence-transformers model name.

    Returns:
        Dict with intra/inter distances, separation ratio, and raw embeddings.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_distances

    embedder = SentenceTransformer(embedding_model_name)

    all_texts = []
    all_labels = []
    for name, texts in generated_texts.items():
        all_texts.extend(texts)
        all_labels.extend([name] * len(texts))

    embeddings = embedder.encode(all_texts, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings)
    label_arr = np.array(all_labels)
    unique_labels = sorted(set(all_labels))

    dist_matrix = cosine_distances(embeddings)

    intra_dists = []
    inter_dists = []

    for label in unique_labels:
        mask = label_arr == label
        idx = np.where(mask)[0]
        for j in range(len(idx)):
            for k in range(j + 1, len(idx)):
                intra_dists.append(dist_matrix[idx[j], idx[k]])

    for i, l1 in enumerate(unique_labels):
        for l2 in unique_labels[i + 1:]:
            idx1 = np.where(label_arr == l1)[0]
            idx2 = np.where(label_arr == l2)[0]
            for j in idx1:
                for k in idx2:
                    inter_dists.append(dist_matrix[j, k])

    return {
        "intra_mean": float(np.mean(intra_dists)) if intra_dists else 0.0,
        "intra_std": float(np.std(intra_dists)) if intra_dists else 0.0,
        "inter_mean": float(np.mean(inter_dists)) if inter_dists else 0.0,
        "inter_std": float(np.std(inter_dists)) if inter_dists else 0.0,
        "separation_ratio": float(np.mean(inter_dists) / max(np.mean(intra_dists), 1e-8)) if intra_dists else 0.0,
        "embeddings": embeddings,
        "labels": all_labels,
    }


def save_fingerprint_results(
    kl_results: Dict,
    vocab_fingerprints: Dict,
    embedding_results: Dict,
    output_dir: str,
) -> str:
    """Save all fingerprint measurements to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Strip non-serializable fields
    emb_save = {k: v for k, v in embedding_results.items()
                if k not in ("embeddings", "labels")}

    results = {
        "kl_divergence": kl_results,
        "vocab_fingerprints": vocab_fingerprints,
        "embedding_distances": emb_save,
    }

    path = os.path.join(output_dir, "fingerprint_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved fingerprint results to %s", path)
    return path
