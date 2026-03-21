"""Temperature erosion experiment: measure persona signal decay under sampling."""

import json
import logging
import os
from typing import Dict, List

import numpy as np
import torch

from .fingerprint import (
    compute_embedding_distances,
    compute_logit_distributions,
    compute_pairwise_kl,
)

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def run_erosion_sweep(
    models: Dict[str, torch.nn.Module],
    tokenizer,
    prompts: List[str],
    temperatures: List[float] = None,
    max_new_tokens: int = 50,
    device: torch.device = None,
) -> Dict:
    """Sweep temperatures and measure persona separation at each level.

    Args:
        models: Dict mapping model_name -> loaded model (include "base").
        tokenizer: Shared tokenizer.
        prompts: List of formatted probe prompts.
        temperatures: Temperature values to sweep.
        max_new_tokens: Tokens to generate per probe.
        device: Torch device.

    Returns:
        Dict with per-temperature KL and embedding metrics.
    """
    if temperatures is None:
        temperatures = DEFAULT_TEMPERATURES

    results = {}

    for temp in temperatures:
        logger.info("=== Temperature %.1f ===", temp)

        # Collect distributions for KL measurement
        distributions = {}
        generated_texts = {}

        for name, model in models.items():
            logger.info("  Generating from %s...", name)

            # Get logit distributions
            dists = compute_logit_distributions(
                model, tokenizer, prompts,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                device=device,
            )
            distributions[name] = dists

            # Also generate text for embedding distances
            texts = []
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temp,
                        do_sample=(temp > 0),
                        top_k=0,
                        top_p=1.0,
                        use_cache=True,
                    )
                response = tokenizer.decode(
                    outputs[0][input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                texts.append(response)
            generated_texts[name] = texts

        # Compute metrics at this temperature
        kl_results = compute_pairwise_kl(distributions, base_key="base")
        emb_results = compute_embedding_distances(generated_texts)

        results[f"t_{temp}"] = {
            "temperature": temp,
            "kl": kl_results,
            "embedding": {
                k: v for k, v in emb_results.items()
                if k not in ("embeddings", "labels")
            },
        }

    return results


def summarize_erosion(results: Dict) -> Dict:
    """Extract summary curves from erosion sweep results.

    Returns dict with:
        - mean_kl_vs_base: temperature -> mean KL across all personas vs base
        - mean_kl_pairwise: temperature -> mean KL across all persona pairs
        - separation_ratio: temperature -> embedding separation ratio
    """
    temps = []
    kl_vs_base = []
    kl_pairwise = []
    sep_ratios = []

    for key in sorted(results.keys()):
        r = results[key]
        temp = r["temperature"]
        temps.append(temp)

        # Mean KL vs base across all personas
        if r["kl"]["vs_base"]:
            kl_vals = [v["mean_kl"] for v in r["kl"]["vs_base"].values()]
            kl_vs_base.append(float(np.mean(kl_vals)))
        else:
            kl_vs_base.append(0.0)

        # Mean pairwise KL
        if r["kl"]["pairwise"]:
            pw_vals = [v["mean_kl"] for v in r["kl"]["pairwise"].values()]
            kl_pairwise.append(float(np.mean(pw_vals)))
        else:
            kl_pairwise.append(0.0)

        # Separation ratio
        sep_ratios.append(r["embedding"].get("separation_ratio", 0.0))

    return {
        "temperatures": temps,
        "mean_kl_vs_base": kl_vs_base,
        "mean_kl_pairwise": kl_pairwise,
        "separation_ratio": sep_ratios,
    }


def save_erosion_results(results: Dict, output_dir: str) -> str:
    """Save erosion sweep results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "erosion_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved erosion results to %s", path)
    return path
