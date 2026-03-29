"""Local Persona Probe — Ollama-based (no LoRA, prompt-only personas).

Tests whether different persona system prompts produce distinguishable
outputs at the embedding level, replicating Pivot A's feasibility test
locally without fine-tuning.

Usage:
  python src/scripts/run_local_persona_probe.py --out-dir results/local_persona
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import requests

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"

# 20 shared probe questions (same as Pivot A)
PROBE_QUESTIONS = [
    "What's the best way to spend a rainy afternoon?",
    "Describe your ideal morning routine.",
    "What's something most people get wrong about happiness?",
    "Tell me about a time you changed your mind about something important.",
    "What advice would you give to your younger self?",
    "What makes a good friend?",
    "Describe a place that feels like home to you.",
    "What's your take on social media?",
    "What do you think about when you can't sleep?",
    "Describe your favorite meal and why it matters to you.",
    "What's something you're proud of that nobody knows about?",
    "How do you deal with a bad day?",
    "What's the most interesting thing you've learned recently?",
    "Describe a stranger who left an impression on you.",
    "What does success mean to you?",
    "Tell me about something you find beautiful.",
    "What's an unpopular opinion you hold?",
    "How do you decide what matters?",
    "Describe a moment that changed how you see the world.",
    "What would you do with an extra hour every day?",
]

# 3 distinct persona system prompts — designed to produce maximally different styles
PERSONAS = {
    "teen_gamer": (
        "You are a 16-year-old who loves video games, anime, and memes. "
        "You write casually with slang, abbreviations, and lots of enthusiasm. "
        "You use 'lol', 'ngl', 'fr', and exclamation marks freely. "
        "Keep responses to 2-3 sentences. Write like you're texting a friend."
    ),
    "academic_philosopher": (
        "You are a 55-year-old philosophy professor who has spent decades studying "
        "existentialism and ethics. You write in measured, precise prose with complex "
        "sentence structure. You reference thinkers like Kierkegaard, Sartre, and "
        "Simone de Beauvoir. Keep responses to 3-4 sentences."
    ),
    "practical_nurse": (
        "You are a 40-year-old emergency room nurse with 15 years of experience. "
        "You are direct, empathetic, and practical. You draw on real-world medical "
        "and caregiving experience. You value efficiency and kindness equally. "
        "Keep responses to 2-3 sentences."
    ),
}


def query_ollama(model: str, system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    """Send a chat request to Ollama."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 256},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        # Strip thinking block if present
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        return content
    except Exception as e:
        logger.warning("Ollama error: %s", e)
        return ""


def run_persona_probe(model: str, temperature: float = 0.7) -> dict:
    """Generate responses from all personas to all probe questions."""
    results = {}
    for persona_name, sys_prompt in PERSONAS.items():
        logger.info("Generating from %s...", persona_name)
        responses = []
        for q in PROBE_QUESTIONS:
            resp = query_ollama(model, sys_prompt, q, temperature)
            responses.append(resp)
        results[persona_name] = responses
        logger.info("  %s: %d responses, avg %.0f chars",
                    persona_name, len(responses), np.mean([len(r) for r in responses]))
    return results


def compute_embedding_distances(generated_texts: dict) -> dict:
    """Compute pairwise embedding distances between persona outputs."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_distances

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
        idx = np.where(label_arr == label)[0]
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
        "intra_mean": float(np.mean(intra_dists)),
        "intra_std": float(np.std(intra_dists)),
        "inter_mean": float(np.mean(inter_dists)),
        "inter_std": float(np.std(inter_dists)),
        "separation_ratio": float(np.mean(inter_dists) / max(np.mean(intra_dists), 1e-8)),
        "embeddings": embeddings,
        "labels": all_labels,
    }


def classify_personas(embeddings: np.ndarray, labels: list) -> dict:
    """Train logistic regression on embeddings → persona."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, embeddings, y, cv=5, scoring="accuracy")
    return {
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "num_classes": len(set(labels)),
        "chance_level": 1.0 / len(set(labels)),
    }


def visualize(emb_results: dict, classification: dict, out_dir: str) -> str:
    """Generate t-SNE + classification summary figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    embeddings = emb_results["embeddings"]
    labels = emb_results["labels"]
    unique = sorted(set(labels))
    label_arr = np.array(labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Local Persona Probe: Prompt-Only Personas (qwen3:1.7b via Ollama)",
                 fontsize=12, fontweight="bold")

    # Panel 1: t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    coords = tsne.fit_transform(embeddings)
    colors = plt.cm.Set1(np.linspace(0, 0.5, len(unique)))
    for i, persona in enumerate(unique):
        mask = label_arr == persona
        axes[0].scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]],
                       label=persona.replace("_", " ").title(), alpha=0.6, s=40)
    axes[0].set_title("t-SNE of Persona Outputs (20 probes each)")
    axes[0].legend(fontsize=9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel 2: Classification + distances
    acc = classification["mean_accuracy"]
    chance = classification["chance_level"]
    sep = emb_results["separation_ratio"]

    axes[1].bar(["Classifier\nAccuracy", "Chance\nLevel"], [acc, chance],
               color=["#228833", "#BBBBBB"], edgecolor="black")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"5-Fold LogReg: {acc:.0%} vs {chance:.0%}\n(Sep ratio: {sep:.2f})")
    for j, v in enumerate([acc, chance]):
        axes[1].text(j, v + 0.02, f"{v:.0%}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "local_persona_probe.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Local Persona Probe (Ollama)")
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--out-dir", default="results/local_persona")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("=== Generating persona responses ===")
    start = time.time()
    generated = run_persona_probe(args.model, args.temperature)
    elapsed = time.time() - start
    logger.info("Generation complete in %.1f minutes", elapsed / 60)

    # Save responses
    with open(os.path.join(args.out_dir, "persona_responses.json"), "w") as f:
        json.dump(generated, f, indent=2)

    logger.info("=== Computing embedding distances ===")
    emb_results = compute_embedding_distances(generated)

    logger.info("=== Classifying personas ===")
    classification = classify_personas(emb_results["embeddings"], emb_results["labels"])

    logger.info("=== Generating visualization ===")
    fig_path = visualize(emb_results, classification, args.out_dir)

    # Save metrics
    metrics = {
        "embedding_distances": {k: v for k, v in emb_results.items() if k not in ("embeddings", "labels")},
        "classification": classification,
        "elapsed_seconds": elapsed,
        "model": args.model,
        "temperature": args.temperature,
        "num_personas": len(PERSONAS),
        "num_probes": len(PROBE_QUESTIONS),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== RESULTS ===")
    print(f"Separation ratio: {emb_results['separation_ratio']:.3f}")
    print(f"Classification: {classification['mean_accuracy']:.0%} vs {classification['chance_level']:.0%} chance")
    print(f"Intra-persona distance: {emb_results['intra_mean']:.4f}")
    print(f"Inter-persona distance: {emb_results['inter_mean']:.4f}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
