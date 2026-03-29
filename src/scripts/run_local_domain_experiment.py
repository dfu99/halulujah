"""Local Cross-Domain Experiment — Ollama-based (no GPU cluster needed).

Runs Pivot B Phase 2 locally using Ollama with prompt-only domain specialization
(system prompts instead of LoRA adapters). Also runs a base model comparison.

Requirements:
  - Ollama running locally (ollama serve)
  - qwen3:1.7b pulled (ollama pull qwen3:1.7b)
  - pip install datasets requests numpy matplotlib

Usage:
  python src/scripts/run_local_domain_experiment.py --out-dir results/local_domain
  python src/scripts/run_local_domain_experiment.py --model qwen3:4b --out-dir results/local_domain_4b
"""

import argparse
import json
import logging
import os
import sys
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

DOMAINS = ["physics", "law", "biology"]

DOMAIN_SYSTEM_PROMPTS = {
    "base": "Answer the multiple-choice question. Give only the letter of the correct answer.",
    "physics": "You are a physics expert. Answer the multiple-choice question. Give only the letter of the correct answer.",
    "law": "You are a legal expert. Answer the multiple-choice question. Give only the letter of the correct answer.",
    "biology": "You are a biology expert. Answer the multiple-choice question. Give only the letter of the correct answer.",
}

DOMAIN_SUBJECTS = {
    "physics": ["college_physics", "high_school_physics", "astronomy", "conceptual_physics"],
    "law": ["professional_law", "jurisprudence", "international_law"],
    "biology": ["college_biology", "high_school_biology", "anatomy", "clinical_knowledge"],
}

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_test_questions(domain: str, n_per_domain: int = 30) -> list:
    """Load MMLU test questions for a domain."""
    from datasets import load_dataset

    entries = []
    for subject in DOMAIN_SUBJECTS[domain]:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for row in ds:
            if len(entries) >= n_per_domain:
                break
            choices = row["choices"]
            answer_idx = row["answer"]
            answer_letter = ANSWER_MAP[answer_idx]
            question_text = row["question"]
            choices_text = "\n".join(f"{ANSWER_MAP[j]}. {c}" for j, c in enumerate(choices))
            entries.append({
                "question": f"{question_text}\n\n{choices_text}",
                "answer_letter": answer_letter,
                "answer": f"{answer_letter}. {choices[answer_idx]}",
                "subject": subject,
                "domain": domain,
            })
    return entries[:n_per_domain]


def query_ollama(model: str, system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    """Send a chat request to Ollama and return the response."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 100,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        logger.warning("Ollama error: %s", e)
        return ""


def extract_answer_letter(response: str) -> str:
    """Extract answer letter (A/B/C/D) from response."""
    response = response.strip().upper()
    if response and response[0] in "ABCD":
        return response[0]
    for letter in "ABCD":
        if f"{letter}." in response or f"{letter})" in response:
            return letter
    for letter in "ABCD":
        if f"ANSWER IS {letter}" in response or f"ANSWER: {letter}" in response:
            return letter
    return "X"


def run_cross_domain_eval(model: str, test_sets: dict, n_per_domain: int, temperature: float = 0.7) -> list:
    """Run all model variants on all domain questions."""
    all_results = []
    model_variants = ["base"] + [f"specialist_{d}" for d in DOMAINS]

    for variant in model_variants:
        if variant == "base":
            sys_prompt = DOMAIN_SYSTEM_PROMPTS["base"]
        else:
            domain = variant.replace("specialist_", "")
            sys_prompt = DOMAIN_SYSTEM_PROMPTS[domain]

        for q_domain, questions in test_sets.items():
            logger.info("  %s answering %s questions (%d)...", variant, q_domain, len(questions))
            correct = 0
            for entry in questions:
                response = query_ollama(model, sys_prompt, entry["question"], temperature)
                predicted = extract_answer_letter(response)
                is_correct = predicted == entry["answer_letter"]
                if is_correct:
                    correct += 1
                all_results.append({
                    "model": variant,
                    "question_domain": q_domain,
                    "subject": entry["subject"],
                    "expected": entry["answer_letter"],
                    "predicted": predicted,
                    "response": response[:200],
                    "correct": is_correct,
                })
            acc = correct / len(questions) if questions else 0
            logger.info("    → %s on %s: %.1f%% (%d/%d)", variant, q_domain, acc * 100, correct, len(questions))

    return all_results


def build_confusion_matrix(all_results: list) -> dict:
    """Build accuracy matrix from results."""
    model_names = sorted(set(r["model"] for r in all_results))
    domains = DOMAINS

    matrix = {}
    for model_name in model_names:
        matrix[model_name] = {}
        for q_domain in domains:
            relevant = [r for r in all_results if r["model"] == model_name and r["question_domain"] == q_domain]
            matrix[model_name][q_domain] = sum(r["correct"] for r in relevant) / max(len(relevant), 1)

    summary = {}
    for model_name in model_names:
        trained_domain = model_name.replace("specialist_", "")
        if trained_domain not in domains:
            all_accs = list(matrix[model_name].values())
            summary[model_name] = {"overall_accuracy": float(np.mean(all_accs))}
            continue
        in_domain = matrix[model_name].get(trained_domain, 0)
        cross_accs = [acc for d, acc in matrix[model_name].items() if d != trained_domain]
        cross_domain = float(np.mean(cross_accs))
        summary[model_name] = {
            "trained_domain": trained_domain,
            "in_domain_accuracy": in_domain,
            "cross_domain_accuracy": cross_domain,
            "accuracy_drop": in_domain - cross_domain,
        }

    return {"matrix": matrix, "summary": summary, "domains": domains, "model_names": model_names}


def visualize_results(confusion: dict, out_dir: str):
    """Generate confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion["matrix"]
    domains = confusion["domains"]
    model_names = confusion["model_names"]

    data = np.zeros((len(model_names), len(domains)))
    for i, model in enumerate(model_names):
        for j, domain in enumerate(domains):
            data[i, j] = matrix.get(model, {}).get(domain, 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([d.capitalize() for d in domains], fontsize=11)
    ax.set_yticks(range(len(model_names)))
    short = [n.replace("specialist_", "expert: ").replace("base", "base (no expert prompt)") for n in model_names]
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel("Question Domain", fontsize=11)
    ax.set_ylabel("Model Variant", fontsize=11)
    ax.set_title("Local Cross-Domain Accuracy (Ollama qwen3:1.7b, prompt-only specialization)", fontsize=12, fontweight="bold")

    for i in range(len(model_names)):
        for j in range(len(domains)):
            val = data[i, j]
            color = "white" if val < 0.35 or val > 0.75 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=12, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")

    # Add summary text
    summary = confusion["summary"]
    summary_lines = []
    for name, stats in summary.items():
        if "accuracy_drop" in stats:
            short_name = name.replace("specialist_", "")
            summary_lines.append(
                f"{short_name}: in={stats['in_domain_accuracy']:.0%}, "
                f"cross={stats['cross_domain_accuracy']:.0%}, "
                f"drop={stats['accuracy_drop']:+.0%}"
            )
    if summary_lines:
        ax.text(0.5, -0.15, "  |  ".join(summary_lines), transform=ax.transAxes,
                ha="center", fontsize=9, style="italic")

    plt.tight_layout()
    path = os.path.join(out_dir, "local_cross_domain_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved %s", path)
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description="Local Cross-Domain Experiment (Ollama)")
    parser.add_argument("--model", default="qwen3:1.7b", help="Ollama model name")
    parser.add_argument("--out-dir", default="results/local_domain", help="Output directory")
    parser.add_argument("--n-per-domain", type=int, default=30, help="Questions per domain")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load test questions
    logger.info("=== Loading MMLU test questions ===")
    test_sets = {}
    for domain in DOMAINS:
        test_sets[domain] = load_test_questions(domain, args.n_per_domain)
        logger.info("  %s: %d questions", domain, len(test_sets[domain]))

    # Save test sets
    with open(os.path.join(args.out_dir, "test_sets.json"), "w") as f:
        json.dump(test_sets, f, indent=2)

    # Run cross-domain evaluation
    logger.info("=== Running cross-domain evaluation ===")
    start = time.time()
    all_results = run_cross_domain_eval(args.model, test_sets, args.n_per_domain, args.temperature)
    elapsed = time.time() - start
    logger.info("Evaluation complete in %.1f minutes", elapsed / 60)

    # Build confusion matrix
    confusion = build_confusion_matrix(all_results)

    # Save results
    with open(os.path.join(args.out_dir, "cross_eval_results.json"), "w") as f:
        json.dump({"confusion_matrix": confusion, "num_results": len(all_results)}, f, indent=2)
    with open(os.path.join(args.out_dir, "cross_eval_details.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n=== RESULTS ({args.model}, {args.n_per_domain}q/domain, T={args.temperature}) ===")
    header = f"{'Model':<30}" + "".join(f"{d:>12}" for d in DOMAINS)
    print(header)
    print("-" * len(header))
    for model_name in confusion["model_names"]:
        row = f"{model_name:<30}"
        for d in DOMAINS:
            acc = confusion["matrix"][model_name][d]
            row += f"{acc:>11.0%} "
        print(row)

    print("\nSummary:")
    for name, stats in confusion["summary"].items():
        if "accuracy_drop" in stats:
            print(f"  {name}: in={stats['in_domain_accuracy']:.0%}, cross={stats['cross_domain_accuracy']:.0%}, drop={stats['accuracy_drop']:+.0%}")

    # Visualize
    logger.info("=== Generating visualization ===")
    fig_path = visualize_results(confusion, args.out_dir)

    # Save metadata
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump({
            "model": args.model,
            "n_per_domain": args.n_per_domain,
            "temperature": args.temperature,
            "elapsed_seconds": elapsed,
            "figure": fig_path,
        }, f, indent=2)

    print(f"\nResults saved to {args.out_dir}")


if __name__ == "__main__":
    main()
