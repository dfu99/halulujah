"""Visualize Pivot B cross-domain hallucination results.

Generates 4 key figures:
1. Confusion matrix: specialist × question domain → accuracy (heatmap)
2. Domain distance map: KL divergence between specialist pairs (triangle heatmap)
3. Scatter plot: KL distance vs accuracy drop (core finding)
4. Summary dashboard combining all panels

Usage:
  python src/scripts/visualize_domain_results.py \
      --cross-eval results/domain/cross_eval/cross_eval_results.json \
      --kl-results results/domain/domain_distance/kl_results.json \
      --kl-hallucination results/domain/domain_distance/kl_vs_hallucination.json \
      --out-dir figures/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cross_eval: dict, ax=None):
    """Heatmap: specialist × question domain → accuracy."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    matrix = cross_eval["confusion_matrix"]["matrix"]
    domains = cross_eval["confusion_matrix"]["domains"]
    model_names = cross_eval["confusion_matrix"]["model_names"]

    data = np.zeros((len(model_names), len(domains)))
    for i, model in enumerate(model_names):
        for j, domain in enumerate(domains):
            data[i, j] = matrix.get(model, {}).get(domain, 0)

    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, fontsize=9)
    ax.set_yticks(range(len(model_names)))
    short_names = [n.replace("specialist_", "").replace("base", "base (no ft)") for n in model_names]
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Question Domain")
    ax.set_ylabel("Model")
    ax.set_title("Cross-Domain Accuracy Matrix")

    for i in range(len(model_names)):
        for j in range(len(domains)):
            val = data[i, j]
            color = "white" if val < 0.4 or val > 0.7 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=10, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")
    return ax


def plot_kl_distance_map(kl_results: dict, ax=None):
    """Triangle heatmap: pairwise KL between specialists."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    pairwise = kl_results.get("pairwise", {})
    # Extract unique model names from pair keys
    names = set()
    for key in pairwise:
        parts = key.split("_vs_")
        names.update(parts)
    names = sorted(names)

    n = len(names)
    kl_matrix = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            key = f"{a}_vs_{b}"
            alt_key = f"{b}_vs_{a}"
            if key in pairwise:
                kl_matrix[i, j] = pairwise[key]["mean_kl"]
            elif alt_key in pairwise:
                kl_matrix[i, j] = pairwise[alt_key]["mean_kl"]

    im = ax.imshow(kl_matrix, cmap="YlOrRd", aspect="equal")
    short = [n.replace("specialist_", "") for n in names]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, fontsize=9, rotation=45)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=9)
    ax.set_title("Pairwise KL Divergence (Domain Distance)")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{kl_matrix[i,j]:.1f}", ha="center", va="center",
                    fontsize=8, color="white" if kl_matrix[i, j] > np.max(kl_matrix) * 0.5 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8, label="KL Divergence (nats)")
    return ax


def plot_kl_vs_hallucination(correlations: list, ax=None):
    """Scatter: KL distance vs accuracy drop."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    if not correlations:
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("KL Distance vs Accuracy Drop")
        return ax

    kl_vals = [c["kl_distance"] for c in correlations]
    drops = [c["accuracy_drop"] * 100 for c in correlations]  # Convert to percentage
    labels = [f"{c['model'].replace('specialist_', '')}→{c['question_domain']}" for c in correlations]

    colors = plt.cm.Set2(np.linspace(0, 1, len(correlations)))

    ax.scatter(kl_vals, drops, c=colors, s=120, edgecolors="black", linewidth=1, zorder=5)

    for i, label in enumerate(labels):
        ax.annotate(label, (kl_vals[i], drops[i]), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    # Fit line if enough points
    if len(kl_vals) >= 3:
        z = np.polyfit(kl_vals, drops, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(kl_vals) * 0.9, max(kl_vals) * 1.1, 50)
        ax.plot(x_range, p(x_range), "--", color="gray", alpha=0.5, label=f"Linear fit")

        # Correlation coefficient
        corr = np.corrcoef(kl_vals, drops)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=10, va="top", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("KL Divergence (domain distance, nats)")
    ax.set_ylabel("Accuracy Drop (%)")
    ax.set_title("Domain Distance vs Hallucination Severity")
    ax.grid(True, alpha=0.3)
    if len(kl_vals) >= 3:
        ax.legend()
    return ax


def generate_all_figures(args):
    """Generate all Pivot B figures."""
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    cross_eval = None
    kl_results = None
    correlations = None

    if args.cross_eval and os.path.exists(args.cross_eval):
        with open(args.cross_eval) as f:
            cross_eval = json.load(f)

    if args.kl_results and os.path.exists(args.kl_results):
        with open(args.kl_results) as f:
            kl_results = json.load(f)

    if args.kl_hallucination and os.path.exists(args.kl_hallucination):
        with open(args.kl_hallucination) as f:
            correlations = json.load(f)

    # Generate combined dashboard
    n_panels = sum([cross_eval is not None, kl_results is not None, correlations is not None])
    if n_panels == 0:
        print("No data files found. Nothing to visualize.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle("Pivot B: Cross-Domain Hallucination Analysis\n(Qwen3-1.7B, MMLU: Physics/Law/Biology)",
                 fontsize=13, fontweight="bold")

    panel_idx = 0
    if cross_eval is not None:
        plot_confusion_matrix(cross_eval, axes[panel_idx])
        panel_idx += 1

    if kl_results is not None:
        plot_kl_distance_map(kl_results, axes[panel_idx])
        panel_idx += 1

    if correlations is not None:
        plot_kl_vs_hallucination(correlations, axes[panel_idx])
        panel_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(args.out_dir, "domain_cross_hallucination.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    # Also save individual figures
    if cross_eval is not None:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        plot_confusion_matrix(cross_eval, ax2)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "domain_confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close()

    if correlations is not None:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        plot_kl_vs_hallucination(correlations, ax3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "kl_vs_hallucination_scatter.png"), dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Pivot B domain results")
    parser.add_argument("--cross-eval", default=None, help="Path to cross_eval_results.json")
    parser.add_argument("--kl-results", default=None, help="Path to kl_results.json")
    parser.add_argument("--kl-hallucination", default=None, help="Path to kl_vs_hallucination.json")
    parser.add_argument("--out-dir", default="figures", help="Output directory for figures")
    args = parser.parse_args()
    generate_all_figures(args)


if __name__ == "__main__":
    main()
