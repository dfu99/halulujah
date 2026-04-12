"""Visualize LoRA rank ablation results.

Generates a multi-panel figure showing:
1. Solo accuracy vs LoRA rank (does specialization improve?)
2. Collaboration delta vs rank (does specialization hurt collaboration?)
3. Switch rates and C2W/W2C vs rank (rigidity mechanism)
4. Training entropy vs rank (confidence/rigidity proxy)
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results/runpod_domain/rank_ablation")

# Training metrics extracted from log (entropy = training entropy at epoch 3)
TRAINING_METRICS = {
    4: {"loss": 1.365, "entropy": 0.865, "token_acc": 0.802},
    8: {"loss": 1.298, "entropy": 0.851, "token_acc": 0.806},
    16: {"loss": None, "entropy": None, "token_acc": None},  # Will fill from original training
    32: {"loss": 1.149, "entropy": 0.749, "token_acc": 0.825},
}


def load_results():
    """Load merged results from the combined file."""
    merged_path = os.path.join(RESULTS_DIR, "rank_ablation.json")
    if os.path.exists(merged_path):
        with open(merged_path) as f:
            data = json.load(f)
            return data["results"]

    # Fallback: merge from partial files
    results = []
    partial_path = os.path.join(RESULTS_DIR, "rank_ablation_partial.json")
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            data = json.load(f)
            results.extend(data["results"])

    r16_path = os.path.join(RESULTS_DIR, "rank_ablation_r16.json")
    if os.path.exists(r16_path):
        with open(r16_path) as f:
            data = json.load(f)
            for r in data["results"]:
                if r.get("rank") == 16:
                    results.append(r)

    return results


def plot_rank_ablation(results, output_path):
    """Generate multi-panel rank ablation figure."""
    # Separate LoRA results from base
    lora_results = sorted(
        [r for r in results if r["type"] == "lora"],
        key=lambda r: r["rank"]
    )
    base_result = next((r for r in results if r["type"] == "base"), None)

    ranks = [r["rank"] for r in lora_results]
    solo_accs = [r["solo_acc"] * 100 for r in lora_results]
    base_accs = [r.get("base_collab_acc", 0) * 100 for r in lora_results]
    deltas = [r.get("base_delta", 0) * 100 for r in lora_results]
    c2w = [r.get("base_c2w", 0) for r in lora_results]
    w2c = [r.get("base_w2c", 0) for r in lora_results]
    switches = [r.get("base_switches", 0) for r in lora_results]
    base_solo = base_result["solo_acc"] * 100 if base_result else 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LoRA Rank Ablation: Medicine Specialist + Base Helper",
                 fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: Solo accuracy vs rank
    ax1 = axes[0, 0]
    ax1.plot(ranks, solo_accs, "o-", color="#2196F3", linewidth=2, markersize=8, label="LoRA specialist")
    ax1.axhline(y=base_solo, color="#9E9E9E", linestyle="--", linewidth=1.5, label=f"Base model ({base_solo:.0f}%)")
    ax1.set_xlabel("LoRA Rank (r)", fontsize=11)
    ax1.set_ylabel("Solo Accuracy (%)", fontsize=11)
    ax1.set_title("Solo Accuracy vs LoRA Rank", fontsize=12, fontweight="bold")
    ax1.set_xticks(ranks)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 80)

    # Panel 2: Collaboration delta vs rank
    ax2 = axes[0, 1]
    colors = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]
    bars = ax2.bar(range(len(ranks)), deltas, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)
    ax2.set_xticks(range(len(ranks)))
    ax2.set_xticklabels([f"r={r}" for r in ranks])
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_xlabel("LoRA Rank", fontsize=11)
    ax2.set_ylabel("Collaboration Delta (pp)", fontsize=11)
    ax2.set_title("Collaboration Effect (Specialist + Base)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, deltas):
        y_pos = bar.get_height() + (1 if val >= 0 else -2.5)
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f"{val:+.0f}pp",
                ha="center", va="bottom" if val >= 0 else "top", fontsize=10, fontweight="bold")

    # Panel 3: Switching behavior vs rank
    ax3 = axes[1, 0]
    x = np.arange(len(ranks))
    width = 0.25
    bars_c2w = ax3.bar(x - width, c2w, width, label="C2W (harmful)", color="#F44336", alpha=0.8)
    bars_w2c = ax3.bar(x, w2c, width, label="W2C (helpful)", color="#4CAF50", alpha=0.8)
    bars_total = ax3.bar(x + width, switches, width, label="Total switches", color="#9E9E9E", alpha=0.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"r={r}" for r in ranks])
    ax3.set_xlabel("LoRA Rank", fontsize=11)
    ax3.set_ylabel("Count (out of 50)", fontsize=11)
    ax3.set_title("Switching Behavior vs Rank", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    # Add switch rate annotation
    for i, (s, r) in enumerate(zip(switches, ranks)):
        ax3.text(i + width, s + 1, f"{s/50*100:.0f}%",
                ha="center", va="bottom", fontsize=9, color="#666")

    # Panel 4: Training entropy vs rank (rigidity proxy)
    ax4 = axes[1, 1]
    ent_ranks = [r for r in ranks if TRAINING_METRICS.get(r, {}).get("entropy") is not None]
    ent_values = [TRAINING_METRICS[r]["entropy"] for r in ent_ranks]
    loss_values = [TRAINING_METRICS[r]["loss"] for r in ent_ranks]

    ax4_loss = ax4
    ax4_ent = ax4.twinx()

    l1, = ax4_loss.plot(ent_ranks, loss_values, "s-", color="#FF9800", linewidth=2, markersize=8, label="Training loss")
    l2, = ax4_ent.plot(ent_ranks, ent_values, "D-", color="#9C27B0", linewidth=2, markersize=8, label="Training entropy")

    ax4_loss.set_xlabel("LoRA Rank (r)", fontsize=11)
    ax4_loss.set_ylabel("Training Loss", fontsize=11, color="#FF9800")
    ax4_ent.set_ylabel("Training Entropy", fontsize=11, color="#9C27B0")
    ax4.set_title("Training Metrics vs Rank (Rigidity Proxy)", fontsize=12, fontweight="bold")
    ax4.set_xticks(ent_ranks)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, fontsize=9, loc="upper right")
    ax4.grid(True, alpha=0.3)

    # Add annotation
    ax4.annotate("Higher rank → lower entropy\n= more confident = more rigid",
                xy=(32, 1.149), xytext=(20, 1.33),
                fontsize=9, fontstyle="italic", color="#666",
                arrowprops=dict(arrowstyle="->", color="#999"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    results = load_results()
    if not results:
        print("No results found. Run rank ablation first.")
        exit(1)

    output_path = os.path.join(RESULTS_DIR, "rank_ablation_curve.png")
    plot_rank_ablation(results, output_path)
