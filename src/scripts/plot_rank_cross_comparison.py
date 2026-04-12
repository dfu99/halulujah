"""Visualize the definitive rank x helper-type comparison.

Shows that the damage comes from the HELPER's expertise, not the specialist's rank.
Base helper: mild C2W/W2C (~1.5x). Cross-domain helper: catastrophic (23x-infinity).
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results/runpod_domain/rank_ablation")


def main():
    with open(os.path.join(RESULTS_DIR, "rank_ablation_cross.json")) as f:
        data = json.load(f)

    lora = sorted([r for r in data["results"] if r["type"] == "lora"], key=lambda r: r["rank"])
    ranks = [r["rank"] for r in lora]
    rank_labels = [f"r={r}" for r in ranks]

    solo = [r["solo_acc"] * 100 for r in lora]
    base_acc = [r["base_collab_acc"] * 100 for r in lora]
    cross_acc = [r["cross_collab_acc"] * 100 for r in lora]
    base_delta = [r["base_delta"] * 100 for r in lora]
    cross_delta = [r["cross_delta"] * 100 for r in lora]
    base_c2w = [r["base_c2w"] for r in lora]
    base_w2c = [r["base_w2c"] for r in lora]
    cross_c2w = [r["cross_c2w"] for r in lora]
    cross_w2c = [r["cross_w2c"] for r in lora]
    base_switches = [r["base_switches"] for r in lora]
    cross_switches = [r["cross_switches"] for r in lora]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Helper Type Controls Collaboration Harm, Not Specialist Rank",
                 fontsize=14, fontweight="bold", y=0.98)

    x = np.arange(len(ranks))
    width = 0.3

    # Panel 1: Accuracy comparison (solo vs base collab vs cross collab)
    ax1 = axes[0, 0]
    ax1.bar(x - width, solo, width, label="Solo", color="#2196F3", alpha=0.8)
    ax1.bar(x, base_acc, width, label="+ Base helper", color="#4CAF50", alpha=0.8)
    ax1.bar(x + width, cross_acc, width, label="+ Physics helper", color="#F44336", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(rank_labels)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Accuracy by Condition", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, 80)

    # Panel 2: Delta comparison
    ax2 = axes[0, 1]
    bars_base = ax2.bar(x - width/2, base_delta, width, label="+ Base delta",
                        color="#4CAF50", alpha=0.8, edgecolor="white", linewidth=1.5)
    bars_cross = ax2.bar(x + width/2, cross_delta, width, label="+ Physics delta",
                         color="#F44336", alpha=0.8, edgecolor="white", linewidth=1.5)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(rank_labels)
    ax2.set_ylabel("Collaboration Delta (pp)", fontsize=11)
    ax2.set_title("Collaboration Effect by Helper Type", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars_base, base_delta):
        y = bar.get_height() + (1 if val >= 0 else -3)
        ax2.text(bar.get_x() + bar.get_width()/2, y, f"{val:+.0f}",
                ha="center", fontsize=8, fontweight="bold", color="#2E7D32")
    for bar, val in zip(bars_cross, cross_delta):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3,
                f"{val:+.0f}", ha="center", fontsize=8, fontweight="bold", color="#C62828")

    # Panel 3: C2W vs W2C — the smoking gun
    ax3 = axes[1, 0]
    w = 0.18
    ax3.bar(x - 1.5*w, base_c2w, w, label="Base C2W", color="#EF9A9A", edgecolor="#F44336", linewidth=1.2)
    ax3.bar(x - 0.5*w, base_w2c, w, label="Base W2C", color="#A5D6A7", edgecolor="#4CAF50", linewidth=1.2)
    ax3.bar(x + 0.5*w, cross_c2w, w, label="Physics C2W", color="#F44336", edgecolor="#B71C1C", linewidth=1.2)
    ax3.bar(x + 1.5*w, cross_w2c, w, label="Physics W2C", color="#4CAF50", edgecolor="#1B5E20", linewidth=1.2)
    ax3.set_xticks(x)
    ax3.set_xticklabels(rank_labels)
    ax3.set_ylabel("Count (out of 50)", fontsize=11)
    ax3.set_title("C2W vs W2C: Base Helper vs Physics Helper", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3, axis="y")

    # Annotate the W2C=0 for cross
    for i, wc in enumerate(cross_w2c):
        if wc == 0:
            ax3.text(i + 1.5*w, 0.5, "0", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#1B5E20")

    # Panel 4: C2W/W2C ratio comparison
    ax4 = axes[1, 1]
    base_ratio = [c / max(w, 0.5) for c, w in zip(base_c2w, base_w2c)]
    cross_ratio = [c / max(w, 0.5) for c, w in zip(cross_c2w, cross_w2c)]

    ax4.bar(x - width/2, base_ratio, width, label="Base helper C2W/W2C",
            color="#4CAF50", alpha=0.7, edgecolor="white", linewidth=1.5)
    ax4.bar(x + width/2, cross_ratio, width, label="Physics helper C2W/W2C",
            color="#F44336", alpha=0.7, edgecolor="white", linewidth=1.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(rank_labels)
    ax4.set_ylabel("C2W / W2C Ratio", fontsize=11)
    ax4.set_title("Harmful Switching Ratio by Helper Type", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    # Annotate infinity symbols
    for i, (cr, wc) in enumerate(zip(cross_c2w, cross_w2c)):
        if wc == 0:
            ratio_val = cr / 0.5  # display value
            ax4.text(i + width/2, ratio_val + 1, f"{cr}:0",
                    ha="center", fontsize=8, color="#B71C1C", fontstyle="italic")

    for i, (br, bw) in enumerate(zip(base_c2w, base_w2c)):
        ratio = br / max(bw, 0.5)
        ax4.text(i - width/2, ratio + 0.5, f"{br}:{bw}",
                ha="center", fontsize=8, color="#2E7D32")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(RESULTS_DIR, "rank_cross_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
