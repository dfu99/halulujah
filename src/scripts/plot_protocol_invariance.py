#!/usr/bin/env python3
"""Visualize the 3.4x harmful switching ratio invariance across protocols.

For each of 3 communication protocols (full-cot, answer-only, structured),
cross-domain LoRA specialist collaboration shows ~3.4x C2W/W2C ratio.
The ratio is invariant — different information-sharing levels don't change
the fundamental collaboration harm.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

PROTOCOLS = ["full-cot", "answer-only", "structured"]
DESCRIPTIONS = {
    "full-cot": "Full chain-of-thought\nshared",
    "answer-only": "1-sentence summary\nonly",
    "structured": "Answer + confidence\n+ key reasoning",
}

data = {}
for proto in PROTOCOLS:
    s = json.load(open(f"results/runpod_domain/collaboration_{proto}/collab_summary.json"))
    conv = s["convergence"]
    data[proto] = {
        "c2w": conv["correct_to_wrong_rate"] * 100,
        "w2c": conv["wrong_to_correct_rate"] * 100,
        "switch_rate": conv["switch_rate"] * 100,
        "pre_acc": conv["pre_collab_agent_a_accuracy"] * 100,
        "post_acc": conv["post_collab_accuracy"] * 100,
        "n": conv["n_questions"],
    }
    data[proto]["ratio"] = data[proto]["c2w"] / max(data[proto]["w2c"], 0.01)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Harmful Switching Ratio Invariant Across Communication Protocols\n"
    "(Cross-domain LoRA specialist pairs, 10 domains × 9 partners × 50Q = 5000 trials)",
    fontsize=12, fontweight="bold", y=1.02,
)

x = np.arange(len(PROTOCOLS))
labels = [DESCRIPTIONS[p] for p in PROTOCOLS]

# Panel 1: C2W vs W2C rates
ax1 = axes[0]
c2w = [data[p]["c2w"] for p in PROTOCOLS]
w2c = [data[p]["w2c"] for p in PROTOCOLS]
w = 0.35
b1 = ax1.bar(x - w/2, c2w, w, label="Correct→Wrong", color="#E53935", alpha=0.85,
             edgecolor="black", linewidth=0.5)
b2 = ax1.bar(x + w/2, w2c, w, label="Wrong→Correct", color="#43A047", alpha=0.85,
             edgecolor="black", linewidth=0.5)
for bar, val in zip(b1, c2w):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, val in zip(b2, w2c):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("Switch Rate (%)")
ax1.set_title("Answer Switch Rates by Direction")
ax1.legend()
ax1.set_ylim(0, 35)

# Panel 2: C2W/W2C ratio (the invariance)
ax2 = axes[1]
ratios = [data[p]["ratio"] for p in PROTOCOLS]
colors = ["#1E88E5", "#FB8C00", "#8E24AA"]
bars = ax2.bar(x, ratios, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, ratios):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.1,
             f"{val:.1f}x", ha="center", va="bottom", fontsize=14, fontweight="bold")
ax2.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Neutral (1.0x)")
ax2.axhline(y=np.mean(ratios), color="red", linestyle="--", alpha=0.4,
            label=f"Mean {np.mean(ratios):.2f}x")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("C2W / W2C Ratio")
ax2.set_title("Harmful Switch Ratio\n(higher = more harmful)")
ax2.legend(fontsize=8)
ax2.set_ylim(0, 4.5)

# Panel 3: Pre vs Post collab accuracy
ax3 = axes[2]
pre = [data[p]["pre_acc"] for p in PROTOCOLS]
post = [data[p]["post_acc"] for p in PROTOCOLS]
b3 = ax3.bar(x - w/2, pre, w, label="Pre-collab (solo)", color="#9E9E9E", alpha=0.85,
             edgecolor="black", linewidth=0.5)
b4 = ax3.bar(x + w/2, post, w, label="Post-collab", color="#E53935", alpha=0.85,
             edgecolor="black", linewidth=0.5)
for bar, val in zip(b3, pre):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.5,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, val in zip(b4, post):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.5,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=9)
ax3.set_ylabel("Accuracy (%)")
ax3.set_title("Accuracy Before vs After Collaboration")
ax3.legend(fontsize=8)
ax3.set_ylim(0, max(pre + post) + 10)

fig.text(0.5, -0.04,
    f"Across three protocols with increasing information restriction, cross-domain LoRA specialists\n"
    f"consistently switch their correct answers to wrong ~3.4× more often than wrong-to-correct.\n"
    f"The protocol does not mitigate the harm — restricting communication doesn't help.",
    ha="center", va="top", fontsize=9, style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("results/runpod_domain/protocol_invariance.png", dpi=150, bbox_inches="tight")
print("Saved to results/runpod_domain/protocol_invariance.png")
