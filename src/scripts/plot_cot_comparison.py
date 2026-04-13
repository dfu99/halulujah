#!/usr/bin/env python3
"""Visualize CoT-preserved vs original specialist comparison."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

results_path = "results/cot_specialist/cot_comparison.json"
output_path = "results/cot_specialist/cot_comparison.png"

with open(results_path) as f:
    data = json.load(f)

orig = data["results"][0]
cot = data["results"][1]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle("CoT-Preserved vs Original Specialist (medicine, r=16)", fontsize=14, fontweight="bold", y=0.98)

# --- Panel 1: Accuracy across conditions ---
ax1 = axes[0]
conditions = ["Solo", "+Base Helper", "+Physics Helper"]
orig_accs = [orig["solo_acc"]*100, orig["base_collab_acc"]*100, orig["cross_collab_acc"]*100]
cot_accs = [cot["solo_acc"]*100, cot["base_collab_acc"]*100, cot["cross_collab_acc"]*100]

x = np.arange(len(conditions))
w = 0.35
bars1 = ax1.bar(x - w/2, orig_accs, w, label="Original", color="#2196F3", edgecolor="white", linewidth=0.5)
bars2 = ax1.bar(x + w/2, cot_accs, w, label="CoT-Preserved", color="#FF9800", edgecolor="white", linewidth=0.5)

for bar, val in zip(bars1, orig_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar, val in zip(bars2, cot_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.axhline(y=21, color="gray", linestyle="--", alpha=0.5, label="Base model (21%)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Accuracy by Condition")
ax1.set_xticks(x)
ax1.set_xticklabels(conditions)
ax1.set_ylim(0, 80)
ax1.legend(fontsize=8)

# --- Panel 2: C2W/W2C ratio (collaboration harm quality) ---
ax2 = axes[1]
conditions_collab = ["+Base", "+Physics"]
orig_ratios = [orig["base_c2w"]/max(orig["base_w2c"],1), orig["cross_c2w"]/max(orig["cross_w2c"],1)]
cot_ratios = [cot["base_c2w"]/max(cot["base_w2c"],1), cot["cross_c2w"]/max(cot["cross_w2c"],1)]

x2 = np.arange(len(conditions_collab))
bars3 = ax2.bar(x2 - w/2, orig_ratios, w, label="Original", color="#2196F3", edgecolor="white", linewidth=0.5)
bars4 = ax2.bar(x2 + w/2, cot_ratios, w, label="CoT-Preserved", color="#FF9800", edgecolor="white", linewidth=0.5)

for bar, val in zip(bars3, orig_ratios):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{val:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar, val in zip(bars4, cot_ratios):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{val:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax2.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Neutral (1.0x)")
ax2.set_ylabel("C2W / W2C Ratio")
ax2.set_title("Collaboration Harm Ratio\n(lower = better)")
ax2.set_xticks(x2)
ax2.set_xticklabels(conditions_collab)
ax2.set_ylim(0, 35)
ax2.legend(fontsize=8)

# --- Panel 3: Raw switch counts ---
ax3 = axes[2]
categories = ["C2W\n(+Base)", "W2C\n(+Base)", "C2W\n(+Physics)", "W2C\n(+Physics)"]
orig_switches = [orig["base_c2w"], orig["base_w2c"], orig["cross_c2w"], orig["cross_w2c"]]
cot_switches = [cot["base_c2w"], cot["base_w2c"], cot["cross_c2w"], cot["cross_w2c"]]

x3 = np.arange(len(categories))
w3 = 0.35
bars5 = ax3.bar(x3 - w3/2, orig_switches, w3, label="Original", color="#2196F3", edgecolor="white", linewidth=0.5)
bars6 = ax3.bar(x3 + w3/2, cot_switches, w3, label="CoT-Preserved", color="#FF9800", edgecolor="white", linewidth=0.5)

for bar, val in zip(bars5, orig_switches):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar, val in zip(bars6, cot_switches):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

ax3.set_ylabel("Count (of 50 questions)")
ax3.set_title("Answer Switches\n(C2W=harmful, W2C=helpful)")
ax3.set_xticks(x3)
ax3.set_xticklabels(categories, fontsize=8)
ax3.set_ylim(0, 38)
ax3.legend(fontsize=8)

# Add annotation box
fig.text(0.5, 0.01,
    "CoT training destroyed domain knowledge (66%→14%) but eliminated epistemic rigidity.\n"
    "Original: physics helper catastrophic (31 C2W, 1 W2C = 31x harm). CoT: physics helper beneficial (7 C2W, 12 W2C = 0.6x).\n"
    "Implication: reasoning preservation prevents collaboration harm — but naive CoT approach loses the knowledge itself.",
    ha="center", va="bottom", fontsize=9, style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.09, 1, 0.95])
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {output_path}")
