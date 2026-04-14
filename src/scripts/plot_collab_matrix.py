#!/usr/bin/env python3
"""Visualize full collaboration matrix results."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("results/collab_matrix/matrix_results.json") as f:
    data = json.load(f)

conds = data["conditions"]
domains = data["config"]["domains"]

def get_conds(ctype):
    return {c.get("specialist", c.get("question_domain", "")): c
            for c in conds if c["type"] == ctype}

base_solo = get_conds("base_solo")
solo = get_conds("solo")
base_pair = get_conds("base_pair")
same_pair = get_conds("same_pair")
mixed_pair = get_conds("mixed_pair")

# Cross-domain: keyed by (specialist, helper)
cross = {}
for c in conds:
    if c["type"] == "cross_pair":
        cross[(c["specialist"], c["helper"])] = c

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Collaboration Matrix: Deliberation Value vs Expert Routing (MoE)\n"
    "5 domains, 50 questions each, reasoning-preserved specialists, Qwen3-1.7B",
    fontsize=13, fontweight="bold", y=1.02,
)

# Colors
C_BASE = "#9E9E9E"
C_SOLO = "#E53935"
C_BPAIR = "#1E88E5"
C_SAME = "#43A047"
C_MIXED = "#FB8C00"
C_CROSS = "#8E24AA"

# --- Panel 1: Condition type comparison ---
ax1 = axes[0]
types = ["Base\nSolo", "Specialist\nSolo (MoE)", "Base\nPair", "Same-\nDomain", "Specialist\n+Base", "Cross-\nDomain"]
means = [
    np.mean([base_solo[d]["accuracy"] for d in domains]) * 100,
    np.mean([solo[d]["accuracy"] for d in domains]) * 100,
    np.mean([base_pair[d]["accuracy"] for d in domains]) * 100,
    np.mean([same_pair[d]["accuracy"] for d in domains]) * 100,
    np.mean([mixed_pair[d]["accuracy"] for d in domains]) * 100,
    np.mean([c["accuracy"] for c in conds if c["type"] == "cross_pair"]) * 100,
]
colors = [C_BASE, C_SOLO, C_BPAIR, C_SAME, C_MIXED, C_CROSS]

bars = ax1.bar(range(len(types)), means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_xticks(range(len(types)))
ax1.set_xticklabels(types, fontsize=8)
ax1.set_ylabel("Mean Accuracy (%)")
ax1.set_title("Mean Accuracy by Condition Type")
ax1.set_ylim(0, 62)
ax1.axhline(y=means[1], color=C_SOLO, linestyle="--", alpha=0.4, linewidth=1)
ax1.text(5.5, means[1] + 1, "MoE\nbaseline", fontsize=7, color=C_SOLO, ha="right")

# --- Panel 2: Per-domain breakdown ---
ax2 = axes[1]
x = np.arange(len(domains))
w = 0.15

for i, (label, getter, color) in enumerate([
    ("Base solo", lambda d: base_solo[d]["accuracy"] * 100, C_BASE),
    ("Specialist solo", lambda d: solo[d]["accuracy"] * 100, C_SOLO),
    ("Base pair", lambda d: base_pair[d]["accuracy"] * 100, C_BPAIR),
    ("Same pair", lambda d: same_pair[d]["accuracy"] * 100, C_SAME),
    ("Spec+base", lambda d: mixed_pair[d]["accuracy"] * 100, C_MIXED),
]):
    vals = [getter(d) for d in domains]
    offset = (i - 2) * w
    ax2.bar(x + offset, vals, w, label=label, color=color, alpha=0.85, edgecolor="black", linewidth=0.3)

ax2.set_xticks(x)
ax2.set_xticklabels([d.capitalize() for d in domains], fontsize=9)
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Per-Domain Accuracy by Condition")
ax2.legend(fontsize=7, loc="upper right")
ax2.set_ylim(0, 82)

# --- Panel 3: Cross-domain heatmap ---
ax3 = axes[2]
matrix = np.zeros((len(domains), len(domains)))
for i, d1 in enumerate(domains):
    for j, d2 in enumerate(domains):
        if d1 == d2:
            # Same-domain delta
            matrix[i, j] = same_pair[d1].get("delta", 0) * 100
        elif (d1, d2) in cross:
            matrix[i, j] = cross[(d1, d2)].get("delta", 0) * 100

im = ax3.imshow(matrix, cmap="RdYlGn", vmin=-20, vmax=10, aspect="auto")
ax3.set_xticks(range(len(domains)))
ax3.set_yticks(range(len(domains)))
ax3.set_xticklabels([d[:4].capitalize() for d in domains], fontsize=8)
ax3.set_yticklabels([d[:4].capitalize() for d in domains], fontsize=8)
ax3.set_xlabel("Helper")
ax3.set_ylabel("Specialist (tested on)")
ax3.set_title("Collaboration Delta (pp)\nvs Solo Specialist")

# Annotate cells
for i in range(len(domains)):
    for j in range(len(domains)):
        val = matrix[i, j]
        color = "white" if abs(val) > 10 else "black"
        ax3.text(j, i, f"{val:+.0f}", ha="center", va="center",
                 fontsize=9, fontweight="bold", color=color)

plt.colorbar(im, ax=ax3, label="Delta (pp)", shrink=0.8)

# Bottom annotation
fig.text(0.5, -0.06,
    "Base pair deliberation (+29pp mean) massively outperforms specialist solo (MoE routing baseline).\n"
    "Two untrained models reasoning together reach 49% vs specialists at 29%. LoRA specialization constrains deliberation benefit.\n"
    "Cross-domain pairs: near-neutral on average (+0.5pp), but physics systematically harmed by all helpers.",
    ha="center", va="top", fontsize=9, style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.savefig("results/collab_matrix/collab_matrix.png", dpi=150, bbox_inches="tight")
print("Saved to results/collab_matrix/collab_matrix.png")
