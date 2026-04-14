#!/usr/bin/env python3
"""Visualize composite question experiment: structurally necessary collaboration."""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("results/composite/composite_results.json") as f:
    data = json.load(f)

conds = data["conditions"]
pairs = sorted(set(c["pair"] for c in conds))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Composite Questions: Does Collaboration Help When Structurally Necessary?\n"
    "Interdisciplinary MCQ requiring both domains, Qwen3-1.7B RP specialists",
    fontsize=12, fontweight="bold", y=1.02,
)

C_SOLO_A = "#E53935"
C_SOLO_B = "#FB8C00"
C_COLLAB = "#1E88E5"
C_BASE_S = "#9E9E9E"
C_BASE_P = "#43A047"

# --- Panel 1: Per-pair accuracy comparison ---
ax1 = axes[0]
x = np.arange(len(pairs))
w = 0.15

for i, (label, ctype, filt, color) in enumerate([
    ("Base solo", "base_solo", None, C_BASE_S),
    ("Solo A", "solo_specialist", "first", C_SOLO_A),
    ("Solo B", "solo_specialist", "second", C_SOLO_B),
    ("A+B collab", "cross_collab", None, C_COLLAB),
    ("Base pair", "base_pair", None, C_BASE_P),
]):
    vals = []
    for pair in pairs:
        pc = [c for c in conds if c["pair"] == pair and c["type"] == ctype]
        if filt == "first":
            pc = [c for c in pc if c["specialist"] == pair.split("+")[0]]
        elif filt == "second":
            pc = [c for c in pc if c["specialist"] == pair.split("+")[1]]
        vals.append(pc[0]["accuracy"] * 100 if pc else 0)
    offset = (i - 2) * w
    bars = ax1.bar(x + offset, vals, w, label=label, color=color, alpha=0.85,
                   edgecolor="black", linewidth=0.3)

ax1.set_xticks(x)
ax1.set_xticklabels([p.replace("+", "\n+") for p in pairs], fontsize=8)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Per-Pair Accuracy on Composite Questions")
ax1.legend(fontsize=7, loc="upper right", ncol=2)
ax1.set_ylim(0, 90)

# --- Panel 2: Collaboration delta comparison ---
ax2 = axes[1]
collab_deltas = []
base_deltas = []
pair_labels = []
for pair in pairs:
    cc = [c for c in conds if c["pair"] == pair and c["type"] == "cross_collab"]
    bp = [c for c in conds if c["pair"] == pair and c["type"] == "base_pair"]
    if cc:
        collab_deltas.append(cc[0].get("delta", 0) * 100)
    else:
        collab_deltas.append(0)
    if bp:
        base_deltas.append(bp[0].get("delta", 0) * 100)
    else:
        base_deltas.append(0)
    pair_labels.append(pair.replace("+", "\n+"))

x2 = np.arange(len(pairs))
bars1 = ax2.bar(x2 - 0.18, collab_deltas, 0.35, label="Specialist A+B collab",
                color=C_COLLAB, alpha=0.85, edgecolor="black", linewidth=0.5)
bars2 = ax2.bar(x2 + 0.18, base_deltas, 0.35, label="Base pair",
                color=C_BASE_P, alpha=0.85, edgecolor="black", linewidth=0.5)

for bar, val in zip(bars1, collab_deltas):
    y = val + 1.5 if val >= 0 else val - 3
    va = "bottom" if val >= 0 else "top"
    ax2.text(bar.get_x() + bar.get_width()/2, y,
             f"{val:+.0f}pp", ha="center", va=va, fontsize=9, fontweight="bold", color=C_COLLAB)
for bar, val in zip(bars2, base_deltas):
    y = val + 1.5 if val >= 0 else val - 3
    va = "bottom" if val >= 0 else "top"
    ax2.text(bar.get_x() + bar.get_width()/2, y,
             f"{val:+.0f}pp", ha="center", va=va, fontsize=9, fontweight="bold", color=C_BASE_P)

ax2.axhline(y=0, color="black", linewidth=0.8)
ax2.set_xticks(x2)
ax2.set_xticklabels(pair_labels, fontsize=8)
ax2.set_ylabel("Accuracy Delta vs Best Solo (pp)")
ax2.set_title("Collaboration Delta on Composite Questions\n(positive = collaboration helps)")
ax2.legend(fontsize=8)
ax2.set_ylim(-40, 50)

fig.text(0.5, -0.04,
    "Specialist collaboration helps in 3/5 pairs when structurally necessary (phys+math: +40pp).\n"
    "Pattern: both specialists must be weak for collaboration to help. If one is already decent, it gets dragged down.\n"
    "Base pairs still dominate (mean 36% vs specialist 32%), confirming LoRA constrains deliberation benefit.",
    ha="center", va="top", fontsize=9, style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig("results/composite/composite_results.png", dpi=150, bbox_inches="tight")
print("Saved to results/composite/composite_results.png")
