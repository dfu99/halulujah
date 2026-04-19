#!/usr/bin/env python3
"""Visualize Full FT vs LoRA collaboration comparison."""

import json
import matplotlib.pyplot as plt
import numpy as np

ft = json.load(open("results/full_ft/full_ft_results.json"))
ft_conds = {c["id"]: c for c in ft["conditions"]}

# LoRA comparison values from matrix experiment
# (medicine and physics RP specialists)
lora = {
    "solo_medicine": 38,
    "solo_physics": 38,
    "lora_medicine_plus_base": 0,    # delta
    "lora_medicine_plus_same": -4,
    "lora_medicine_plus_cross": 0,   # med+phys cross delta
    "lora_physics_plus_base": 0,
    "lora_physics_plus_same": 8,
    "lora_physics_plus_cross": -16,  # phys+med cross delta
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Full Fine-Tuning vs LoRA: Does the Training Method Kill Collaborativeness?\n"
    "Medicine + Physics specialists, Qwen3-1.7B, reasoning-preserved format",
    fontsize=13, fontweight="bold", y=1.02,
)

C_FT = "#1E88E5"
C_LORA = "#E53935"
C_BASE = "#9E9E9E"

# --- Panel 1: Medicine collaboration deltas ---
ax1 = axes[0]
conditions = ["+Base", "+Same\nDomain", "+Cross\n(Physics)"]
ft_deltas_med = [
    ft_conds["full_ft_medicine_plus_base"]["delta"] * 100,
    ft_conds["full_ft_medicine_plus_same"]["delta"] * 100,
    ft_conds["full_ft_medicine_plus_physics"]["delta"] * 100,
]
lora_deltas_med = [
    lora["lora_medicine_plus_base"],
    lora["lora_medicine_plus_same"],
    lora["lora_medicine_plus_cross"],
]

x = np.arange(len(conditions))
w = 0.35
bars1 = ax1.bar(x - w/2, ft_deltas_med, w, label="Full FT", color=C_FT, alpha=0.85,
                edgecolor="black", linewidth=0.5)
bars2 = ax1.bar(x + w/2, lora_deltas_med, w, label="LoRA (r=16)", color=C_LORA, alpha=0.85,
                edgecolor="black", linewidth=0.5)

for bar, val in zip(bars1, ft_deltas_med):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1,
             f"{val:+.0f}pp", ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_FT)
for bar, val in zip(bars2, lora_deltas_med):
    y = val - 2 if val < 0 else val + 1
    va = "top" if val < 0 else "bottom"
    ax1.text(bar.get_x() + bar.get_width()/2, y,
             f"{val:+.0f}pp", ha="center", va=va, fontsize=10, fontweight="bold", color=C_LORA)

ax1.axhline(y=0, color="black", linewidth=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(conditions)
ax1.set_ylabel("Accuracy Delta vs Solo (pp)")
ax1.set_title("Medicine Specialist")
ax1.legend(fontsize=9)
ax1.set_ylim(-10, 35)

# --- Panel 2: Physics collaboration deltas ---
ax2 = axes[1]
ft_deltas_phys = [
    ft_conds["full_ft_physics_plus_base"]["delta"] * 100,
    ft_conds["full_ft_physics_plus_same"]["delta"] * 100,
    ft_conds["full_ft_physics_plus_medicine"]["delta"] * 100,
]
lora_deltas_phys = [
    lora["lora_physics_plus_base"],
    lora["lora_physics_plus_same"],
    lora["lora_physics_plus_cross"],
]

bars3 = ax2.bar(x - w/2, ft_deltas_phys, w, label="Full FT", color=C_FT, alpha=0.85,
                edgecolor="black", linewidth=0.5)
bars4 = ax2.bar(x + w/2, lora_deltas_phys, w, label="LoRA (r=16)", color=C_LORA, alpha=0.85,
                edgecolor="black", linewidth=0.5)

for bar, val in zip(bars3, ft_deltas_phys):
    y = val + 1 if val >= 0 else val - 2
    va = "bottom" if val >= 0 else "top"
    ax2.text(bar.get_x() + bar.get_width()/2, y,
             f"{val:+.0f}pp", ha="center", va=va, fontsize=10, fontweight="bold", color=C_FT)
for bar, val in zip(bars4, lora_deltas_phys):
    y = val - 2 if val < 0 else val + 1
    va = "top" if val < 0 else "bottom"
    ax2.text(bar.get_x() + bar.get_width()/2, y,
             f"{val:+.0f}pp", ha="center", va=va, fontsize=10, fontweight="bold", color=C_LORA)

ax2.axhline(y=0, color="black", linewidth=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(["+Base", "+Same\nDomain", "+Cross\n(Medicine)"])
ax2.set_ylabel("Accuracy Delta vs Solo (pp)")
ax2.set_title("Physics Specialist")
ax2.legend(fontsize=9)
ax2.set_ylim(-22, 18)

# --- Panel 3: Summary comparison ---
ax3 = axes[2]
categories = ["Full FT\nmean delta", "LoRA\nmean delta", "Base pair\nmean delta"]
ft_mean = np.mean(ft_deltas_med + ft_deltas_phys)
lora_mean = np.mean(lora_deltas_med + lora_deltas_phys)
base_pair_mean = 29.2  # from matrix experiment

vals = [ft_mean, lora_mean, base_pair_mean]
colors = [C_FT, C_LORA, C_BASE]
bars5 = ax3.bar(range(3), vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

for bar, val in zip(bars5, vals):
    y = val + 1 if val >= 0 else val - 2
    va = "bottom" if val >= 0 else "top"
    ax3.text(bar.get_x() + bar.get_width()/2, y,
             f"{val:+.1f}pp", ha="center", va=va, fontsize=12, fontweight="bold")

ax3.axhline(y=0, color="black", linewidth=0.8)
ax3.set_xticks(range(3))
ax3.set_xticklabels(categories)
ax3.set_ylabel("Mean Collaboration Delta (pp)")
ax3.set_title("Overall: Training Method Comparison")
ax3.set_ylim(-8, 35)

fig.text(0.5, -0.06,
    f"Full FT mean delta: {ft_mean:+.1f}pp | LoRA mean delta: {lora_mean:+.1f}pp | "
    f"Base pair: {base_pair_mean:+.1f}pp\n"
    "Full fine-tuning preserves collaborativeness (+14.7pp mean) while LoRA destroys it (-2pp mean).\n"
    "The bottleneck is LoRA's rank constraint, not specialization itself.",
    ha="center", va="top", fontsize=10, style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig("results/full_ft/full_ft_vs_lora.png", dpi=150, bbox_inches="tight")
print("Saved to results/full_ft/full_ft_vs_lora.png")
