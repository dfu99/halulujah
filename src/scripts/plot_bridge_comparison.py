#!/usr/bin/env python3
"""Visualize bridge vs direct collaboration with full FT models."""

import json
import matplotlib.pyplot as plt
import numpy as np

med = json.load(open("results/full_ft_mediator/full_ft_mediator_results.json"))
br = json.load(open("results/bridged/bridged_results.json"))

med_conds = {c["id"]: c for c in med["conditions"]}
br_conds = {c["id"]: c for c in br["conditions"]}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Full FT: Direct vs Bridged vs Mediator Collaboration\n"
    "Medicine + Physics, Qwen3-1.7B full fine-tuning",
    fontsize=13, fontweight="bold", y=1.02,
)

C_SOLO = "#9E9E9E"
C_BASE = "#43A047"
C_CROSS = "#FB8C00"
C_MED = "#8E24AA"
C_DIRECT = "#1E88E5"
C_BRIDGE = "#E53935"

# --- Panel 1: Medicine domain - all helpers compared ---
ax1 = axes[0]
labels = ["Solo", "+Base\n(untrained)", "+Cross\n(FT phys)", "+FT\nMediator", "Direct\n2-agent", "Bridged\n3-agent"]
accs = [
    med_conds["solo_medicine"]["accuracy"] * 100,
    med_conds["specialist_medicine_plus_base"]["accuracy"] * 100,
    med_conds["specialist_medicine_plus_ft_physics"]["accuracy"] * 100,
    med_conds["specialist_medicine_plus_ft_mediator"]["accuracy"] * 100,
    br_conds["direct_medicine_physics_on_medicine"]["accuracy"] * 100,
    br_conds["bridged_medicine_physics_on_medicine"]["accuracy"] * 100,
]
colors = [C_SOLO, C_BASE, C_CROSS, C_MED, C_DIRECT, C_BRIDGE]

bars = ax1.bar(range(len(labels)), accs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Medicine Questions")
ax1.set_ylim(0, 85)

# --- Panel 2: Physics domain - all helpers compared ---
ax2 = axes[1]
labels2 = ["Solo", "+Base\n(untrained)", "+Cross\n(FT med)", "+FT\nMediator", "Direct\n2-agent", "Bridged\n3-agent"]
accs2 = [
    med_conds["solo_physics"]["accuracy"] * 100,
    med_conds["specialist_physics_plus_base"]["accuracy"] * 100,
    med_conds["specialist_physics_plus_ft_medicine"]["accuracy"] * 100,
    med_conds["specialist_physics_plus_ft_mediator"]["accuracy"] * 100,
    br_conds["direct_medicine_physics_on_physics"]["accuracy"] * 100,
    br_conds["bridged_medicine_physics_on_physics"]["accuracy"] * 100,
]

bars2 = ax2.bar(range(len(labels2)), accs2, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars2, accs2):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_xticks(range(len(labels2)))
ax2.set_xticklabels(labels2, fontsize=8)
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Physics Questions")
ax2.set_ylim(0, 60)

# --- Panel 3: Architecture comparison (delta) ---
ax3 = axes[2]
arch_labels = ["Direct\n2-agent", "Bridged\n3-agent", "+Base\n(best 2-agent)", "+FT\nMediator"]
med_deltas = [
    br_conds["direct_medicine_physics_on_medicine"].get("delta", 0) * 100,
    br_conds["bridged_medicine_physics_on_medicine"].get("delta", 0) * 100,
    med_conds["specialist_medicine_plus_base"].get("delta", 0) * 100,
    med_conds["specialist_medicine_plus_ft_mediator"].get("delta", 0) * 100,
]
phys_deltas = [
    br_conds["direct_medicine_physics_on_physics"].get("delta", 0) * 100,
    br_conds["bridged_medicine_physics_on_physics"].get("delta", 0) * 100,
    med_conds["specialist_physics_plus_base"].get("delta", 0) * 100,
    med_conds["specialist_physics_plus_ft_mediator"].get("delta", 0) * 100,
]

x3 = np.arange(len(arch_labels))
w = 0.35
bars3 = ax3.bar(x3 - w/2, med_deltas, w, label="Medicine Qs", color="#E53935", alpha=0.7)
bars4 = ax3.bar(x3 + w/2, phys_deltas, w, label="Physics Qs", color="#1E88E5", alpha=0.7)

for bar, val in zip(bars3, med_deltas):
    y = val + 1 if val >= 0 else val - 2
    ax3.text(bar.get_x() + bar.get_width()/2, y,
             f"{val:+.0f}", ha="center", va="bottom" if val >= 0 else "top",
             fontsize=9, fontweight="bold")
for bar, val in zip(bars4, phys_deltas):
    y = val + 1 if val >= 0 else val - 2
    ax3.text(bar.get_x() + bar.get_width()/2, y,
             f"{val:+.0f}", ha="center", va="bottom" if val >= 0 else "top",
             fontsize=9, fontweight="bold")

ax3.axhline(y=0, color="black", linewidth=0.8)
ax3.set_xticks(x3)
ax3.set_xticklabels(arch_labels, fontsize=8)
ax3.set_ylabel("Delta vs Solo (pp)")
ax3.set_title("Architecture Comparison\n(collaboration delta)")
ax3.legend(fontsize=9)
ax3.set_ylim(-10, 35)

fig.text(0.5, -0.06,
    "Direct 2-agent (specialist+specialist) outperforms bridged 3-agent (specialist+mediator+specialist).\n"
    "Adding an intermediary degrades signal quality — the mediator is an information bottleneck.\n"
    "Best overall: Full FT specialist + untrained base helper (medicine: +26pp, 1.5x ratio).",
    ha="center", va="top", fontsize=9, style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig("results/bridged/bridge_comparison.png", dpi=150, bbox_inches="tight")
print("Saved to results/bridged/bridge_comparison.png")
