#!/usr/bin/env python3
"""Visualize reasoning-preserved vs original specialist comparison."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("results/reasoning_preserved/rp_comparison.json") as f:
    data = json.load(f)

conditions = data["conditions"]

# Helper to get a specific condition
def get(specialist, helper, typ):
    for c in conditions:
        if c["specialist"] == specialist and c["helper"] == helper and c["type"] == typ:
            return c
    return None

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(
    "Reasoning-Preserved vs Original Specialists\n(medicine + physics, r=16, n=50)",
    fontsize=14, fontweight="bold", y=1.0,
)

C_OLD = "#E53935"   # red
C_RP = "#1E88E5"    # blue
C_BASE = "#9E9E9E"  # gray

# --- Panel 1: Accuracy across all conditions ---
ax1 = axes[0]
labels = ["Med\nSolo", "Med\n+Base", "Med\n+Cross", "Phys\nSolo", "Phys\n+Base", "Phys\n+Cross"]
old_accs = [
    get("medicine", "none", "old")["accuracy"] * 100,
    get("medicine", "base", "old")["accuracy"] * 100,
    get("medicine", "physics_old", "old")["accuracy"] * 100,
    get("physics", "none", "old")["accuracy"] * 100,
    get("physics", "base", "old")["accuracy"] * 100,
    get("physics", "medicine_old", "old")["accuracy"] * 100,
]
rp_accs = [
    get("medicine", "none", "rp")["accuracy"] * 100,
    get("medicine", "base", "rp")["accuracy"] * 100,
    get("medicine", "physics_rp", "rp")["accuracy"] * 100,
    get("physics", "none", "rp")["accuracy"] * 100,
    get("physics", "base", "rp")["accuracy"] * 100,
    get("physics", "medicine_rp", "rp")["accuracy"] * 100,
]

x = np.arange(len(labels))
w = 0.35
bars1 = ax1.bar(x - w/2, old_accs, w, label="Original (no reasoning)", color=C_OLD, alpha=0.85)
bars2 = ax1.bar(x + w/2, rp_accs, w, label="Reasoning-preserved", color=C_RP, alpha=0.85)

for bar, val in zip(bars1, old_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar, val in zip(bars2, rp_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax1.axhline(y=21, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
ax1.text(5.7, 22, "base model", fontsize=7, color="gray")
ax1.axvline(x=2.5, color="gray", linestyle=":", alpha=0.3)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Accuracy by Condition")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylim(0, 78)
ax1.legend(fontsize=7, loc="upper right")

# --- Panel 2: Collaboration delta (cross-domain only) ---
ax2 = axes[1]
cross_labels = ["Medicine\n+Physics", "Physics\n+Medicine"]
old_deltas = [
    get("medicine", "physics_old", "old")["delta"] * 100,
    get("physics", "medicine_old", "old")["delta"] * 100,
]
rp_deltas = [
    get("medicine", "physics_rp", "rp")["delta"] * 100,
    get("physics", "medicine_rp", "rp")["delta"] * 100,
]

x2 = np.arange(len(cross_labels))
bars3 = ax2.bar(x2 - w/2, old_deltas, w, label="Original", color=C_OLD, alpha=0.85)
bars4 = ax2.bar(x2 + w/2, rp_deltas, w, label="Reasoning-preserved", color=C_RP, alpha=0.85)

for bar, val in zip(bars3, old_deltas):
    ax2.text(bar.get_x() + bar.get_width()/2, min(val, 0) - 3,
             f"{val:+.0f}pp", ha="center", va="top", fontsize=10, fontweight="bold", color=C_OLD)
for bar, val in zip(bars4, rp_deltas):
    y = val + 2 if val >= 0 else val - 3
    va = "bottom" if val >= 0 else "top"
    ax2.text(bar.get_x() + bar.get_width()/2, y,
             f"{val:+.0f}pp", ha="center", va=va, fontsize=10, fontweight="bold", color=C_RP)

ax2.axhline(y=0, color="black", linewidth=0.8)
ax2.set_ylabel("Accuracy Delta (pp)")
ax2.set_title("Cross-Domain Collaboration Impact")
ax2.set_xticks(x2)
ax2.set_xticklabels(cross_labels)
ax2.set_ylim(-75, 25)
ax2.legend(fontsize=8)

# --- Panel 3: C2W/W2C harm ratio (cross-domain) ---
ax3 = axes[2]
old_ratios = [
    get("medicine", "physics_old", "old")["c2w"] / max(get("medicine", "physics_old", "old")["w2c"], 0.5),
    get("physics", "medicine_old", "old")["c2w"] / max(get("physics", "medicine_old", "old")["w2c"], 0.5),
]
rp_ratios = [
    get("medicine", "physics_rp", "rp")["c2w"] / max(get("medicine", "physics_rp", "rp")["w2c"], 1),
    get("physics", "medicine_rp", "rp")["c2w"] / max(get("physics", "medicine_rp", "rp")["w2c"], 1),
]

bars5 = ax3.bar(x2 - w/2, old_ratios, w, label="Original", color=C_OLD, alpha=0.85)
bars6 = ax3.bar(x2 + w/2, rp_ratios, w, label="Reasoning-preserved", color=C_RP, alpha=0.85)

for bar, val in zip(bars5, old_ratios):
    label = f"{val:.0f}x" if val < 50 else "∞"
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             label, ha="center", va="bottom", fontsize=11, fontweight="bold", color=C_OLD)
for bar, val in zip(bars6, rp_ratios):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f"{val:.1f}x", ha="center", va="bottom", fontsize=11, fontweight="bold", color=C_RP)

ax3.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Neutral (1.0x)")
ax3.set_ylabel("C2W / W2C Ratio")
ax3.set_title("Collaboration Harm Ratio\n(higher = more harmful switches)")
ax3.set_xticks(x2)
ax3.set_xticklabels(cross_labels)
ax3.set_ylim(0, 42)
ax3.legend(fontsize=7)

# Annotation
fig.text(0.5, -0.04,
    "Reasoning-preserved training eliminates catastrophic cross-domain collaboration failure.\n"
    "Old specialists: -64pp / -40pp accuracy drop, 34x / 22x harmful switch ratio.\n"
    "RP specialists: -2pp / +12pp delta, 1.8x / 1.6x ratio. Trade-off: solo accuracy 66%→42% (med), 64%→20% (phys).",
    ha="center", va="top", fontsize=9, style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("results/reasoning_preserved/rp_comparison.png", dpi=150, bbox_inches="tight")
print("Saved to results/reasoning_preserved/rp_comparison.png")
