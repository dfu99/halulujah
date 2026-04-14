#!/usr/bin/env python3
"""Visualize RP mediator results vs old mediator baseline."""

import json
import matplotlib.pyplot as plt
import numpy as np

with open("results/rp_mediator/rp_mediator_results.json") as f:
    data = json.load(f)

conds = data["conditions"]
solos = {c["specialist"]: c["accuracy"] * 100 for c in conds if c["type"] == "solo"}
mediated = sorted([c for c in conds if c["type"] == "mediated"], key=lambda x: x["id"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "RP Mediators (thinking-enabled) vs Old Mediators\n"
    "Does reasoning preservation fix the mediator catastrophe?",
    fontsize=13, fontweight="bold", y=1.02,
)

pair_labels = [c["mediator_pair"].replace("+", "\n+") for c in mediated]
x = np.arange(len(mediated))

# --- Panel 1: Accuracy and delta ---
ax1 = axes[0]
solo_accs = [solos[c["specialist"]] for c in mediated]
med_accs = [c["accuracy"] * 100 for c in mediated]
deltas = [c["delta"] * 100 for c in mediated]

w = 0.35
bars1 = ax1.bar(x - w/2, solo_accs, w, label="Solo specialist", color="#E53935", alpha=0.85,
                edgecolor="black", linewidth=0.5)
bars2 = ax1.bar(x + w/2, med_accs, w, label="+ RP mediator", color="#1E88E5", alpha=0.85,
                edgecolor="black", linewidth=0.5)

for bar, val in zip(bars1, solo_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, val, d in zip(bars2, med_accs, deltas):
    color = "#1565C0" if d >= 0 else "#C62828"
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.0f}%\n({d:+.0f}pp)", ha="center", va="bottom", fontsize=8,
             fontweight="bold", color=color)

ax1.axhline(y=0, color="black", linewidth=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(pair_labels, fontsize=8)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Solo vs Mediated Accuracy")
ax1.legend(fontsize=9)
ax1.set_ylim(0, 55)

# --- Panel 2: C2W/W2C ratio comparison ---
ax2 = axes[1]
rp_ratios = [c["c2w_w2c_ratio"] for c in mediated]

# Old mediator reference: 7.1x mean (from planning.md)
old_ref = 7.1

bars3 = ax2.bar(x, rp_ratios, 0.5, label="RP mediator", color="#1E88E5", alpha=0.85,
                edgecolor="black", linewidth=0.5)
for bar, val in zip(bars3, rp_ratios):
    color = "#C62828" if val > 3 else "#2E7D32"
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{val:.1f}x", ha="center", va="bottom", fontsize=11, fontweight="bold",
             color=color)

ax2.axhline(y=old_ref, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.7,
            label=f"Old mediator mean ({old_ref}x)")
ax2.axhline(y=1.0, color="#43A047", linestyle="--", linewidth=1, alpha=0.5,
            label="Neutral (1.0x)")

ax2.set_xticks(x)
ax2.set_xticklabels(pair_labels, fontsize=8)
ax2.set_ylabel("C2W / W2C Ratio")
ax2.set_title("Harmful Switch Ratio\n(lower = better, 1.0 = neutral)")
ax2.legend(fontsize=8)
ax2.set_ylim(0, 10)

# Annotation
mean_delta = np.mean(deltas)
mean_ratio = np.mean(rp_ratios)
fig.text(0.5, -0.04,
    f"RP mediators: mean delta {mean_delta:+.1f}pp, mean C2W/W2C {mean_ratio:.1f}x\n"
    f"Old mediators: mean delta -32.8pp, mean C2W/W2C 7.1x\n"
    f"Reasoning preservation eliminates catastrophic harm but switching remains biased toward harmful.",
    ha="center", va="top", fontsize=9, style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig("results/rp_mediator/rp_mediator_results.png", dpi=150, bbox_inches="tight")
print("Saved to results/rp_mediator/rp_mediator_results.png")
