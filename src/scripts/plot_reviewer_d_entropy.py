#!/usr/bin/env python3
"""Reviewer D response: calibration-based analysis using switch rates as proxy.

Can't compute per-turn logit entropy directly (logits not saved). But switch
classification (C2W, W2C, held, other) is a direct calibration proxy:

  - C2W (correct → wrong after seeing helper): the specialist abandoned a
    correct answer under pressure from a wrong helper. High C2W = low
    resistance = poor confidence calibration (too eager to switch).
  - W2C (wrong → correct): the specialist updated toward truth. High W2C =
    good updating = well-calibrated confidence.
  - Held: no change. If solo was correct, this is 'confident correct'.
    If solo was wrong, this is 'confident wrong'.

The C2W/W2C ratio = overconfidence in wrong direction / self-correction ratio.
Ratio >> 1 means the model confidently replaces correct with wrong; it fails
to discount the helper's information relative to its own.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

ft17 = json.load(open("results/paper_sweep/full_ft_5domain/full_ft_5domain.json"))
rs17 = json.load(open("results/paper_sweep/rank_sweep_rp/rank_sweep_rp.json"))
rs4 = json.load(open("results/paper_sweep/qwen3_4b/rank_sweep_rp.json"))
ft4 = json.load(open("results/paper_sweep/qwen3_4b_ft/4b_full_ft.json"))

def get_switches(data, cid):
    c = next((x for x in data["conditions"] if x["id"] == cid), None)
    if c is None:
        return None
    n = c.get("n", 200)
    c2w = c.get("c2w", 0)
    w2c = c.get("w2c", 0)
    switches = c.get("switches", c2w + w2c)
    held = n - switches
    return {"c2w": c2w, "w2c": w2c, "held": held, "n": n,
            "c2w_rate": c2w / n * 100, "w2c_rate": w2c / n * 100,
            "held_rate": held / n * 100,
            "ratio": c.get("c2w_w2c_ratio", c2w / max(w2c, 1))}

# Compare across training methods for medicine + physics at 4B
configs = [
    ("Base pair (medicine)", get_switches(ft17, "base_pair_medicine"), "#9E9E9E"),
    ("1.7B FT+base (medicine)", get_switches(ft17, "ft_medicine_plus_base"), "#1E88E5"),
    ("4B LoRA r=16 +base (med)", get_switches(rs4, "collab_medicine_r16_plus_base"), "#FB8C00"),
    ("4B LoRA r=128 +base (med)", get_switches(rs4, "collab_medicine_r128_plus_base"), "#E53935"),
    ("4B Full FT +base (med)", get_switches(ft4, "ft_4b_medicine_plus_base"), "#43A047"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Reviewer D Response: Calibration Proxy via Switch Classification\n"
    "(C2W = specialist confidently wrong after seeing helper; W2C = correctly updated; ratio = miscalibration)",
    fontsize=12, fontweight="bold", y=1.02,
)

# Panel 1: stacked bar of switch types
ax = axes[0]
labels = [c[0] for c in configs]
held_rates = [c[1]["held_rate"] for c in configs]
w2c_rates = [c[1]["w2c_rate"] for c in configs]
c2w_rates = [c[1]["c2w_rate"] for c in configs]
# "other" switches = switches - c2w - w2c
other_rates = [100 - h - w - c for h, w, c in zip(held_rates, w2c_rates, c2w_rates)]

x = np.arange(len(configs))
ax.bar(x, held_rates, label="Held (no switch)", color="#BDBDBD", alpha=0.85,
       edgecolor="black", linewidth=0.5)
ax.bar(x, w2c_rates, bottom=held_rates, label="W2C (helpful)", color="#43A047",
       alpha=0.85, edgecolor="black", linewidth=0.5)
ax.bar(x, c2w_rates,
       bottom=[h + w for h, w in zip(held_rates, w2c_rates)],
       label="C2W (harmful)", color="#E53935",
       alpha=0.85, edgecolor="black", linewidth=0.5)
ax.bar(x, other_rates,
       bottom=[h + w + c for h, w, c in zip(held_rates, w2c_rates, c2w_rates)],
       label="Other switches", color="#9E9E9E", alpha=0.5,
       edgecolor="black", linewidth=0.5)

for i, c in enumerate(configs):
    d = c[1]
    ax.text(i, 103, f"C2W:W2C\n{d['ratio']:.1f}×",
            ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("% of trials")
ax.set_title("Switch Type Distribution")
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim(0, 115)

# Panel 2: C2W rate (miscalibration proxy)
ax = axes[1]
c2w_vals = [c[1]["c2w_rate"] for c in configs]
w2c_vals = [c[1]["w2c_rate"] for c in configs]
ratios = [c[1]["ratio"] for c in configs]

x = np.arange(len(configs))
w = 0.35
bars1 = ax.bar(x - w/2, c2w_vals, w, label="C2W rate (miscal.)", color="#E53935",
               alpha=0.85, edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + w/2, w2c_vals, w, label="W2C rate (updating)", color="#43A047",
               alpha=0.85, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars1, c2w_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
            f"{val:.0f}%", ha="center", fontsize=8, fontweight="bold")
for bar, val in zip(bars2, w2c_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
            f"{val:.0f}%", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("% of trials")
ax.set_title("C2W vs W2C — Calibration Under Collaboration")
ax.legend(fontsize=8)
ax.set_ylim(0, max(c2w_vals) + 5)

fig.text(0.5, -0.06,
    "Key pattern: base pair (no training) has balanced W2C > C2W — the agent is well-calibrated, updating toward truth more often than away.\n"
    "LoRA specialists show C2W >> W2C at high rank (r=128: 19× ratio) — they confidently switch correct answers to wrong ones. Full FT at matched solo accuracy keeps ratio near 1× (well-calibrated).\n"
    "Interpretation: LoRA rank constraint creates overconfidence in the specialist's decision-making machinery, not just the domain knowledge.",
    ha="center", va="top", fontsize=9, style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
import os
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/reviewer_d_entropy_by_turn.png", dpi=150, bbox_inches="tight")
print("Saved to figures/reviewer_d_entropy_by_turn.png")
