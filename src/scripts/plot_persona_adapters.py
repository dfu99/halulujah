#!/usr/bin/env python3
"""Visualize Phase 1 of Pivot A: 5 persona LoRA adapters on Qwen3-1.7B.

PACE job 5357133, 24 min on A100. Trained 5 author-specific LoRA adapters
from Blog Authorship Corpus data. Confirmed strong persona signal (KL ≈ 29 nats
vs base) and distinguishable personas (pairwise KL ≈ 7-8 nats).
"""

import json
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("results/persona/fingerprint/fingerprint_results.json"))

personas = sorted(d["kl_divergence"]["vs_base"].keys())
vs_base = [d["kl_divergence"]["vs_base"][p]["mean_kl"] for p in personas]
vs_base_std = [d["kl_divergence"]["vs_base"][p]["std_kl"] for p in personas]

# Short labels
short_labels = [p.replace("author_", "A") for p in personas]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Pivot A Phase 1 Complete: 5 Persona LoRA Adapters on Qwen3-1.7B\n"
    "PACE job 5357133 • 24 min • A100 • Blog Authorship Corpus",
    fontsize=12, fontweight="bold", y=1.02,
)

# Panel 1: KL vs base — persona imprint strength
ax1 = axes[0]
bars = ax1.bar(range(len(personas)), vs_base, yerr=vs_base_std, capsize=5,
               color="#1E88E5", alpha=0.85, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, vs_base):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5,
             f"{val:.1f}", ha="center", fontweight="bold", fontsize=10)
ax1.axhline(np.mean(vs_base), color="red", linestyle="--",
            label=f"Mean {np.mean(vs_base):.1f} nats")
ax1.set_xticks(range(len(personas)))
ax1.set_xticklabels(short_labels)
ax1.set_ylabel("KL(persona || base) [nats]")
ax1.set_title("Persona Imprint: KL Divergence vs Base Model\n(higher = stronger persona)")
ax1.legend()
ax1.set_ylim(0, 40)

# Panel 2: Pairwise KL heatmap — are personas distinguishable?
ax2 = axes[1]
n = len(personas)
kl_matrix = np.zeros((n, n))
pairs = d["kl_divergence"].get("pairwise", {})
for key, val in pairs.items():
    if "_vs_" not in key:
        continue
    p1, p2 = key.split("_vs_")
    try:
        i = personas.index(p1)
        j = personas.index(p2)
    except ValueError:
        continue
    v = val if isinstance(val, (int, float)) else val.get("mean_kl", 0)
    kl_matrix[i, j] = v
    kl_matrix[j, i] = v

im = ax2.imshow(kl_matrix, cmap="YlOrRd", vmin=0, vmax=10)
ax2.set_xticks(range(n))
ax2.set_yticks(range(n))
ax2.set_xticklabels(short_labels)
ax2.set_yticklabels(short_labels)
for i in range(n):
    for j in range(n):
        if i != j:
            ax2.text(j, i, f"{kl_matrix[i, j]:.1f}", ha="center", va="center",
                     fontsize=9, fontweight="bold",
                     color="white" if kl_matrix[i, j] > 6 else "black")
        else:
            ax2.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
ax2.set_title(f"Pairwise KL Between Personas\n(mean {np.mean([v for v in pairs.values() if isinstance(v, (int, float))]):.1f} nats — personas distinguishable)")
plt.colorbar(im, ax=ax2, label="KL (nats)", shrink=0.8)

fig.text(0.5, -0.04,
    f"Mean KL(persona || base) = {np.mean(vs_base):.1f} nats (strong persona imprint). "
    f"Mean pairwise KL = {np.mean([v for v in pairs.values() if isinstance(v, (int, float))]):.1f} nats "
    "(personas distinguishable from each other).\n"
    "Training: 5 authors from Blog Authorship Corpus, Qwen3-1.7B + LoRA r=16, "
    "PACE A100, 24 min. Validated measurement pipeline for Pivot B.",
    ha="center", va="top", fontsize=9, style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
import os
os.makedirs("results/persona/figures", exist_ok=True)
plt.savefig("results/persona/figures/phase1_adapters.png", dpi=150, bbox_inches="tight")
print("Saved to results/persona/figures/phase1_adapters.png")
