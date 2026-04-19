#!/usr/bin/env python3
"""Visualize bootstrap CI widths at n=20 to show sample size limitation.

Mean CI half-width is ±17.9pp. This means any effect smaller than ~36pp is
statistically indistinguishable from noise at n=20. Justified pushing to N=200
in later experiments.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

d = json.load(open("results/pace_domain_10/bootstrap_cis.json"))

solo_widths = [v["ci_width"]*100 for v in d["solo"].values()]
collab_widths = []
collab_deltas = []
for pair_key, pd in d.get("collab", {}).items():
    w = pd.get("ci_width")
    if w is not None and w > 0:
        collab_widths.append(w * 100)
        collab_deltas.append(pd.get("delta", 0) * 100)

all_widths = solo_widths + collab_widths
mean_half = np.mean(all_widths) / 2

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Bootstrap 95% CI Widths Reveal N=20 Sample Size Limitation\n"
    "(10-domain × 9-partner collaboration study, PACE April 2026)",
    fontsize=12, fontweight="bold", y=1.02,
)

# Panel 1: CI width distribution
ax1 = axes[0]
ax1.hist(all_widths, bins=15, color="#1E88E5", alpha=0.85, edgecolor="black")
ax1.axvline(np.mean(all_widths), color="red", linestyle="--", linewidth=2,
            label=f"Mean width {np.mean(all_widths):.1f}pp")
ax1.set_xlabel("95% CI Width (pp)")
ax1.set_ylabel("Number of Conditions")
ax1.set_title(f"CI Width Distribution\n(N={len(all_widths)} conditions, all at n=20)")
ax1.legend()

# Panel 2: Signal vs noise — effect sizes vs their CIs
ax2 = axes[1]
deltas = np.array(collab_deltas)
widths = np.array(collab_widths)
half_widths = widths / 2
significant = np.abs(deltas) > half_widths
sig_frac = np.mean(significant) * 100

sort_idx = np.argsort(deltas)
y = np.arange(len(deltas))
ax2.errorbar(deltas[sort_idx], y, xerr=half_widths[sort_idx],
             fmt="none", color="gray", alpha=0.5, linewidth=0.8)
colors = ["#E53935" if s else "#9E9E9E" for s in significant[sort_idx]]
ax2.scatter(deltas[sort_idx], y, c=colors, s=15)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Collaboration Delta (pp)")
ax2.set_ylabel("Collaboration Pairs (sorted)")
ax2.set_title(f"Effect Sizes with 95% CIs\n{sig_frac:.0f}% of pairs have |effect| > half-width")

# Panel 3: Detection threshold
ax3 = axes[2]
effects = np.linspace(0, 50, 50)
# Power analysis approximation: need effect > half-width to be distinguishable from noise
detectable_n20 = effects > mean_half
ax3.fill_between(effects, 0, 1, where=detectable_n20, color="#43A047", alpha=0.3,
                 label=f"Detectable at n=20 (> {mean_half:.0f}pp)")
ax3.fill_between(effects, 0, 1, where=~detectable_n20, color="#E53935", alpha=0.3,
                 label="Lost in noise")
ax3.axvline(mean_half, color="red", linestyle="--",
            label=f"Threshold: ±{mean_half:.1f}pp")
# Show typical effect sizes we care about
ax3.axvline(9, color="blue", linestyle=":", alpha=0.7, label="Base helper +9pp")
ax3.axvline(22, color="purple", linestyle=":", alpha=0.7, label="FT +base +22pp")

ax3.set_xlabel("Effect Size (pp)")
ax3.set_ylabel("")
ax3.set_yticks([])
ax3.set_title("Minimum Detectable Effect at n=20")
ax3.legend(fontsize=8, loc="upper right")
ax3.set_xlim(0, 50)

fig.text(0.5, -0.04,
    f"At n=20 per condition, bootstrap 95% CIs have mean half-width ±{mean_half:.1f}pp. "
    f"Effects smaller than ~{mean_half*2:.0f}pp are indistinguishable from noise.\n"
    f"This justified later experiments at n=200 (expected CI half-width ±5.7pp, 3.2× tighter).",
    ha="center", va="top", fontsize=9, style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("results/pace_domain_10/bootstrap_cis.png", dpi=150, bbox_inches="tight")
print(f"Saved. Mean CI half-width: {mean_half:.1f}pp")
