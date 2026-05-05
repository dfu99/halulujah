"""Visualize the 2026-04-27 question-clustered bootstrap that produced
the 22.3x cluster-corrected primary/helper variance ratio - now suspended
because the underlying chains had empty <think> blocks (obj-042).

Two panels:
  (left) Row means per domain with 95% question-clustered bootstrap CIs
    from the polluted PACE 10-domain corpus. Medicine -38 pp / chemistry
    -29 pp / physics -24 pp dominate the negative tail.
  (right) Asymmetry ratio comparison: polluted 22.3x [10.2, 48.0] vs
    verified 4.47x (obj-040, 5x5 LoRA pair-grid on Qwen3-1.7B).

Output: figures/fig_clustered_bootstrap_suspended.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "archive/pre_RP_polluted_2026-04-27/stats_rework/summary.json"
OUT = ROOT / "figures/fig_clustered_bootstrap_suspended.png"

d = json.load(open(SRC))
row_data = d["row_means_clustered_bootstrap"]
anova = d["anova_clustered_resample"]
fdr = d["fdr_corrected_pairs"]

# Sort domains by row_mean ascending
ordered = sorted(row_data.items(), key=lambda kv: kv[1]["row_mean_pp"])
domains = [k for k, _ in ordered]
means = [v["row_mean_pp"] for _, v in ordered]
ci_lo = [v["ci_lo_pp"] for _, v in ordered]
ci_hi = [v["ci_hi_pp"] for _, v in ordered]
err_low = [m - lo for m, lo in zip(means, ci_lo)]
err_high = [hi - m for m, hi in zip(means, ci_hi)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                         gridspec_kw={"width_ratios": [1.6, 1]})

# Left: row means with bootstrap CIs
ax = axes[0]
y = np.arange(len(domains))
colors = ["#c0392b" if m < -5 else ("#fbc02d" if abs(m) <= 5 else "#2e7d32")
          for m in means]
ax.errorbar(means, y, xerr=[err_low, err_high], fmt="o", color="black",
            ecolor="gray", elinewidth=1.5, capsize=4, markersize=6, zorder=3)
ax.barh(y, means, color=colors, alpha=0.55, edgecolor="black",
        linewidth=0.5, zorder=2)
for yi, m, lo, hi in zip(y, means, ci_lo, ci_hi):
    ax.text(m + (1.5 if m >= 0 else -1.5), yi,
            f"{m:+.1f}\n[{lo:+.1f}, {hi:+.1f}]",
            ha="left" if m >= 0 else "right", va="center",
            fontsize=8.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels(domains, fontsize=10)
ax.set_xlabel("Row mean delta vs solo (pp), 95% question-clustered bootstrap",
              fontsize=10)
ax.set_xlim(min(ci_lo) - 12, max(ci_hi) + 12)
ax.grid(axis="x", alpha=0.3)
ax.set_title("PACE pre-RP 10-domain row means (suspended)\n"
             "All chains had empty <think></think> blocks (obj-042)",
             fontsize=10.5, fontweight="bold")

# Right: asymmetry ratio comparison
ax = axes[1]
labels = ["Polluted PACE\n(2026-04-27)\n22.3x [10.2, 48.0]",
         "Verified 5x5\n(2026-05-03, obj-040)\n4.47x"]
ratios = [anova["ratio_median"], 4.47]
ratio_lo = [anova["ratio_ci_lo"], np.nan]
ratio_hi = [anova["ratio_ci_hi"], np.nan]
err_lo = [r - lo if not np.isnan(lo) else 0
          for r, lo in zip(ratios, ratio_lo)]
err_hi = [hi - r if not np.isnan(hi) else 0
          for r, hi in zip(ratios, ratio_hi)]
x = np.arange(len(labels))
ax.bar(x, ratios, color=["#c0392b", "#2e7d32"], edgecolor="black",
       linewidth=0.6, width=0.5)
ax.errorbar(x, ratios,
            yerr=[err_lo, err_hi],
            fmt="none", ecolor="black", elinewidth=1.5, capsize=8)
for xi, r in zip(x, ratios):
    ax.text(xi, r + 1.5, f"{r:.2f}x", ha="center", fontsize=11,
            fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("primary / helper variance ratio", fontsize=10)
ax.set_ylim(0, max(ratio_hi[0], 5) + 8)
ax.grid(axis="y", alpha=0.3)
ax.set_title("Asymmetry ratio: suspended 22.3x replaced by verified 4.47x",
             fontsize=10.5, fontweight="bold")

plt.suptitle(
    f"2026-04-27 question-clustered bootstrap (PACE 10-domain n=20, 1000 resamples)\n"
    f"Headline 22.3x [10.2, 48.0] suspended after empty-think audit. "
    f"22 of 90 BH-FDR-significant pairs SUSPENDED.",
    fontsize=12.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
print()
print(f"Polluted ratio: {anova['ratio_median']:.2f}x "
      f"[{anova['ratio_ci_lo']:.2f}, {anova['ratio_ci_hi']:.2f}]")
print(f"primary_var_mean: {anova['primary_var_mean']:.4f}")
print(f"helper_var_mean: {anova['helper_var_mean']:.4f}")
print(f"BH-FDR significant pairs: {fdr['n_sig_after_BH_q_05']} of {fdr['n_pairs']}")
