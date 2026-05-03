"""Pair-grid visualization for the verified Qwen3-1.7B 5x5 LoRA collaboration matrix.

Builds:
  1. Heatmap of delta (specialist+helper accuracy - solo specialist accuracy)
     across (primary, helper) pairs, with row and column marginals.
  2. Row-mean / col-mean asymmetry summary.

Output: figures/fig_verified_pair_grid_5x5.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "figures/fig_verified_pair_grid_5x5.png"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

d = json.load(open(SRC))
conds = d["conditions"]

# Build delta matrix (rows = primary domain, cols = helper)
mat = np.full((len(DOMAINS), len(HELPERS)), np.nan)
for i, dd in enumerate(DOMAINS):
    for j, h in enumerate(HELPERS):
        c = conds.get(f"pair_{dd}_{h}")
        if c is not None:
            mat[i, j] = c.get("delta", 0) * 100

# Row and column means
row_means = np.nanmean(mat, axis=1)
col_means = np.nanmean(mat, axis=0)

# Spread
row_spread = float(np.nanmax(row_means) - np.nanmin(row_means))
col_spread = float(np.nanmax(col_means) - np.nanmin(col_means))
ratio = row_spread / col_spread if col_spread > 0 else float("inf")

# Solo + base_pair for reference
solo_acc = {dd: conds[f"solo_{dd}"]["accuracy"] * 100 for dd in DOMAINS}
base_pair_acc = {dd: conds[f"base_pair_{dd}"]["accuracy"] * 100 for dd in DOMAINS}
base_pair_delta = {dd: conds[f"base_pair_{dd}"].get("delta", 0) * 100
                   for dd in DOMAINS}

# ------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 6.5))
gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 0.5],
                      wspace=0.35, hspace=0.55)

# Top-left: heatmap with marginals
ax0 = fig.add_subplot(gs[0, 0])
vmax = float(max(abs(np.nanmin(mat)), abs(np.nanmax(mat))))
im = ax0.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
ax0.set_xticks(range(len(HELPERS)))
ax0.set_xticklabels(HELPERS, fontsize=10)
ax0.set_yticks(range(len(DOMAINS)))
ax0.set_yticklabels(DOMAINS, fontsize=10)
ax0.set_xlabel("helper", fontsize=10)
ax0.set_ylabel("primary", fontsize=10)
for i in range(len(DOMAINS)):
    for j in range(len(HELPERS)):
        v = mat[i, j]
        ax0.text(j, i, f"{v:+.1f}", ha="center", va="center",
                 fontsize=9.5, fontweight="bold",
                 color="white" if abs(v) > 25 else "black")
plt.colorbar(im, ax=ax0, fraction=0.04,
             label="pair_acc - solo_primary_acc (pp)")
ax0.set_title("5x5 LoRA pair-grid: deliberation gain over solo primary (pp)\n"
              "Rows = primary specialist; Cols = helper; cell = pair_acc - solo_primary_acc",
              fontsize=10.5, fontweight="bold")

# Top-right: row & col means
ax1 = fig.add_subplot(gs[0, 1])
y = np.arange(len(DOMAINS))
ax1.barh(y, row_means, color="#5d8aa8", edgecolor="black", linewidth=0.5,
         label="row mean (per primary)")
for yi, v in zip(y, row_means):
    ax1.annotate(f"{v:+.1f}", (v + 0.6, yi), va="center", fontsize=9.5,
                 fontweight="bold")
ax1.set_yticks(y)
ax1.set_yticklabels(DOMAINS)
ax1.invert_yaxis()
ax1.axvline(0, color="black", linewidth=0.5)
ax1.set_xlabel("mean delta across helpers (pp)")
ax1.set_title(f"Row means (primary)\nrow spread = {row_spread:.1f} pp",
              fontsize=10.5, fontweight="bold")
ax1.grid(axis="x", alpha=0.3)

# Bottom-left: column means
ax2 = fig.add_subplot(gs[1, 0])
x = np.arange(len(HELPERS))
bars = ax2.bar(x, col_means, color="#888", edgecolor="black", linewidth=0.5)
for xi, v in zip(x, col_means):
    ax2.annotate(f"{v:+.1f}", (xi, v + 0.7), ha="center", fontsize=9.5,
                 fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(HELPERS)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("mean delta across primaries (pp)")
ax2.set_title(f"Column means (helper)\ncol spread = {col_spread:.1f} pp",
              fontsize=10.5, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

# Bottom-right: asymmetry summary
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis("off")
text = (
    f"WHO-asymmetry ratio:\n  row spread / col spread\n"
    f"  = {row_spread:.1f} / {col_spread:.1f}\n"
    f"  = *{ratio:.2f}x*\n\n"
    f"Solo accuracy (3-rd CoT):\n  " +
    "\n  ".join(f"{dd}: {solo_acc[dd]:.1f}%" for dd in DOMAINS) +
    f"\n\nBase-pair (base+base) delta:\n  " +
    "\n  ".join(f"{dd}: {base_pair_delta[dd]:+.1f}" for dd in DOMAINS)
)
ax3.text(0, 1, text, fontsize=10, fontfamily="monospace", va="top",
         ha="left", transform=ax3.transAxes)

plt.suptitle("Verified-specialist 5x5 LoRA pair-grid (Qwen3-1.7B, N=50, 3 CoT rounds)\n"
             f"WHO-asymmetry ratio = {ratio:.2f}x  ::  "
             f"row spread {row_spread:.1f} pp >> col spread {col_spread:.1f} pp",
             fontsize=12.5, fontweight="bold", y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
print()
print(f"row_spread={row_spread:.1f} pp")
print(f"col_spread={col_spread:.1f} pp")
print(f"WHO-asymmetry ratio = {ratio:.2f}x")
print()
print("Row means (primary):")
for dd, m in zip(DOMAINS, row_means):
    print(f"  {dd}: {m:+.1f}")
print("Col means (helper):")
for h, m in zip(HELPERS, col_means):
    print(f"  {h}: {m:+.1f}")
