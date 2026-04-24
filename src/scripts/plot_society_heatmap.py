"""
Figure 1 for the pivoted paper spine: collaboration-delta heatmap over the
10x10 grid of ordered (primary, helper) domain pairs at Qwen3-1.7B, N=20
per pair (PACE study). Lead figure for the "societies of agents" claim.

Key visual story: the ROWS (primary agent's domain) differ dramatically;
the COLUMNS (helper's domain) are nearly flat within a row. Medicine is
harmed by every helper. Philosophy is helped by every helper. This is
the ANOVA result made visible (primary 13.1% of variance, helper 0.5%).

Output: figures/fig1_society_heatmap.png
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/pace_domain_10/collaboration/collab_summary.json"
OUT = ROOT / "figures/fig1_society_heatmap.png"


def main():
    d = json.load(open(SRC))
    deltas = d["collab_delta"]

    domains = sorted({k.split("+")[0] for k in deltas.keys()})
    n = len(domains)
    idx = {name: i for i, name in enumerate(domains)}

    grid = np.full((n, n), np.nan)
    for key, value in deltas.items():
        primary, helper = key.split("+")
        grid[idx[primary], idx[helper]] = value * 100.0

    row_means = np.nanmean(grid, axis=1)
    col_means = np.nanmean(grid, axis=0)

    fig, (ax, ax_marg) = plt.subplots(
        1, 2, figsize=(10.5, 6.5),
        gridspec_kw={"width_ratios": [3, 1]})

    im = ax.imshow(grid, cmap="RdBu_r",
                   vmin=-max(abs(np.nanmin(grid)), abs(np.nanmax(grid))),
                   vmax=max(abs(np.nanmin(grid)), abs(np.nanmax(grid))))
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(domains, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(domains, fontsize=9)
    ax.set_xlabel("helper domain", fontsize=10)
    ax.set_ylabel("primary agent's domain", fontsize=10)
    ax.set_title("Collaboration delta per ordered (primary, helper) pair\n"
                 "Qwen3-1.7B LoRA specialists, N=20 per pair, 10 MMLU domains",
                 fontsize=10)

    for i in range(n):
        for j in range(n):
            v = grid[i, j]
            if np.isnan(v):
                continue
            color = "white" if abs(v) > 20 else "black"
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                    color=color, fontsize=7)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="delta (pp)")

    y = np.arange(n)
    ax_marg.barh(y, row_means, color="#b71c1c", alpha=0.7,
                 label="row mean (primary)")
    ax_marg.barh(y + 0.4, col_means, height=0.4,
                 color="#1565c0", alpha=0.7, label="col mean (helper)")
    ax_marg.set_yticks(y + 0.2)
    ax_marg.set_yticklabels(domains, fontsize=9)
    ax_marg.invert_yaxis()
    ax_marg.axvline(0, color="black", linewidth=0.6)
    ax_marg.set_xlabel("mean collaboration delta (pp)", fontsize=10)
    ax_marg.set_title("Row means (primary) vs column means (helper)\n"
                      "ANOVA: primary 13.1% of variance, helper 0.5%  -> 26x more predictive",
                      fontsize=10)
    ax_marg.legend(loc="lower right", fontsize=8)
    ax_marg.grid(axis="x", alpha=0.3)

    fig.suptitle("Figure 1.  Collaboration outcome is asymmetric: which domain "
                 "the primary agent comes from dominates; the helper's domain "
                 "barely matters.",
                 fontsize=11, fontweight="bold", y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUT.parent, exist_ok=True)
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")

    print("\nRow means (primary-agent's domain):")
    order = np.argsort(row_means)
    for i in order:
        print(f"  {domains[i]:20s} {row_means[i]:+6.1f} pp")
    print("\nColumn means (helper's domain):")
    order = np.argsort(col_means)
    for i in order:
        print(f"  {domains[i]:20s} {col_means[i]:+6.1f} pp")


if __name__ == "__main__":
    main()
