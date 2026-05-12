"""Plot 4B Full FT pair-grid deltas + WHO-asymmetry summary.

Reads who_summary_v2.json (matrix_results_v2.json applies the same
parser-repair logic as the 1.7B grid) and renders a 2-panel figure:
heatmap of delta (pair - solo) on the left, row/col means + WHO
ratio on the right. Saved to figures/4b_ft_pair_grid_2026-05-12.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/ft_pair_grid_4b_2026-05-10/who_summary_v2.json"
OUT = ROOT / "figures/4b_ft_pair_grid_2026-05-12.png"


def main() -> None:
    s = json.loads(SUMMARY.read_text())
    delta = np.asarray(s["delta_v2"])
    primaries = s["primaries"]
    helpers = s["helpers"]
    row_means = [s["row_means_pp"][p] for p in primaries]
    col_means = [s["col_means_pp"][h] for h in helpers]

    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.9, 0.9],
                          wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(delta, vmin=-0.15, vmax=0.30, cmap="RdBu_r",
                    aspect="auto")
    ax0.set_xticks(range(len(helpers)))
    ax0.set_xticklabels(helpers, rotation=30, ha="right")
    ax0.set_yticks(range(len(primaries)))
    ax0.set_yticklabels(primaries)
    ax0.set_xlabel("helper")
    ax0.set_ylabel("primary")
    for i in range(len(primaries)):
        for j in range(len(helpers)):
            ax0.text(j, i, f"{delta[i, j]:+.2f}", ha="center",
                     va="center", fontsize=9,
                     color="white" if abs(delta[i, j]) > 0.15 else "black")
    plt.colorbar(im, ax=ax0, shrink=0.85, label="Δ acc (pair − solo)")
    ax0.set_title("4B Full FT pair-grid (v2-repaired predictions)\n"
                  "Δ acc primary × helper")

    ax1 = fig.add_subplot(gs[1])
    y = np.arange(len(primaries))
    ax1.barh(y, row_means, color="#2c7bb6")
    ax1.set_yticks(y)
    ax1.set_yticklabels(primaries)
    ax1.axvline(0, color="black", linewidth=0.6)
    ax1.set_xlabel("mean Δ across helpers")
    ax1.set_title(f"row means (primary side)\nvar = {s['var_rows']:.5f}")
    for i, v in enumerate(row_means):
        ax1.text(v + 0.005, i, f"{v:+.3f}", va="center", fontsize=9)
    ax1.set_xlim(-0.10, 0.25)

    ax2 = fig.add_subplot(gs[2])
    y = np.arange(len(helpers))
    ax2.barh(y, col_means, color="#d7191c")
    ax2.set_yticks(y)
    ax2.set_yticklabels(helpers)
    ax2.axvline(0, color="black", linewidth=0.6)
    ax2.set_xlabel("mean Δ across primaries")
    ax2.set_title(f"col means (helper side)\nvar = {s['var_cols']:.5f}")
    for i, v in enumerate(col_means):
        ax2.text(v + 0.005, i, f"{v:+.3f}", va="center", fontsize=9)
    ax2.set_xlim(-0.05, 0.25)

    fig.suptitle(
        f"4B Full FT pair-grid 2026-05-10 — "
        f"WHO-asymmetry (v2) = {s['who_ratio_v2']:.2f}×  "
        f"(primary-side dominance)",
        fontsize=12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
