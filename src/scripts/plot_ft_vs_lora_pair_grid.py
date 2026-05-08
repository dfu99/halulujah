"""Side-by-side comparison: LoRA pair-grid vs Full FT pair-grid.

Reads:
  results/verified_pair_grid_qwen3_1p7b/matrix_results.json  (LoRA)
  results/ft_pair_grid_2026-05-08/matrix_results.json        (Full FT)

For each primary domain, plots:
  - Mean accuracy across helpers (1 bar per helper)
  - C2W and W2C counts overlaid (with delta)
  - Net delta per helper (C2W - W2C) as a horizontal line per cell

Also computes the WHO-asymmetry primary/helper variance ratio for
both grids so we can see if the asymmetry HOLDS under Full FT.

Output:
  figures/ft_vs_lora_pair_grid_2026-05-08.png  (3 rows × 5 col grid)
  figures/ft_vs_lora_who_asymmetry_2026-05-08.png  (asymmetry summary)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LORA_MATRIX = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
FT_MATRIX = ROOT / "results/ft_pair_grid_2026-05-08/matrix_results.json"
OUT_GRID = ROOT / "figures/ft_vs_lora_pair_grid_2026-05-08.png"
OUT_WHO = ROOT / "figures/ft_vs_lora_who_asymmetry_2026-05-08.png"

DOMAINS = ["medicine", "math", "biology", "law", "physics"]
HELPERS = ["medicine", "math", "biology", "law", "physics", "base"]
COLOR = {
    "math": "#1f77b4",
    "medicine": "#ff7f0e",
    "biology": "#2ca02c",
    "law": "#d62728",
    "physics": "#9467bd",
    "base": "#7f7f7f",
}


def load_grid(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def get_acc(grid: dict, cell_id: str) -> float | None:
    c = grid.get("conditions", {}).get(cell_id)
    if c is None:
        return None
    return c.get("accuracy")


def primary_helper_variance(grid: dict, primaries: list,
                             helpers: list) -> tuple[float, float]:
    """Compute primary-effect vs helper-effect variance.

    Within-helper-rows variance: for each helper, compute mean acc
    across primaries; variance of those means estimates 'primary effect'
    when projected through helper. (We use the simpler standard:
    take the 5x6 matrix of acc[primary, helper], compute row variance
    (across helpers per primary, averaged) vs column variance (across
    primaries per helper, averaged).
    """
    matrix = []
    for p in primaries:
        row = []
        for h in helpers:
            acc = get_acc(grid, f"pair_{p}_{h}")
            row.append(acc if acc is not None else float("nan"))
        matrix.append(row)
    arr = np.array(matrix)
    # Variance attributable to primary: spread of row means
    row_means = np.nanmean(arr, axis=1)
    col_means = np.nanmean(arr, axis=0)
    primary_var = float(np.nanvar(row_means))
    helper_var = float(np.nanvar(col_means))
    return primary_var, helper_var


def main() -> None:
    lora = load_grid(LORA_MATRIX)
    ft = load_grid(FT_MATRIX)
    if not ft:
        print(f"FT matrix missing at {FT_MATRIX}; skipping plot.")
        return

    # ----- Plot 1: per-primary accuracy bar charts (LoRA top, FT bottom)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey="row")
    for col, primary in enumerate(DOMAINS):
        for row, (label, grid) in enumerate([("LoRA", lora), ("Full FT", ft)]):
            ax = axes[row, col]
            accs = []
            xticks = []
            colors = []
            for h in HELPERS:
                acc = get_acc(grid, f"pair_{primary}_{h}")
                accs.append(acc if acc is not None else 0)
                xticks.append(h[:3])
                colors.append(COLOR[h])
            solo_acc = get_acc(grid, f"solo_{primary}")
            ax.bar(range(len(HELPERS)), accs, color=colors,
                   edgecolor="black", lw=0.5)
            for i, a in enumerate(accs):
                if a > 0:
                    ax.text(i, a + 0.012, f"{a:.2f}", fontsize=7,
                            ha="center")
            if solo_acc is not None:
                ax.axhline(solo_acc, color="black", linestyle=":",
                           lw=0.8, alpha=0.7,
                           label=f"solo={solo_acc:.2f}")
                ax.legend(fontsize=7, loc="upper right")
            ax.set_xticks(range(len(HELPERS)))
            ax.set_xticklabels(xticks, fontsize=8, rotation=30, ha="right")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", linestyle=":", alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"{label}\naccuracy", fontsize=10)
            if row == 0:
                ax.set_title(f"primary = {primary}",
                             fontsize=11, fontweight="bold",
                             color=COLOR[primary])
    fig.suptitle(
        "Pair-grid accuracy by helper (top: LoRA r=4-128 sweep; bottom: Full FT)\n"
        "Each cell = primary specialist deliberating with helper, 50 q × 3 rounds, full-CoT",
        fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_GRID, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_GRID}")

    # ----- Plot 2: WHO-asymmetry comparison (primary-var / helper-var)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    rows = []
    for label, grid in [("LoRA pair-grid", lora), ("Full FT pair-grid", ft)]:
        if not grid:
            continue
        pv, hv = primary_helper_variance(grid, DOMAINS, HELPERS)
        ratio = pv / hv if hv > 0 else float("inf")
        rows.append((label, pv, hv, ratio))
    if rows:
        labels = [r[0] for r in rows]
        ratios = [r[3] for r in rows]
        x = np.arange(len(rows))
        ax.bar(x, ratios, color=["#1f77b4", "#ff7f0e"][:len(rows)],
               edgecolor="black", lw=0.5)
        for i, r in enumerate(rows):
            ax.text(i, r[3] + 0.5, f"{r[3]:.1f}×", fontsize=12,
                    ha="center", fontweight="bold")
            ax.text(i, -2,
                    f"primary var={r[1]:.4f}\nhelper var={r[2]:.4f}",
                    fontsize=8, ha="center", va="top", color="#444")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("primary variance / helper variance "
                       "(WHO-asymmetry ratio)", fontsize=10)
        ax.set_title(
            "WHO-asymmetry holds under both training methods?\n"
            "Higher ratio = more variance attributable to who-holds-question",
            fontsize=11, fontweight="bold")
        ax.axhline(1, color="red", linestyle="--", lw=0.8, alpha=0.5)
        ax.text(len(rows) - 0.5, 1.05, "ratio=1 (no asymmetry)",
                fontsize=8, ha="right", color="red", style="italic")
        ax.set_ylim(min(0, -3), max(ratios) * 1.2)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_WHO, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_WHO}")


if __name__ == "__main__":
    main()
