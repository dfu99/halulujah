"""Plot the recovered 1.7B FT WHO bootstrap (cells_v3) side-by-side
with the 1.7B LoRA bootstrap. Paper-ship-grade figure for the
54.74x headline.

Run: python -m src.scripts.plot_clustered_boot_ft_recovered
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LORA = ROOT / "results/verified_pair_grid_qwen3_1p7b/clustered_bootstrap.json"
FT = ROOT / "results/ft_pair_grid_2026-05-08/clustered_bootstrap_v3.json"
OUT = ROOT / "figures/clustered_bootstrap_ft_recovered_2026-05-13.png"

PRIMARIES = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base", "math", "medicine", "biology", "law", "physics"]

N_ITER = 2000
SEED = 2026


def replay_ft_bootstrap() -> np.ndarray:
    """Re-run the bootstrap to recover per-iteration variance ratios for the
    histogram panel (the JSON stores only summary stats). Cheap (~1 s).
    """
    from src.scripts.clustered_bootstrap_who_ft_recovered import (
        load_grid,
        who_metrics,
    )

    grid = load_grid()
    n_per_primary = {p: len(grid[p]["base"]) for p in PRIMARIES}
    np_rng = np.random.default_rng(SEED)
    vratio = []
    for _ in range(N_ITER):
        idx = {
            p: np_rng.integers(0, n_per_primary[p], size=n_per_primary[p])
            for p in PRIMARIES
        }
        _, _, _, vr = who_metrics(grid, idx)
        vratio.append(vr)
    return np.array(vratio)


def main() -> None:
    lora = json.loads(LORA.read_text())
    ft = json.loads(FT.read_text())

    lora_sr = lora["bootstrap_spread_ratio"]
    lora_vr = lora["bootstrap_variance_ratio"]
    ft_sr = ft["bootstrap_spread_ratio"]
    ft_vr = ft["bootstrap_variance_ratio"]

    boot_vr_ft = replay_ft_bootstrap()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (A) Variance ratio: LoRA vs FT recovered, with 95% CI
    ax = axes[0, 0]
    labels = ["1.7B LoRA\n(verified pair-grid)", "1.7B Full FT\n(recovered cells_v3)"]
    points = [
        lora["point"]["variance_ratio_§6m"],
        ft["point"]["variance_ratio_ss_primary_over_ss_helper"],
    ]
    ci_lo = [lora_vr["p2.5"], ft_vr["p2.5"]]
    ci_hi = [lora_vr["p97.5"], ft_vr["p97.5"]]
    err = [
        [p - l for p, l in zip(points, ci_lo)],
        [h - p for p, h in zip(points, ci_hi)],
    ]
    colors = ["#1f77b4", "#d62728"]
    ax.bar(labels, points, yerr=err, capsize=10, color=colors, alpha=0.85,
           edgecolor="black", linewidth=1.2)
    for i, (p, lo, hi) in enumerate(zip(points, ci_lo, ci_hi)):
        ax.text(i, p + (hi - p) + 4, f"{p:.2f}×\n[{lo:.1f}, {hi:.1f}]",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Variance ratio  SS$_{primary}$ / SS$_{helper}$", fontsize=11)
    ax.set_title("(A) WHO-asymmetry: LoRA vs Full FT  (1.7B, recovered grid)",
                 fontsize=12)
    ax.set_ylim(0, max(ci_hi) * 1.18)
    ax.grid(axis="y", alpha=0.3)

    # (B) Spread ratio: LoRA vs FT, side-by-side
    ax = axes[0, 1]
    points_sr = [
        lora["point"]["spread_ratio_§6b"],
        ft["point"]["spread_ratio_row_over_col"],
    ]
    ci_lo_sr = [lora_sr["p2.5"], ft_sr["p2.5"]]
    ci_hi_sr = [lora_sr["p97.5"], ft_sr["p97.5"]]
    err_sr = [
        [p - l for p, l in zip(points_sr, ci_lo_sr)],
        [h - p for p, h in zip(points_sr, ci_hi_sr)],
    ]
    ax.bar(labels, points_sr, yerr=err_sr, capsize=10, color=colors,
           alpha=0.85, edgecolor="black", linewidth=1.2)
    for i, (p, lo, hi) in enumerate(zip(points_sr, ci_lo_sr, ci_hi_sr)):
        ax.text(i, p + (hi - p) + 0.4, f"{p:.2f}×\n[{lo:.2f}, {hi:.2f}]",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Spread ratio  row-spread / col-spread", fontsize=11)
    ax.set_title("(B) Spread ratio (max-min row / max-min col)", fontsize=12)
    ax.set_ylim(0, max(ci_hi_sr) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.6, label="null (no asymmetry)")
    ax.legend(loc="upper left", fontsize=9)

    # (C) FT bootstrap histogram (variance ratio) on log x-axis
    ax = axes[1, 0]
    ax.hist(np.clip(boot_vr_ft, 1, 500), bins=50,
            color="#d62728", alpha=0.7, edgecolor="black")
    ax.axvline(ft["point"]["variance_ratio_ss_primary_over_ss_helper"],
               color="black", lw=2, label=f"point = {ft['point']['variance_ratio_ss_primary_over_ss_helper']:.2f}×")
    ax.axvline(ft_vr["p2.5"], color="darkblue", ls="--",
               label=f"2.5%ile = {ft_vr['p2.5']:.1f}×")
    ax.axvline(ft_vr["p97.5"], color="darkblue", ls="--",
               label=f"97.5%ile = {ft_vr['p97.5']:.1f}×")
    ax.set_xscale("log")
    ax.set_xlabel("Variance ratio (log scale)", fontsize=11)
    ax.set_ylabel("Bootstrap iterations", fontsize=11)
    ax.set_title("(C) FT recovered: bootstrap distribution of variance ratio",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # (D) P(ratio > k) bar chart, both grids
    ax = axes[1, 1]
    ks = [1, 2, 5, 10, 20, 30]
    p_ft = []
    p_lora = []
    for k in ks:
        p_ft.append(float((boot_vr_ft > k).mean()))
        # LoRA only has p_gt_1/2/3/5 stored, so recompute via spread_ratio
        # available keys. Use variance_ratio bootstrap stats stored.
        key = f"p_gt_{k}"
        p_lora.append(lora_vr.get(key, np.nan))
    x = np.arange(len(ks))
    w = 0.38
    ax.bar(x - w / 2, p_lora, w, label="1.7B LoRA", color="#1f77b4", alpha=0.85)
    ax.bar(x + w / 2, p_ft, w, label="1.7B Full FT (recovered)",
           color="#d62728", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f">{k}×" for k in ks])
    ax.set_ylabel("P(variance ratio > k)", fontsize=11)
    ax.set_xlabel("Threshold k", fontsize=11)
    ax.set_title("(D) Bootstrap probability of exceeding threshold k", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for xi, (pl, pf) in enumerate(zip(p_lora, p_ft)):
        if not np.isnan(pl):
            ax.text(xi - w / 2, pl + 0.02, f"{pl:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, pf + 0.02, f"{pf:.2f}", ha="center", fontsize=8)

    fig.suptitle(
        "Cluster-respecting bootstrap on 1.7B WHO ratio  —  "
        "Full FT (recovered cells_v3) vs LoRA  (2026-05-13)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
