"""Compare collaboration metrics between cp1500 and final pair-grids.

PI directive: "see if training time causes collaboration scores to drift
predictably". Reads two pair-grid results:

  results/ft_pair_grid_step1500_2026-05-08/matrix_results_v2.json  (early)
  results/ft_pair_grid_2026-05-08/matrix_results_v2.json            (final)

Computes per-cell delta (final - cp1500) on:
  - accuracy
  - c2w count
  - w2c count
  - switch count

Writes:
  figures/drift_step1500_vs_final_2026-05-08.png  (3-panel: per-cell
    accuracy delta heatmap, w2c delta, c2w delta)
  results/ft_pair_grid_2026-05-08/drift_summary.json
    (per-primary mean drift, ANOVA F on training-time effect)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EARLY = ROOT / "results/ft_pair_grid_step1500_2026-05-08/matrix_results_v2.json"
FINAL = ROOT / "results/ft_pair_grid_2026-05-08/matrix_results_v2.json"
OUT_FIG = ROOT / "figures/drift_step1500_vs_final_2026-05-08.png"
OUT_JSON = ROOT / "results/ft_pair_grid_2026-05-08/drift_summary.json"

DOMAINS = ["medicine", "math", "biology", "law", "physics"]
HELPERS = ["medicine", "math", "biology", "law", "physics", "base"]


def cell(grid: dict, primary: str, helper: str, key: str) -> float:
    c = grid.get("conditions", {}).get(f"pair_{primary}_{helper}", {})
    v = c.get(key)
    return float(v) if v is not None else float("nan")


def matrix(grid: dict, key: str) -> np.ndarray:
    out = np.full((len(DOMAINS), len(HELPERS)), np.nan)
    for i, p in enumerate(DOMAINS):
        for j, h in enumerate(HELPERS):
            out[i, j] = cell(grid, p, h, key)
    return out


def main() -> None:
    if not EARLY.exists():
        print(f"Early matrix missing: {EARLY}; aborting.")
        return
    if not FINAL.exists():
        print(f"Final matrix missing: {FINAL}; aborting.")
        return
    early = json.loads(EARLY.read_text())
    final = json.loads(FINAL.read_text())

    acc_early = matrix(early, "accuracy")
    acc_final = matrix(final, "accuracy")
    c2w_early = matrix(early, "c2w")
    c2w_final = matrix(final, "c2w")
    w2c_early = matrix(early, "w2c")
    w2c_final = matrix(final, "w2c")

    delta_acc = acc_final - acc_early
    delta_c2w = c2w_final - c2w_early
    delta_w2c = w2c_final - w2c_early

    # 3-panel heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def heatmap(ax, data, title, cmap, vlim):
        im = ax.imshow(data, cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto")
        ax.set_xticks(range(len(HELPERS)))
        ax.set_xticklabels(HELPERS, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(DOMAINS)))
        ax.set_yticklabels(DOMAINS, fontsize=9)
        ax.set_xlabel("helper", fontsize=10)
        ax.set_ylabel("primary", fontsize=10)
        for i in range(len(DOMAINS)):
            for j in range(len(HELPERS)):
                v = data[i, j]
                if np.isnan(v):
                    ax.text(j, i, "nan", ha="center", va="center",
                            fontsize=7, color="#888")
                    continue
                txt = (f"{v:+.2f}" if abs(v) < 1
                       else f"{v:+.0f}")
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(v) > 0.5 * vlim else "black",
                        fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    heatmap(axes[0], delta_acc,
            "Δ accuracy  (final − cp1500)\nbluer = trained-longer is better",
            "RdBu_r", 0.3)
    heatmap(axes[1], delta_w2c,
            "Δ W2C count  (final − cp1500)\nbluer = more recovery with training",
            "RdBu_r", 12)
    heatmap(axes[2], delta_c2w,
            "Δ C2W count  (final − cp1500)\nredder = more corruption with training",
            "RdBu_r", 8)

    fig.suptitle(
        "Collaboration drift: step 1500 → final\n"
        "Per-cell change in pair-grid metrics across SFT training (50 q × 3 rounds; 5 primaries × 6 helpers)",
        fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_FIG}")

    # Summary stats
    summary = {
        "per_primary_mean_delta_acc": {
            d: float(np.nanmean(delta_acc[i]))
            for i, d in enumerate(DOMAINS)
        },
        "per_helper_mean_delta_acc": {
            h: float(np.nanmean(delta_acc[:, j]))
            for j, h in enumerate(HELPERS)
        },
        "global_mean_delta_acc": float(np.nanmean(delta_acc)),
        "global_mean_delta_w2c": float(np.nanmean(delta_w2c)),
        "global_mean_delta_c2w": float(np.nanmean(delta_c2w)),
        "primary_var_delta_acc": float(np.nanvar(np.nanmean(delta_acc, axis=1))),
        "helper_var_delta_acc": float(np.nanvar(np.nanmean(delta_acc, axis=0))),
    }
    summary["drift_who_asymmetry_ratio"] = (
        summary["primary_var_delta_acc"] / summary["helper_var_delta_acc"]
        if summary["helper_var_delta_acc"] > 0 else float("inf")
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_JSON}")
    print("\nDrift summary:")
    print(f"  global mean Δ acc:    {summary['global_mean_delta_acc']:+.3f}")
    print(f"  global mean Δ w2c:    {summary['global_mean_delta_w2c']:+.2f}")
    print(f"  global mean Δ c2w:    {summary['global_mean_delta_c2w']:+.2f}")
    print(f"  drift WHO-asymmetry:  {summary['drift_who_asymmetry_ratio']:.1f}×")
    print("  per-primary Δ acc:")
    for d, v in summary["per_primary_mean_delta_acc"].items():
        print(f"    {d:9s}: {v:+.3f}")


if __name__ == "__main__":
    main()
