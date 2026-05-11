"""Plot 4B Full FT verification gate (specialist - base delta, 5 domains).

Reads results/full_ft_4b_streaming/mmlu_5shot/scan.json and produces a heatmap
of (specialist row, eval-domain col) deltas vs. the base model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DOMAINS = ["medicine", "math", "biology", "law", "physics"]
SCAN_PATH = Path("results/full_ft_4b_streaming/mmlu_5shot/scan.json")
OUT_PATH = Path("figures/4b_verification_gate_2026-05-10.png")


def main() -> None:
    data = json.loads(SCAN_PATH.read_text())
    per = data["per_ckpt"]
    base = per["base"]["domains"]
    rows = [f"{d}-final" for d in DOMAINS]
    delta = np.full((len(rows), len(DOMAINS)), np.nan)
    abs_acc = np.full((len(rows), len(DOMAINS)), np.nan)
    for i, name in enumerate(rows):
        if name not in per:
            continue
        d = per[name]["domains"]
        for j, ed in enumerate(DOMAINS):
            if ed in d:
                abs_acc[i, j] = d[ed]
                delta[i, j] = d[ed] - base[ed]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    im0 = axes[0].imshow(abs_acc, vmin=0.35, vmax=0.80, cmap="viridis",
                         aspect="auto")
    axes[0].set_title("4B Full FT (cp-1500 for bio) — abs 5-shot accuracy")
    axes[0].set_xticks(range(len(DOMAINS)))
    axes[0].set_xticklabels(DOMAINS, rotation=30, ha="right")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(DOMAINS)):
            v = abs_acc[i, j]
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=9, color="white" if v < 0.55 else "black")
    plt.colorbar(im0, ax=axes[0], shrink=0.8, label="accuracy")

    im1 = axes[1].imshow(delta, vmin=-0.10, vmax=0.10, cmap="RdBu_r",
                         aspect="auto")
    axes[1].set_title("Delta vs Qwen3-4B base (specialist − base)")
    axes[1].set_xticks(range(len(DOMAINS)))
    axes[1].set_xticklabels(DOMAINS, rotation=30, ha="right")
    axes[1].set_yticks(range(len(rows)))
    axes[1].set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(DOMAINS)):
            v = delta[i, j]
            txt = f"{v:+.2f}"
            color = "white" if abs(v) > 0.06 else "black"
            star = "*" if v >= 0.05 else ""
            axes[1].text(j, i, txt + star, ha="center", va="center",
                         fontsize=9, color=color)
    plt.colorbar(im1, ax=axes[1], shrink=0.8, label="delta")

    # Gate annotation
    gate_pass = [int((delta[i] >= 0.05).sum()) for i in range(len(rows))]
    pass_str = ", ".join(
        f"{r.replace('-final', '')}: {g}/5"
        for r, g in zip(rows, gate_pass)
    )
    fig.suptitle(
        f"Verification gate (specialist >= base+5pp on >=1 OOD)  |  pass: "
        f"{pass_str}",
        fontsize=11)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
