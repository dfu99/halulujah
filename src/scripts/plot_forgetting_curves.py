"""Plot per-domain accuracy across training checkpoints.

Five panels (one per primary specialist domain). Each panel shows the
trajectory of MMLU-5shot accuracy on all 5 benchmark domains as the
specialist trains. Includes the base-model baseline and a final-marker.

Reads results/full_ft_streaming/mmlu_5shot/scan.json
Writes figures/forgetting_curve_full_ft_2026-05-08.png
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "results/full_ft_streaming/mmlu_5shot/scan.json"
OUT = ROOT / "figures/forgetting_curve_full_ft_2026-05-08.png"

DOMAINS = ["medicine", "math", "biology", "law", "physics"]
COLOR = {
    "math": "#1f77b4",
    "medicine": "#ff7f0e",
    "biology": "#2ca02c",
    "law": "#d62728",
    "physics": "#9467bd",
}
FINAL_STEP = {  # last per-step checkpoint we saw
    "medicine": 7500, "math": 5604, "biology": 7500,
    "law": 7500, "physics": 7500,
}


def parse_step(name: str, domain: str) -> int | None:
    m = re.match(rf"{domain}-step(\d+)", name)
    if m:
        return int(m.group(1))
    if name == f"{domain}-final":
        return FINAL_STEP[domain] + 200  # plot finals slightly past last step
    return None


def main() -> None:
    data = json.loads(SCAN.read_text())
    per = data["per_ckpt"]
    base_acc = per["base"]["domains"]
    base_mean = per["base"]["mean_acc"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "MMLU 5-shot per-checkpoint forgetting curves "
        "(Full FT Qwen3-1.7B, 5 domain specialists)\n"
        "Each panel: one specialist's training trajectory across all 5 "
        "benchmark domains. Dashed = base baseline.",
        fontsize=12, fontweight="bold", y=0.985,
    )

    for idx, primary in enumerate(DOMAINS):
        ax = axes[idx // 3, idx % 3]
        # Collect per-step trajectory for this primary
        traj = defaultdict(list)  # bench_domain -> list of (step, acc)
        for name, v in per.items():
            step = parse_step(name, primary)
            if step is None:
                continue
            for bench in DOMAINS:
                if bench in v["domains"]:
                    traj[bench].append((step, v["domains"][bench]))
        for bench in DOMAINS:
            pts = sorted(traj[bench])
            if not pts:
                continue
            xs = [0] + [p[0] for p in pts]   # prepend step=0 with base acc
            ys = [base_acc[bench]] + [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", color=COLOR[bench],
                    label=bench, lw=2,
                    markeredgecolor="black", markeredgewidth=0.5)
            # Highlight the final
            final_x = FINAL_STEP[primary] + 200
            for x, y in pts:
                if x == final_x:
                    ax.scatter([x], [y], marker="*", s=180,
                               color=COLOR[bench], edgecolor="black",
                               lw=0.7, zorder=5)
        # Base mean line
        ax.axhline(base_mean, color="gray", linestyle=":", lw=1, alpha=0.5)
        ax.text(0, base_mean + 0.005, f"base mean={base_mean:.2f}",
                fontsize=7, color="gray")
        # In-domain shaded marker
        ax.axhline(base_acc[primary], color=COLOR[primary],
                   linestyle="--", lw=0.7, alpha=0.5)
        ax.set_title(
            f"{primary} specialist (* = final)",
            fontsize=11, fontweight="bold",
            color=COLOR[primary],
        )
        ax.set_xlabel("training step", fontsize=9)
        ax.set_ylabel("MMLU 5-shot accuracy", fontsize=9)
        ax.set_ylim(0.30, 0.85)
        ax.grid(linestyle=":", alpha=0.3)
        ax.legend(fontsize=7, loc="lower left", ncol=2)

    # Last panel: summary heatmap of (specialist final - base) per bench
    ax = axes[1, 2]
    deltas = []
    for primary in DOMAINS:
        row = []
        final_key = f"{primary}-final"
        if final_key not in per:
            row = [0.0] * 5
        else:
            for bench in DOMAINS:
                d = per[final_key]["domains"].get(bench, 0)
                row.append(d - base_acc[bench])
        deltas.append(row)
    im = ax.imshow(deltas, cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(DOMAINS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"{d}-final" for d in DOMAINS], fontsize=9)
    ax.set_xlabel("benchmark domain", fontsize=9)
    ax.set_ylabel("specialist", fontsize=9)
    for i in range(5):
        for j in range(5):
            v = deltas[i][j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(v) > 0.10 else "black",
                    fontweight="bold")
    ax.set_title("Δ accuracy vs base (final - base)",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
