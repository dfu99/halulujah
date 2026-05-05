"""Visualize the A4500 -> A40 pod transition: GPU memory budget for LoRA
vs Full FT on Qwen3-1.7B.

Justifies why the A40 48 GB switch was needed before launching Full FT.

Output: figures/fig_a40_bootstrap_memory_budget.png
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures/fig_a40_bootstrap_memory_budget.png"

# Memory components in GB (estimated for Qwen3-1.7B = 1.72B params)
# bf16 model weights:    1.72e9 * 2 / 1e9 = 3.44 GB
# fp32 Adam state (m+v): 1.72e9 * 8 / 1e9 = 13.76 GB
# fp32 gradients:        1.72e9 * 4 / 1e9 = 6.88 GB
# bf16 activations + grad checkpoint: ~5 GB (estimated)
# LoRA r=64 trainable params + Adam state (fp32): ~0.05 + ~0.4 = ~0.5 GB
# LoRA r=16: even smaller, ~0.15 GB

scenarios = [
    {
        "name": "LoRA r=16\n(A4500)",
        "components": [
            ("model bf16", 3.44, "#5d8aa8"),
            ("LoRA params + Adam", 0.15, "#fbc02d"),
            ("activations + checkpoint", 5.0, "#f57c00"),
        ],
        "pod_cap": 20.0,
        "pod_label": "A4500 20 GB",
    },
    {
        "name": "LoRA r=64\n(A4500)",
        "components": [
            ("model bf16", 3.44, "#5d8aa8"),
            ("LoRA params + Adam", 0.50, "#fbc02d"),
            ("activations + checkpoint", 5.0, "#f57c00"),
        ],
        "pod_cap": 20.0,
        "pod_label": "A4500 20 GB",
    },
    {
        "name": "Full FT bf16+8bit-Adam\n(A4500, compromised)",
        "components": [
            ("model bf16", 3.44, "#5d8aa8"),
            ("Adam state 8-bit", 3.44, "#fbc02d"),
            ("gradients fp32", 6.88, "#c0392b"),
            ("activations", 5.0, "#f57c00"),
        ],
        "pod_cap": 20.0,
        "pod_label": "A4500 20 GB",
    },
    {
        "name": "Full FT clean\n(A40, no quantization)",
        "components": [
            ("model bf16", 3.44, "#5d8aa8"),
            ("Adam state fp32", 13.76, "#2e7d32"),
            ("gradients fp32", 6.88, "#c0392b"),
            ("activations + checkpoint", 5.0, "#f57c00"),
        ],
        "pod_cap": 48.0,
        "pod_label": "A40 48 GB",
    },
]

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(scenarios))
width = 0.6

# Plot stacked bars
seen_labels = set()
for i, s in enumerate(scenarios):
    bottom = 0
    for label, h, color in s["components"]:
        bar_label = label if label not in seen_labels else None
        if bar_label:
            seen_labels.add(label)
        ax.bar(i, h, width, bottom=bottom, color=color,
               edgecolor="black", linewidth=0.5, label=bar_label)
        if h > 0.5:
            ax.text(i, bottom + h / 2, label, ha="center", va="center",
                    fontsize=8, color="white"
                    if color in ["#5d8aa8", "#c0392b", "#2e7d32"] else "black")
        bottom += h

    total = sum(c[1] for c in s["components"])
    fits = total <= s["pod_cap"]
    cap = s["pod_cap"]
    ax.hlines(cap, i - width / 2, i + width / 2,
              color="red", linestyle="--", linewidth=2)
    ax.text(i, cap + 0.6, s["pod_label"], ha="center", fontsize=9,
            color="red", fontweight="bold")
    status = "FITS" if fits else "OOM"
    status_color = "#2e7d32" if fits else "#c0392b"
    ax.text(i, total + 1.0, f"{total:.1f} GB total\n[{status}]",
            ha="center", fontsize=9.5, color=status_color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([s["name"] for s in scenarios], fontsize=10)
ax.set_ylabel("GPU memory (GB)", fontsize=10.5)
ax.set_ylim(0, 55)
ax.set_title("Why A40 was needed: Qwen3-1.7B memory budget under 4 training scenarios\n"
             "Full FT on A4500 requires quantization compromises; A40 fits clean Full FT with headroom",
             fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
