"""
Does any 1.7B combination beat a single 4B model on its own domain?

Answer: *No, not on matched-domain* — but 1.7B Full-FT specialists are
surprisingly close on their own domain, and 1.7B Full-FT biology solo
(87.5%) actually exceeds 4B Full-FT medicine solo (84%) on its own task.

Data (N=200 per condition) from results/paper_sweep/:
  - 1.7B: full_ft_5domain/full_ft_5domain.json
  - 4B Full FT: qwen3_4b_ft/4b_full_ft.json
  - 4B LoRA (r=16, r=128): qwen3_4b/rank_sweep_rp.json

Output: figures/fig_1p7b_vs_4b_medicine.png
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWEEP = os.path.join(ROOT, "results/paper_sweep")
OUT = os.path.join(ROOT, "figures/fig_1p7b_vs_4b_medicine.png")

d17 = json.load(open(f"{SWEEP}/full_ft_5domain/full_ft_5domain.json"))["conditions"]
d4b_ft = json.load(open(f"{SWEEP}/qwen3_4b_ft/4b_full_ft.json"))["conditions"]
d4b_lora = json.load(open(f"{SWEEP}/qwen3_4b/rank_sweep_rp.json"))["conditions"]


def acc(rows, **filt):
    for r in rows:
        if all(r.get(k) == v for k, v in filt.items()):
            return r["accuracy"] * 100
    raise KeyError(filt)


# medicine-only comparison (head-to-head on the same task)
conditions = [
    ("1.7B solo\n(full FT)", acc(d17, id="solo_medicine"), "#1565c0"),
    ("1.7B base+base\npair", acc(d17, id="base_pair_medicine"), "#1565c0"),
    ("1.7B FT med +\nbase (pair)", acc(d17, id="ft_medicine_plus_base"), "#1976d2"),
    ("1.7B FT med +\nFT med (same)", acc(d17, id="same_medicine"), "#1976d2"),
    ("1.7B FT med +\nFT phys (cross)", acc(d17, id="ft_medicine_plus_ft_physics"), "#1976d2"),
    ("4B LoRA r=16\nsolo", acc(d4b_lora, id="solo_medicine_r16"), "#e67e22"),
    ("4B LoRA r=128\nsolo", acc(d4b_lora, id="solo_medicine_r128"), "#e67e22"),
    ("4B full FT\nsolo", acc(d4b_ft, id="solo_4b_medicine"), "#2e7d32"),
    ("4B full FT +\nbase (pair)", acc(d4b_ft, id="ft_4b_medicine_plus_base"), "#2e7d32"),
]

labels = [c[0] for c in conditions]
values = [c[1] for c in conditions]
colors = [c[2] for c in conditions]

fig, ax = plt.subplots(figsize=(12, 4.8))
x = np.arange(len(labels))
bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.7)
for bar, v in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        v + 0.8,
        f"{v:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# reference lines
best_17b = max(values[:5])
best_4b_ft = max(values[5:])
ax.axhline(
    best_17b, color="#1976d2", linestyle="--", alpha=0.5, linewidth=1,
    label=f"Best 1.7B combo: {best_17b:.1f}%",
)
ax.axhline(
    best_4b_ft, color="#2e7d32", linestyle="--", alpha=0.5, linewidth=1,
    label=f"Best 4B single: {best_4b_ft:.1f}%",
)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.set_title(
    "Does any 1.7B combination beat a single 4B model? — medicine, N=200\n"
    "Answer: No. Best 1.7B pair (FT med + FT phys, 77.5%) still loses by "
    f"{best_4b_ft - best_17b:.1f} pp to best 4B single (full FT pair).",
    fontsize=10,
    fontweight="bold",
)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# group annotations beneath x-axis
ax.annotate(
    "1.7B configurations",
    xy=(2, -10),
    xycoords=("data", "axes points"),
    ha="center",
    fontsize=10,
    fontweight="bold",
    color="#1565c0",
)
ax.annotate(
    "4B configurations",
    xy=(7, -10),
    xycoords=("data", "axes points"),
    ha="center",
    fontsize=10,
    fontweight="bold",
    color="#2e7d32",
)

plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
print("\nSummary table:")
for lab, v, _ in conditions:
    print(f"  {lab.replace(chr(10), ' '):40s} {v:5.1f}%")
print(f"\nBest 1.7B combo: {best_17b:.1f}%")
print(f"Best 4B single:  {best_4b_ft:.1f}%")
print(f"Gap: {best_4b_ft - best_17b:.1f} pp  →  1.7B DOES NOT beat 4B on matched domain.")
