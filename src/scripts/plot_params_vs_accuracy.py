"""
Compute-matched re-framing of the 1.7B-pair vs 4B-single question.

PI reframing: 2 * 1.7B = 3.4B active params < 1 * 4B = 4B active params.
So a 1.7B pair is 15% LESS memory than a single 4B. If it matches
accuracy, collaboration is the memory-conservative strategy.

X-axis: total resident parameters (GB in bf16).
Y-axis: medicine accuracy (%).
Shows the Pareto frontier and which configs beat/lose on accuracy-per-byte.

Output: figures/fig_params_vs_accuracy_medicine.png
"""
import json
import os

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWEEP = os.path.join(ROOT, "results/paper_sweep")
OUT = os.path.join(ROOT, "figures/fig_params_vs_accuracy_medicine.png")

d17 = json.load(open(f"{SWEEP}/full_ft_5domain/full_ft_5domain.json"))["conditions"]
d4b_ft = json.load(open(f"{SWEEP}/qwen3_4b_ft/4b_full_ft.json"))["conditions"]
d4b_lora = json.load(open(f"{SWEEP}/qwen3_4b/rank_sweep_rp.json"))["conditions"]


def acc(rows, **filt):
    for r in rows:
        if all(r.get(k) == v for k, v in filt.items()):
            return r["accuracy"] * 100
    raise KeyError(filt)


# (label, resident_params_gb_bf16, accuracy, size_group, is_pair)
configs = [
    ("1.7B solo FT", 3.4, acc(d17, id="solo_medicine"), "1.7B", False),
    ("1.7B base+base pair", 6.8, acc(d17, id="base_pair_medicine"), "1.7B pair", True),
    ("1.7B FT med + base", 6.8, acc(d17, id="ft_medicine_plus_base"), "1.7B pair", True),
    ("1.7B FT med + FT med", 6.8, acc(d17, id="same_medicine"), "1.7B pair", True),
    ("1.7B FT med + FT phys*", 6.8, acc(d17, id="ft_medicine_plus_ft_physics"), "1.7B pair", True),
    ("4B LoRA r=16 solo", 8.0, acc(d4b_lora, id="solo_medicine_r16"), "4B", False),
    ("4B LoRA r=128 solo", 8.0, acc(d4b_lora, id="solo_medicine_r128"), "4B", False),
    ("4B full FT solo", 8.0, acc(d4b_ft, id="solo_4b_medicine"), "4B", False),
    ("4B full FT + base", 8.0 + 3.4, acc(d4b_ft, id="ft_4b_medicine_plus_base"), "4B pair", True),
]

colors = {"1.7B": "#1565c0", "1.7B pair": "#42a5f5", "4B": "#2e7d32", "4B pair": "#66bb6a"}

fig, ax = plt.subplots(figsize=(10.5, 5.5))
for label, params, a, group, _ in configs:
    ax.scatter(params, a, s=160, color=colors[group], edgecolor="black",
               linewidth=0.8, zorder=3)
    # label with offset to avoid overlap
    offset = (0.2, 1.5) if a > 70 else (0.2, -3.5)
    ax.annotate(label, (params, a), xytext=(params + offset[0], a + offset[1]),
                fontsize=8.5, ha="left", va="center")

# Pareto-frontier (best accuracy at each param level)
pareto = []
seen_params = sorted(set(c[1] for c in configs))
for p in seen_params:
    best = max(c[2] for c in configs if c[1] == p)
    pareto.append((p, best))
px, py = zip(*pareto)
ax.plot(px, py, color="black", linestyle="--", alpha=0.3, label="Pareto frontier (acc @ params)")

# reference shading
ax.axvspan(6.7, 6.9, color="#bbdefb", alpha=0.35, zorder=0,
           label="1.7B pair footprint (~6.8 GB)")
ax.axvspan(7.9, 8.1, color="#c8e6c9", alpha=0.45, zorder=0,
           label="4B single footprint (~8 GB)")

ax.set_xlabel("Resident parameters (GB, bf16)  →  memory cost", fontsize=11)
ax.set_ylabel("Medicine accuracy (%)  →  task performance", fontsize=11)
ax.set_title(
    "Compute-matched re-framing: does 2×1.7B beat 1×4B on accuracy-per-byte?\n"
    "Best 1.7B pair (77.5%) at 6.8 GB loses by 6.5 pp to 4B solo (84%) at 8.0 GB.\n"
    "4B solo dominates the Pareto front. (* composite task, not pure medicine)",
    fontsize=10, fontweight="bold",
)
ax.set_xlim(2.5, 12.5)
ax.set_ylim(45, 95)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")

# print summary table
print("\nConfig                               params  acc  acc/GB")
for label, p, a, _, _ in configs:
    print(f"  {label:34s}  {p:4.1f}   {a:5.1f}  {a/p:5.1f}")
