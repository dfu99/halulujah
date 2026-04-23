"""
Figure 1 candidate for the ACL 2026 paper.

Two panels at 4B Qwen3-medicine (solo accuracy 84% for all three conditions):
  (A) Collaboration delta (pp) for LoRA r=16, LoRA r=128, Full FT
  (B) C2W/W2C switch-quality ratio for the same three conditions.

This is the "money shot" of the paper: at matched solo accuracy, rank alone
does not recover full FT's collaborativeness. Full FT gets a larger delta AND
a vastly better switch-quality ratio.

Inputs: results/paper_sweep/qwen3_4b/rank_sweep_rp.json,
        results/paper_sweep/qwen3_4b_ft/4b_full_ft.json

Output: figures/figure1_matched_solo_4b_medicine.png
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWEEP = os.path.join(ROOT, "results/paper_sweep")
OUT = os.path.join(ROOT, "figures/figure1_matched_solo_4b_medicine.png")

lora = json.load(open(f"{SWEEP}/qwen3_4b/rank_sweep_rp.json"))
ft = json.load(open(f"{SWEEP}/qwen3_4b_ft/4b_full_ft.json"))


def get(d, domain, rank=None):
    # find collab condition (has delta) for given domain/rank
    for c in d["conditions"]:
        if c.get("domain") != domain:
            continue
        if rank is not None and c.get("rank") != rank:
            continue
        if "delta" in c:
            return c
    raise KeyError(f"no collab row for {domain} rank={rank}")


lora16 = get(lora, "medicine", 16)
lora128 = get(lora, "medicine", 128)
full_ft = get(ft, "medicine")

conditions = ["LoRA r=16", "LoRA r=128", "Full FT"]
rows = [lora16, lora128, full_ft]
deltas_pp = [r["delta"] * 100 for r in rows]
c2w = [r["c2w"] for r in rows]
w2c = [r["w2c"] for r in rows]
ratios = [c / max(w, 1) for c, w in zip(c2w, w2c)]

colors = ["#c0392b", "#e67e22", "#2e7d32"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# --- Panel A: collaboration delta
ax = axes[0]
bars = ax.bar(conditions, deltas_pp, color=colors, edgecolor="black", linewidth=0.8)
for bar, v in zip(bars, deltas_pp):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        v + 0.15,
        f"+{v:.1f} pp" if v >= 0 else f"{v:.1f} pp",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("Collaboration delta (pp)", fontsize=11)
ax.set_title(
    "(A)  Collaboration gain at matched solo accuracy\n"
    "Qwen3-4B medicine specialist + base partner, N=200, solo=84%",
    fontsize=10,
)
ax.set_ylim(-2.5, max(deltas_pp) + 1.8)
ax.grid(axis="y", alpha=0.3)

# --- Panel B: C2W/W2C switch ratio (log scale)
ax = axes[1]
bars = ax.bar(conditions, ratios, color=colors, edgecolor="black", linewidth=0.8)
for bar, r, cw, wc in zip(bars, ratios, c2w, w2c):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        r * 1.05,
        f"{r:.1f}×\n({cw}/{wc})",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
ax.axhline(1, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
ax.text(
    -0.35,
    1.05,
    "1× = balanced switches",
    fontsize=8,
    style="italic",
    color="gray",
)
ax.set_ylabel("C2W / W2C ratio  (↓ is better)", fontsize=11)
ax.set_yscale("log")
ax.set_title(
    "(B)  Switch-quality ratio (C2W / W2C)\n"
    "Higher = specialist abandons correct answers more than it recovers",
    fontsize=10,
)
ax.set_ylim(0.6, 40)
ax.grid(axis="y", alpha=0.3, which="both")

fig.suptitle(
    "Figure 1.  At matched solo accuracy, rank does not recover full-FT's "
    "collaborativeness.\n"
    "Full FT gets 3.3× larger delta AND 13× better switch quality vs. LoRA r=128.",
    fontsize=11,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"deltas: {dict(zip(conditions, deltas_pp))}")
print(f"ratios: {dict(zip(conditions, ratios))}")
