"""Visualize medicine HP sweep: per-rank delta vs base on 6 MMLU medicine
subjects + MedQA-test held-out.

Output: figures/fig_medicine_sweep_verification.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "results/specialist_verification/medicine_sweep"
OUT = ROOT / "figures/fig_medicine_sweep_verification.png"

RANKS = [8, 16, 32, 64]
BENCHES = [
    "anatomy", "clinical_knowledge", "college_medicine",
    "medical_genetics", "professional_medicine", "virology",
    "MedQA-test",
]


def get_base_acc(d, b):
    if b == "MedQA-test":
        return d["base"]["medqa"]["accuracy"] * 100
    return d["base"]["subjects"][b]["accuracy"] * 100


def get_spec_acc(d, b):
    if b == "MedQA-test":
        return d["specialist"]["medqa"]["accuracy"] * 100
    return d["specialist"]["subjects"][b]["accuracy"] * 100


# Build delta matrix: rows = benches, cols = ranks
deltas = np.zeros((len(BENCHES), len(RANKS)))
base_accs = []
for ri, r in enumerate(RANKS):
    d = json.load(open(SRC_DIR / f"medicine_r{r}.json"))
    if ri == 0:
        base_accs = [get_base_acc(d, b) for b in BENCHES]
    for bi, b in enumerate(BENCHES):
        deltas[bi, ri] = get_spec_acc(d, b) - get_base_acc(d, b)

fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                         gridspec_kw={"width_ratios": [3, 2]})

# Left: delta heatmap (rows = benchmarks, cols = ranks)
ax = axes[0]
vmax = max(abs(deltas.min()), abs(deltas.max()))
im = ax.imshow(deltas, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(RANKS)))
ax.set_xticklabels([f"r={r}" for r in RANKS], fontsize=10)
ax.set_yticks(range(len(BENCHES)))
ax.set_yticklabels([b.replace("_", " ") for b in BENCHES], fontsize=9)
for bi in range(len(BENCHES)):
    for ri in range(len(RANKS)):
        v = deltas[bi, ri]
        ax.text(ri, bi, f"{v:+.1f}", ha="center", va="center",
                fontsize=9.5, fontweight="bold",
                color="white" if abs(v) > 5 else "black")
ax.set_title("Delta vs base (pp), per (rank, benchmark)\n"
             "Trained on MedQA-USMLE-4-options train, Qwen3-1.7B + LoRA",
             fontsize=10.5, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.04, label="spec - base (pp)")

# Right: pass count bar chart
ax = axes[1]
pass_counts = []
for r in RANKS:
    d = json.load(open(SRC_DIR / f"medicine_r{r}.json"))
    pass_counts.append(d["pass_count"])
n_total = len(BENCHES)
bars = ax.bar([f"r={r}" for r in RANKS], pass_counts,
              color=["#5d8aa8", "#5d8aa8", "#5d8aa8", "#2e7d32"],
              edgecolor="black", linewidth=0.6)
for b, c in zip(bars, pass_counts):
    ax.annotate(f"{c}/{n_total}", (b.get_x() + b.get_width() / 2,
                c + 0.06), ha="center", fontsize=11, fontweight="bold")
ax.axhline(1, color="red", linestyle="--", linewidth=1,
           label="Pass threshold (>=1 of 7)")
ax.set_ylim(0, n_total)
ax.set_ylabel("Benchmarks at spec >= base + 5 pp",
              fontsize=10)
ax.set_title("Verification gate per rank\n"
             "All 4 ranks PASS; r=64 strongest (3/7).",
             fontsize=10.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("Medicine specialist HP sweep on MedQA-USMLE-train\n"
             "Verified by per-MMLU-subject 5-shot + MedQA-USMLE-test held-out",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
print("\nSweep summary:")
for ri, r in enumerate(RANKS):
    d = json.load(open(SRC_DIR / f"medicine_r{r}.json"))
    print(f"  r={r}: pass_count={d['pass_count']}/{n_total} verified={d['verified']}")
