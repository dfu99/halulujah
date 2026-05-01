"""Visualize biology HP sweep: per-rank delta vs base on 2 MMLU biology
subjects + PubMedQA-test held-out.

Output: figures/fig_biology_sweep_verification.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "results/specialist_verification/biology_sweep"
OUT = ROOT / "figures/fig_biology_sweep_verification.png"

RANKS = [8, 16, 32, 64]
BENCHES = ["college_biology", "high_school_biology", "PubMedQA-test"]


def get_base_acc(d, b):
    if b == "PubMedQA-test":
        return d["base"]["pubmedqa"]["accuracy"] * 100
    return d["base"]["subjects"][b]["accuracy"] * 100


def get_spec_acc(d, b):
    if b == "PubMedQA-test":
        return d["specialist"]["pubmedqa"]["accuracy"] * 100
    return d["specialist"]["subjects"][b]["accuracy"] * 100


deltas = np.zeros((len(BENCHES), len(RANKS)))
for ri, r in enumerate(RANKS):
    d = json.load(open(SRC_DIR / f"biology_r{r}.json"))
    for bi, b in enumerate(BENCHES):
        deltas[bi, ri] = get_spec_acc(d, b) - get_base_acc(d, b)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4),
                         gridspec_kw={"width_ratios": [3, 2]})

ax = axes[0]
vmax = max(abs(deltas.min()), abs(deltas.max()))
im = ax.imshow(deltas, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(RANKS)))
ax.set_xticklabels([f"r={r}" for r in RANKS], fontsize=10)
ax.set_yticks(range(len(BENCHES)))
ax.set_yticklabels([b.replace("_", " ") for b in BENCHES], fontsize=10)
for bi in range(len(BENCHES)):
    for ri in range(len(RANKS)):
        v = deltas[bi, ri]
        ax.text(ri, bi, f"{v:+.1f}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if abs(v) > 5 else "black")
ax.set_title("Delta vs base (pp), per (rank, benchmark)\n"
             "Trained on PubMedQA pqa_artificial (10K subsample), Qwen3-1.7B + LoRA",
             fontsize=10.5, fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.04, label="spec - base (pp)")

ax = axes[1]
pass_counts = []
for r in RANKS:
    d = json.load(open(SRC_DIR / f"biology_r{r}.json"))
    pass_counts.append(d["pass_count"])
n_total = len(BENCHES)
bars = ax.bar([f"r={r}" for r in RANKS], pass_counts,
              color=["#5d8aa8"] * 4,
              edgecolor="black", linewidth=0.6)
for b, c in zip(bars, pass_counts):
    ax.annotate(f"{c}/{n_total}", (b.get_x() + b.get_width() / 2,
                c + 0.04), ha="center", fontsize=11, fontweight="bold")
ax.axhline(1, color="red", linestyle="--", linewidth=1,
           label="Pass threshold (>=1 of 3)")
ax.set_ylim(0, n_total)
ax.set_ylabel("Benchmarks at spec >= base + 5 pp", fontsize=10)
ax.set_title("Verification gate per rank\n"
             "All 4 ranks PASS (1/3 each); flat across ranks.",
             fontsize=10.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("Biology specialist HP sweep on PubMedQA-train (10K subsample)\n"
             "Verified by MMLU bio subjects 5-shot + PubMedQA pqa_labeled held-out",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
print("\nSweep summary:")
for ri, r in enumerate(RANKS):
    d = json.load(open(SRC_DIR / f"biology_r{r}.json"))
    print(f"  r={r}: pass_count={d['pass_count']}/{n_total} verified={d['verified']}")
