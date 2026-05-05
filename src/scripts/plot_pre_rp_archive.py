"""Visualize the 2026-04-27 empty-think-tag audit that triggered the
pre-RP archive.

Two panels:
  (left) Collaboration outcome breakdown across the 1800 PACE-domain-10
    trials: 61.2% failed, dominated by confident_wrong + extraction_failure
    + answer_switch modes.
  (right) Per-domain failure counts.

Annotates: the chains had EMPTY <think></think> blocks - the agents were
not deliberating. Headline 22.3x cluster-corrected variance ratio computed
from these chains is suspended.

Output: figures/fig_empty_think_audit.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "archive/pre_RP_polluted_2026-04-27/pace_domain_10/chain_analysis.json"
OUT = ROOT / "figures/fig_empty_think_audit.png"

d = json.load(open(SRC))

fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                         gridspec_kw={"width_ratios": [1.0, 1.5]})

# ---- LEFT: outcome breakdown
ax = axes[0]
total = d["total_collab"]
n_fail = d["total_failed"]
n_ok = d["total_succeeded"]

bars = []
labels = []
colors = []
for k, v in d["failure_modes"].items():
    bars.append(v)
    labels.append(f"FAIL\n{k.replace('_', ' ')}")
    colors.append("#c0392b")
for k, v in d["success_modes"].items():
    bars.append(v)
    labels.append(f"OK\n{k}")
    colors.append("#2e7d32")

x = np.arange(len(bars))
ax.bar(x, bars, color=colors, edgecolor="black", linewidth=0.5)
for xi, v in zip(x, bars):
    ax.annotate(f"{v}\n({v/total*100:.1f}%)",
                (xi, v + 8), ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Trials (of 1800)", fontsize=10)
ax.set_title(f"Collaboration outcomes across 1800 trials\n"
             f"61.2% FAIL  /  38.8% OK  - chains had empty think blocks",
             fontsize=10.5, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.4)

# ---- RIGHT: per-domain failure breakdown
ax = axes[1]
domains = list(d["failure_by_domain"].keys())
modes_keys = ["confident_wrong", "extraction_failure", "answer_switch"]
mode_colors = ["#c0392b", "#f57c00", "#fbc02d"]

stacked = {m: [] for m in modes_keys}
for dom in domains:
    for m in modes_keys:
        stacked[m].append(d["failure_by_domain"][dom].get(m, 0))

x = np.arange(len(domains))
bottom = np.zeros(len(domains))
for m, color in zip(modes_keys, mode_colors):
    vals = np.array(stacked[m])
    ax.bar(x, vals, bottom=bottom, color=color, edgecolor="black",
           linewidth=0.5, label=m.replace("_", " "))
    for xi, v, bot in zip(x, vals, bottom):
        if v >= 10:
            ax.text(xi, bot + v / 2, str(int(v)),
                    ha="center", va="center", fontsize=8.5,
                    color="white" if m == "confident_wrong" else "black")
    bottom += vals

for xi, dom in enumerate(domains):
    total_fail = bottom[xi]
    ax.text(xi, total_fail + 6, f"{int(total_fail)}",
            ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=9, rotation=30, ha="right")
ax.set_ylabel("Failed collab trials", fontsize=10)
ax.set_title("Per-domain failure breakdown\n"
             "medicine + chemistry dominate via extraction_failure (empty answers)",
             fontsize=10.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.4)

plt.suptitle(
    "2026-04-27 audit: empty <think></think> chains across the PACE 10-domain run\n"
    "1101/1800 (61.2%) trials failed. 22.3x primary/helper variance ratio "
    "ARCHIVED pending RP-format regeneration.",
    fontsize=12.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
