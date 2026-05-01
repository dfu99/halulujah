"""Bar plot: Qwen3-1.7B + LoRA r=16 on CaseHOLD-train (specialist) vs Qwen3-1.7B (base).
3 MMLU law subjects + CaseHOLD-test held-out.

Output: figures/fig_law_qwen3_verification.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/specialist_verification/law_qwen3/law_r16.json"
OUT = ROOT / "figures/fig_law_qwen3_verification.png"

d = json.load(open(SRC))
spec, base = d["specialist"], d["base"]
labels, base_acc, spec_acc = [], [], []
for s, v in spec["subjects"].items():
    labels.append(s.replace("_", "\n"))
    base_acc.append(base["subjects"][s]["accuracy"] * 100)
    spec_acc.append(v["accuracy"] * 100)
labels.append("CaseHOLD\n-test")
base_acc.append(base["casehold"]["accuracy"] * 100)
spec_acc.append(spec["casehold"]["accuracy"] * 100)

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(9.5, 5))
ax.bar(x - 0.18, base_acc, 0.34, label="Qwen3-1.7B (base)",
       color="#888", edgecolor="black", linewidth=0.6)
ax.bar(x + 0.18, spec_acc, 0.34, label="Qwen3-1.7B + LoRA r=16 on CaseHOLD-train",
       color="#5d3a8a", edgecolor="black", linewidth=0.6)
for i, (b, s) in enumerate(zip(base_acc, spec_acc)):
    delta = s - b
    color = "#2e7d32" if delta >= 5 else ("#f57c00" if delta >= 0 else "#c0392b")
    ax.annotate(f"{delta:+.1f}", (i, max(b, s) + 1.5), ha="center",
                fontsize=10, color=color, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("5-shot accuracy (%)", fontsize=10)
ax.set_ylim(0, max(spec_acc + base_acc) + 14)
ax.set_title("Qwen3-1.7B law specialist verification (LoRA r=16 on CaseHOLD-train, 3 epochs)\n"
             "PASS gate: +24 pp on CaseHOLD-test (training-aligned). MASSIVE LOSS on every MMLU law subject\n"
             "(jurisprudence -25, intl_law -13, prof_law -7). Textbook LoRA overfit to training distribution.",
             fontsize=10, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.4)
plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
