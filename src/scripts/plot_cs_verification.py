"""Bar plot: Qwen2.5-Coder-1.5B-Instruct (specialist) vs Qwen2.5-1.5B-Instruct (base)
across 4 MMLU CS subjects.

Output: figures/fig_cs_specialist_verification.png
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/specialist_verification/qwen2_5_coder_1_5b.json"
OUT = ROOT / "figures/fig_cs_specialist_verification.png"

d = json.load(open(SRC))
spec, base = d["specialist"], d["base"]
labels, base_acc, spec_acc = [], [], []
for s, v in spec["subjects"].items():
    labels.append(s.replace("_", "\n"))
    base_acc.append(base["subjects"][s]["accuracy"] * 100)
    spec_acc.append(v["accuracy"] * 100)

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.bar(x - 0.18, base_acc, 0.34, label="Qwen2.5-1.5B-Instruct (base)",
       color="#888", edgecolor="black", linewidth=0.6)
ax.bar(x + 0.18, spec_acc, 0.34, label="Qwen2.5-Coder-1.5B-Instruct (specialist)",
       color="#c0392b", edgecolor="black", linewidth=0.6)
for i, (b, s) in enumerate(zip(base_acc, spec_acc)):
    delta = s - b
    color = "#2e7d32" if delta >= 5 else ("#f57c00" if delta >= 0 else "#c0392b")
    ax.annotate(f"{delta:+.1f}", (i, max(b, s) + 1.5), ha="center",
                fontsize=9.5, color=color, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("5-shot accuracy (%)", fontsize=10)
ax.set_ylim(0, max(base_acc + spec_acc) + 12)
ax.set_title("CS specialist verification: code-instruction tuning REDUCES MMLU-CS knowledge\n"
             "Qwen2.5-Coder vs Qwen2.5-Instruct base. Result: 0/4 pass; mean delta -4.5 pp.\n"
             "Domain-specialist != task-specialist for MCQ knowledge benchmarks.",
             fontsize=10, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.4)
plt.tight_layout()
os.makedirs(OUT.parent, exist_ok=True)
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
