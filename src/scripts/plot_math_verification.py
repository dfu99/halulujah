"""Bar plot: Qwen2.5-Math-1.5B-Instruct (specialist) vs Qwen2.5-1.5B-Instruct (base)
across 4 MMLU math subjects + GSM8K-test, per-benchmark accuracy.

Output: figures/fig_math_specialist_verification.png
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/specialist_verification/qwen2_5_math_1_5b.json"
OUT = ROOT / "figures/fig_math_specialist_verification.png"

d = json.load(open(SRC))
spec, base = d["specialist"], d["base"]
labels, base_acc, spec_acc = [], [], []
for s, v in spec["subjects"].items():
    labels.append(s.replace("_", "\n"))
    base_acc.append(base["subjects"][s]["accuracy"] * 100)
    spec_acc.append(v["accuracy"] * 100)
labels.append("GSM8K\n-test")
base_acc.append(base["gsm8k"]["accuracy"] * 100)
spec_acc.append(spec["gsm8k"]["accuracy"] * 100)

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.bar(x - 0.18, base_acc, 0.34, label="Qwen2.5-1.5B-Instruct (base)",
       color="#888", edgecolor="black", linewidth=0.6)
ax.bar(x + 0.18, spec_acc, 0.34, label="Qwen2.5-Math-1.5B-Instruct (specialist)",
       color="#2e7d32", edgecolor="black", linewidth=0.6)
for i, (b, s) in enumerate(zip(base_acc, spec_acc)):
    delta = s - b
    color = "#2e7d32" if delta >= 5 else ("#f57c00" if delta >= 0 else "#c0392b")
    ax.annotate(f"{delta:+.1f}", (i, max(b, s) + 1.5), ha="center",
                fontsize=9, color=color, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("5-shot accuracy (%)", fontsize=10)
ax.set_ylim(0, max(spec_acc) + 12)
ax.set_title("Math specialist verification (Qwen2.5-Math-1.5B-Instruct)\n"
             "PASS gate: spec >= base + 5 pp on at least one benchmark. "
             "Result: 2/5 pass (abstract_algebra +14, GSM8K +15.5).",
             fontsize=10.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="black", linewidth=0.4)
plt.tight_layout()
os.makedirs(OUT.parent, exist_ok=True)
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
