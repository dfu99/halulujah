"""Roster: all 5 verified Qwen3-1.7B LoRA specialists at a glance.

Per-specialist deltas vs base on the specialist's eval suite.
Output: figures/fig_specialist_roster_qwen3_1p7b.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_deltas(json_path, mmlu_extra_keys=()):
    """Return list of (label, base, spec, delta) tuples in order."""
    d = json.load(open(json_path))
    out = []
    for s, v in d["specialist"]["subjects"].items():
        b = d["base"]["subjects"][s]["accuracy"] * 100
        sp = v["accuracy"] * 100
        out.append((s.replace("_", " "), b, sp, sp - b))
    for k, label in mmlu_extra_keys:
        if d["specialist"].get(k):
            b = d["base"][k]["accuracy"] * 100
            sp = d["specialist"][k]["accuracy"] * 100
            out.append((label, b, sp, sp - b))
    return out


SPECIALISTS = [
    ("math\n(GSM8K -> Qwen3+LoRA r=16)",
     ROOT / "results/specialist_verification/math_qwen3/math_r16.json",
     [("gsm8k", "GSM8K-test")]),
    ("medicine\n(MedQA -> Qwen3+LoRA r=64)",
     ROOT / "results/specialist_verification/medicine_sweep/medicine_r64.json",
     [("medqa", "MedQA-test")]),
    ("biology\n(PubMedQA -> Qwen3+LoRA r=16)",
     ROOT / "results/specialist_verification/biology_sweep/biology_r16.json",
     [("pubmedqa", "PubMedQA-test")]),
    ("law\n(CaseHOLD -> Qwen3+LoRA r=16)",
     ROOT / "results/specialist_verification/law_qwen3/law_r16.json",
     [("casehold", "CaseHOLD-test")]),
    ("physics\n(SciQ -> Qwen3+LoRA r=16)",
     ROOT / "results/specialist_verification/physics_qwen3/physics_r16.json",
     [("sciq", "SciQ-test")]),
]

# Build figure: 5 rows of horizontal-bar per-bench delta
fig, axes = plt.subplots(5, 1, figsize=(11.5, 13),
                         gridspec_kw={"hspace": 0.7})

for ax, (title, src, extras) in zip(axes, SPECIALISTS):
    rows = load_deltas(src, extras)
    labels = [r[0] for r in rows]
    deltas = [r[3] for r in rows]
    colors = ["#2e7d32" if d >= 5 else
              ("#fbc02d" if d >= 0 else "#c0392b") for d in deltas]
    y = np.arange(len(rows))
    ax.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    for yi, d in enumerate(deltas):
        ax.annotate(f"{d:+.1f}",
                    (d + (0.6 if d >= 0 else -0.6), yi),
                    va="center",
                    ha="left" if d >= 0 else "right",
                    fontsize=9, fontweight="bold")
    ax.axvline(5, color="#2e7d32", linestyle="--", linewidth=0.7,
               label="+5 pp pass gate")
    ax.axvline(0, color="black", linewidth=0.5)
    pass_count = sum(1 for d in deltas if d >= 5)
    n_total = len(deltas)
    ax.set_title(f"{title}    PASS {pass_count}/{n_total}",
                 fontsize=10.5, fontweight="bold", loc="left")
    ax.set_xlim(min(min(deltas) - 4, -10), max(max(deltas) + 4, 10))
    ax.set_xlabel("Specialist - base (pp)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

plt.suptitle("Five verified Qwen3-1.7B LoRA specialists, per-benchmark delta vs base\n"
             "All pass the verification gate (>=1 of N benchmarks at +5 pp)",
             fontsize=12.5, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.985])
out = ROOT / "figures/fig_specialist_roster_qwen3_1p7b.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}")
