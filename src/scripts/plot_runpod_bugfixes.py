#!/usr/bin/env python3
"""Visualize the 3 RunPod compatibility bugs and their fixes (2026-03-31).

Diagnostic chart showing each bug's symptom, fix, and impact on time-to-working-pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(13, 7))
fig.suptitle(
    "2026-03-31: Three RunPod Compatibility Bugs Blocked Pipeline Launch\n"
    "Fixed across 3 commits over 6.5 hours before first successful run",
    fontsize=12, fontweight="bold", y=0.98,
)

# Timeline: 01:05 → 07:33 (fix commits), then ~08:00 successful run
bugs = [
    {
        "time": 0.0, "end": 0.7, "label": "Bug 1: SFTConfig",
        "sha": "14c12de", "time_str": "01:05",
        "symptom": "TypeError: unexpected\nkwarg 'overwrite_output_dir'",
        "cause": "Newer TRL (0.12+) removed\noverwrite_output_dir from SFTConfig",
        "fix": "Removed the kwarg\nfrom run_domain_experiment.py",
        "file": "run_domain_experiment.py",
        "color": "#E53935",
    },
    {
        "time": 0.72, "end": 1.4, "label": "Bug 2: DOMAINS ref",
        "sha": "6751ed8", "time_str": "01:48",
        "symptom": "NameError: name 'DOMAINS'\nis not defined",
        "cause": "Incomplete refactor from global\nDOMAINS to get_domains(args)",
        "fix": "Replaced 6 remaining refs in\nphase_evaluate",
        "file": "run_domain_experiment.py",
        "color": "#FB8C00",
    },
    {
        "time": 6.4, "end": 7.1, "label": "Bug 3: BatchEncoding",
        "sha": "8e30370", "time_str": "07:33",
        "symptom": "AttributeError: 'BatchEncoding'\nobject has no attribute 'shape'",
        "cause": "Newer transformers returns\nBatchEncoding not tensor",
        "fix": "Extract .input_ids before .shape\nin cross_eval.py",
        "file": "cross_eval.py",
        "color": "#8E24AA",
    },
]

ax.set_xlim(-0.5, 9)
ax.set_ylim(-3.5, 4)
ax.axis("off")

# Timeline bar
ax.annotate("", xy=(8.5, 0), xytext=(-0.3, 0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
for h in [0, 2, 4, 6, 8]:
    ax.text(h, -0.3, f"{h}h", ha="center", fontsize=8, color="gray")
ax.text(4, 0.3, "Time since first failure attempt", ha="center", fontsize=9, color="gray")

# Bugs
for i, b in enumerate(bugs):
    x = b["time"]
    # Bug marker
    ax.scatter(x, 0, s=200, color=b["color"], zorder=5, edgecolor="black", linewidth=1)
    # Box above: symptom
    box_x = x - 0.8 if i < 2 else x - 1.3
    box_y = 1.0
    ax.text(x, box_y + 2.3, b["label"], ha="center", fontweight="bold", fontsize=10,
            color=b["color"])
    ax.text(x, box_y + 1.9, f"[{b['time_str']}] {b['sha']}", ha="center", fontsize=8,
            color="gray", style="italic")
    ax.text(x, box_y + 1.3, "Symptom:", ha="center", fontsize=8, fontweight="bold")
    ax.text(x, box_y + 0.7, b["symptom"], ha="center", fontsize=7, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor=b["color"]))

    # Box below: cause + fix
    ax.text(x, -1.0, "Root cause:", ha="center", fontsize=8, fontweight="bold")
    ax.text(x, -1.6, b["cause"], ha="center", fontsize=7)
    ax.text(x, -2.3, "Fix:", ha="center", fontsize=8, fontweight="bold", color="#2E7D32")
    ax.text(x, -2.9, b["fix"], ha="center", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#2E7D32"))

# Success marker at end
ax.scatter(7.5, 0, s=300, color="#43A047", marker="*", zorder=5,
           edgecolor="black", linewidth=1)
ax.text(7.5, 0.5, "Pipeline\nworks", ha="center", fontsize=9, fontweight="bold",
        color="#2E7D32")

ax.text(0.5, -3.3,
    "Lessons: All 3 bugs were version incompatibilities between PACE's older HF stack and RunPod's newer one.\n"
    "Bug 1 (SFTConfig) and Bug 3 (BatchEncoding) were breaking changes in TRL/transformers. "
    "Bug 2 (DOMAINS) was an incomplete prior refactor.\nDocumented in tasks/lessons.md.",
    fontsize=8, style="italic",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
import os
os.makedirs("results/bugfixes", exist_ok=True)
plt.savefig("results/bugfixes/runpod_bugfixes_timeline.png", dpi=150, bbox_inches="tight")
print("Saved to results/bugfixes/runpod_bugfixes_timeline.png")
