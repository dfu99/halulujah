#!/usr/bin/env python3
"""Visualize the 2026-03-20 literature survey for both pivots.

Two pivot ideas confirmed novel against existing work. Key gaps found:
  Pivot A: personality measurement papers use prompting, not fine-tuning.
           Per-human LoRA + distributional fingerprinting is an open lane.
  Pivot B: multi-agent LLM debate literature does not test LoRA-fine-tuned
           specialists in alternating CoT — the closest work is homogeneous
           model debate.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle(
    "Literature Survey 2026-03-20: Both Pivots Confirmed Novel\n"
    "Landscape map of related work vs our proposed contributions",
    fontsize=12, fontweight="bold", y=0.98,
)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# Two-pivot split
ax.axvline(5, color="gray", linestyle="--", alpha=0.5)
ax.text(2.5, 9.5, "Pivot A: Personality Fingerprinting", ha="center",
        fontweight="bold", fontsize=12, color="#1E88E5")
ax.text(7.5, 9.5, "Pivot B: Cross-Domain Collaboration", ha="center",
        fontweight="bold", fontsize=12, color="#E53935")

# Pivot A related work
pivot_a_related = [
    ("Machine unlearning\n(Harry Potter, TOFU)", "Erases content post-hoc,\ndoesn't measure style imprint", 2.5, 8.2),
    ("Personality measurement\n(Big-5, MBTI LLM probes)", "Uses PROMPTING to elicit\npersonas, not fine-tuning", 2.5, 7.0),
    ("PERSIST (AAAI 2026)", "Measures personality\ninstability by temperature\nbut doesn't isolate temp\nas a separate variable", 2.5, 5.8),
    ("Style transfer (authorship\nobfuscation)", "Classification-level, not\ndistributional fingerprint", 2.5, 4.6),
]
for label, note, x, y in pivot_a_related:
    ax.add_patch(mpatches.FancyBboxPatch((x - 1.7, y - 0.4), 3.4, 0.8,
                                         boxstyle="round,pad=0.05",
                                         facecolor="#E3F2FD", edgecolor="#1E88E5"))
    ax.text(x, y + 0.15, label, ha="center", fontsize=8, fontweight="bold")
    ax.text(x, y - 0.25, note, ha="center", fontsize=7, style="italic")

# Pivot A gap
ax.add_patch(mpatches.FancyBboxPatch((0.8, 2.5), 3.4, 1.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#FFF9C4", edgecolor="#F57F17", linewidth=2))
ax.text(2.5, 3.7, "GAP", ha="center", fontweight="bold", fontsize=11, color="#E65100")
ax.text(2.5, 3.2, "Per-human LoRA fine-tuning +\ntoken-level KL fingerprint\n+ temperature erosion sweep", ha="center", fontsize=8)
ax.text(2.5, 1.8, "Our contribution: measure KL(persona || base)\nat multiple temperatures, show erosion",
        ha="center", fontsize=8, fontweight="bold", color="#1E88E5")

# Pivot B related work
pivot_b_related = [
    ("Du et al. 2023\n(Multiagent Debate)", "Homogeneous GPT-4 instances,\nno domain specialization", 7.5, 8.2),
    ("MetaGPT, ChatDev", "Role-based, sequential,\nnot alternating CoT debate", 7.5, 7.0),
    ("ICLR 2025 MAD blog", "Documents debate failures,\nbut no LoRA specialist study", 7.5, 5.8),
    ("Catastrophic forgetting\nin LoRA (NeurIPS 24)", "Documents forgetting,\ndoesn't test collaboration", 7.5, 4.6),
]
for label, note, x, y in pivot_b_related:
    ax.add_patch(mpatches.FancyBboxPatch((x - 1.7, y - 0.4), 3.4, 0.8,
                                         boxstyle="round,pad=0.05",
                                         facecolor="#FFEBEE", edgecolor="#E53935"))
    ax.text(x, y + 0.15, label, ha="center", fontsize=8, fontweight="bold")
    ax.text(x, y - 0.25, note, ha="center", fontsize=7, style="italic")

# Pivot B gap
ax.add_patch(mpatches.FancyBboxPatch((5.8, 2.5), 3.4, 1.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#FFF9C4", edgecolor="#F57F17", linewidth=2))
ax.text(7.5, 3.7, "GAP", ha="center", fontweight="bold", fontsize=11, color="#E65100")
ax.text(7.5, 3.2, "Per-domain LoRA specialists in\nalternating CoT collaboration,\nmeasure hallucination rate", ha="center", fontsize=8)
ax.text(7.5, 1.8, "Our contribution: quantify when LoRA\ncollaboration helps vs hurts",
        ha="center", fontsize=8, fontweight="bold", color="#E53935")

# Summary at bottom
ax.add_patch(mpatches.FancyBboxPatch((0.5, 0.2), 9, 1.2,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2))
ax.text(5, 1.1, "Outcome", ha="center", fontweight="bold", fontsize=11, color="#1B5E20")
ax.text(5, 0.7, "Both pivots confirmed novel. Pivot A validates the measurement pipeline (per-human LoRA + KL); "
        "Pivot B applies it to\ncross-domain collaboration. Slack report delivered to PI. Data source pivoted "
        "from PANDORA (gated) → Blog Authorship Corpus (open).",
        ha="center", fontsize=9)

plt.tight_layout()
import os
os.makedirs("results/lit_survey", exist_ok=True)
plt.savefig("results/lit_survey/pivot_novelty_map.png", dpi=150, bbox_inches="tight")
print("Saved to results/lit_survey/pivot_novelty_map.png")
