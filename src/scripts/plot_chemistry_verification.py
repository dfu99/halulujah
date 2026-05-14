"""Bar chart of the 2026-05-14 chemistry verification gate.

Run: python -m src.scripts.plot_chemistry_verification
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/specialist_verification/chemistry_qwen3/chemistry_r16.json"
OUT = ROOT / "figures/chemistry_verification_2026-05-14.png"


def main() -> None:
    d = json.loads(SRC.read_text())
    spec = d["specialist"]
    base = d["base"]

    bench_order = [
        ("high_school_chemistry", "MMLU\nhs_chem", "subjects"),
        ("college_chemistry", "MMLU\ncollege_chem", "subjects"),
        ("sciq", "SciQ-test\n(on-domain)", "flat"),
        ("mmlu_pro_chem", "MMLU-Pro\nchemistry", "flat"),
    ]
    labels, base_pct, spec_pct = [], [], []
    for key, label, where in bench_order:
        if where == "subjects":
            b = base["subjects"][key]["accuracy"] * 100
            s = spec["subjects"][key]["accuracy"] * 100
        else:
            b = base[key]["accuracy"] * 100
            s = spec[key]["accuracy"] * 100
        labels.append(label)
        base_pct.append(b)
        spec_pct.append(s)

    deltas = [s - b for s, b in zip(spec_pct, base_pct)]
    pass_color = "#2ca02c"
    fail_color = "#bbbbbb"
    bar_colors = [pass_color if d >= 5 else fail_color for d in deltas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    x = np.arange(len(labels))
    w = 0.38
    ax1.bar(x - w / 2, base_pct, w, label="Qwen3-1.7B base", color="#888")
    ax1.bar(x + w / 2, spec_pct, w, label="chemistry r=16 spec", color="#1f77b4")
    for i, (b, s) in enumerate(zip(base_pct, spec_pct)):
        ax1.text(i - w / 2, b + 1.0, f"{b:.0f}", ha="center", fontsize=9, color="#444")
        ax1.text(i + w / 2, s + 1.0, f"{s:.0f}", ha="center", fontsize=9,
                 color="#1f77b4", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Chemistry specialist (Qwen3-1.7B + LoRA r=16 on SciQ)",
                  fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, deltas, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax2.axhline(5, ls="--", color="green", lw=1.0,
                label="verification gate (+5 pp)")
    ax2.axhline(0, color="black", lw=0.8)
    for i, d_ in enumerate(deltas):
        sign = "+" if d_ >= 0 else ""
        ax2.text(i, d_ + (0.5 if d_ >= 0 else -1.5),
                 f"{sign}{d_:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Δ accuracy  spec − base  (pp)", fontsize=11)
    ax2.set_title("Per-benchmark delta", fontsize=12)
    ax2.set_ylim(-5, max(20, max(deltas) + 4))
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Chemistry verification gate — PASS (1 of 4 ≥+5pp).  "
        "MMLU-Pro/chemistry +14.6 pp is the strongest OOD transfer signal.",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
