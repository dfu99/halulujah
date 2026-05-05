"""Figure-1 candidate: pair_X_Y vs pair_Y_X scatter — primary effect dominates.

Audit context: tasks/audit-2026-05-05.md §6j and audit follow-up #11.

For every off-diagonal cross-domain pair in the verified 5x5 LoRA grid,
we have two cells: pair_X_Y (X is primary, Y helps) and pair_Y_X
(Y is primary, X helps). If the conversation were symmetric in agent
identity — i.e., who-holds-the-question were not the dominant axis —
the two cells should land on the diagonal.

They don't. Mean |swap diff| is 20pp; the biology-law swap is +44pp.
This is the WHO-asymmetry mechanism expressed at single-pair
granularity, before any cross-cell averaging.

Output: figures/audit_pair_swap_asymmetry.png

Run: python -m src.scripts.plot_pair_swap_asymmetry
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/verified_pair_grid_qwen3_1p7b/matrix_results.json"
OUT = ROOT / "figures/audit_pair_swap_asymmetry.png"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
DOM_COLOR = {
    "math": "#1f77b4",
    "medicine": "#d62728",
    "biology": "#2ca02c",
    "law": "#9467bd",
    "physics": "#ff7f0e",
}


def main() -> None:
    data = json.loads(RESULTS.read_text())
    C = data["conditions"]

    # IMPORTANT: pair_X_Y uses X's primary-domain question set; pair_Y_X uses
    # Y's. The two cells are NOT on the same question set, so absolute-accuracy
    # swap conflates "primary's domain difficulty" with "WHO-asymmetry of help".
    # Subtract the primary's own solo accuracy on its own questions to isolate
    # the gain-from-help on this primary's domain.
    pairs = []
    for i, p1 in enumerate(DOMAINS):
        for p2 in DOMAINS[i + 1:]:
            a_acc = C[f"pair_{p1}_{p2}"]["accuracy"] * 100
            b_acc = C[f"pair_{p2}_{p1}"]["accuracy"] * 100
            a_solo = C[f"solo_{p1}"]["accuracy"] * 100
            b_solo = C[f"solo_{p2}"]["accuracy"] * 100
            a_delta = a_acc - a_solo  # gain on p1's questions when p1 is primary
            b_delta = b_acc - b_solo  # gain on p2's questions when p2 is primary
            pairs.append((p1, p2, a_delta, b_delta, a_delta - b_delta))

    pairs.sort(key=lambda r: -abs(r[4]))
    for p1, p2, a, b, diff in pairs:
        print(
            f"  pair_{p1}_{p2} = {a:5.1f}% | "
            f"pair_{p2}_{p1} = {b:5.1f}% | diff = {diff:+5.1f}pp"
        )
    mean_abs_diff = np.mean([abs(r[4]) for r in pairs])
    print(f"\n  mean |diff| = {mean_abs_diff:.2f} pp")
    print(f"  max |diff|  = {max(abs(r[4]) for r in pairs):.2f} pp")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(
        "Pair-swap asymmetry (delta cells, n=50/cell): does help "
        "depend on which agent holds the question?",
        fontsize=12,
        fontweight="bold",
    )

    # Panel L: scatter delta_pair_X_Y vs delta_pair_Y_X
    ax = axes[0]
    bound = 50
    ax.plot([-bound, bound], [-bound, bound], "k--", lw=1, alpha=0.5,
            label="diagonal\n(symmetric help)")
    ax.axhline(0, color="black", lw=0.5, alpha=0.6)
    ax.axvline(0, color="black", lw=0.5, alpha=0.6)

    for p1, p2, a, b, diff in pairs:
        higher = p1 if a >= b else p2
        ax.scatter(
            a, b, s=130, color=DOM_COLOR[higher], edgecolors="black",
            linewidths=1.0, zorder=4,
        )
        label = f"{p1[:3]}↔{p2[:3]}  Δ={diff:+.0f}"
        offset = (5, 5) if abs(diff) > 5 else (5, -10)
        weight = "bold" if abs(diff) > 30 else "normal"
        ax.annotate(
            label,
            (float(a), float(b)),
            fontsize=9,
            xytext=offset,
            textcoords="offset points",
            fontweight=weight,
        )
    ax.set_xlabel(
        "delta_pair_X_Y (pp)\n"
        "X = primary; gain on X's domain questions",
        fontsize=10,
    )
    ax.set_ylabel(
        "delta_pair_Y_X (pp)\n"
        "Y = primary; gain on Y's domain questions",
        fontsize=10,
    )
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(
        f"L. Pair-swap delta scatter — mean |swap| = {mean_abs_diff:.1f}pp",
        fontsize=11,
    )

    # Panel R: ranked diffs as horizontal bar chart
    ax = axes[1]
    ranked = sorted(pairs, key=lambda r: r[4])
    labels = [f"{r[0]} (P) vs {r[1]} (P)" for r in ranked]
    diffs = [r[4] for r in ranked]
    colors = ["#d62728" if d < 0 else "#2ca02c" for d in diffs]
    ax.barh(range(len(diffs)), diffs, color=colors, edgecolor="black")
    for i, (lab, d) in enumerate(zip(labels, diffs)):
        if d >= 0:
            ax.text(d + 1, i, f"{d:+.0f}pp", va="center", fontsize=9)
        else:
            ax.text(d - 1, i, f"{d:+.0f}pp", va="center", ha="right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(
        "delta_pair_X_Y − delta_pair_Y_X (pp)\n"
        "delta = post-collab acc − solo acc (on primary's domain questions)",
        fontsize=10,
    )
    ax.set_xlim(-50, 50)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(
        "R. Ranked swap diffs (delta) — primary identity matters even after\n"
        "removing primary's solo competence",
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
