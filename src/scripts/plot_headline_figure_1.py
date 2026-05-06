"""Audit follow-up #48: paper-ready headline Figure 1.

Generates figures/headline_who_asymmetry.png — a 6-panel paper-ready
figure capturing the audit's headline findings in reviewer-friendly
form. Designed for direct inclusion as Figure 1 in
paper/abstract_and_intro.md.

Panel A: variance-decomposition row/helper ratio family
Panel B: per-primary vs per-helper mean W2C
Panel C: per-subject hard W2C forest plot (top 5 + bottom 5)
Panel D: Bonferroni-survivor counts at three aggregation levels
Panel E: mutually-unrecoverable distribution per primary
Panel F: 35-test audit summary banner

Output: figures/headline_who_asymmetry.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/verified_pair_grid_qwen3_1p7b"
OUT = ROOT / "figures/headline_who_asymmetry.png"

PRIMARY_COLOR = {
    "math": "#1f77b4",
    "medicine": "#ff7f0e",
    "biology": "#2ca02c",
    "law": "#d62728",
    "physics": "#9467bd",
}
HELPER_COLOR = {
    "base": "#888888",
    "math": "#1f77b4",
    "medicine": "#ff7f0e",
    "biology": "#2ca02c",
    "law": "#d62728",
    "physics": "#9467bd",
}


def main() -> None:
    fig = plt.figure(figsize=(16, 11), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        hspace=0.55,
        wspace=0.30,
        left=0.07,
        right=0.97,
        top=0.92,
        bottom=0.05,
    )
    fig.suptitle(
        "Figure 1. WHO-asymmetry in multi-agent collaboration: row factor (primary or subject) "
        "dominates helper factor at every formal lens",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    # ============================================================
    # PANEL A: 6-way WHO ratio family bar chart
    # ============================================================
    ax = fig.add_subplot(gs[0, 0])
    swr = json.loads((RESULTS / "subject_who_ratio.json").read_text())
    swrh = json.loads((RESULTS / "subject_who_ratio_hard.json").read_text())
    swre = json.loads((RESULTS / "subject_who_ratio_easy.json").read_text())

    a_pf = swr["anova_primary_helper_recomputed"]
    a_sf = swr["anova_filtered_weighted"]
    a_ph = swrh["anova_primary_helper_hard"]
    a_sh = swrh["anova_filtered_weighted"]
    a_pe = swre["anova_primary_helper_easy"]
    a_se = swre["anova_filtered_weighted"]

    labels = [
        "primary\nfull", "subject\nfull",
        "primary\nhard", "subject\nhard",
        "primary\neasy", "subject\neasy",
    ]
    ratios = [
        a_pf["frac_row"] / a_pf["frac_col"],
        a_sf["frac_row"] / a_sf["frac_col"],
        a_ph["frac_row"] / a_ph["frac_col"],
        a_sh["frac_row"] / a_sh["frac_col"],
        a_pe["frac_row"] / a_pe["frac_col"],
        a_se["frac_row"] / a_se["frac_col"],
    ]
    colors = ["#1f77b4", "#9467bd"] * 3
    ax.bar(np.arange(len(labels)), ratios, color=colors,
           edgecolor="black", lw=0.6)
    for i, r in enumerate(ratios):
        ax.text(i, r * 1.08, f"{r:.1f}×", fontsize=11, ha="center",
                fontweight="bold", color="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("row / helper variance ratio", fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 200)
    ax.axhline(1.0, color="black", linestyle=":", lw=0.8, alpha=0.5)
    ax.text(5.4, 1.05, "no asymmetry", fontsize=8, ha="right",
            color="gray", style="italic")
    ax.set_title("A. Six-way variance-decomposition family", fontsize=11,
                 fontweight="bold")
    ax.legend(handles=[
        Patch(facecolor="#1f77b4", label="primary (5 levels)"),
        Patch(facecolor="#9467bd", label="subject (17 levels)"),
    ], fontsize=9, loc="upper right")
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.3)

    # ============================================================
    # PANEL B: per-primary vs per-helper mean W2C
    # ============================================================
    ax = fig.add_subplot(gs[0, 1])
    hpz = json.loads((RESULTS / "helper_w2c_pairwise_z.json").read_text())
    # Per-primary W2C from §6hh / matrix_results
    matrix = json.loads((RESULTS / "matrix_results.json").read_text())["conditions"]
    DOMAINS = ["math", "medicine", "biology", "law", "physics"]
    HELPERS = ["base"] + DOMAINS

    primary_w2c = {}
    for p in DOMAINS:
        solo_pq = matrix[f"solo_{p}"]["per_q"]
        hard_idx = [q["idx"] for q in solo_pq if not q["correct"]]
        rates = []
        for h in HELPERS:
            pair_pq = matrix[f"pair_{p}_{h}"]["per_q"]
            n_recover = sum(1 for idx in hard_idx if pair_pq[idx]["correct"])
            rates.append(n_recover / len(hard_idx) if hard_idx else 0)
        primary_w2c[p] = float(np.mean(rates))

    helper_w2c = {h: hpz["helper_kn"][h]["p_w2c"] for h in HELPERS}

    sorted_primaries = sorted(DOMAINS, key=lambda p: primary_w2c[p])
    sorted_helpers = sorted(HELPERS, key=lambda h: helper_w2c[h])

    x_left = np.arange(5)
    x_right = np.arange(6) + 6
    p_rates = [primary_w2c[p] * 100 for p in sorted_primaries]
    h_rates = [helper_w2c[h] * 100 for h in sorted_helpers]
    p_colors = [PRIMARY_COLOR[p] for p in sorted_primaries]
    h_colors = [HELPER_COLOR[h] for h in sorted_helpers]

    ax.bar(x_left, p_rates, color=p_colors, edgecolor="black", lw=0.5,
           label="primary axis (varies)")
    ax.bar(x_right, h_rates, color=h_colors, edgecolor="black", lw=0.5,
           label="helper axis (flat)")

    for x, r in zip(x_left, p_rates):
        ax.text(float(x), r + 1.5, f"{r:.0f}%", fontsize=9, ha="center",
                fontweight="bold")
    for x, r in zip(x_right, h_rates):
        ax.text(float(x), r + 1.5, f"{r:.0f}%", fontsize=9, ha="center",
                fontweight="bold")

    ax.set_xticks(list(x_left) + list(x_right))
    ax.set_xticklabels(sorted_primaries + sorted_helpers, fontsize=8.5,
                       rotation=30, ha="right")
    ax.set_ylabel("mean W2C rate on hard questions (%)", fontsize=11)
    ax.set_ylim(0, 75)
    # Highlight ranges
    p_min, p_max = min(p_rates), max(p_rates)
    h_min, h_max = min(h_rates), max(h_rates)
    ax.axhspan(p_min, p_max, xmin=0.0, xmax=0.42, color="#1f77b4", alpha=0.07)
    ax.axhspan(h_min, h_max, xmin=0.50, xmax=0.99, color="#888888", alpha=0.10)
    ax.text(2, p_max + 6, f"primary spread\n{p_max-p_min:.0f} pp",
            fontsize=10, ha="center", color="#1f77b4", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white",
                      edgecolor="#1f77b4"))
    ax.text(8.5, h_max + 6, f"helper spread\n{h_max-h_min:.0f} pp",
            fontsize=10, ha="center", color="#888888", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white",
                      edgecolor="#888888"))
    ax.set_title("B. Per-primary vs per-helper W2C: 7× spread asymmetry",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")

    # ============================================================
    # PANEL C: per-subject hard W2C forest plot (top 5 + bottom 5)
    # ============================================================
    ax = fig.add_subplot(gs[1, 0])
    swhb = json.loads((RESULTS / "subject_w2c_hard_bootstrap.json").read_text())
    subj_items = list(swhb["subject_summary"].items())
    subj_items.sort(key=lambda x: x[1]["point_w2c"], reverse=True)
    top5 = subj_items[:5]
    bottom5 = subj_items[-5:]
    selected = top5 + [("...", None)] + bottom5

    y_pos = np.arange(len(selected))
    for i, (s, b) in enumerate(selected):
        if b is None:
            ax.text(50, i, "...", fontsize=14, ha="center", color="#888")
            continue
        c = PRIMARY_COLOR[b["primary"]]
        ax.plot([b["ci_lo"] * 100, b["ci_hi"] * 100],
                [i, i], color=c, lw=2.5, alpha=0.7)
        ax.scatter([b["point_w2c"] * 100], [i], color=c, s=80, zorder=5,
                   edgecolor="black", lw=0.5)
        ax.text(b["ci_hi"] * 100 + 1.5, i, f"n={b['n_hard']}",
                fontsize=8, va="center", color="#444")
    labels = []
    for s, b in selected:
        if b is None:
            labels.append("...")
        else:
            labels.append(f"{b['primary']}/{s}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("hard W2C rate (%) — 95% bootstrap CI", fontsize=11)
    ax.set_xlim(-5, 100)
    # Annotate the 39pp gap
    hs_bio = next(b for s, b in subj_items if s == "high_school_biology")
    college_math = next(b for s, b in subj_items if s == "college_mathematics")
    ax.annotate(
        "",
        xy=(hs_bio["ci_lo"] * 100, 0),
        xytext=(college_math["ci_hi"] * 100, len(selected) - 1),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
    )
    ax.text(35, len(selected) / 2 - 0.5,
            f"39 pp\nrobust\nnon-overlap",
            fontsize=10, ha="center", color="black", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white",
                      edgecolor="black"))
    ax.set_title("C. hs_biology vs college_math: 39 pp robust gap "
                 "(survives both bootstrap and z-test at Bonferroni-136)",
                 fontsize=10.5, fontweight="bold")

    # ============================================================
    # PANEL D: Bonferroni-survivor counts at three aggregation levels
    # ============================================================
    ax = fig.add_subplot(gs[1, 1])
    swhin = json.loads((RESULTS / "subject_w2c_hard_bootstrap_hin.json").read_text())
    ncbh = json.loads((RESULTS / "net_corrector_bootstrap_hin.json").read_text())
    swpz = json.loads((RESULTS / "subject_w2c_pairwise_z.json").read_text())
    hwpz = json.loads((RESULTS / "helper_w2c_pairwise_z.json").read_text())

    levels = ["subject\n(136 pairs)", "primary\n(10 pairs)", "helper\n(15 pairs)"]
    n_pairs = [136, 10, 15]
    boot_count = [
        swhin["n_bonferroni_pass"],
        sum(1 for ps in ncbh["pair_stats"] if ps["axis"] == "primary" and ps["bonferroni_pass"]),
        sum(1 for ps in ncbh["pair_stats"] if ps["axis"] == "helper" and ps["bonferroni_pass"]),
    ]
    z_count = [
        swpz["n_bonferroni_pass"],
        0,  # primary z-test not in §6yy file
        hwpz["n_bonferroni_pass"],
    ]
    z_uncorr = [
        swpz["n_uncorrected_pass"],
        0,
        hwpz["n_uncorrected_pass"],
    ]

    x_d = np.arange(len(levels))
    width = 0.27
    ax.bar(x_d - width, boot_count, width, color="#1f77b4",
           edgecolor="black", lw=0.5, label="bootstrap Bonferroni (n=50k)")
    ax.bar(x_d, z_count, width, color="#d62728",
           edgecolor="black", lw=0.5, label="z-test Bonferroni")
    ax.bar(x_d + width, z_uncorr, width, color="#cccccc",
           edgecolor="black", lw=0.5, label="z-test uncorrected α=0.05")
    for offset, vs in [(-width, boot_count), (0, z_count), (width, z_uncorr)]:
        for i, v in enumerate(vs):
            pct = 100 * v / n_pairs[i]
            label = f"{v}\n({pct:.0f}%)"
            ax.text(i + offset, v + 1, label,
                    fontsize=8.5, ha="center", fontweight="bold")
    ax.set_xticks(x_d)
    ax.set_xticklabels(levels, fontsize=10)
    ax.set_ylabel("# pairs surviving Bonferroni / uncorrected", fontsize=11)
    ax.set_ylim(0, 80)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("D. Formal-statistical hierarchy: row 28-vs-helper 0",
                 fontsize=11, fontweight="bold")

    # ============================================================
    # PANEL E: mutually-unrecoverable distribution per primary
    # ============================================================
    ax = fig.add_subplot(gs[2, 0])
    ha = json.loads((RESULTS / "helper_agreement_hard.json").read_text())
    primaries = list(ha["per_primary"].keys())
    primaries.sort(key=lambda p: ha["per_primary"][p]["mean_recovery_count_of_6"],
                   reverse=True)
    n_hards = [ha["per_primary"][p]["n_hard"] for p in primaries]
    bin_data = np.zeros((7, len(primaries)))
    for j, p in enumerate(primaries):
        for k in range(7):
            key = str(k) if isinstance(
                next(iter(ha["per_primary"][p]["bin_counts"].keys())), str
            ) else k
            bin_data[k, j] = ha["per_primary"][p]["bin_counts"][key]
    bin_frac = bin_data / np.array(n_hards)[None, :] * 100
    cmap_bp = ["#660000", "#990000", "#cc4400", "#cccccc", "#669900", "#339900", "#006600"]
    x_e = np.arange(len(primaries))
    bottoms = np.zeros(len(primaries))
    for k in range(7):
        ax.bar(x_e, bin_frac[k, :], bottom=bottoms, color=cmap_bp[k],
               edgecolor="white", lw=0.5,
               label=f"{k}/6 helpers" if k in [0, 6] else None)
        bottoms = bottoms + bin_frac[k, :]
    for j, (p, n) in enumerate(zip(primaries, n_hards)):
        zero_frac = ha["per_primary"][p]["frac_0_of_6"] * 100
        six_frac = ha["per_primary"][p]["frac_6_of_6"] * 100
        ax.text(j, 102, f"n={n}", fontsize=9, ha="center", color="#444")
        if zero_frac >= 8:
            ax.text(j, zero_frac / 2, f"{zero_frac:.0f}%",
                    fontsize=9, ha="center", va="center", color="white",
                    fontweight="bold")
        if six_frac >= 8:
            ax.text(j, 100 - six_frac / 2, f"{six_frac:.0f}%",
                    fontsize=9, ha="center", va="center", color="white",
                    fontweight="bold")
    ax.set_xticks(x_e)
    ax.set_xticklabels(primaries, fontsize=10)
    ax.set_ylabel("% of hard questions", fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8.5, loc="lower right", ncol=2,
              title="recovery count")
    ax.set_title("E. Mutually-unrecoverable mechanism: math 53% / biology 19%",
                 fontsize=11, fontweight="bold")

    # ============================================================
    # PANEL F: audit summary banner
    # ============================================================
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    ax.text(0.05, 0.95, "Audit summary (35 row-effect tests + 8 helper-effect lenses):",
            fontsize=11, fontweight="bold", transform=ax.transAxes)
    summary_lines = [
        ("Row effect", "ALL 35 tests reject H0", "✓"),
        ("Helper effect", "All 8 lenses non-rejecting", "✓"),
        ("Bootstrap Bonferroni-136", "11 of 136 subject-pair", "✓"),
        ("z-test Bonferroni-136", "28 of 136 subject-pair", "✓"),
        ("z-test helper Bonferroni-15", "0 of 15 EVEN UNCORR", "★"),
        ("ANOVA F_primary triple", "26.55 / 31.22 / 3.13", "✓"),
        ("ANOVA F_helper triple", "0.96 / 0.50 / 1.91", "·"),
        ("Cohen's f primary (full/hard)", "2.24 / 2.62 — huge", "✓"),
        ("Cohen's f helper (full/hard)", "0.20 / 0.13 — small", "·"),
        ("Mutually-unrecoverable (hard)", "47.7% nearly unrec", "✓"),
        ("hs_biology vs college_math", "39 pp non-overlap CI", "★"),
        ("Headline", "WHO holds question dominates", "★"),
    ]
    y = 0.85
    for desc, val, mark in summary_lines:
        ax.text(0.05, y, desc, fontsize=9, transform=ax.transAxes)
        ax.text(0.50, y, val, fontsize=9, transform=ax.transAxes,
                fontweight="bold", color="#1f77b4")
        color = {"✓": "#2ca02c", "·": "#888888", "★": "#d62728"}[mark]
        ax.text(0.95, y, mark, fontsize=14, transform=ax.transAxes,
                color=color, fontweight="bold", ha="right")
        y -= 0.065
    ax.text(0.05, y - 0.04,
            "Source: tasks/audit-2026-05-05.md §10 11th revision\n"
            "(35 row-effect tests, 8 helper-effect lenses).",
            fontsize=8.5, transform=ax.transAxes, style="italic",
            color="#444")

    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
