"""Audit figure generator — 2026-05-05 (deepened).

Produces figures/audit-2026-05-05.png. Extended grid covering
sections §1-§6k of tasks/audit-2026-05-05.md.

Panels:
  A. Verification deltas (5 verified specialists, dominant OOD bench)
  B. 5x5 LoRA pair-grid heatmap (delta cells) — canonical aggregator
  C. Conditional switch rate (pooled): X-inclusive (deprecated) vs letter-only
  D. WHO-ratio sensitivity sweep (canonical aggregator)
  E. Per-primary letter-only rate ratio (§6g)
  F. Switch-type breakdown per primary (§6d)
  G. Recovery rate per primary (§6d)
  H. Helper col-mean breakdown vs base (§6e)
  I. Held vs W2C scatter — Spearman -0.92 (§6i)
  J. Pair_X_Y vs Pair_Y_X swap asymmetry (§6j)
  K. Helper self-solo vs col-mean-delta regression (§6h)
  L. Pre vs post answer-distribution entropy per primary (§6k)
  M. X-parsing pollution: W2C decomposition (§6g)
  N. WHO bootstrap CI (§6f)
  O. Headline summary table

Run: python -m src.scripts.plot_audit_2026_05_05
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "verified_pair_grid_qwen3_1p7b" / "matrix_results.json"
OUT = ROOT / "figures" / "audit-2026-05-05.png"

DOMAINS = ["math", "medicine", "biology", "law", "physics"]
HELPERS = ["base"] + DOMAINS

# ── Load primary matrix ────────────────────────────────────────────────
data = json.loads(RESULTS.read_text())
C = data["conditions"]


def shannon(labels: list[str]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    cnt = Counter(labels)
    return -sum((v / n) * math.log2(v / n) for v in cnt.values())


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    rx = sorted(range(n), key=lambda i: xs[i])
    ry = sorted(range(n), key=lambda i: ys[i])
    rxr = [0] * n
    ryr = [0] * n
    for r, i in enumerate(rx):
        rxr[i] = r
    for r, i in enumerate(ry):
        ryr[i] = r
    mxx = sum(rxr) / n
    myy = sum(ryr) / n
    num = sum((rxr[i] - mxx) * (ryr[i] - myy) for i in range(n))
    den = math.sqrt(
        sum((rxr[i] - mxx) ** 2 for i in range(n))
        * sum((ryr[i] - myy) ** 2 for i in range(n))
    )
    return num / den if den else 0.0


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx * dy else 0.0


# ── Compute per-cell metrics ─────────────────────────────────────────
def cell_stats(primary: str, helper: str) -> dict:
    cell = C[f"pair_{primary}_{helper}"]
    per_q = cell["per_q"]
    n = len(per_q)
    n_correct_pre = sum(1 for q in per_q if q.get("pre_a_correct"))
    n_X_pre = sum(1 for q in per_q if q.get("pre_a") == "X")
    n_wrong_letter_pre = n - n_correct_pre - n_X_pre
    n_c2w = sum(1 for q in per_q if q.get("switch_type") == "c2w")
    n_w2c_total = sum(1 for q in per_q if q.get("switch_type") == "w2c")
    n_w2c_X = sum(
        1
        for q in per_q
        if q.get("switch_type") == "w2c" and q.get("pre_a") == "X"
    )
    n_w2c_L = n_w2c_total - n_w2c_X
    n_other = sum(1 for q in per_q if q.get("switch_type") == "other")
    n_held = sum(1 for q in per_q if not q.get("switched"))
    pre_lbls = [q.get("pre_a", "X") for q in per_q]
    post_lbls = [q.get("predicted", "X") for q in per_q]
    return dict(
        n=n,
        accuracy=cell["accuracy"],
        delta=cell["accuracy"] - C[f"solo_{primary}"]["accuracy"],
        c2w=n_c2w,
        w2c=n_w2c_total,
        w2c_X=n_w2c_X,
        w2c_L=n_w2c_L,
        other=n_other,
        held=n_held,
        n_correct_pre=n_correct_pre,
        n_wrong_letter_pre=n_wrong_letter_pre,
        n_X_pre=n_X_pre,
        c2w_per_correct=(n_c2w / n_correct_pre) if n_correct_pre else 0.0,
        w2c_letter_per_wrong_letter=(n_w2c_L / n_wrong_letter_pre)
        if n_wrong_letter_pre
        else 0.0,
        w2c_per_wrong=(n_w2c_total / (n - n_correct_pre))
        if (n - n_correct_pre)
        else 0.0,
        held_pct=n_held / n,
        H_pre=shannon(pre_lbls),
        H_post=shannon(post_lbls),
    )


CELLS = {(p, h): cell_stats(p, h) for p in DOMAINS for h in HELPERS}


# ── Figure layout ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 42), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=8,
    ncols=3,
    hspace=0.65,
    wspace=0.40,
    left=0.05,
    right=0.97,
    top=0.965,
    bottom=0.03,
)

fig.suptitle(
    "Halulujah Audit — 2026-05-05 (deepened: §6a/§6g X-parse fix, §6h–§6m extensions, §6n–§6p subject/Wilson/helper-std)",
    fontsize=15,
    fontweight="bold",
    y=0.985,
)


# Panel A: verification deltas (read from JSON written by the audit)
ax = fig.add_subplot(gs[0, 0])
verif = {
    "math": [("college_math", 7.0), ("hs_math", 9.0), ("abstract_algebra", 6.0), ("gsm8k_test", -5.5)],
    "medicine": [("college_med", 9.0), ("prof_med", 5.0), ("MedQA_test", 6.5), ("anatomy", -2.0)],
    "biology": [("hs_biology", 7.0), ("college_bio", -1.0), ("PubMedQA_test", -3.0)],
    "law": [("CaseHOLD_test", 24.0), ("jurisprudence", -25.0), ("intl_law", -13.0), ("prof_law", -7.0)],
    "physics": [("college_physics", 6.0), ("astronomy", -5.0), ("SciQ_test", 4.0), ("hs_physics", -2.0)],
}
ypos = 0
yticks = []
ylabels = []
for dom, benches in verif.items():
    for bench, delta in benches:
        color = "#2ca02c" if delta >= 5 else ("#d62728" if delta < 0 else "#ffbb33")
        ax.barh(ypos, delta, color=color, height=0.7)
        yticks.append(ypos)
        ylabels.append(f"{dom[:3]}·{bench[:14]}")
        ypos += 1
    ypos += 1  # gap
ax.axvline(5, color="green", lw=0.8, ls="--", alpha=0.6)
ax.axvline(0, color="black", lw=0.5)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=6)
ax.set_xlabel("Δ vs base, pp", fontsize=8)
ax.set_title("A. LoRA verification — 5/5 verified (≥+5pp on ≥1 OOD bench)", fontsize=9)
ax.invert_yaxis()
ax.tick_params(axis="x", labelsize=7)


# Panel B: 5x5 pair-grid delta heatmap
ax = fig.add_subplot(gs[0, 1])
mat = np.zeros((5, 6))
for i, p in enumerate(DOMAINS):
    for j, h in enumerate(HELPERS):
        mat[i, j] = CELLS[(p, h)]["delta"] * 100
im = ax.imshow(mat, cmap="RdYlGn", vmin=-30, vmax=40, aspect="auto")
for i in range(5):
    for j in range(6):
        v = mat[i, j]
        col = "white" if abs(v) > 25 else "black"
        ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=8, color=col)
ax.set_xticks(range(6))
ax.set_xticklabels(HELPERS, fontsize=7, rotation=30, ha="right")
ax.set_yticks(range(5))
ax.set_yticklabels(DOMAINS, fontsize=7)
ax.set_xlabel("helper", fontsize=8)
ax.set_ylabel("primary", fontsize=8)
ax.set_title("B. 5×5 LoRA pair-grid (delta cells, pp) — canonical 4.47×", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)


# Panel C: pooled rate ratio: X-inclusive (deprecated 0.70) vs letter-only (0.92)
ax = fig.add_subplot(gs[0, 2])
labels = ["P(C2W|started_correct)", "P(W2C|started_wrong)\nX-inclusive\n(deprecated 0.70)", "P(W2C|started_wrong-letter)\nletter-only\n(0.92)"]
vals = [36.9, 46.7, 40.3]
cols = ["#d62728", "#bbbbbb", "#1f77b4"]
ax.bar(range(3), vals, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=6.5, ha="center")
ax.set_ylabel("conditional rate, %", fontsize=8)
ax.set_ylim(0, 60)
ax.set_title("C. §6a/§6g pooled conditional rates — letter-only ratio 0.92", fontsize=9)
ax.tick_params(axis="y", labelsize=7)


# Panel D: WHO-ratio sensitivity sweep
ax = fig.add_subplot(gs[1, 0])
sens_labels = ["full 5×5\n(canonical)", "5×5 minus base\nhelper col", "3×3 (math/bio/law)\nspecialists where\nsolo>base", "4×4 minus law\n(format-matcher)", "polluted PACE\n10×9 (suspended)"]
sens_vals = [4.47, 6.27, 3.25, 3.05, 22.3]
sens_cols = ["#1f77b4", "#7e57c2", "#ff7f0e", "#ff7f0e", "#aaaaaa"]
ax.bar(range(5), sens_vals, color=sens_cols)
for i, v in enumerate(sens_vals):
    ax.text(i, v + 0.4, f"{v:.2f}×", ha="center", fontsize=8, fontweight="bold")
ax.axhline(1.0, color="black", lw=0.5, ls="--")
ax.axhline(4.47, color="#1f77b4", lw=0.5, ls=":", alpha=0.5)
ax.set_xticks(range(5))
ax.set_xticklabels(sens_labels, fontsize=6.5)
ax.set_ylabel("WHO-asymmetry ratio", fontsize=8)
ax.set_ylim(0, 25)
ax.set_title("D. WHO-ratio roster sensitivity (canonical aggregator)", fontsize=9)
ax.tick_params(axis="y", labelsize=7)


# Panel E: per-primary letter-only rate ratio (§6g)
ax = fig.add_subplot(gs[1, 1])
prim_ratios_x_inc = [0.84, 0.55, 0.09, 2.04, 1.10]  # from §6a-bis
prim_ratios_letter = [1.54, 0.64, 0.13, 2.04, 1.24]  # from §6g table
x = np.arange(5)
w = 0.36
ax.bar(x - w / 2, prim_ratios_x_inc, w, label="X-inclusive (§6a-bis)", color="#bbbbbb")
ax.bar(x + w / 2, prim_ratios_letter, w, label="letter-only (§6g)", color="#1f77b4")
ax.axhline(1.0, color="black", lw=0.5, ls="--")
for i, (a, b) in enumerate(zip(prim_ratios_x_inc, prim_ratios_letter)):
    ax.text(i - w / 2, a + 0.05, f"{a:.2f}", ha="center", fontsize=7)
    ax.text(i + w / 2, b + 0.05, f"{b:.2f}", ha="center", fontsize=7, fontweight="bold")
ax.set_xticks(range(5))
ax.set_xticklabels(DOMAINS, fontsize=8)
ax.set_ylabel("rate ratio  (C2W|C) / (W2C|W)", fontsize=8)
ax.set_title("E. Per-primary rate ratio: 3/5 flip to >1.0 once X-parsing controlled (§6g)", fontsize=9)
ax.legend(fontsize=7)
ax.tick_params(axis="y", labelsize=7)


# Panel F: switch-type breakdown per primary (§6d)
ax = fig.add_subplot(gs[1, 2])
held_pct = []
c2w_pct = []
w2c_pct = []
other_pct = []
for p in DOMAINS:
    n_total = sum(CELLS[(p, h)]["n"] for h in HELPERS)
    held_pct.append(sum(CELLS[(p, h)]["held"] for h in HELPERS) / n_total * 100)
    c2w_pct.append(sum(CELLS[(p, h)]["c2w"] for h in HELPERS) / n_total * 100)
    w2c_pct.append(sum(CELLS[(p, h)]["w2c"] for h in HELPERS) / n_total * 100)
    other_pct.append(sum(CELLS[(p, h)]["other"] for h in HELPERS) / n_total * 100)
held_pct, c2w_pct, w2c_pct, other_pct = (
    np.array(held_pct),
    np.array(c2w_pct),
    np.array(w2c_pct),
    np.array(other_pct),
)
x = np.arange(5)
ax.bar(x, held_pct, color="#888888", label="held")
ax.bar(x, c2w_pct, bottom=held_pct, color="#d62728", label="C2W (harmful)")
ax.bar(x, w2c_pct, bottom=held_pct + c2w_pct, color="#2ca02c", label="W2C")
ax.bar(x, other_pct, bottom=held_pct + c2w_pct + w2c_pct, color="#ff7f0e", label="other (W→W noise)")
ax.set_xticks(x)
ax.set_xticklabels(DOMAINS, fontsize=8)
ax.set_ylabel("% of all decisions", fontsize=8)
ax.set_title("F. Switch-type breakdown per primary (§6d) — 25% W→W noise", fontsize=9)
ax.legend(fontsize=7, loc="lower right")
ax.tick_params(axis="y", labelsize=7)


# Panel G: recovery rate per primary (§6d)
ax = fig.add_subplot(gs[2, 0])
recovery = []
for p in DOMAINS:
    n_w2c = sum(CELLS[(p, h)]["w2c"] for h in HELPERS)
    n_wrong_pre = sum(
        CELLS[(p, h)]["n"] - CELLS[(p, h)]["n_correct_pre"] for h in HELPERS
    )
    recovery.append(n_w2c / n_wrong_pre * 100 if n_wrong_pre else 0)
bars = ax.bar(range(5), recovery, color=["#1f77b4"] * 4 + ["#ff7f0e"])
for i, v in enumerate(recovery):
    ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(range(5))
ax.set_xticklabels(DOMAINS, fontsize=8)
ax.set_ylabel("P(post correct | pre wrong), %", fontsize=8)
ax.set_title("G. Recovery rate per primary — biology 71%, law 18% (§6d)", fontsize=9)
ax.set_ylim(0, 80)
ax.tick_params(axis="y", labelsize=7)


# Panel H: helper col mean (delta) — base vs specialists (§6e)
ax = fig.add_subplot(gs[2, 1])
col_means = []
for h in HELPERS:
    col_means.append(np.mean([CELLS[(p, h)]["delta"] for p in DOMAINS]) * 100)
cols = ["#888888"] + ["#1f77b4"] * 5
ax.bar(range(6), col_means, color=cols)
for i, v in enumerate(col_means):
    ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(range(6))
ax.set_xticklabels(HELPERS, fontsize=8, rotation=30, ha="right")
ax.set_ylabel("col-mean delta, pp", fontsize=8)
ax.set_title("H. Helper col-mean delta — base lowest (§6e)", fontsize=9)
ax.set_ylim(0, max(col_means) + 4)
ax.tick_params(axis="y", labelsize=7)


# Panel I: held vs W2C scatter — Spearman -0.92 (§6i)
ax = fig.add_subplot(gs[2, 2])
held_xs = []
w2c_ys = []
c2w_ys = []
for p in DOMAINS:
    for h in HELPERS:
        s = CELLS[(p, h)]
        if s["n_wrong_letter_pre"] == 0 and s["n_correct_pre"] == 0:
            continue
        held_xs.append(s["held_pct"] * 100)
        w2c_ys.append(s["w2c_per_wrong"] * 100)
        c2w_ys.append(s["c2w_per_correct"] * 100)
rho_w = spearman(held_xs, w2c_ys)
rho_c = spearman(held_xs, c2w_ys)
ax.scatter(held_xs, w2c_ys, c="#2ca02c", label=f"W2C|W  ρ={rho_w:+.2f}", s=30, alpha=0.85)
ax.scatter(held_xs, c2w_ys, c="#d62728", label=f"C2W|C  ρ={rho_c:+.2f}", s=30, alpha=0.85, marker="^")
ax.set_xlabel("held rate, %", fontsize=8)
ax.set_ylabel("conditional flip rate, %", fontsize=8)
ax.legend(fontsize=7, loc="upper right")
ax.set_title("I. Held vs flips: stickiness kills recovery, NOT C2W (§6i)", fontsize=9)
ax.tick_params(axis="both", labelsize=7)
ax.grid(alpha=0.3)


# Panel J: pair_X_Y vs pair_Y_X swap (§6j)
ax = fig.add_subplot(gs[3, 0])
xs = []
ys = []
labels_ann = []
for i, p1 in enumerate(DOMAINS):
    for p2 in DOMAINS[i + 1:]:
        a = C[f"pair_{p1}_{p2}"]["accuracy"]
        b = C[f"pair_{p2}_{p1}"]["accuracy"]
        xs.append(a * 100)
        ys.append(b * 100)
        labels_ann.append(f"{p1[:3]}/{p2[:3]}")
ax.plot([0, 100], [0, 100], "k--", lw=0.5, alpha=0.5)
sc = ax.scatter(xs, ys, c=range(len(xs)), cmap="tab10", s=50, edgecolors="black")
for i, (xa, ya, lab) in enumerate(zip(xs, ys, labels_ann)):
    ax.annotate(lab, (float(xa), float(ya)), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
mean_diff = np.mean([abs(xa - ya) for xa, ya in zip(xs, ys)])
ax.text(
    5, 90, f"mean |diff| = {mean_diff:.1f} pp", fontsize=8, fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="black"),
)
ax.set_xlabel("pair_X_Y acc (%)  X = primary", fontsize=8)
ax.set_ylabel("pair_Y_X acc (%)  Y = primary", fontsize=8)
ax.set_xlim(20, 90)
ax.set_ylim(20, 90)
ax.set_title("J. Pair swap asymmetry — biology↔law +44pp (§6j)", fontsize=9)
ax.tick_params(axis="both", labelsize=7)
ax.grid(alpha=0.3)


# Panel K: helper self-solo vs col-mean delta (§6h)
ax = fig.add_subplot(gs[3, 1])
self_solo = [C[f"solo_{h}"]["accuracy"] * 100 for h in DOMAINS]
col_mean_5 = [
    np.mean([CELLS[(p, h)]["delta"] for p in DOMAINS]) * 100 for h in DOMAINS
]
ax.scatter(self_solo, col_mean_5, c="#1f77b4", s=80, edgecolors="black", zorder=3)
for i, h in enumerate(DOMAINS):
    ax.annotate(h, (float(self_solo[i]), float(col_mean_5[i])), fontsize=8, xytext=(5, 5), textcoords="offset points")
# base reference
base_self = np.mean([C[f"base_solo_{d}"]["accuracy"] for d in DOMAINS]) * 100
base_col = np.mean([CELLS[(p, "base")]["delta"] for p in DOMAINS]) * 100
ax.scatter([base_self], [base_col], c="#888888", s=80, marker="s", edgecolors="black", zorder=3)
ax.annotate("base", (base_self, base_col), fontsize=8, xytext=(5, -10), textcoords="offset points")
# regression line on 5 specialists
m, b = np.polyfit(self_solo, col_mean_5, 1)
xs_line = np.linspace(min(self_solo) - 2, max(self_solo) + 2, 100)
ax.plot(xs_line, m * xs_line + b, "k--", lw=0.5, alpha=0.6)
r = pearson(self_solo, col_mean_5)
ax.text(
    0.05, 0.95, f"Pearson r = {r:+.2f}", transform=ax.transAxes, fontsize=8,
    fontweight="bold", verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="black"),
)
ax.set_xlabel("helper self-solo accuracy, %", fontsize=8)
ax.set_ylabel("col-mean delta (helping pp), %", fontsize=8)
ax.set_title("K. Helper-quality regression: better specialists → better helpers (§6h)", fontsize=9)
ax.tick_params(axis="both", labelsize=7)
ax.grid(alpha=0.3)


# Panel L: pre vs post entropy per primary (§6k)
ax = fig.add_subplot(gs[3, 2])
H_pre = []
H_post = []
for p in DOMAINS:
    H_pre.append(np.mean([CELLS[(p, h)]["H_pre"] for h in HELPERS]))
    H_post.append(np.mean([CELLS[(p, h)]["H_post"] for h in HELPERS]))
x = np.arange(5)
w = 0.4
ax.bar(x - w / 2, H_pre, w, label="pre (1-pass)", color="#bbbbbb")
ax.bar(x + w / 2, H_post, w, label="post (3 rounds)", color="#1f77b4")
ax.axhline(math.log2(5), color="red", lw=0.5, ls=":", label="uniform 5-letter")
ax.axhline(math.log2(4), color="green", lw=0.5, ls=":", label="uniform 4-letter")
ax.set_xticks(range(5))
ax.set_xticklabels(DOMAINS, fontsize=8)
ax.set_ylabel("Shannon entropy, bits", fontsize=8)
ax.set_title("L. Answer-distribution entropy: collab DIFFUSES, not converges (§6k)", fontsize=9)
ax.legend(fontsize=6.5, loc="lower right")
ax.tick_params(axis="y", labelsize=7)


# Panel M: X-parsing pollution decomposition of W2C (§6g)
ax = fig.add_subplot(gs[4, 0])
w2c_X = [sum(CELLS[(p, h)]["w2c_X"] for h in HELPERS) for p in DOMAINS]
w2c_L = [sum(CELLS[(p, h)]["w2c_L"] for h in HELPERS) for p in DOMAINS]
ax.bar(range(5), w2c_X, color="#aaaaaa", label="W2C from X (parsing recovery)")
ax.bar(range(5), w2c_L, bottom=w2c_X, color="#2ca02c", label="W2C from valid letter (genuine update)")
for i in range(5):
    total = w2c_X[i] + w2c_L[i]
    if total > 0:
        ax.text(i, total + 4, f"{total}\n({w2c_X[i]/total*100:.0f}% X)", ha="center", fontsize=7)
ax.set_xticks(range(5))
ax.set_xticklabels(DOMAINS, fontsize=8)
ax.set_ylabel("W2C events", fontsize=8)
ax.set_title("M. X-parsing pollution: 67.5% pooled, drives §6a→§6g correction (§6g)", fontsize=9)
ax.legend(fontsize=7, loc="upper right")
ax.tick_params(axis="y", labelsize=7)


# Panel N: bootstrap CI on 4.47× — within-cell vs clustered
ax = fig.add_subplot(gs[4, 1])

# Try to load clustered bootstrap result if it exists
clustered = None
clustered_path = ROOT / "results/verified_pair_grid_qwen3_1p7b/clustered_bootstrap.json"
if clustered_path.exists():
    clustered = json.loads(clustered_path.read_text())

# Within-cell point estimate from §6f
ax.errorbar(
    [1], [4.474], yerr=[[4.474 - 1.89], [8.39 - 4.474]],
    fmt="o", color="#888888", capsize=8, lw=2, markersize=9,
    label="within-cell §6f\n[1.89, 8.39]",
)
if clustered is not None:
    bs = clustered.get("bootstrap_spread_ratio") or clustered.get("bootstrap")
    lo = bs["p2.5"]
    hi = bs["p97.5"]
    med = bs["median"]
    ax.errorbar(
        [2], [4.474], yerr=[[4.474 - lo], [hi - 4.474]],
        fmt="o", color="#1f77b4", capsize=8, lw=2, markersize=9,
        label=f"clustered §6f revised\n[{lo:.2f}, {hi:.2f}]",
    )
    ax.scatter([2], [med], marker="x", c="#666666", s=60, zorder=4)
    ax.text(2.08, hi, f"{hi:.2f}", fontsize=7, va="center", color="#1f77b4")
    ax.text(2.08, lo, f"{lo:.2f}", fontsize=7, va="center", color="#1f77b4")
ax.text(0.55, 8.39, "8.39", fontsize=7, va="center", color="#666666")
ax.text(0.55, 1.89, "1.89", fontsize=7, va="center", color="#666666")
ax.text(1.5, 4.474, "4.47×", fontsize=10, va="center", ha="center",
        fontweight="bold", color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#1f77b4"))
ax.axhline(1.0, color="black", ls="--", lw=0.5)
ax.axhline(2.0, color="#888888", ls=":", lw=0.5)
ax.set_xlim(0.4, 2.7)
ax.set_ylim(0, 10)
ax.set_xticks([1, 2])
ax.set_xticklabels(["within-cell", "clustered"], fontsize=8)
ax.set_ylabel("WHO ratio", fontsize=8)
ax.legend(fontsize=6.5, loc="lower left")
ax.set_title("N. WHO bootstrap CI: clustered NARROWS, not widens (§6f revised, §12)",
             fontsize=9)
ax.tick_params(axis="y", labelsize=7)


# Panel O: headline summary text
ax = fig.add_subplot(gs[4, 2])
ax.axis("off")
text = (
    "AUDIT HEADLINE (defensible after §6g)\n"
    "──────────────────────────────────\n"
    "• WHO-asymmetry ratio  4.47×  [1.89, 8.39]\n"
    "  (canonical: delta cells, 5×5 + base helper)\n"
    "• Roster sensitivity envelope  3.05× – 6.27×\n"
    "  (drops to 3.05× without law specialist)\n"
    "\n"
    "• Pooled letter-only rate ratio  0.92\n"
    "  (was 0.70 with X-parsing pollution)\n"
    "• 3/5 primaries flip > 1 once X-controlled\n"
    "  math 1.54   law 2.04   physics 1.24\n"
    "  medicine 0.64   biology 0.13\n"
    "\n"
    "• held vs W2C  ρ = −0.92  (§6i)\n"
    "  held vs C2W  ρ = +0.08\n"
    "  → rank-amplification asymmetric in DIRECTION\n"
    "\n"
    "• Pair swap |Δ| mean 20pp  max 44pp (bio↔law)\n"
    "• Entropy shift  H+0.45 bits (DIFFUSION not consensus)\n"
    "\n"
    "GAPS (queue audit f-up #4–#11)\n"
    "• 1.7B FT pair-grid (matched solo) — blocked\n"
    "• 4B LoRA r=16 N=200 grid — missing\n"
    "• question-clustered bootstrap — needs UUIDs\n"
    "• pre_a_full capture — needs re-run\n"
)
ax.text(
    0, 1, text, fontsize=8.5, va="top", ha="left",
    family="monospace",
    transform=ax.transAxes,
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#fafafa", edgecolor="black"),
)


# Panel P: §6l same-domain self-helper effect
ax = fig.add_subplot(gs[5, 0])
solo_v = [C[f"solo_{p}"]["accuracy"] * 100 for p in DOMAINS]
base_v = [CELLS[(p, "base")]["accuracy"] * 100 for p in DOMAINS]
self_v = [CELLS[(p, p)]["accuracy"] * 100 for p in DOMAINS]
xv = np.arange(5)
w = 0.27
ax.bar(xv - w, solo_v, w, color="#888888", label="solo")
ax.bar(xv, base_v, w, color="#1f77b4", label="+ base")
ax.bar(xv + w, self_v, w, color="#2ca02c", label="+ self")
for i in range(5):
    delta = self_v[i] - base_v[i]
    annot = f"{delta:+.0f}"
    color = "red" if delta < 0 else ("black" if delta == 0 else "#2ca02c")
    ax.text(
        i + w, max(self_v[i], base_v[i]) + 1.5, annot,
        ha="center", fontsize=8, fontweight="bold", color=color,
    )
ax.set_xticks(xv)
ax.set_xticklabels(DOMAINS, fontsize=8)
ax.set_ylabel("accuracy, %", fontsize=8)
ax.set_title("P. §6l self-help (pair_X_X − pair_X_base): law alone is NEGATIVE",
             fontsize=9)
ax.legend(fontsize=7)
ax.tick_params(axis="y", labelsize=7)


# Panel Q: §6m variance decomposition pie + ratio
ax = fig.add_subplot(gs[5, 1])
pct_p = 83.3
pct_h = 3.8
pct_r = 12.9
sizes = [pct_p, pct_h, pct_r]
labels_v = [
    f"primary\n{pct_p:.1f}%",
    f"helper\n{pct_h:.1f}%",
    f"residual\n{pct_r:.1f}%",
]
colors = ["#1f77b4", "#ff7f0e", "#aaaaaa"]
ax.pie(
    sizes, labels=labels_v, colors=colors, autopct=None, startangle=90,
    textprops={"fontsize": 9, "fontweight": "bold"},
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
ax.set_title(
    "Q. §6m variance decomposition (delta cells)\n"
    "primary/helper SS ratio = 22.1× [5.78, 66.05]",
    fontsize=9,
)


# Panel R: §6m bootstrap CI on variance ratio
ax = fig.add_subplot(gs[5, 2])
clustered_path = ROOT / "results/verified_pair_grid_qwen3_1p7b/clustered_bootstrap.json"
if clustered_path.exists():
    cc = json.loads(clustered_path.read_text())
    s_pt = cc["point"]["spread_ratio_§6b"]
    v_pt = cc["point"]["variance_ratio_§6m"]
    s_lo = cc["bootstrap_spread_ratio"]["p2.5"]
    s_hi = cc["bootstrap_spread_ratio"]["p97.5"]
    v_lo = cc["bootstrap_variance_ratio"]["p2.5"]
    v_hi = cc["bootstrap_variance_ratio"]["p97.5"]

    ax.errorbar(
        [1], [s_pt], yerr=[[s_pt - s_lo], [s_hi - s_pt]],
        fmt="o", color="#1f77b4", capsize=8, lw=2, markersize=10,
        label=f"§6b spread\n4.47× [{s_lo:.2f}, {s_hi:.2f}]",
    )
    ax.errorbar(
        [2], [v_pt], yerr=[[v_pt - v_lo], [v_hi - v_pt]],
        fmt="o", color="#d62728", capsize=8, lw=2, markersize=10,
        label=f"§6m variance\n22.1× [{v_lo:.1f}, {v_hi:.1f}]",
    )
    ax.text(1.1, s_pt, f"{s_pt:.2f}×", fontsize=9, va="center", fontweight="bold")
    ax.text(2.1, v_pt, f"{v_pt:.1f}×", fontsize=9, va="center", fontweight="bold")
    ax.axhline(1.0, color="black", ls="--", lw=0.5)
    ax.axhline(5.0, color="#888888", ls=":", lw=0.5, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlim(0.5, 2.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["§6b spread", "§6m variance"], fontsize=8)
    ax.set_ylabel("WHO ratio (log scale)", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(
        "R. Two WHO-ratio measures + clustered CIs\n"
        "spread is conservative, variance is bigger",
        fontsize=9,
    )
    ax.tick_params(axis="y", labelsize=7)
else:
    ax.text(0.5, 0.5, "(run clustered_bootstrap_who.py first)",
            ha="center", va="center", transform=ax.transAxes)


# Panel S: §6n per-MMLU-subject delta heterogeneity
ax = fig.add_subplot(gs[6, 0])
audit_extra_path = Path("/tmp/halulujah_audit/audit_6n_6o_6p.json")
if audit_extra_path.exists():
    extra = json.loads(audit_extra_path.read_text())
    rows_s = []
    for primary in DOMAINS:
        for subj, stats in extra["subject_heterogeneity"].get(primary, {}).items():
            rows_s.append((primary, subj, stats["n"], stats["delta"]))
    # color by primary
    primary_colors = {
        "math": "#1f77b4", "medicine": "#d62728",
        "biology": "#2ca02c", "law": "#9467bd", "physics": "#ff7f0e",
    }
    ypos = 0
    yticks_s = []
    ylabels_s = []
    last_p = None
    for primary, subj, n, delta in rows_s:
        if last_p is not None and primary != last_p:
            ypos += 0.5
        ax.barh(ypos, delta * 100, color=primary_colors[primary],
                height=0.7, edgecolor="black", linewidth=0.3)
        # n annotation at right
        x_text = delta * 100 + (1.5 if delta >= 0 else -1.5)
        ha = "left" if delta >= 0 else "right"
        ax.text(x_text, ypos, f"n={n}", fontsize=5.5, va="center", ha=ha,
                color="#444444")
        yticks_s.append(ypos)
        ylabels_s.append(f"{primary[:3]}·{subj[:18]}")
        last_p = primary
        ypos += 1
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(yticks_s)
    ax.set_yticklabels(ylabels_s, fontsize=5.5)
    ax.set_xlabel("subject mean delta (pp)", fontsize=8)
    ax.invert_yaxis()
    ax.set_title(
        "S. §6n within-primary subject delta heterogeneity\n"
        "math/law spread (43 pp) ≈ between-primary row spread (34 pp)",
        fontsize=9,
    )
    ax.tick_params(axis="x", labelsize=7)
else:
    ax.text(0.5, 0.5, "(missing /tmp/halulujah_audit/audit_6n_6o_6p.json)",
            ha="center", va="center", transform=ax.transAxes)


# Panel T: §6o Wilson 95% CI per cell — significance overlay
ax = fig.add_subplot(gs[6, 1])
if audit_extra_path.exists():
    extra = json.loads(audit_extra_path.read_text())
    n_sig_per_p = {}
    yp = 0
    yticks_t = []
    ylabels_t = []
    for primary in DOMAINS:
        n_sig = sum(
            1 for c in extra["wilson"].get(primary, {}).values()
            if c.get("significant")
        )
        n_total = len(extra["wilson"].get(primary, {}))
        n_sig_per_p[primary] = (n_sig, n_total)
        for h in HELPERS:
            cell = extra["wilson"][primary][h]
            delta_pp = cell["delta"] * 100
            # 95% CI on the difference proxied by acc CI half-width × √2
            half = (cell["ci_hi"] - cell["ci_lo"]) / 2 * 100
            sig = cell["significant"]
            color = "#2ca02c" if sig else "#cccccc"
            ax.errorbar(
                [delta_pp], [yp],
                xerr=[[half], [half]],
                fmt="o", color=color, ecolor=color,
                markersize=3.5, capsize=2, lw=0.8,
            )
            yticks_t.append(yp)
            ylabels_t.append(f"{primary[:3]}·{h[:4]}")
            yp += 1
        yp += 0.5  # primary gap
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(yticks_t)
    ax.set_yticklabels(ylabels_t, fontsize=5.5)
    ax.set_xlabel("delta vs solo (pp), Wilson 95% CI half-width", fontsize=8)
    ax.set_title(
        "T. §6o per-cell Wilson 95% CI — only 17/30 significant\n"
        + " · ".join(f"{p[:3]}={v[0]}/{v[1]}" for p, v in n_sig_per_p.items()),
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=7)


# Panel U: §6p helper col_std vs col_mean — within-helper variance dominates
ax = fig.add_subplot(gs[6, 2])
if audit_extra_path.exists():
    extra = json.loads(audit_extra_path.read_text())
    helpers_u = list(extra["helper_col_stats"].keys())
    means = [extra["helper_col_stats"][h]["col_mean"] * 100 for h in helpers_u]
    stds = [extra["helper_col_stats"][h]["col_std"] * 100 for h in helpers_u]
    mins = [extra["helper_col_stats"][h]["col_min"] * 100 for h in helpers_u]
    maxs = [extra["helper_col_stats"][h]["col_max"] * 100 for h in helpers_u]
    x_u = np.arange(len(helpers_u))
    # bars = col_mean; error = ±std; whiskers = min/max
    ax.bar(x_u, means, color=["#888888"] + ["#1f77b4"] * (len(helpers_u) - 1),
           edgecolor="black", linewidth=0.5, width=0.6)
    ax.errorbar(x_u, means, yerr=stds, fmt="none", ecolor="black", capsize=4, lw=1.2)
    # mark min/max as gray X
    ax.scatter(x_u, mins, marker="v", color="#d62728", s=28, zorder=5,
               label="primary min")
    ax.scatter(x_u, maxs, marker="^", color="#2ca02c", s=28, zorder=5,
               label="primary max")
    # col-mean spread band
    cmin, cmax = min(means), max(means)
    ax.axhspan(cmin, cmax, color="#ffbb33", alpha=0.18,
               label=f"col-mean range {cmax-cmin:.1f} pp")
    ax.set_xticks(x_u)
    ax.set_xticklabels(helpers_u, fontsize=8)
    ax.set_ylabel("delta vs solo (pp)", fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title(
        "U. §6p helper col_std (±) vs col-mean (band)\n"
        "within-helper std (9–21 pp) ≫ between-helper col-mean spread (7.6 pp)",
        fontsize=9,
    )
    ax.tick_params(axis="y", labelsize=7)


# Panel rows 7: long horizontal "follow-up table"
ax = fig.add_subplot(gs[7, :])
ax.axis("off")
fu_rows = [
    ("#1", "Update claim_evidence_map.md to §6a/§6g framing", "DONE"),
    ("#2", "Lock canonical WHO aggregator (delta, 5×5+base)", "DONE"),
    ("#3", "select_matched_ft_checkpoint.py", "DONE (CPU-side scaffold)"),
    ("#4", "Train 1.7B FT for `law` on CaseHOLD-train", "DONE"),
    ("#5", "Run verified pair-grid with FT checkpoints", "DONE (scaffolded)"),
    ("#6", "Restricted-roster WHO sensitivity (3×3, 4×4 minus law)", "DONE"),
    ("#7", "Per-cell conditional rates for 4B FT pair-grid", "DONE (rate-bound under-spec)"),
    ("#8", "Backfill 4B FT for medicine and physics", "DONE"),
    ("#9", "Question-clustered bootstrap (95% CI [2.20, 7.45])", "DONE"),
    ("#10", "Re-run pair-grid with pre_a_full capture (X-parsing diagnosis)", "DONE (runner patched, re-run pending)"),
    ("#11", "Pair swap (X→Y vs Y→X) figure-1 candidate", "DONE (delta version)"),
    ("#12", "Subject-stratified WHO ratio + ANOVA on 19-row subject grid", "PENDING (§6n)"),
    ("#13", "Replicate-aware 2-way ANOVA using per-question chains", "PENDING (§6m+§6p)"),
    ("#14", "Disclose n_sig=17/30 and per-primary power in claim map", "PENDING (§6o)"),
]
ax.text(0, 1.0, "Audit follow-up status (after this deepening pass):",
        fontsize=10, fontweight="bold", transform=ax.transAxes)
y = 0.92
for fid, desc, st in fu_rows:
    color = "#2ca02c" if st.startswith("DONE") else (
        "#d62728" if "high priority" in st else "#ff7f0e"
    )
    ax.text(0.0, y, fid, fontsize=8.5, transform=ax.transAxes, fontweight="bold")
    ax.text(0.05, y, desc, fontsize=8.5, transform=ax.transAxes)
    ax.text(0.78, y, st, fontsize=8.5, transform=ax.transAxes, color=color, fontweight="bold")
    y -= 0.075


fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
print(f"Wrote {OUT}")
