"""
Mechanism follow-ups #1 (subtopic breadth) and #2 (reasoning vs recall, heuristic).

#1: Number of MMLU subtopics per domain in our training cluster, correlated
    with the row-mean collab delta. A "narrow" domain (chemistry: 2 subjects)
    might produce a more rigid specialist than a "broad" domain (philosophy:
    4 subjects: logic + ethics + moral + Western).

#2: Heuristic reasoning-vs-recall scoring of each MMLU question. Reasoning
    cues: "calculate", "compute", "prove", "given", "if ... then", "derive",
    "solve for", "show that", numeric patterns. Recall cues: "what is", "name",
    "which of the following is", "according to", proper-noun-heavy questions.

Output:
  figures/fig6_mechanism_followups.png
  results/cka/mechanism_followups.json
"""
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
import sys
sys.path.insert(0, str(ROOT / "src"))

from halulujah.domain.data_prep import DOMAIN_SUBJECTS_EXTENDED, load_mmlu_domain

DOMAINS = [
    "biology", "chemistry", "computer_science", "economics", "history",
    "law", "math", "medicine", "philosophy", "physics",
]

REASONING_PATTERNS = [
    r"\bcalculat", r"\bcompute", r"\bderive", r"\bsolve\b", r"\bsolve for\b",
    r"\bprove", r"\bshow that\b", r"\bif\s.+\bthen\b", r"\bgiven\b",
    r"\bevaluate\b", r"\bdetermine\b", r"\bmaximize\b", r"\bminimize\b",
    r"\bratio\b", r"\bequation\b", r"\binfer\b", r"\bjustif",
    r"=\s*[\?\d]", r"\d+\s*[\+\-\*/]\s*\d+", r"\\frac",
    r"\bproof\b", r"\bargument\b", r"\bvalid\b", r"\bsound\b",
]
RECALL_PATTERNS = [
    r"^what\s+is\b", r"^who\s+is\b", r"^which\s+of\s+the\s+following\b",
    r"^name\b", r"^list\b", r"\baccording to\b",
    r"\bthe\s+definition\b", r"\bbest known for\b", r"\bdescribed as\b",
    r"\bin\s+\d{4}\b",  # year reference
]

REASONING_RE = re.compile("|".join(REASONING_PATTERNS), re.IGNORECASE)
RECALL_RE = re.compile("|".join(RECALL_PATTERNS), re.IGNORECASE)


def score_question(q):
    r = bool(REASONING_RE.search(q))
    c = bool(RECALL_RE.search(q))
    if r and not c:
        return "reasoning"
    if c and not r:
        return "recall"
    if r and c:
        return "both"
    return "neither"


def main():
    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED

    subtopic_counts = {d: len(DOMAIN_SUBJECTS_EXTENDED[d]) for d in DOMAINS}
    print("subtopic counts:")
    for d in DOMAINS:
        print(f"  {d:20s} {subtopic_counts[d]} subtopics: {DOMAIN_SUBJECTS_EXTENDED[d]}")

    print("\nclassifying MMLU questions per domain (heuristic)...")
    pct_reasoning = {}
    counts_per_domain = {}
    for d in DOMAINS:
        entries = load_mmlu_domain(d, split="test")
        try:
            entries.extend(load_mmlu_domain(d, split="validation"))
        except Exception:
            pass
        cats = [score_question(e.get("question", "")) for e in entries]
        n = len(cats)
        n_r = cats.count("reasoning")
        n_c = cats.count("recall")
        n_both = cats.count("both")
        n_neither = cats.count("neither")
        pct_reasoning[d] = (n_r + 0.5 * n_both) / n if n > 0 else 0.0
        counts_per_domain[d] = {"n": n, "reasoning": n_r, "recall": n_c,
                                "both": n_both, "neither": n_neither,
                                "pct_reasoning": pct_reasoning[d]}
        print(f"  {d:20s} n={n:5d} reasoning={n_r:4d} recall={n_c:4d} "
              f"pct_reasoning={pct_reasoning[d]:.2%}")

    # Load row-mean collab delta
    collab = json.load(open("results/pace_domain_10/collaboration/collab_summary.json"))
    delta = collab["collab_delta"]
    row_means = {}
    for primary in DOMAINS:
        vals = [v * 100 for k, v in delta.items() if k.startswith(primary + "+")]
        row_means[primary] = float(np.mean(vals)) if vals else 0.0

    sub_x = np.array([subtopic_counts[d] for d in DOMAINS])
    pct_x = np.array([pct_reasoning[d] for d in DOMAINS])
    y = np.array([row_means[d] for d in DOMAINS])

    r_sub = float(np.corrcoef(sub_x, y)[0, 1])
    r_pct = float(np.corrcoef(pct_x, y)[0, 1])
    print(f"\nr(subtopic count, row-mean delta) = {r_sub:+.3f}  n={len(DOMAINS)}")
    print(f"r(% reasoning, row-mean delta)    = {r_pct:+.3f}  n={len(DOMAINS)}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))

    ax = axes[0]
    ax.scatter(sub_x, y, s=90, c=y, cmap="RdBu_r", edgecolor="black", linewidth=0.5,
               vmin=-max(abs(y.min()), abs(y.max())),
               vmax=max(abs(y.min()), abs(y.max())))
    for i, d in enumerate(DOMAINS):
        ax.annotate(d, (sub_x[i], y[i]), xytext=(5, 5),
                    textcoords="offset points", fontsize=9)
    if len(sub_x) > 1:
        coef = np.polyfit(sub_x, y, 1)
        xs = np.linspace(sub_x.min(), sub_x.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "r--", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("number of MMLU subtopics in training cluster", fontsize=10)
    ax.set_ylabel("row-mean collab delta (pp)", fontsize=10)
    ax.set_title(f"#1 subtopic breadth vs row-mean delta\n  r = {r_sub:+.3f}  (n=10)",
                 fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(pct_x, y, s=90, c=y, cmap="RdBu_r", edgecolor="black", linewidth=0.5,
               vmin=-max(abs(y.min()), abs(y.max())),
               vmax=max(abs(y.min()), abs(y.max())))
    for i, d in enumerate(DOMAINS):
        ax.annotate(d, (pct_x[i], y[i]), xytext=(5, 5),
                    textcoords="offset points", fontsize=9)
    if len(pct_x) > 1:
        coef = np.polyfit(pct_x, y, 1)
        xs = np.linspace(pct_x.min(), pct_x.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "r--", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("fraction of questions classified as reasoning (heuristic)", fontsize=10)
    ax.set_ylabel("row-mean collab delta (pp)", fontsize=10)
    ax.set_title(f"#2 % reasoning vs row-mean delta\n  r = {r_pct:+.3f}  (n=10)",
                 fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Figure 6.  Mechanism follow-ups: subtopic breadth and "
                 "reasoning-vs-recall content as candidate predictors of the "
                 "primary-agent asymmetry.",
                 fontsize=10.5, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results/cka", exist_ok=True)
    out = "figures/fig6_mechanism_followups.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")

    summary = {
        "domains": DOMAINS,
        "subtopic_counts": subtopic_counts,
        "pct_reasoning": pct_reasoning,
        "row_mean_delta_pp": row_means,
        "counts_per_domain": counts_per_domain,
        "r_subtopic_vs_delta": r_sub,
        "r_pct_reasoning_vs_delta": r_pct,
    }
    with open("results/cka/mechanism_followups.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote results/cka/mechanism_followups.json")


if __name__ == "__main__":
    main()
