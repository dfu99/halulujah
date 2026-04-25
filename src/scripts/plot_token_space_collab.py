"""
Token-space divergence between domain training sets vs collaboration delta.

Hypothesis (paper §5.1): universally-harmed primary domains have narrower
or more peaked training-token distributions. A helper's reasoning tokens
are then more out-of-distribution for the primary, causing decoder lock-in.

Measurement:
  1. For each of 10 MMLU-derived domain training sets, tokenize with the
     Qwen3 tokenizer and compute a normalized unigram token distribution.
  2. Compute pairwise symmetric KL (Jensen-Shannon) between every pair
     of domain distributions.
  3. Correlate JS(domain_A, domain_B) with collab_delta(primary=A, helper=B)
     across 90 ordered pairs from results/pace_domain_10/.
  4. Also compute per-domain entropy (width of training-token distribution)
     and correlate with the row-mean collab delta (primary's universal
     harm-or-help level).

Output:
  figures/fig5_token_space_vs_collab.png
  results/cka/token_space_summary.json
"""
import json
import os
from collections import Counter
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
TOKENIZER_NAME = "Qwen/Qwen3-1.7B"


def build_token_dist(domain, tokenizer, cache_dir=None):
    entries = load_mmlu_domain(domain, split="test", cache_dir=cache_dir)
    try:
        aux = load_mmlu_domain(domain, split="validation", cache_dir=cache_dir)
        entries.extend(aux)
    except Exception:
        pass
    texts = []
    for e in entries:
        q = e.get("question", "")
        choices = e.get("choices", [])
        ans = e.get("answer_letter", "")
        texts.append(q + " " + " ".join(choices) + " " + ans)
    joined = "\n".join(texts)
    ids = tokenizer.encode(joined, add_special_tokens=False)
    counts = Counter(ids)
    total = sum(counts.values())
    vocab_size = tokenizer.vocab_size
    dist = np.zeros(vocab_size, dtype=np.float64)
    for tid, c in counts.items():
        if 0 <= tid < vocab_size:
            dist[tid] = c / total
    return dist, total


def kl(p, q, eps=1e-12):
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def js(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    return 0.5 * (kl(p, m) + kl(q, m))


def entropy(p, eps=1e-12):
    return float(-np.sum(p * np.log(p + eps)))


def main():
    print("loading Qwen3 tokenizer...")
    from halulujah.domain import data_prep
    data_prep.DOMAIN_SUBJECTS = DOMAIN_SUBJECTS_EXTENDED
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, trust_remote_code=True)

    print("building per-domain token distributions...")
    dists = {}
    totals = {}
    for d in DOMAINS:
        p, n_tokens = build_token_dist(d, tokenizer)
        dists[d] = p
        totals[d] = n_tokens
        H = entropy(p)
        nnz = int((p > 0).sum())
        print(f"  {d:20s} n_tokens={n_tokens:7d} entropy={H:6.2f} nats "
              f"nnz={nnz:6d}")

    n = len(DOMAINS)
    js_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            js_mat[i, j] = js(dists[DOMAINS[i]], dists[DOMAINS[j]])
            js_mat[j, i] = js_mat[i, j]

    ents = np.array([entropy(dists[d]) for d in DOMAINS])

    collab = json.load(open("results/pace_domain_10/collaboration/collab_summary.json"))
    delta = collab["collab_delta"]

    rows = []
    for i, primary in enumerate(DOMAINS):
        for j, helper in enumerate(DOMAINS):
            if i == j:
                continue
            key = f"{primary}+{helper}"
            if key not in delta:
                continue
            rows.append({
                "primary": primary, "helper": helper,
                "js": float(js_mat[i, j]),
                "delta_pp": float(delta[key]) * 100.0,
                "primary_entropy": float(ents[i]),
            })
    js_vals = np.array([r["js"] for r in rows])
    delta_vals = np.array([r["delta_pp"] for r in rows])
    ent_vals = np.array([r["primary_entropy"] for r in rows])

    r_js = float(np.corrcoef(js_vals, delta_vals)[0, 1])

    row_means = np.zeros(n)
    for i, primary in enumerate(DOMAINS):
        vals = [r["delta_pp"] for r in rows if r["primary"] == primary]
        row_means[i] = float(np.mean(vals)) if vals else 0.0
    r_ent = float(np.corrcoef(ents, row_means)[0, 1])

    print(f"\nr(JS, collab_delta) pairwise = {r_js:+.3f}  (n={len(rows)})")
    print(f"r(primary training-token entropy, row-mean delta) = {r_ent:+.3f} (n={n})")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

    ax = axes[0]
    im = ax.imshow(js_mat, cmap="viridis")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(DOMAINS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(DOMAINS, fontsize=8)
    ax.set_title("(A)  Pairwise JS divergence between\ndomain training-token distributions",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    ax.scatter(js_vals, delta_vals, s=25, alpha=0.7,
               color="#1E88E5", edgecolor="black", linewidth=0.3)
    if len(js_vals) > 1:
        coef = np.polyfit(js_vals, delta_vals, 1)
        xs = np.linspace(js_vals.min(), js_vals.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "r--", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("JS divergence (primary, helper)", fontsize=10)
    ax.set_ylabel("collaboration delta (pp)", fontsize=10)
    ax.set_title(f"(B)  Pairwise JS vs collab delta  r = {r_js:+.3f}\n"
                 f"(90 ordered pairs)", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    colors = plt.cm.RdBu_r((row_means - row_means.min()) /
                           (row_means.max() - row_means.min() + 1e-9))
    ax.scatter(ents, row_means, s=90, c=colors, edgecolor="black", linewidth=0.5)
    for i, d in enumerate(DOMAINS):
        ax.annotate(d, (ents[i], row_means[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    if len(ents) > 1:
        coef = np.polyfit(ents, row_means, 1)
        xs = np.linspace(ents.min(), ents.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "r--", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("primary training-token entropy (nats)", fontsize=10)
    ax.set_ylabel("row-mean collab delta (pp)", fontsize=10)
    ax.set_title(f"(C)  Primary entropy vs row-mean delta  r = {r_ent:+.3f}\n"
                 f"(n=10 domains, one point per primary)", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Figure 5.  Token-space divergence and primary training-distribution "
        "entropy as candidate mechanism signals.",
        fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results/cka", exist_ok=True)
    out = "figures/fig5_token_space_vs_collab.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")

    summary = {
        "domains": DOMAINS,
        "js_matrix": js_mat.tolist(),
        "entropy_per_domain": ents.tolist(),
        "row_mean_collab_delta_pp": row_means.tolist(),
        "r_js_vs_delta": r_js,
        "r_entropy_vs_row_mean_delta": r_ent,
        "n_pairs": len(rows),
    }
    with open("results/cka/token_space_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote results/cka/token_space_summary.json")


if __name__ == "__main__":
    main()
