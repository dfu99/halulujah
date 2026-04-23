"""
Streaming, memory-efficient CKA on LoRA ΔW matrices.

Previous plot_reviewer_e_cka.py materialized every ΔW = (α/r)·B@A to
a flat float32 vector (768 MB per adapter × 10 adapters = 7.7 GB before
similarity computation). The rewrite below never materializes ΔW. For
two LoRA adapters with A∈R^{r×in}, B∈R^{out×r}:

    ⟨vec(B_i A_i), vec(B_j A_j)⟩ = Tr(A_i^T B_i^T B_j A_j)
                                 = Tr((B_i^T B_j) (A_j A_i^T))
    ‖vec(B_i A_i)‖² = Tr((B_i^T B_i) (A_i A_i^T))

Everything on the right is an r×r product (r=16 → 256 entries per
module). Memory per adapter reduces from ~4 GB (flat ΔW) to ~1 MB
(A and B stored in bf16). Peak usage across 10 adapters: ~10 MB.

Output:
  figures/fig_lora_cka_intruder_dimensions.png — similarity matrix +
      scatter of CKA-like similarity vs collab delta.
  results/cka/lora_cka_similarity.npz — raw similarity matrix.
  results/cka/summary.json — correlations and metadata.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ROOT = Path("/media/dan/WD_BLACK/models/halulujah/domain_10_adapters")
DOMAINS = [
    "biology", "chemistry", "computer_science", "economics", "history",
    "law", "math", "medicine", "philosophy", "physics",
]


def load_lora_pairs(adapter_dir):
    """Return dict: module_key -> (A, B) in float32 torch tensors on CPU."""
    pairs = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"),
                   framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k).float()
            if k.endswith("lora_A.weight"):
                mk = k.replace(".lora_A.weight", "")
                pairs.setdefault(mk, {})["A"] = t
            elif k.endswith("lora_B.weight"):
                mk = k.replace(".lora_B.weight", "")
                pairs.setdefault(mk, {})["B"] = t
    return {mk: (p["A"], p["B"]) for mk, p in pairs.items() if "A" in p and "B" in p}


def per_module_cosine(pairs_i, pairs_j, alpha_over_r):
    """Average per-module cosine between ΔW_i and ΔW_j without materializing ΔW."""
    cos_list = []
    shared = sorted(set(pairs_i) & set(pairs_j))
    for mk in shared:
        A_i, B_i = pairs_i[mk]
        A_j, B_j = pairs_j[mk]
        BtB = (B_i.T @ B_j)
        AAt = (A_j @ A_i.T)
        num = float(torch.trace(BtB @ AAt))
        BtB_ii = (B_i.T @ B_i)
        AAt_ii = (A_i @ A_i.T)
        BtB_jj = (B_j.T @ B_j)
        AAt_jj = (A_j @ A_j.T)
        norm_i = float(torch.trace(BtB_ii @ AAt_ii))
        norm_j = float(torch.trace(BtB_jj @ AAt_jj))
        cos = (alpha_over_r * alpha_over_r * num) / (
            np.sqrt(alpha_over_r ** 2 * norm_i) *
            np.sqrt(alpha_over_r ** 2 * norm_j) + 1e-12)
        cos_list.append(cos)
    return float(np.mean(cos_list)) if cos_list else 0.0


def main():
    print("Loading 10 LoRA adapter A/B pairs (streaming-safe)...")
    all_pairs = {}
    for d in DOMAINS:
        all_pairs[d] = load_lora_pairs(ADAPTER_ROOT / f"adapter_{d}")
        n_modules = len(all_pairs[d])
        print(f"  {d}: {n_modules} modules")

    alpha_over_r = 32.0 / 16.0
    n = len(DOMAINS)
    sim = np.zeros((n, n), dtype=np.float64)

    print("Computing pairwise per-module cosine similarities...")
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            sim[i, j] = per_module_cosine(
                all_pairs[DOMAINS[i]], all_pairs[DOMAINS[j]], alpha_over_r)
            sim[j, i] = sim[i, j]
        print(f"  row {i+1}/{n} ({DOMAINS[i]}) done")

    dist = 1.0 - sim

    collab_path = ROOT / "results/pace_domain_10/collaboration/collab_summary.json"
    r_sim = None
    if collab_path.exists():
        collab = json.load(open(collab_path))
        collab_delta = collab.get("collab_delta", {})
        pairs = []
        for i, di in enumerate(DOMAINS):
            for j, dj in enumerate(DOMAINS):
                if i == j:
                    continue
                key = f"{di}+{dj}"
                if key not in collab_delta:
                    continue
                pairs.append({
                    "target": di, "helper": dj,
                    "similarity": float(sim[i, j]),
                    "distance": float(dist[i, j]),
                    "delta_pp": float(collab_delta[key]) * 100.0,
                })
        if pairs:
            sims = np.array([p["similarity"] for p in pairs])
            deltas = np.array([p["delta_pp"] for p in pairs])
            r_sim = float(np.corrcoef(sims, deltas)[0, 1])
            print(f"Paired {len(pairs)} points. r(sim, delta) = {r_sim:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    fig.suptitle(
        "LoRA weight-space similarity between 10 Qwen3-1.7B domain specialists\n"
        "(streaming per-module cosine of ΔW = (α/r)·B·A, no materialization)",
        fontsize=11, fontweight="bold", y=1.02)

    ax = axes[0]
    off_diag = sim[~np.eye(n, dtype=bool)]
    im = ax.imshow(sim, cmap="viridis",
                   vmin=off_diag.min(), vmax=off_diag.max())
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(DOMAINS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(DOMAINS, fontsize=8)
    ax.set_title("(A)  Pairwise cosine similarity of LoRA ΔW", fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                    color="white" if sim[i,j] < 0.5 else "black", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    if r_sim is not None:
        ax.scatter(sims, deltas, s=30, alpha=0.7,
                   color="#1E88E5", edgecolor="black", linewidth=0.3)
        coef = np.polyfit(sims, deltas, 1)
        xs = np.linspace(sims.min(), sims.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "r--", linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("LoRA ΔW cosine similarity")
        ax.set_ylabel("Collaboration delta (pp)")
        ax.set_title(f"(B)  Similarity vs collab delta  r = {r_sim:+.3f}\n"
                     "(prior KL cross-eval proxy: r = 0.197)", fontsize=10)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5,
                "No collab_summary.json found;\n"
                "similarity matrix only.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
        ax.set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results/cka", exist_ok=True)
    out_fig = "figures/fig_lora_cka_intruder_dimensions.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"wrote {out_fig}")

    np.savez("results/cka/lora_cka_similarity.npz",
             similarity=sim, distance=dist, domains=np.array(DOMAINS))
    with open("results/cka/summary.json", "w") as f:
        json.dump({
            "domains": DOMAINS,
            "similarity_matrix": sim.tolist(),
            "r_sim_vs_collab_delta": r_sim,
            "prior_kl_proxy_r": 0.197,
        }, f, indent=2)
    print("wrote results/cka/{lora_cka_similarity.npz, summary.json}")


if __name__ == "__main__":
    main()
