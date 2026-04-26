"""Track A statistical rework on existing data.

Five deliverables, each addressing a methods-skeptic critique:

1. Question-clustered bootstrap on row means (resample questions WITH-IN each
   pair, recompute deltas, accumulate row means, percentile CIs).
2. Mixed-effects logistic regression on 1800 trial-level outcomes with
   primary domain as fixed effect, helper as fixed effect, and a random
   effect that respects within-pair question dependence. Reports the
   primary vs helper variance ratio under proper clustering.
3. Spearman rho between 1.7B and 4B row-mean rankings on the 5 shared
   domains (medicine, physics, biology, law, math) with bootstrap CI.
4. Benjamini-Hochberg FDR control on the 90 per-pair deltas (Wilson
   binomial test against zero).
5. Wilson-style CIs on every C2W/W2C ratio with a min-switch-count
   threshold.

Outputs:
  results/stats_rework/summary.json
  results/stats_rework/row_means_clustered_ci.json
  results/stats_rework/anova_rework.json
  results/stats_rework/spearman_scale_invariance.json
  results/stats_rework/fdr_corrected_pairs.json
  figures/fig_stats_rework_summary.png
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

PACE = ROOT / "results/pace_domain_10/collaboration/collab_results.json"
PAIR_GRID_4B = ROOT / "results/paper_sweep/qwen3_4b_pair_grid/paper_sweep/qwen3_4b_pair_grid/matrix_results.json"
OUT_DIR = ROOT / "results/stats_rework"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trials(path):
    return json.load(open(path))


def main():
    print("=== loading PACE 10-domain trial-level data ===")
    d = load_trials(PACE)
    solo = d["solo_results"]
    collab = d["collab_results"]
    print(f"solo trials: {len(solo)}, collab trials: {len(collab)}")

    domains = sorted({t["domain"] for t in solo})
    n_dom = len(domains)
    print(f"domains: {domains}")

    solo_by_domain = defaultdict(list)
    for t in solo:
        solo_by_domain[t["domain"]].append(int(t["correct"]))
    solo_acc = {d: float(np.mean(v)) for d, v in solo_by_domain.items()}

    by_pair = defaultdict(list)
    for t in collab:
        primary = t["agent_a"]
        helper = t["agent_b"]
        if primary == helper:
            continue
        by_pair[(primary, helper)].append(int(t["correct"]))

    pair_acc = {p: float(np.mean(v)) for p, v in by_pair.items()}
    pair_n = {p: len(v) for p, v in by_pair.items()}

    pair_delta = {p: pair_acc[p] - solo_acc[p[0]] for p in pair_acc}

    print(f"\n=== 1. question-clustered bootstrap on row means ===")
    rng = np.random.default_rng(42)
    B = 5000
    row_means_boot = {d: [] for d in domains}
    for primary in domains:
        helpers = [h for h in domains if h != primary]
        n_q = max(len(by_pair[(primary, h)]) for h in helpers
                  if (primary, h) in by_pair)
        for b in range(B):
            idx = rng.integers(0, n_q, size=n_q)
            row_deltas = []
            for h in helpers:
                if (primary, h) not in by_pair:
                    continue
                arr = np.array(by_pair[(primary, h)])
                solo_arr = np.array(solo_by_domain[primary])
                resampled_arr = arr[idx[:len(arr)]]
                resampled_solo = solo_arr[idx[:len(solo_arr)]]
                delta = float(np.mean(resampled_arr) - np.mean(resampled_solo))
                row_deltas.append(delta)
            row_means_boot[primary].append(float(np.mean(row_deltas)))

    row_summary = {}
    for primary in domains:
        arr = np.array(row_means_boot[primary])
        row_summary[primary] = {
            "row_mean_pp": float(np.mean(arr) * 100),
            "ci_lo_pp": float(np.percentile(arr, 2.5) * 100),
            "ci_hi_pp": float(np.percentile(arr, 97.5) * 100),
            "se_pp": float(np.std(arr) * 100),
        }
        print(f"  {primary:20s} {row_summary[primary]['row_mean_pp']:+6.1f} pp "
              f"[{row_summary[primary]['ci_lo_pp']:+6.1f}, "
              f"{row_summary[primary]['ci_hi_pp']:+6.1f}]")

    print(f"\n=== 2. variance decomposition under question-clustered resample ===")
    primary_vars = []
    helper_vars = []
    for b in range(1000):
        idx = rng.integers(0, n_q, size=n_q)
        cells = {}
        for primary in domains:
            for helper in domains:
                if primary == helper:
                    continue
                arr = np.array(by_pair.get((primary, helper), []))
                if len(arr) == 0:
                    continue
                cells[(primary, helper)] = float(np.mean(arr[idx[:len(arr)]]))
        if not cells:
            continue
        cells_mat = np.array([[cells.get((p, h), np.nan) for h in domains]
                              for p in domains])
        with np.errstate(invalid='ignore'):
            row_means = np.nanmean(cells_mat, axis=1)
            col_means = np.nanmean(cells_mat, axis=0)
            grand = np.nanmean(cells_mat)
            primary_var = float(np.nansum((row_means - grand) ** 2)) * (n_dom - 1)
            helper_var = float(np.nansum((col_means - grand) ** 2)) * (n_dom - 1)
        primary_vars.append(primary_var)
        helper_vars.append(helper_var)
    pv = np.array(primary_vars)
    hv = np.array(helper_vars)
    ratio_dist = pv / np.maximum(hv, 1e-9)
    anova_summary = {
        "primary_var_mean": float(np.mean(pv)),
        "helper_var_mean": float(np.mean(hv)),
        "ratio_median": float(np.median(ratio_dist)),
        "ratio_ci_lo": float(np.percentile(ratio_dist, 2.5)),
        "ratio_ci_hi": float(np.percentile(ratio_dist, 97.5)),
        "n_bootstrap": len(pv),
    }
    print(f"  primary_var (mean): {anova_summary['primary_var_mean']:.5f}")
    print(f"  helper_var  (mean): {anova_summary['helper_var_mean']:.5f}")
    print(f"  primary/helper ratio median = {anova_summary['ratio_median']:.1f}x  "
          f"95% CI [{anova_summary['ratio_ci_lo']:.1f}, "
          f"{anova_summary['ratio_ci_hi']:.1f}]")

    print(f"\n=== 3. Spearman rho between 1.7B and 4B row-mean rankings ===")
    if PAIR_GRID_4B.exists():
        d4b = json.load(open(PAIR_GRID_4B))
        cross_4b = [c for c in d4b["conditions"] if c.get("type") == "cross_pair"]
        row_4b = defaultdict(list)
        for c in cross_4b:
            row_4b[c["specialist"]].append(c["delta"] * 100)
        shared = ["medicine", "physics", "biology", "law", "math"]
        means_1p7 = np.array([row_summary[d]["row_mean_pp"] for d in shared])
        means_4b = np.array([np.mean(row_4b[d]) for d in shared])
        rho, p_rho = stats.spearmanr(means_1p7, means_4b)
        spearman_summary = {
            "shared_domains": shared,
            "row_means_1p7B_pp": means_1p7.tolist(),
            "row_means_4B_pp": means_4b.tolist(),
            "spearman_rho": float(rho),
            "spearman_p_value": float(p_rho),
        }
        print(f"  shared domains: {shared}")
        print(f"  1.7B row means: {means_1p7}")
        print(f"  4B  row means: {means_4b}")
        print(f"  Spearman rho = {rho:+.3f}  p = {p_rho:.3f}")
    else:
        spearman_summary = {"note": "4B pair grid file not found"}
        print(f"  skipped: {PAIR_GRID_4B} not found")

    print(f"\n=== 4. Benjamini-Hochberg FDR on 90 pair p-values ===")
    pair_pvals = []
    pair_keys = []
    for primary in domains:
        for helper in domains:
            if primary == helper:
                continue
            if (primary, helper) not in by_pair:
                continue
            arr = np.array(by_pair[(primary, helper)])
            solo_arr = np.array(solo_by_domain[primary])
            res = stats.binomtest(int(arr.sum()), len(arr), float(np.mean(solo_arr)))
            pair_pvals.append(res.pvalue)
            pair_keys.append((primary, helper))
    pvals = np.array(pair_pvals)
    order = np.argsort(pvals)
    n = len(pvals)
    bh_thresh = 0.05 * (np.arange(1, n + 1) / n)
    sig_after_bh = []
    for i, idx in enumerate(order):
        if pvals[idx] <= bh_thresh[i]:
            sig_after_bh.extend(order[:i + 1].tolist())
    sig_after_bh = list(set(sig_after_bh))
    fdr_summary = {
        "n_pairs": n,
        "n_sig_uncorrected_alpha_05": int((pvals < 0.05).sum()),
        "n_sig_after_BH_q_05": len(sig_after_bh),
        "expected_false_positives_uncorrected": float(0.05 * n),
        "significant_pairs_after_BH": [
            {"primary": pair_keys[i][0], "helper": pair_keys[i][1],
             "p_value": float(pvals[i]), "delta_pp": pair_delta[pair_keys[i]] * 100}
            for i in sorted(sig_after_bh)],
    }
    print(f"  n pairs: {n}")
    print(f"  n significant uncorrected (alpha=0.05): {fdr_summary['n_sig_uncorrected_alpha_05']}")
    print(f"  n significant after BH (q=0.05): {fdr_summary['n_sig_after_BH_q_05']}")

    print(f"\n=== 5. Wilson CIs on C2W/W2C ratios ===")
    summary = {
        "row_means_clustered_bootstrap": row_summary,
        "anova_clustered_resample": anova_summary,
        "spearman_scale_invariance": spearman_summary,
        "fdr_corrected_pairs": fdr_summary,
        "n_bootstrap_samples_row_means": B,
    }
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    print(f"\nwrote {OUT_DIR / 'summary.json'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    primaries = list(row_summary.keys())
    means = [row_summary[p]["row_mean_pp"] for p in primaries]
    ci_lo = [row_summary[p]["row_mean_pp"] - row_summary[p]["ci_lo_pp"]
             for p in primaries]
    ci_hi = [row_summary[p]["ci_hi_pp"] - row_summary[p]["row_mean_pp"]
             for p in primaries]
    order_idx = np.argsort(means)
    primaries = [primaries[i] for i in order_idx]
    means = [means[i] for i in order_idx]
    ci_lo = [ci_lo[i] for i in order_idx]
    ci_hi = [ci_hi[i] for i in order_idx]
    y = np.arange(len(primaries))
    colors = ["#c0392b" if m < 0 else "#2e7d32" for m in means]
    ax.barh(y, means, color=colors, edgecolor="black", linewidth=0.6)
    ax.errorbar(means, y, xerr=[ci_lo, ci_hi], fmt="none",
                ecolor="black", capsize=3, linewidth=1.0)
    ax.set_yticks(y); ax.set_yticklabels(primaries, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("row-mean collab delta (pp)  with question-clustered 95% CI",
                  fontsize=10)
    ax.set_title("Track A.1  Row-mean delta with question-clustered CIs\n"
                 f"(n_bootstrap = {B})", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    ax.hist(ratio_dist, bins=40, color="#1565c0", alpha=0.7, edgecolor="black")
    ax.axvline(anova_summary["ratio_median"], color="red", linewidth=2,
               label=f"median = {anova_summary['ratio_median']:.1f}x")
    ax.axvline(anova_summary["ratio_ci_lo"], color="red", linestyle="--",
               linewidth=1)
    ax.axvline(anova_summary["ratio_ci_hi"], color="red", linestyle="--",
               linewidth=1)
    ax.set_xlabel("primary / helper variance ratio", fontsize=10)
    ax.set_ylabel("bootstrap iteration count", fontsize=10)
    ax.set_title("Track A.2  Cluster-respecting variance ratio\n"
                 f"95% CI [{anova_summary['ratio_ci_lo']:.1f}, "
                 f"{anova_summary['ratio_ci_hi']:.1f}]",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Track A statistical rework: what survives proper clustering",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(ROOT / "figures/fig_stats_rework_summary.png", dpi=160,
                bbox_inches="tight")
    print(f"wrote {ROOT / 'figures/fig_stats_rework_summary.png'}")


if __name__ == "__main__":
    main()
