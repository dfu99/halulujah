#!/usr/bin/env python3
"""Generate a 4-panel summary figure for the paper revision.
Combines: heatmap, ANOVA pie, failure modes, and CI overview.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_PATH = "results/pace_domain_10/collaboration/collab_results.json"
SUMMARY_PATH = "results/pace_domain_10/collaboration/collab_summary.json"
BOOTSTRAP_PATH = "results/pace_domain_10/bootstrap_cis.json"
CHAIN_PATH = "results/pace_domain_10/chain_analysis.json"
ANOVA_PATH = "results/pace_domain_10/variance_decomposition.json"
OUTPUT = "results/pace_domain_10/figures/paper_summary_4panel.png"

DOMAINS = ['physics', 'law', 'biology', 'computer_science', 'history',
           'math', 'chemistry', 'economics', 'philosophy', 'medicine']


def main():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    with open(BOOTSTRAP_PATH) as f:
        bootstrap = json.load(f)
    with open(CHAIN_PATH) as f:
        chain = json.load(f)
    with open(ANOVA_PATH) as f:
        anova = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # === Panel A: Collaboration delta heatmap ===
    ax = axes[0, 0]
    # Build delta matrix
    n = len(DOMAINS)
    delta_matrix = np.full((n, n), np.nan)
    for pair_key, v in summary['collab'].items():
        parts = pair_key.split('+')
        primary, helper = parts[0], parts[1]
        if primary in DOMAINS and helper in DOMAINS:
            i = DOMAINS.index(primary)
            j = DOMAINS.index(helper)
            delta_matrix[i, j] = v['delta']

    # Sort by mean delta (worst at top)
    mean_deltas = [np.nanmean(delta_matrix[i, :]) for i in range(n)]
    sort_idx = np.argsort(mean_deltas)
    sorted_domains = [DOMAINS[i] for i in sort_idx]
    sorted_matrix = delta_matrix[sort_idx][:, sort_idx]

    im = ax.imshow(sorted_matrix, cmap='RdYlGn', vmin=-0.5, vmax=0.3, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_domains, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_domains, fontsize=8)
    ax.set_xlabel('Helper Domain', fontsize=9)
    ax.set_ylabel('Primary Domain', fontsize=9)
    ax.set_title('A. Collaboration Delta Heatmap\n(Primary × Helper → Accuracy Change)', fontsize=11)
    plt.colorbar(im, ax=ax, label='Delta', shrink=0.8)

    # Annotate values
    for i in range(n):
        for j in range(n):
            if not np.isnan(sorted_matrix[i, j]):
                val = sorted_matrix[i, j]
                color = 'white' if abs(val) > 0.3 else 'black'
                ax.text(j, i, f'{val:+.0%}', ha='center', va='center', fontsize=6, color=color)

    # === Panel B: Variance decomposition ===
    ax = axes[0, 1]
    anova_data = anova['anova']
    labels = ['Primary\nDomain\n(13.1%)', 'Helper\nDomain\n(0.5%)', 'Interaction\n(2.0%)', 'Residual\n(84.6%)']
    sizes = [anova_data['primary_domain']['eta_sq'],
             anova_data['helper_domain']['eta_sq'],
             anova_data['interaction']['eta_sq'],
             anova_data['residual']['eta_sq']]
    colors = ['#1976d2', '#f57c00', '#7b1fa2', '#e0e0e0']
    explode = (0.08, 0.03, 0.03, 0)
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                       autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                                       colors=colors, textprops={'fontsize': 9},
                                       startangle=90)
    for t in autotexts:
        t.set_fontweight('bold')
        t.set_fontsize(9)
    ax.set_title('B. Variance Decomposition (Two-Way ANOVA)\nPrimary domain dominates (F=29.25, p<0.001)', fontsize=11)

    # === Panel C: Failure mode breakdown ===
    ax = axes[1, 0]
    failure_modes = chain['failure_modes']
    success_modes = chain['success_modes']

    # Stacked bar: failures and successes
    categories = ['Failures\n(n=1,101)', 'Successes\n(n=699)']
    failure_labels = ['Confident Wrong', 'Extraction Failure', 'Answer Switch']
    success_labels = ['Confirmation', 'Correction', 'Elaboration']

    failure_vals = [failure_modes.get('confident_wrong', 0),
                    failure_modes.get('extraction_failure', 0),
                    failure_modes.get('answer_switch', 0)]
    success_vals = [success_modes.get('confirmation', 0),
                    success_modes.get('correction', 0),
                    success_modes.get('elaboration', 0)]

    # Normalize to percentages
    f_total = sum(failure_vals)
    s_total = sum(success_vals)
    f_pcts = [v/f_total for v in failure_vals]
    s_pcts = [v/s_total for v in success_vals]

    f_colors = ['#e53935', '#ff8a65', '#d81b60']
    s_colors = ['#43a047', '#81c784', '#a5d6a7']

    # Failures bar
    bottom = 0
    for val, label, color in zip(f_pcts, failure_labels, f_colors):
        ax.barh(0, val, left=bottom, color=color, label=label, height=0.6)
        if val > 0.1:
            ax.text(bottom + val/2, 0, f'{val:.0%}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        bottom += val

    # Successes bar
    bottom = 0
    for val, label, color in zip(s_pcts, success_labels, s_colors):
        ax.barh(1, val, left=bottom, color=color, label=label, height=0.6)
        if val > 0.1:
            ax.text(bottom + val/2, 1, f'{val:.0%}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        bottom += val

    ax.set_yticks([0, 1])
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel('Proportion', fontsize=9)
    ax.set_title('C. Failure Mode Analysis\n(what happens in failed vs successful collaborations)', fontsize=11)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.set_xlim(0, 1.0)

    # === Panel D: Domain-level delta with CIs ===
    ax = axes[1, 1]
    domain_stats = {}
    for d in DOMAINS:
        deltas = []
        for pair_key, v in bootstrap['collab'].items():
            if pair_key.startswith(d + '+'):
                deltas.append(v['delta'])
        if deltas:
            domain_stats[d] = {
                'mean': np.mean(deltas),
                'se': np.std(deltas) / np.sqrt(len(deltas)),
            }

    sorted_ds = sorted(domain_stats.items(), key=lambda x: x[1]['mean'])
    domains_sorted = [d for d, _ in sorted_ds]
    means = [s['mean'] for _, s in sorted_ds]
    ses = [s['se'] * 1.96 for _, s in sorted_ds]

    colors = ['#d32f2f' if m < -0.05 else '#388e3c' if m > 0.05 else '#ffa726' for m in means]
    ax.barh(range(len(domains_sorted)), means, xerr=ses, capsize=4,
            color=colors, alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(len(domains_sorted)))
    ax.set_yticklabels(domains_sorted, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Mean Collaboration Delta (± 95% CI)', fontsize=9)
    ax.set_title('D. Domain-Level Collaboration Effect\n(mean delta across all helpers)', fontsize=11)
    for i, (d, m) in enumerate(zip(domains_sorted, means)):
        ax.text(m + ses[i] + 0.02 if m > 0 else m - ses[i] - 0.02, i,
                f'{m:+.1%}', ha='left' if m > 0 else 'right', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout(pad=2.0)
    plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT}")


if __name__ == '__main__':
    main()
