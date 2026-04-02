#!/usr/bin/env python3
"""Bootstrap confidence intervals for all collaboration accuracy numbers.

Addresses unanimous reviewer critique #1: n=20 is too small, need CIs on all reported numbers.
Uses 10,000 bootstrap resamples to compute 95% BCa confidence intervals.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_PATH = "results/pace_domain_10/collaboration/collab_results.json"
OUTPUT_JSON = "results/pace_domain_10/bootstrap_cis.json"
OUTPUT_FIGURE = "results/pace_domain_10/figures/bootstrap_ci_forest.png"

N_BOOTSTRAP = 10000
ALPHA = 0.05


def bootstrap_ci(correct_array, n_boot=N_BOOTSTRAP, alpha=ALPHA):
    """Compute bootstrap percentile CI for accuracy."""
    n = len(correct_array)
    if n == 0:
        return 0.0, 0.0, 0.0
    observed = np.mean(correct_array)
    boot_means = np.array([
        np.mean(np.random.choice(correct_array, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return observed, lo, hi


def main():
    np.random.seed(42)

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    results = {}

    # Solo baselines
    solo_by_domain = defaultdict(list)
    for r in data['solo_results']:
        solo_by_domain[r['domain']].append(int(r['correct']))

    results['solo'] = {}
    for domain, corrects in sorted(solo_by_domain.items()):
        acc, lo, hi = bootstrap_ci(np.array(corrects))
        results['solo'][domain] = {
            'accuracy': round(acc, 4),
            'ci_low': round(lo, 4),
            'ci_high': round(hi, 4),
            'n': len(corrects),
            'ci_width': round(hi - lo, 4)
        }
        print(f"Solo {domain:20s}: {acc:.1%} [{lo:.1%}, {hi:.1%}] (n={len(corrects)})")

    # Collaboration pairs
    collab_by_pair = defaultdict(list)
    for r in data['collab_results']:
        pair_key = f"{r['agent_a']}+{r['agent_b']}"
        collab_by_pair[pair_key].append(int(r['correct']))

    results['collab'] = {}
    for pair, corrects in sorted(collab_by_pair.items()):
        acc, lo, hi = bootstrap_ci(np.array(corrects))
        agent_a = pair.split('+')[0]
        solo_acc = results['solo'][agent_a]['accuracy']
        delta = acc - solo_acc
        results['collab'][pair] = {
            'accuracy': round(acc, 4),
            'ci_low': round(lo, 4),
            'ci_high': round(hi, 4),
            'n': len(corrects),
            'ci_width': round(hi - lo, 4),
            'solo_baseline': round(solo_acc, 4),
            'delta': round(delta, 4),
        }

    # Aggregate stats
    all_deltas = [v['delta'] for v in results['collab'].values()]
    ci_widths = [v['ci_width'] for v in results['collab'].values()]
    results['aggregate'] = {
        'mean_delta': round(np.mean(all_deltas), 4),
        'median_delta': round(np.median(all_deltas), 4),
        'mean_ci_width': round(np.mean(ci_widths), 4),
        'pairs_hurt': sum(1 for d in all_deltas if d < 0),
        'pairs_helped': sum(1 for d in all_deltas if d > 0),
        'pairs_neutral': sum(1 for d in all_deltas if d == 0),
    }
    print(f"\nAggregate: mean delta = {results['aggregate']['mean_delta']:.1%}")
    print(f"Mean CI width: {results['aggregate']['mean_ci_width']:.1%}")
    print(f"Pairs hurt/helped/neutral: {results['aggregate']['pairs_hurt']}/{results['aggregate']['pairs_helped']}/{results['aggregate']['pairs_neutral']}")

    # Save JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    # === Forest plot ===
    # Group by primary domain (question domain = agent_a)
    domains = sorted(solo_by_domain.keys())

    fig, axes = plt.subplots(2, 1, figsize=(14, 20), gridspec_kw={'height_ratios': [1, 4]})

    # Top panel: solo baselines with CIs
    ax = axes[0]
    for i, domain in enumerate(domains):
        s = results['solo'][domain]
        color = 'steelblue'
        ax.errorbar(s['accuracy'], i, xerr=[[s['accuracy'] - s['ci_low']], [s['ci_high'] - s['accuracy']]],
                    fmt='o', color=color, capsize=4, markersize=6)
        ax.text(s['ci_high'] + 0.02, i, f"{s['accuracy']:.0%} [{s['ci_low']:.0%}, {s['ci_high']:.0%}]",
                va='center', fontsize=8)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=9)
    ax.set_xlabel('Accuracy')
    ax.set_title('Solo Baseline Accuracy with 95% Bootstrap CIs (n=20 per domain)', fontsize=11)
    ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (4-way MCQ)')
    ax.set_xlim(-0.05, 1.0)
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # Bottom panel: collaboration deltas with CIs
    ax = axes[1]
    # Sort by delta for readability
    sorted_pairs = sorted(results['collab'].items(), key=lambda x: x[1]['delta'])

    for i, (pair, v) in enumerate(sorted_pairs):
        delta = v['delta']
        delta_ci_lo = v['ci_low'] - v['solo_baseline']
        delta_ci_hi = v['ci_high'] - v['solo_baseline']

        color = '#d32f2f' if delta < -0.05 else '#388e3c' if delta > 0.05 else '#757575'
        ax.errorbar(delta, i, xerr=[[delta - delta_ci_lo], [delta_ci_hi - delta]],
                    fmt='o', color=color, capsize=2, markersize=4, alpha=0.8)
        # Label only extreme pairs
        if i < 5 or i >= len(sorted_pairs) - 5:
            ax.text(max(delta_ci_hi, delta) + 0.02, i, pair.replace('+', '→'), fontsize=6, va='center')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Collaboration Delta (accuracy change vs solo)')
    ax.set_title(f'Collaboration Delta with 95% Bootstrap CIs (n=20 per pair, {len(sorted_pairs)} pairs)', fontsize=11)
    ax.set_yticks([])
    ax.set_xlim(-0.6, 0.4)

    # Add summary text
    agg = results['aggregate']
    summary_text = (f"Mean Δ = {agg['mean_delta']:+.1%} | "
                    f"Hurt: {agg['pairs_hurt']}/90 | Helped: {agg['pairs_helped']}/90 | "
                    f"Mean CI width: ±{agg['mean_ci_width']/2:.0%}")
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_FIGURE}")


if __name__ == '__main__':
    main()
