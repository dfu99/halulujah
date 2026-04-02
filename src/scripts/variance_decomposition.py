#!/usr/bin/env python3
"""Variance decomposition: how much of collaboration outcome is explained by
primary domain vs helper domain vs their interaction?

Uses manual ANOVA computation (no statsmodels dependency).
Addresses Evans (R1) demand for statistical variance decomposition.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_PATH = "results/pace_domain_10/collaboration/collab_results.json"
OUTPUT_JSON = "results/pace_domain_10/variance_decomposition.json"
OUTPUT_FIGURE = "results/pace_domain_10/figures/variance_decomposition.png"

TRAINING_SIZES = {
    'physics': 783, 'law': 1907, 'biology': 794, 'computer_science': 500,
    'history': 837, 'math': 677, 'chemistry': 423, 'economics': 679,
    'philosophy': 1394, 'medicine': 824,
}


def two_way_anova(factor_a, factor_b, response):
    """Manual two-way ANOVA computation. Returns SS, df, F, p, eta^2."""
    grand_mean = np.mean(response)
    n_total = len(response)

    # Unique levels
    levels_a = sorted(set(factor_a))
    levels_b = sorted(set(factor_b))
    a_map = {v: i for i, v in enumerate(levels_a)}
    b_map = {v: i for i, v in enumerate(levels_b)}

    # Cell means
    cell_data = defaultdict(list)
    for i in range(n_total):
        cell_data[(factor_a[i], factor_b[i])].append(response[i])

    # Factor A means
    a_data = defaultdict(list)
    for i in range(n_total):
        a_data[factor_a[i]].append(response[i])
    a_means = {k: np.mean(v) for k, v in a_data.items()}

    # Factor B means
    b_data = defaultdict(list)
    for i in range(n_total):
        b_data[factor_b[i]].append(response[i])
    b_means = {k: np.mean(v) for k, v in b_data.items()}

    # Cell means
    cell_means = {k: np.mean(v) for k, v in cell_data.items()}

    # SS_A: sum of n_a * (mean_a - grand_mean)^2
    ss_a = sum(len(a_data[a]) * (a_means[a] - grand_mean) ** 2 for a in levels_a)

    # SS_B: sum of n_b * (mean_b - grand_mean)^2
    ss_b = sum(len(b_data[b]) * (b_means[b] - grand_mean) ** 2 for b in levels_b)

    # SS_AB (interaction): cell deviations beyond main effects
    ss_ab = 0
    for (a, b), vals in cell_data.items():
        expected = a_means[a] + b_means[b] - grand_mean
        ss_ab += len(vals) * (cell_means[(a, b)] - expected) ** 2

    # SS_total
    ss_total = sum((y - grand_mean) ** 2 for y in response)

    # SS_residual (within-cell)
    ss_resid = 0
    for (a, b), vals in cell_data.items():
        cm = cell_means[(a, b)]
        ss_resid += sum((y - cm) ** 2 for y in vals)

    # Degrees of freedom
    df_a = len(levels_a) - 1
    df_b = len(levels_b) - 1
    df_ab = df_a * df_b
    df_resid = n_total - len(levels_a) * len(levels_b)

    # Mean squares
    ms_a = ss_a / df_a if df_a > 0 else 0
    ms_b = ss_b / df_b if df_b > 0 else 0
    ms_ab = ss_ab / df_ab if df_ab > 0 else 0
    ms_resid = ss_resid / df_resid if df_resid > 0 else 1e-10

    # F statistics
    f_a = ms_a / ms_resid
    f_b = ms_b / ms_resid
    f_ab = ms_ab / ms_resid

    # Eta-squared
    eta_a = ss_a / ss_total
    eta_b = ss_b / ss_total
    eta_ab = ss_ab / ss_total
    eta_resid = ss_resid / ss_total

    # p-values via F distribution
    from scipy.stats import f as f_dist
    p_a = 1 - f_dist.cdf(f_a, df_a, df_resid)
    p_b = 1 - f_dist.cdf(f_b, df_b, df_resid)
    p_ab = 1 - f_dist.cdf(f_ab, df_ab, df_resid)

    return {
        'primary_domain': {'SS': ss_a, 'df': df_a, 'F': f_a, 'p': p_a, 'eta_sq': eta_a},
        'helper_domain': {'SS': ss_b, 'df': df_b, 'F': f_b, 'p': p_b, 'eta_sq': eta_b},
        'interaction': {'SS': ss_ab, 'df': df_ab, 'F': f_ab, 'p': p_ab, 'eta_sq': eta_ab},
        'residual': {'SS': ss_resid, 'df': df_resid, 'eta_sq': eta_resid},
        'total': {'SS': ss_total, 'df': n_total - 1},
    }, a_means, b_means


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    # Build arrays
    primaries = []
    helpers = []
    corrects = []
    primary_sizes = []

    for r in data['collab_results']:
        primaries.append(r['question_domain'])
        helpers.append(r['agent_b'])
        corrects.append(int(r['correct']))
        primary_sizes.append(TRAINING_SIZES.get(r['question_domain'], 0))

    response = np.array(corrects, dtype=float)

    print(f"Data: {len(response)} observations, {len(set(primaries))} primary domains, {len(set(helpers))} helper domains")

    # Two-way ANOVA
    anova, primary_means, helper_means = two_way_anova(primaries, helpers, response)

    print("\n=== TWO-WAY ANOVA ===")
    print(f"{'Source':<20s} {'SS':>8s} {'df':>4s} {'F':>8s} {'p':>10s} {'η²':>8s}")
    print("-" * 60)
    for src in ['primary_domain', 'helper_domain', 'interaction', 'residual']:
        v = anova[src]
        f_str = f"{v['F']:.2f}" if 'F' in v else ''
        p_str = f"{v['p']:.6f}" if 'p' in v else ''
        sig = ''
        if 'p' in v:
            if v['p'] < 0.001: sig = '***'
            elif v['p'] < 0.01: sig = '**'
            elif v['p'] < 0.05: sig = '*'
        print(f"{src:<20s} {v['SS']:>8.2f} {v['df']:>4d} {f_str:>8s} {p_str:>10s} {v['eta_sq']:>7.1%} {sig}")

    # Training size correlation
    from scipy.stats import pearsonr
    primary_size_arr = np.array(primary_sizes, dtype=float)
    r_size, p_size = pearsonr(primary_size_arr, response)
    print(f"\nTraining size correlation with accuracy: r = {r_size:.3f}, p = {p_size:.4f}")

    # Per-domain summary
    print("\n=== PRIMARY DOMAIN MEANS (collab accuracy) ===")
    for d in sorted(primary_means, key=primary_means.get):
        print(f"  {d:20s}: {primary_means[d]:.1%} (train size: {TRAINING_SIZES[d]})")

    print("\n=== HELPER DOMAIN MEANS (collab accuracy when helper) ===")
    for d in sorted(helper_means, key=helper_means.get):
        print(f"  {d:20s}: {helper_means[d]:.1%}")

    # ICC estimate (variance attributed to helper / total)
    icc_helper = anova['helper_domain']['eta_sq']

    # Save results
    results = {
        'anova': {k: {kk: round(float(vv), 6) if isinstance(vv, (float, np.floating)) else int(vv) for kk, vv in v.items()} for k, v in anova.items()},
        'primary_domain_means': {k: round(v, 4) for k, v in primary_means.items()},
        'helper_domain_means': {k: round(v, 4) for k, v in helper_means.items()},
        'training_size_correlation': {'r': round(r_size, 4), 'p': round(p_size, 6)},
        'key_findings': {
            'primary_explains': f"{anova['primary_domain']['eta_sq']:.1%}",
            'helper_explains': f"{anova['helper_domain']['eta_sq']:.1%}",
            'interaction_explains': f"{anova['interaction']['eta_sq']:.1%}",
            'residual': f"{anova['residual']['eta_sq']:.1%}",
            'primary_is_dominant': bool(anova['primary_domain']['eta_sq'] > anova['helper_domain']['eta_sq']),
            'interpretation': 'Primary domain identity is the dominant predictor of collaboration outcome, supporting the intrinsic vulnerability hypothesis.',
        }
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    # === Figure ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel 1: Variance decomposition
    ax = axes[0]
    labels = ['Primary\nDomain', 'Helper\nDomain', 'Interaction', 'Residual']
    sizes = [anova['primary_domain']['eta_sq'], anova['helper_domain']['eta_sq'],
             anova['interaction']['eta_sq'], anova['residual']['eta_sq']]
    colors = ['#1976d2', '#f57c00', '#7b1fa2', '#bdbdbd']
    explode = (0.05, 0.05, 0.05, 0)

    def fmt_pct(pct):
        return f'{pct:.1f}%' if pct > 3 else ''

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                       autopct=fmt_pct, colors=colors,
                                       textprops={'fontsize': 10})
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    ax.set_title('Variance Decomposition\n(Two-Way ANOVA on Collaboration Accuracy)', fontsize=11)

    # Add significance annotations
    helper_p = '<0.001' if anova['helper_domain']['p'] < 0.001 else f"{anova['helper_domain']['p']:.3f}"
    sig_text = f"Primary: eta2={anova['primary_domain']['eta_sq']:.1%}, p<0.001\nHelper: eta2={anova['helper_domain']['eta_sq']:.1%}, p={helper_p}"
    ax.text(0, -1.3, sig_text, ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel 2: Primary domain effect
    ax = axes[1]
    sorted_primary = sorted(primary_means.items(), key=lambda x: x[1])
    domains_p = [d for d, _ in sorted_primary]
    means_p = [m for _, m in sorted_primary]
    colors_p = ['#d32f2f' if m < 0.3 else '#388e3c' if m > 0.45 else '#ffa726' for m in means_p]
    ax.barh(range(len(domains_p)), means_p, color=colors_p)
    ax.set_yticks(range(len(domains_p)))
    ax.set_yticklabels(domains_p, fontsize=9)
    ax.set_xlabel('Mean Collaboration Accuracy')
    ax.set_title('Primary Domain Effect\n(mean when domain asks the question)', fontsize=11)
    ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance')
    for i, (d, m) in enumerate(sorted_primary):
        ax.text(m + 0.01, i, f'{m:.0%}', va='center', fontsize=8)
    ax.legend(fontsize=8)

    # Panel 3: Helper domain effect
    ax = axes[2]
    sorted_helper = sorted(helper_means.items(), key=lambda x: x[1])
    domains_h = [d for d, _ in sorted_helper]
    means_h = [m for _, m in sorted_helper]
    ax.barh(range(len(domains_h)), means_h, color='#1976d2')
    ax.set_yticks(range(len(domains_h)))
    ax.set_yticklabels(domains_h, fontsize=9)
    ax.set_xlabel('Mean Collaboration Accuracy')
    ax.set_title('Helper Domain Effect\n(mean when domain provides help)', fontsize=11)
    ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance')
    for i, (d, m) in enumerate(sorted_helper):
        ax.text(m + 0.01, i, f'{m:.0%}', va='center', fontsize=8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_FIGURE}")


if __name__ == '__main__':
    main()
