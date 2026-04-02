#!/usr/bin/env python3
"""Qualitative analysis of collapsed vs successful reasoning chains.

Addresses unanimous reviewer critique #3: need to SEE what collapsed reasoning looks like.
Categorizes failure modes: sycophantic deference, domain confusion, confident-wrong, extraction failure.
"""

import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

RESULTS_PATH = "results/pace_domain_10/collaboration/collab_results.json"
OUTPUT_JSON = "results/pace_domain_10/chain_analysis.json"
OUTPUT_FIGURE = "results/pace_domain_10/figures/chain_failure_modes.png"
OUTPUT_EXAMPLES = "results/pace_domain_10/chain_examples.md"


def classify_failure(record):
    """Classify a failed collaboration into a failure mode."""
    chain = record.get('chain', [])
    final = record.get('final_response', '')
    expected = record['expected']
    predicted = record['predicted']

    if not chain:
        return 'no_chain'

    # Extract text from chain
    thoughts = [step.get('thought', '') for step in chain]
    full_chain = ' '.join(thoughts)

    # Check if predicted is X (extraction failure)
    if predicted == 'X':
        return 'extraction_failure'

    # Check for answer switching: did agent_a initially have the right answer but switch?
    agent_a_thoughts = [s['thought'] for s in chain if s.get('agent') == record['agent_a']]
    agent_b_thoughts = [s['thought'] for s in chain if s.get('agent') == record['agent_b']]

    # Check if the expected answer letter appears in agent_a's first thought
    a_first = agent_a_thoughts[0] if agent_a_thoughts else ''
    a_had_correct = expected in a_first.split('\n')[-1] if a_first else False

    # Check for deference patterns
    deference_patterns = [
        r'agree', r'correct', r'you\'re right', r'good point',
        r'I think .* is right', r'as .* mentioned', r'following',
        r'building on', r'as suggested'
    ]
    deference_count = sum(1 for p in deference_patterns
                         if re.search(p, full_chain, re.IGNORECASE))

    # Check for domain jargon confusion (helper domain terms in primary reasoning)
    helper_domain = record['agent_b']

    # Classify
    if a_had_correct and predicted != expected:
        return 'answer_switch'  # Had right answer, switched after collaboration
    elif deference_count >= 2:
        return 'sycophantic_deference'  # Excessive agreement patterns
    elif len(full_chain) < 50:
        return 'truncated_reasoning'  # Very short chain
    elif len(full_chain) > 500 and predicted != expected:
        return 'overthinking'  # Long chain, wrong answer
    else:
        return 'confident_wrong'  # Confident but incorrect


def classify_success(record):
    """Classify how a successful collaboration succeeded."""
    chain = record.get('chain', [])
    if not chain:
        return 'no_chain'

    thoughts = [step.get('thought', '') for step in chain]
    full_chain = ' '.join(thoughts)

    agent_a_thoughts = [s['thought'] for s in chain if s.get('agent') == record['agent_a']]

    # Check if agent_a had wrong answer initially but corrected
    a_first = agent_a_thoughts[0] if agent_a_thoughts else ''
    expected = record['expected']

    if a_first and expected not in a_first.split('\n')[-1]:
        return 'correction'  # Helper corrected agent_a
    elif len(full_chain) > 300:
        return 'elaboration'  # Extended reasoning helped
    else:
        return 'confirmation'  # Helper confirmed correct answer


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    # Separate solo correct/incorrect for each domain
    solo_correct = defaultdict(list)
    for r in data['solo_results']:
        solo_correct[r['domain']].append(r['correct'])

    # Analyze collab results
    failed = []
    succeeded = []

    for r in data['collab_results']:
        if not r['correct']:
            r['failure_mode'] = classify_failure(r)
            failed.append(r)
        else:
            r['success_mode'] = classify_success(r)
            succeeded.append(r)

    # Count failure modes
    failure_counts = Counter(r['failure_mode'] for r in failed)
    success_counts = Counter(r['success_mode'] for r in succeeded)

    print("=== FAILURE MODES ===")
    for mode, count in failure_counts.most_common():
        print(f"  {mode:25s}: {count:4d} ({count/len(failed):.1%})")

    print(f"\n=== SUCCESS MODES ===")
    for mode, count in success_counts.most_common():
        print(f"  {mode:25s}: {count:4d} ({count/len(succeeded):.1%})")

    # Failure modes by primary domain
    failure_by_domain = defaultdict(lambda: Counter())
    for r in failed:
        failure_by_domain[r['question_domain']][r['failure_mode']] += 1

    success_by_domain = defaultdict(lambda: Counter())
    for r in succeeded:
        success_by_domain[r['question_domain']][r['success_mode']] += 1

    # Chain length analysis
    failed_lengths = [len(' '.join(s['thought'] for s in r.get('chain', []))) for r in failed]
    succeeded_lengths = [len(' '.join(s['thought'] for s in r.get('chain', []))) for r in succeeded]

    print(f"\n=== CHAIN LENGTH ===")
    print(f"  Failed mean:    {np.mean(failed_lengths):.0f} chars")
    print(f"  Succeeded mean: {np.mean(succeeded_lengths):.0f} chars")

    # Save analysis JSON
    analysis = {
        'total_collab': len(data['collab_results']),
        'total_failed': len(failed),
        'total_succeeded': len(succeeded),
        'failure_modes': dict(failure_counts),
        'success_modes': dict(success_counts),
        'failure_by_domain': {d: dict(c) for d, c in failure_by_domain.items()},
        'success_by_domain': {d: dict(c) for d, c in success_by_domain.items()},
        'chain_length': {
            'failed_mean': round(np.mean(failed_lengths), 1),
            'failed_median': round(np.median(failed_lengths), 1),
            'succeeded_mean': round(np.mean(succeeded_lengths), 1),
            'succeeded_median': round(np.median(succeeded_lengths), 1),
        }
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    # === Generate examples markdown ===
    with open(OUTPUT_EXAMPLES, 'w') as f:
        f.write("# Reasoning Chain Analysis: Collapsed vs Successful Examples\n\n")

        f.write("## Collapsed Chains (Wrong Answer After Collaboration)\n\n")
        # Pick examples from most harmful pairs
        shown = 0
        for r in sorted(failed, key=lambda x: x.get('failure_mode', '')):
            if shown >= 10:
                break
            mode = r.get('failure_mode', 'unknown')
            f.write(f"### Example {shown+1}: {r['agent_a']}+{r['agent_b']} on {r['question_domain']} — {mode}\n")
            f.write(f"Expected: {r['expected']} | Predicted: {r['predicted']}\n\n")
            for step in r.get('chain', []):
                agent = step.get('agent', '?')
                thought = step.get('thought', '')[:300]
                f.write(f"**{agent}**: {thought}\n\n")
            f.write("---\n\n")
            shown += 1

        f.write("\n## Successful Chains (Correct Answer After Collaboration)\n\n")
        shown = 0
        for r in sorted(succeeded, key=lambda x: x.get('success_mode', '')):
            if shown >= 10:
                break
            mode = r.get('success_mode', 'unknown')
            f.write(f"### Example {shown+1}: {r['agent_a']}+{r['agent_b']} on {r['question_domain']} — {mode}\n")
            f.write(f"Expected: {r['expected']} | Predicted: {r['predicted']}\n\n")
            for step in r.get('chain', []):
                agent = step.get('agent', '?')
                thought = step.get('thought', '')[:300]
                f.write(f"**{agent}**: {thought}\n\n")
            f.write("---\n\n")
            shown += 1

    print(f"Saved: {OUTPUT_EXAMPLES}")

    # === Figure: failure mode breakdown ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Panel 1: Overall failure mode distribution
    ax = axes[0]
    modes = list(failure_counts.keys())
    counts = [failure_counts[m] for m in modes]
    colors_map = {
        'extraction_failure': '#e53935',
        'confident_wrong': '#fb8c00',
        'sycophantic_deference': '#8e24aa',
        'answer_switch': '#d81b60',
        'truncated_reasoning': '#546e7a',
        'overthinking': '#1e88e5',
        'no_chain': '#bdbdbd',
    }
    colors = [colors_map.get(m, '#9e9e9e') for m in modes]
    bars = ax.barh(range(len(modes)), counts, color=colors)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([m.replace('_', ' ').title() for m in modes], fontsize=9)
    ax.set_xlabel('Count')
    ax.set_title('Failure Mode Distribution\n(all incorrect collab responses)', fontsize=11)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{count} ({count/len(failed):.0%})', va='center', fontsize=8)

    # Panel 2: Failure modes by domain (stacked)
    ax = axes[1]
    domains = sorted(failure_by_domain.keys())
    all_modes = sorted(set(m for c in failure_by_domain.values() for m in c))
    bottom = np.zeros(len(domains))
    for mode in all_modes:
        values = [failure_by_domain[d].get(mode, 0) for d in domains]
        ax.barh(range(len(domains)), values, left=bottom,
                label=mode.replace('_', ' ').title(),
                color=colors_map.get(mode, '#9e9e9e'), alpha=0.85)
        bottom += np.array(values)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=9)
    ax.set_xlabel('Failed Responses')
    ax.set_title('Failure Modes by Primary Domain', fontsize=11)
    ax.legend(fontsize=7, loc='lower right')

    # Panel 3: Chain length comparison
    ax = axes[2]
    ax.hist(failed_lengths, bins=30, alpha=0.6, color='#d32f2f', label=f'Failed (n={len(failed)})', density=True)
    ax.hist(succeeded_lengths, bins=30, alpha=0.6, color='#388e3c', label=f'Succeeded (n={len(succeeded)})', density=True)
    ax.set_xlabel('Chain Length (characters)')
    ax.set_ylabel('Density')
    ax.set_title('Reasoning Chain Length:\nFailed vs Succeeded', fontsize=11)
    ax.legend(fontsize=9)
    ax.axvline(np.mean(failed_lengths), color='#d32f2f', linestyle='--', alpha=0.7)
    ax.axvline(np.mean(succeeded_lengths), color='#388e3c', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_FIGURE}")


if __name__ == '__main__':
    main()
