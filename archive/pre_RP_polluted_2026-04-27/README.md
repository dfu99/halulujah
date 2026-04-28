# Archived: pre-reasoning-preserved (pre-RP) data and derived artifacts

*Archived: 2026-04-27.* These are the original PACE 10-domain n=20
trial-level outcomes plus everything computed from them. They are
*not* current research data and *must not* be used in any paper claim
or analysis pipeline going forward.

## Why archived

Inspection of the trial-level chain field on 2026-04-27 surfaced that
the 1.7B specialists used in this run were trained *before* we
implemented the reasoning-preserved (RP) format fix. As a result, the
"reasoning chains" in `pace_domain_10/collaboration/collab_results.json`
show empty `<think></think>` blocks followed by direct multiple-choice
answer letters across all three rounds, with later agents copying the
primary's verbatim answer rather than producing independent reasoning.

The 22x cluster-corrected primary/helper variance ratio in
`stats_rework/summary.json` was computed from these chains. Although the
statistical method is sound, the underlying chains were not actually
deliberating, so the asymmetry attribution may be measuring an
empty-think-tag artifact rather than a true reasoning collaboration
effect.

The current paper draft (paper/paper_v1.md v3.x) cited the 22x ratio,
the 10x10 ANOVA result, and the per-domain row-mean CIs. *All of these
need to be re-derived from N=200 reasoning-preserved-format chains*
before they go in the abstract.

## What is in this archive

- `pace_domain_10/`: trial-level outcomes, per-pair summaries, ANOVA
  variance decomposition, bootstrap CIs, chain-failure-mode
  classifications. Pre-RP format. Empty think tags.
- `stats_rework/`: question-clustered bootstrap, mixed-effects
  variance decomposition, BH-FDR pair table. *Computed from
  pre-RP data; not valid until re-run on RP data.*
- `cka/`: streaming CKA on LoRA delta-W matrices for the 10 pre-RP
  adapters; token-space divergence summary. The CKA result
  (r = -0.055) is at the weight-space level and is not directly
  invalidated by the empty-think-tag issue, but the adapters
  themselves were trained for the empty-think regime.
- `fig1_society_heatmap_PRE_RP.png`,
  `fig_stats_rework_summary_PRE_RP.png`,
  `fig_lora_cka_PRE_RP.png`, `fig5_token_space_PRE_RP.png`,
  `fig6_mechanism_PRE_RP.png`: figures derived from the above.

## Replacement schedule

The 1.7B 10x10 N=200 run currently in progress on /workspace/adapters_1p7b
trains specialists with the RP format. When that completes, the
following must be regenerated against the RP data and committed to
`results/` and `figures/`:

1. 10x10 collab-delta heatmap with marginal row/col means.
2. Question-clustered bootstrap row means + variance ratio + Spearman
   rho between scales.
3. CKA on the new RP adapters (different from these archived ones).
4. Token-space and subtopic mechanism follow-ups against RP row-mean.

After regeneration, the paper abstract and §1.3 / §4.1 / §5 numbers
must be updated. Do not cite the archived numbers anywhere going
forward.

## Provenance

- Original training: PACE A100 batch, 2026-03-31, n=20 questions per
  pair, alternating CoT, no reasoning preservation.
- Original run code: `src/scripts/run_domain_experiment.py` (see git
  history pre-2026-04-19).
- Lessons.md flagged the empty-think-tag pathology with the fix on
  2026-04-12, but the resulting bad chains were not retroactively
  removed from the analysis pipeline. The 2026-04-27 audit caught
  it; this archive is the cleanup.
