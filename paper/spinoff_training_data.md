# Spinoff Outline: What Training Data Produces Collaboration-Robust Specialists

*Target venue:* EMNLP 2026 long paper or NeurIPS 2026 main track
(with sufficient ablations).
*Lead argument:* Our 75/25 reasoning+domain data mix experiment and
the observed asymmetry between narrow-domain specialists (chemistry;
2 subjects) and broad-domain specialists (philosophy; logic + ethics
+ scenarios) suggest that training data composition predicts a
specialist's collaborativeness. We make this hypothesis testable and
isolate the contribution of data composition vs adaptation method.

## Key figure

A 2×2 ablation grid: {narrow, broad} × {LoRA, full FT} training data
on 5 domains, with collaboration delta per cell. Demonstrates which
factor is primary and whether they interact.

## Evidence we already have

- Solo baselines for 10 LoRA specialists on Qwen3-1.7B, each trained
  on a specific MMLU subject cluster (chemistry 2 subjects, philosophy
  4 subjects, law 3 subjects, etc.).
- Collaboration deltas per specialist, from the 10-domain study.
- Preliminary 75/25 reasoning+domain mix result on the RP mediator
  (obj-021), showing the mix repairs the catastrophic mediator case.

## Evidence we still need

- Controlled training-set-size ablation: train one specialist per
  domain with an equal-sized training set, so training-data quantity
  is held constant.
- Controlled training-set-breadth ablation: train two specialists on
  the same domain, one with broad subtopic coverage and one with
  narrow coverage, matched on total examples.
- Repeat both ablations under LoRA and full FT to disentangle the
  data-composition effect from the adaptation-method effect.

## Why it does not belong in the main paper

The main paper claims the *training method* is the dominant variable.
A full data-composition analysis would require 6-12 additional
specialists and a second round of collab evaluations. That work
belongs in a follow-up paper that specifically isolates data effects
while holding method constant.

## Draft section sketch

1. Introduction: the overlooked role of training-data composition in
   specialist behaviour. Prior evidence of an asymmetry (our pilot,
   Bayesian-LoRA training data sensitivity).
2. Setup: controlled training-set ablations at Qwen3-1.7B and 4B.
3. Results: 2×2 ablation grid; which factor dominates (data vs
   method); whether they interact.
4. Discussion: implications for specialist-curation policies.
5. Limitations: MMLU-bounded domain definitions; single base family.

## Status

Stubbed 2026-04-23. Requires 6-12 new specialist training runs
(controlled ablations) and a second round of collab evaluations.
Estimated 3 months of additional work.
