# Spinoff Outline: Post-Fine-Tuning Calibration in Specialist-Specialist Collaboration

*Target venue:* NeurIPS 2026 workshop short paper (Efficient Methods /
ML for Alignment / Safe AI) or EMNLP 2026 Findings.
*Lead argument:* Bayesian-LoRA (arXiv 2601.21003) showed that post
fine-tuning, adapter-trained models are systematically overconfident at
matched-MAP accuracy. We measure the behavioural consequence of that
calibration deterioration in a two-agent collaboration protocol and
show that overconfidence translates directly into a pathological
switch-quality ratio (C2W / W2C) that prevents the primary agent from
recovering incorrect peer answers.

## Key figure

C2W / W2C ratio as a function of post-fine-tuning calibration
deterioration, across a rank sweep (r=4..128 at 4B). Shows a monotonic
relationship between calibration sharpness and switch pathology.

## Evidence we already have

- N=200 C2W / W2C decomposition across 1.7B 5-domain sweep + 4B medicine/physics
  (full FT and LoRA r=16, r=128).
- Switch classification from `results/paper_sweep/` JSONs.
- A rank sweep that shows C2W / W2C ratio worsens (from 1.4× to 19×)
  as rank grows past r=16.

## Evidence we still need

- Direct calibration metric on each specialist (expected calibration
  error, ECE; or Brier score) at matched solo. Requires a one-shot
  logit-capture re-run on the saved chain data (CPU-feasible).
- Cross-domain calibration: does the LoRA specialist show worse ECE on
  its home domain than the FT specialist does on its home domain?

## Why it does not belong in the main paper

The main paper's single claim is about the *training method* and the
rank-constrained weight-space insertion. Calibration is a proximate
cause but not the mechanism, and bundling the two confuses scope. By
keeping the calibration story as a separate short paper, the main paper
stays disciplined around the weight-space mechanism (intruder dimensions,
rank constraint) while the calibration paper inherits the empirical
C2W / W2C data and extends it to an explicit calibration metric.

## Draft section sketch

1. Introduction: post-fine-tuning overconfidence is a documented
   single-agent phenomenon (Bayesian-LoRA). We extend to collab.
2. Setup: matched-MAP accuracy, rank sweep, N=200 collab pairs.
3. Results: C2W / W2C rises monotonically with rank past r=16; ECE rises
   in parallel; the two correlate at ρ>0.9 across conditions.
4. Discussion: deployment implications for multi-agent scaffolds built
   on LoRA specialists. Gating strategies (abstain when ECE>threshold).
5. Limitations: single base-model family, no cross-architecture control.

## Status

Stubbed 2026-04-23. Reuses all of the main paper's existing data with
one additional logit-capture inference pass.
