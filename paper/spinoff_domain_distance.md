# Spinoff Outline: KL-Divergence Domain Distance as a Predictor of Collaboration Outcomes

*Target venue:* ACL 2026 ARR or EMNLP 2026 Findings.
*Lead argument:* A principled distance metric between two domain
specialists (KL divergence between their output distributions on shared
probes) predicts whether their pairwise collaboration is net helpful or
harmful. Our cross-evaluation accuracy gap (r=0.197 at n=20 in the PACE
10-domain study) was a weak proxy for this quantity; replacing it with
KL divergence from the Pivot A pipeline promotes the correlation to a
publishable signal.

## Key figure

Scatter of KL(specialist_A || specialist_B) against collaboration delta
over 90 ordered pairs from the 10-domain study. Regression line with
bootstrap confidence band; color by whether the pair helped or harmed.

## Evidence we already have

- 10 LoRA specialist adapters on Qwen3-1.7B across {medicine, physics,
  biology, chemistry, mathematics, computer science, law, philosophy,
  history, economics}.
- 90 ordered collaboration pairs at n=20, with accuracy deltas per pair.
- A cross-evaluation accuracy matrix (which domain gets what accuracy
  on another's test set).
- Pivot A's persona-fingerprint KL-divergence pipeline, which computes
  token-level KL between two models on the same probe inputs.

## Evidence we still need

- Run the Pivot A KL pipeline on the 10 domain specialists, over a
  shared probe set of MMLU questions (not just the cross-eval pairs).
  This gives a 10×10 KL matrix.
- Redo the scatter with the KL metric instead of the cross-eval gap;
  compute Pearson r, Spearman ρ, and bootstrap confidence.

## Why it does not belong in the main paper

The main paper claims the *training method* is the dominant control
variable; partner selection (including domain distance) is secondary.
Adding a domain-distance analysis to the main paper risks diluting the
single claim. As a separate paper, it completes the picture for readers
who are specifically interested in partner selection rather than
training-method effects.

## Draft section sketch

1. Introduction: partner selection in multi-agent LLM systems. The
   distance-predicts-outcome hypothesis, tested previously only with
   weak proxies.
2. Setup: 10 domain specialists at Qwen3-1.7B. KL divergence
   computation using shared MMLU probe set.
3. Results: KL vs collaboration delta; comparison to the cross-eval
   proxy; partial regression controlling for solo accuracy.
4. Discussion: routing implications for multi-agent systems.
5. Limitations: small n per pair (n=20); single base-model family.

## Status

Stubbed 2026-04-23. Reuses all 10-domain collab data plus a one-off
KL-divergence pass with Pivot A's existing scripts.
