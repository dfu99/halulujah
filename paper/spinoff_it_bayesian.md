# Spinoff Outline: An Information-Theoretic Framework for LoRA Multi-Agent Collaboration

*Target venue:* AISTATS 2027 or a NeurIPS workshop on information
theory in deep learning.
*Lead argument:* Collaboration success between two LLM agents is, at
the decision level, a conditional mutual information problem. The
primary agent's final answer distribution, conditioned on the partner's
reasoning chain, should have lower entropy about the correct answer than
the unconditioned distribution. We formalize this as
I(chain_B ; Y | chain_A, X) and show that LoRA's rank-constrained
adaptation produces specialists with reduced conditional MI at matched
solo MAP accuracy, explaining the main-paper result from a first-
principles angle.

## Key theoretical construct

- Bayesian view: posterior precision = prior precision + likelihood
  precision. LoRA fine-tuning over-sharpens the prior P_A(y | x) at
  matched MAP, leaving the likelihood term P_A(chain_B | y, x) with
  less room to move the posterior.
- IT view: I(chain_B ; Y | chain_A, X) is the conditional MI that
  measures how much B's reasoning informs Y given A's own reasoning
  and the input. A low conditional MI predicts the low collaboration
  delta + high C2W / W2C we observe empirically.

## Evidence we already have

- The main paper's empirical C2W / W2C gap and compute-matched
  deliberation control provide a behavioural signature of the
  conditional-MI gap. This is the *empirical illustration* that would
  accompany the theoretical treatment.

## Evidence we still need

- A tractable estimator for I(chain_B ; Y | chain_A, X) in an LLM
  context. Candidates: variational lower bounds (MINE), k-NN entropy
  estimators on token-level distributions, or a shuffled-partner
  null distribution control.
- A posterior treatment of LoRA vs full-FT that makes the prior-
  precision claim measurable: Laplace approximation around the
  fine-tuned minimum, or an ensemble over LoRA random seeds to
  estimate epistemic uncertainty.
- Rate-distortion analysis of LoRA ΔW as a compressed representation
  of the full-FT ΔW, connecting the compression ratio (rank / full
  rank) to the behavioural consequence.

## Why it does not belong in the main paper

The conditional-MI / Bayesian framework is elegant and predictive but
requires substantial new machinery (Laplace posteriors, ensemble
training, MI estimation). Including it in the main paper would either
thin the theoretical rigor or bloat the page budget. A separate paper
does the theory justice with an empirical illustration that cites the
main paper for detail.

## Draft section sketch

1. Introduction: multi-agent collaboration as conditional MI.
   Bayesian framing of prior-precision vs likelihood trade-off.
2. Theory: the I(chain_B ; Y | chain_A, X) quantity, lower bounds,
   and its connection to posterior precision.
3. Main theorem: under rank-r adaptation with α/r scaling, the
   prior-precision term grows as O(f(r, α)) in a way that is
   quantifiable from the adapter's singular-value spectrum.
4. Empirical illustration: our C2W / W2C and compute-matched
   deliberation findings as behavioural realizations of the
   theoretical prediction.
5. Discussion: implications for adaptation methods that aim to
   preserve collaborative capacity.

## Status

Stubbed 2026-04-23. Needs substantial new theoretical development
plus a Laplace / ensemble posterior treatment that is out of scope
for the main submission. Estimated 2-3 months of additional work.
