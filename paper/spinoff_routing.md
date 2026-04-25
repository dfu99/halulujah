# Spinoff Outline: Task-Type Routing for Multi-Agent LLM Inference

*Target venue:* NeurIPS 2026 workshop short paper (Efficient Methods,
Agents, or LLM Systems) or ICLR 2027 main.
*Lead argument:* If the WHO-asymmetry from the main paper holds at
deployment time (recall-shaped requests are universally harmed by
multi-agent collaboration; reasoning-shaped requests are universally
helped), then a small auxiliary classifier that pre-labels incoming
requests as recall vs reasoning can decide whether to spend compute
on multi-agent inference. This saves compute on the recall side and
avoids the universally-harmed C2W collapse.

## Distinction from prior adaptive-collaboration work

- *Debate Only When Necessary* (arXiv 2504.05047) routes on question
  difficulty / model uncertainty.
- *SID* (arXiv 2510.06843) gates early-exit on per-agent confidence.
- *Our axis is task-type*: recall vs reasoning shape. Orthogonal to
  difficulty and confidence; informed by the primary-agent asymmetry
  finding.

## Pipeline sketch

1. Lightweight classifier (a small encoder or LLM-as-judge zero-shot
   prompt) labels each incoming request: {recall, reasoning, judgment}.
2. Recall-shaped requests are routed to single-agent inference (skip
   multi-agent compute, avoid C2W harm).
3. Reasoning-shaped requests are routed to multi-agent deliberation.
4. Judgment / mixed requests can fall through to a confidence-gated
   default (e.g. SID-style early-exit).

## Evaluation plan

- *Hold-out test*: classify all 10-domain MMLU questions. Compare end-to-end
  accuracy and total compute under three regimes: (a) always single-agent,
  (b) always multi-agent, (c) router-gated. Predict (c) > (a) > (b) on
  accuracy-per-flop.
- *Calibration of the router*: confusion matrix of router labels vs
  ground-truth task type (manual or LLM-judge labeled), per domain.
- *Robustness*: does the router transfer across model families?

## Evidence we already have (from main paper)

- 10-domain row-mean delta vector. Recall-y primaries (medicine, chemistry,
  physics, biology) are harmed. Reasoning-y (philosophy, law, math) are
  helped. The asymmetry is large enough that a 60-70%-accurate router
  would already produce a positive accuracy-per-flop result.
- 4B follow-up shows the WHO-asymmetry persists *as a phenomenon* but the
  specific harmed/helped domain identities are not preserved across scale.
  The router must therefore label individual requests, not domains, to be
  robust under model scale.

## Evidence we still need

- A labeled task-type ground-truth set. Manual labeling of 1000 MMLU
  questions, or LLM-judge labels validated on a 200-question sample.
- A trained classifier (small encoder fine-tuned on labels, or a
  prompt-based LLM-judge with measured calibration).
- An accuracy-per-flop comparison across the three regimes on the
  same model family.

## Why it does not belong in the main paper

The main paper documents the asymmetry. This paper acts on it. Mixing
the observation paper with a deployment-system paper would dilute both.
A separate spinoff lets the asymmetry paper stay disciplined while this
spinoff develops the routing system properly.

## Status

Stubbed 2026-04-25 after PI suggested the eval-pipeline angle. Reuses
all the main paper's observational evidence; needs a classifier and a
labeled task-type set as new artifacts.
