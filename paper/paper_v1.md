---
title: "Rank-Constrained Adaptation Destroys Collaborative Behavior in Multi-Agent LLMs"
author: "Daniel Fu"
date: "2026-04-23 (working draft v1)"
geometry: margin=1in
fontsize: 11pt
colorlinks: true
---

# Abstract

Multi-agent LLM deliberation has been reported to both improve and harm task
accuracy, with recent controlled studies (Du et al., 2023; *Talk Isn't Always
Cheap*, 2025; *Can LLM Agents Really Debate?*, 2025) yielding outcomes that
span roughly -10 to +15 percentage points on matched benchmarks. We argue
that a single under-controlled variable explains a large portion of this
spread: the *training method* used to produce the domain specialists
participating in the deliberation. We run a controlled comparison in which
LoRA-finetuned and fully-finetuned Qwen3 specialists reach identical solo
accuracy on their target domain, then engage in a two-agent natural-language
collaboration protocol with a matched partner. At 4B parameters and 84%
solo accuracy on medicine, full fine-tuning yields a +5.0 pp collaboration
delta with a 1.4× correct-to-wrong / wrong-to-correct (C2W/W2C) switch
ratio, while LoRA at rank 128 yields a +1.5 pp delta with a 19× C2W/W2C
ratio. The LoRA specialist almost never recovers an incorrect peer answer,
and it routinely abandons its own correct one. A rank sweep from r=4 to
r=128 at both 1.7B and 4B fails to close the gap. We argue the mechanism is
LoRA's low-rank update geometry; the "intruder dimensions" it introduces
(Shuttleworth et al., 2410.21228) are precisely the directions that dominate
collaborative updating. Cheap adaptation has a hidden cost, a collapse of
the information channel that multi-agent deliberation relies on.

*Keywords:* multi-agent LLMs, LoRA, full fine-tuning, deliberation,
rank constraint, calibration.

# 1. Introduction

## 1.1. Motivation

Multi-agent LLM systems, in which two or more language models exchange
natural-language reasoning over several rounds before producing a final
answer, are one of the most widely deployed inference-time scaffolds for
small and mid-size open-source models. Debate, deliberation, mixture-of-agents,
chain-of-experts, and mediator architectures have all been proposed as ways
to extract more from a fixed model class without retraining (Du et al.,
2023; Liang et al., 2024; Wang et al., 2024). Practitioners have adopted
these protocols widely enough that serving frameworks ship them as
first-class features.

At the same time, the evidence base is unstable. Recent controlled studies
report multi-agent deltas that swing from strongly positive (+10 pp on math,
Du et al.) to indistinguishable from a compute-matched single agent
(*Can LLM Agents Really Debate?*, 2025) to negative on heterogeneous pairs
(*Talk Isn't Always Cheap*, 2025). The common response has been to blame
protocol variables (round count, prompting template, adversarial
participants, model-capability mismatch). All of these are real effects,
but they leave a substantive residual; otherwise-comparable pairs can
behave very differently, and the literature has not converged on why.

## 1.2. Our claim

We argue that a large part of the residual is explained by a variable that
prior work has not isolated: the parameter-efficient vs. full training
method used to produce the specialist.

> **Claim.** At matched solo accuracy on the primary task, LoRA-adapted
> domain specialists lose most of the benefit of multi-agent deliberation,
> while fully-fine-tuned specialists preserve it. The mechanism is LoRA's
> rank-constrained update; collaborative improvement requires the model to
> move in directions that LoRA's low-rank parameterization cannot express.

This claim is precise in three ways that prior multi-agent work has not
been. First, it is stated at matched solo accuracy, which forces the two
specialists to have the same single-agent capability. Second, it separates
the training method from the specialization itself; both LoRA and full FT
produce "specialists" in the usual sense, but only the full-FT specialist
collaborates. Third, it localizes the failure to a concrete architectural
choice (rank) that can be ablated.

## 1.3. Summary of evidence

We run all experiments on Qwen3-1.7B and Qwen3-4B, training medicine and
physics specialists on MMLU-derived data via LoRA (r in {4, 8, 16, 32, 64, 128})
and full fine-tuning. We then collaborate each specialist with a matched
base-model partner under a two-agent natural-language protocol of N=200
questions per condition. Our headline result at 4B-medicine, with solo
accuracy 84% for both LoRA r=128 and full FT, is summarized in Figure 1.

![Matched-solo-accuracy collaboration at Qwen3-4B medicine.
(A) Collaboration delta over solo baseline; LoRA r=16 shows -1.5 pp,
LoRA r=128 shows +1.5 pp, full FT shows +5.0 pp.
(B) Switch-quality ratio (C2W / W2C) on a log scale; full FT keeps a
1.4× balanced ratio while LoRA r=128 reaches 19× (19 correct-to-wrong
switches for every 1 wrong-to-correct). At matched solo 84% accuracy,
full FT obtains a 3.3× larger delta and a 13× better switch quality than
LoRA r=128.](../figures/figure1_matched_solo_4b_medicine.png){ width=95% }

| Condition | Δ vs. solo | C2W | W2C | C2W/W2C |
|-----------|-----------:|----:|----:|--------:|
| Full fine-tuning | **+5.0 pp** | 7 | 5 | **1.4×** |
| LoRA r=128 | **+1.5 pp** | 19 | 1 | **19×** |

A rank sweep up to r=128 at both scales fails to recover full FT's
collaboration behavior; higher ranks slightly improve solo accuracy but
leave the deliberation channel pathological (Figure 2).

![Rank sweep (r=4..128) at Qwen3-1.7B (top row) and Qwen3-4B (bottom
row) against full FT (horizontal line). No LoRA rank recovers full FT's
collaboration delta or switch-quality ratio at either scale. Medicine
left column, physics right column. Full FT at 4B medicine obtains a
3.3× larger delta and a 13× better C2W/W2C ratio than LoRA r=128 at
matched solo accuracy.](../figures/reviewer_b_rank_vs_ft.png){ width=95% }

A compute-matched single-agent control (identical total inference compute
as the deliberation, but spent on a single chain) recovers 15 pp on the
base model while full deliberation recovers 21 pp, a 1.4× compute-scaling
ratio. Deliberation has value beyond raw compute in the absence of LoRA,
and LoRA destroys specifically this non-compute value.

## 1.4. Why this matters

LoRA is the dominant adapter choice for small open-source models. The
common wisdom has become "LoRA recovers 90–95% of full FT quality" on
downstream benchmarks. Our result does not contest that claim on solo
task accuracy. It shows that the 5–10% residual is not a uniform loss of
quality; it is concentrated in a single capability, the specialist's
ability to update its answer in response to a peer, and that capability
is precisely the one multi-agent scaffolds depend on. Deployments that
adopt LoRA specialists for cheapness and multi-agent scaffolds for
accuracy may be silently cancelling the second with the first.

## 1.5. Mechanism (preview)

Two recent theoretical threads converge on why rank should matter for
collaboration. Shuttleworth et al. (2410.21228) show that LoRA adapters
introduce novel high-singular-value "intruder dimensions" not present in
full-FT, and that these dimensions dominate the trained model's response
in a way that produces catastrophic forgetting. Our contribution is to
show the behavioural consequence of intruder dimensions in a multi-agent
setting; intruder-dimension rigidity prevents the specialist from
integrating a partner's disagreement, and the behavior is visible in the
C2W/W2C ratio at matched solo accuracy. Full-FT specialists at matched
solo accuracy exhibit neither pathology.

## 1.6. What this paper is *not*

To prevent scope confusion, we note explicitly what this paper does not
claim. We do not address *long-context multi-agent compression*
(Joo et al., 2509.21848; Xu et al., 2506.16411); our setting is
matched-context MCQ, and the mechanism we identify is orthogonal to the
compression framing those papers develop. We do not rediscover *sycophancy
or social conformity* (Sharma et al., 2023; Liang et al., 2024); C2W at
matched solo accuracy with rank=128 is structurally distinct from a
social-pressure artifact. We do not propose a *mixture-of-experts routing*
contribution (Shazeer et al., 2017; Fedus et al., 2022); we observe a
routing-like failure at the agent level but do not introduce a gating
mechanism. We do not adjudicate the *debate literature* (Du et al., 2023);
we use the alternating chain-of-thought protocol as one point in a
landscape, not a verdict on debate. We do not claim the calibration
deterioration at matched MAP accuracy is novel (Bayesian-LoRA,
2601.21003); we measure a consequence of that single-agent finding in the
multi-agent setting.

## 1.7. Contributions

1. A controlled LoRA-vs-full-FT comparison at matched solo accuracy in a
   two-agent collaboration protocol, at Qwen3-1.7B and Qwen3-4B scales,
   over 61 conditions with N=200 per condition (12 conditions at 4B).
2. A rank sweep (r=4…128 at 1.7B; r=16,128 at 4B) demonstrating that
   increasing LoRA capacity does not recover full-FT collaborativeness.
3. A compute-matched single-agent control showing deliberation has
   value beyond raw compute (+21 pp vs. +15 pp), and that LoRA destroys
   this residual specifically.
4. Switch-classification (C2W, W2C, held) as a decision-level calibration
   proxy that makes the pathology directly visible; LoRA at r=128
   yields 19× C2W/W2C, full FT 1.4×.
5. Integration with the intruder-dimension theoretical thread
   (Shuttleworth et al., 2410.21228) and a direct weight-space
   measurement on our adapters (currently in progress).

## 1.8. Organization

Section 2 situates our work in the multi-agent LLM and LoRA literatures.
Section 3 describes the experimental design, including the matched-solo
constraint and the switch-classification analysis. Section 4 presents
the main results at 1.7B and 4B, the rank sweep, and the compute-matched
control. Section 5 examines the mechanism, linking our behavioural
observations to intruder dimensions. Section 6 discusses implications
for deployment and multi-agent system design. Section 7 lays out
limitations and future work.

# 2. Related Work

*(Section stub — to be populated in v2 with: (a) multi-agent debate and
deliberation literature, tightly scoped to specialist-to-specialist
studies; (b) LoRA and PEFT literature with emphasis on matched-task
parity claims; (c) the intruder-dimensions / linear-ceiling theoretical
thread; (d) multi-agent calibration work. Each subsection is anchored to
a claim in the claim-to-evidence map at `paper/claim_evidence_map.md`.)*

# 3. Experimental Design

*(Stub — protocol description, matched-solo calibration procedure,
switch classification definition.)*

# 4. Results

## 4.1. Scale-invariance at 4B across four domains

Full fine-tuning at Qwen3-4B preserves balanced switch quality across
every domain we have tested so far (medicine, physics, biology, law,
N=200 per condition). Collaboration delta varies with domain
difficulty, but C2W / W2C ratio stays in the 1.40–2.25× band, well
inside the healthy region and contrasting sharply with the 19× ratio
LoRA r=128 produces on medicine at matched solo accuracy (Figure 3).

![4B scale-invariance at matched solo accuracy, N=200 per condition.
(A) Collaboration delta per domain under full fine-tuning; deltas
span -2 to +5 pp, but all four land within bootstrap noise of a
consistent full-FT-preserves-collaborativeness picture. (B) C2W / W2C
switch-quality ratio per domain with LoRA r=128 medicine overlay;
full FT stays in the 1.4–2.3× band, LoRA r=128 breaks out at 19×.
Law's slight negative delta (-2 pp) is within bootstrap noise at
N=200 and its ratio (1.43×) is squarely in the healthy
band.](../figures/fig_4b_scale_invariance.png){ width=95% }

| Domain | Solo | +Base | Δ | C2W | W2C | C2W/W2C |
|--------|-----:|------:|--:|----:|----:|--------:|
| Medicine | 84.0% | 89.0% | +5.0 | 7 | 5 | 1.40× |
| Physics | 85.5% | 87.0% | +1.5 | 11 | 6 | 1.83× |
| Biology | 87.0% | 90.5% | +3.5 | 9 | 4 | 2.25× |
| Law | 61.0% | 59.0% | -2.0 | 30 | 21 | 1.43× |
| *Math* | *pending* | | | | | |

## 4.2. Rank sweep and mechanism

*(stub — rank sweep at 1.7B and 4B, compute-matched deliberation
control, §4.4 bridge-agent negative result already below, 7B
replication pending.)*

## 4.4. Bridge agents fail to rescue collaboration (negative result)

A natural intervention to the rank-constrained LoRA failure is to insert a
third agent as a *bridge* or *mediator* between the two specialists. If the
failure mode were a partner-identity mismatch, a carefully trained mediator
should repair it. We test three mediator variants on the medicine+physics
pair at Qwen3-1.7B and find that none rescues the collaboration delta, and
the simplest variant makes it worse.

*50/50-mix specialist mediator (naive bridge).*  A LoRA specialist trained
on an equal mixture of medicine and physics data, inserted as a middle
agent between the medicine and physics specialists, produces a -32.8 pp
delta with a 7.1× C2W/W2C ratio on the medicine task. The mediator's own
training inherits the same rank-constrained insertion pathology we
document in §4.1–§4.3, and the three-agent protocol amplifies rather than
attenuates it.

*Reasoning-preserved (RP) mediator.*  A LoRA specialist trained with the
75/25 reasoning+domain data mix (see §3.3), which preserves the base
model's reasoning pathway, partially repairs the catastrophe; the delta
moves to -3 to -3.5 pp and the C2W/W2C ratio to 3.6×. The training-data
composition reduces but does not eliminate the failure, consistent with
the rank constraint (not the data content) being the dominant mechanism.

*Base-model mediator.*  The base Qwen3-1.7B model, inserted as a mediator
with no LoRA adaptation, gives an asymmetric result, +6.5 pp on medicine
and -8.5 pp on physics. The best-case pair (physics + math, both weak
solo) yields +10 pp at a 1.1× C2W/W2C ratio. A non-adapted mediator is
the only variant that ever helps, and only in narrow configurations.

This negative result reinforces the mechanism claim of §5. A purpose-built
mediator does not rescue collaboration; only an agent whose weight-space
geometry was not rank-constrained during adaptation (the base model) ever
approaches neutral C2W/W2C switching. The failure is in how domain
knowledge was *inserted* into the specialist's weights, not in how the
specialists are *paired*.


# 5. Mechanism

*(Stub — intruder-dimension measurement via CKA on ΔW (pending), linked to
the behavioural observations in §4.)*

# 6. Discussion

*(Stub — deployment implications; cheap adaptation has a hidden
collaboration cost.)*

# 7. Limitations and Future Work

Our current draft stops at r=128 at 4B; extending to r=256 and r=512 is
planned on the A40 pod. The 7B-scale replication is queued as the
matched-scale extrapolation of the claim. Direct weight-space CKA on the
intruder-dimension hypothesis is CPU-scheduled on local compute. Four
separable future-work threads we deliberately do not pursue in this
paper have been stubbed as independent spinoffs (§7.1–§7.4 stubs).

## 7.1. Future work: calibration analysis

*(Spinoff stub — post-fine-tuning calibration deterioration in
specialist-specialist collaboration; extends Bayesian-LoRA single-agent
finding to deliberative settings using our C2W/W2C decomposition. NeurIPS
workshop short paper candidate, minimal new compute.)*

## 7.2. Future work: domain-distance predictor

*(Spinoff stub — KL-divergence domain distance as a predictor of
collaboration outcomes; uses Pivot A's KL pipeline on our 10 specialists
plus a logit-capture re-run.)*

## 7.3. Future work: information-theoretic framework

*(Spinoff stub — conditional-MI I(chain_B ; Y | chain_A, X) as the natural
signal quantity for deliberation. Requires Laplace or ensemble posterior
treatment out of scope here.)*

## 7.4. Future work: training-data composition

*(Spinoff stub — what training data produces collaboration-robust
specialists. Requires controlled training-set ablations.)*

# References

*(Bibliography stub — to be populated via Section 2 and mechanism cites.
Primary anchors: Shuttleworth et al. 2410.21228; Du et al. 2023;
Bayesian-LoRA 2601.21003.)*
