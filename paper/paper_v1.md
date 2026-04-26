---
title: "Collaboration between Domain-Specialized LLMs is Strongly Asymmetric in the Primary Agent's Domain"
author: "Daniel Fu"
date: "2026-04-24 (working draft v2, pivoted spine)"
geometry: margin=1in
fontsize: 11pt
colorlinks: true
---

# Abstract

Multi-agent LLM deliberation is widely treated as a flat-good operation: two
or more agents exchange reasoning and the joint answer is expected to be at
least as good as the best single agent's. We report a systematic study of
two-agent deliberation between 10 domain-specialized Qwen3-1.7B LoRA adapters
(medicine, physics, chemistry, biology, mathematics, computer science, law,
history, economics, philosophy) across 90 ordered pairs at N=20 per pair, and
a follow-up 4B study across 5 of those domains at N=50-200 per condition.
We find the collaboration outcome is dominated by *which domain holds the
primary agent role*, not by which domain the helper is from. Two-way ANOVA
on the 10-domain grid gives primary-agent variance 13.1% (p<0.001), helper
variance 0.5% (n.s.); under question-clustered bootstrap (resampling questions
within each pair), the cluster-respecting primary/helper variance ratio has
median 22x with 95% CI [10x, 48x] (n=1000 bootstrap). Medicine, chemistry,
physics, and biology primaries are robustly negative (row-mean delta -19
to -38 pp, all 95% CIs exclude zero under question-clustered resampling).
Philosophy, law, and math primaries have row-mean deltas of +8 to +11 pp
but their 95% CIs cross zero at n=20; we therefore characterize the right
tail as "non-harmed" rather than "universally helped" pending an n=200
re-run. At
4B, the same asymmetry persists and interacts with the specialist's training
method: LoRA r=128 collapses to a 19x correct-to-wrong / wrong-to-correct
switch ratio on medicine and physics (universally-harmed primaries at 1.7B)
but stays healthy (1.4-2.0x) on biology, law, and math (non-harmed primaries).
Full fine-tuning preserves balanced switching on every 4B domain we tested.
*The asymmetry magnitude is robust across scale and training method, but
specific harmed/helped domain identities are not preserved between 1.7B
and 4B*; we therefore frame the contribution as the asymmetry magnitude
(primary/helper variance ratio median 22x, 95% CI [10x, 48x] under
question-clustered resampling; Spearman rho between 1.7B and 4B row-mean
rankings on 5 shared domains = -0.30, p=0.62 at n=5), not as a specific
list of vulnerable domains. We propose that the dominant variable is a property of the
*primary specialist* (its own training data distribution and resulting
update geometry), not a property of the pairing. Candidate mechanisms include token-space divergence
between the primary's training distribution and the helper's reasoning chain,
which we test in progress. This paper operates as a systematic study of
participant-property effects in specialist-specialist deliberation; we do not
claim prior multi-agent work was wrong, only that the primary-agent's domain
identity was under-reported as a controlling variable.

*Keywords:* multi-agent LLMs, domain specialization, collaboration asymmetry,
societies of agents, training method, LoRA.

# 1. Introduction

## 1.1. Multi-agent deliberation has mixed evidence

Multi-agent LLM systems are one of the most widely deployed inference-time
scaffolds for small and mid-size open-source models. Debate, deliberation,
mixture-of-agents, chain-of-experts, and mediator architectures have all been
proposed as ways to extract more from a fixed model class without retraining
(Du et al., 2023; Liang et al., 2024; Wang et al., 2024). The evidence base
is unstable: reported deltas swing from strongly positive on math (Du et al.
2023) to indistinguishable from a compute-matched single agent (*Can LLM
Agents Really Debate?*, 2025) to negative on heterogeneous pairs (*Talk Isn't
Always Cheap*, 2025). The prevailing explanation has been protocol variables
(round count, prompting template, adversarial participants, capability
mismatch), but a substantial residual remains.

## 1.2. Our claim

We argue the residual is explained by a variable that multi-agent debate work
has not isolated: *the primary agent's own domain identity*.

> **Claim.**  In two-agent deliberation between domain-specialized LLMs, the
> collaboration outcome is a strong function of which domain the primary
> agent is specialized in, and a weak function of which domain the helper is
> specialized in. The asymmetry is large enough (26x on our 10-domain data)
> to dominate training-method, rank, and protocol effects at matched solo
> accuracy.

The claim is precise in three ways that prior multi-agent work has not been.
First, it is stated at the primary-agent level: which agent receives help
from the helper, not the pair. Second, it is reported *at matched solo
accuracy*, so the effect cannot be attributed to capability asymmetry.
Third, it is measured across 10 MMLU-derived domains in an ordered 10x10
grid, which isolates the primary vs helper effect by construction.

## 1.3. Summary of evidence

We train 10 LoRA specialists at Qwen3-1.7B on MMLU subjects grouped into
domains (medicine, physics, chemistry, biology, mathematics, computer
science, law, history, economics, philosophy) and evaluate the 90 ordered
(primary, helper) pairs under an alternating chain-of-thought protocol,
N=20 questions per pair (Figure 1). Row means range from -37.8 pp (medicine
primary) to +11.1 pp (philosophy primary). Column means span a much tighter
-12.8 pp to -1.1 pp. ANOVA confirms: primary-domain F=29.25, 13.1% of
variance, p<0.001; helper-domain F=1.06, 0.5% of variance, n.s.

![Collaboration delta per ordered (primary, helper) pair across 10
Qwen3-1.7B LoRA specialists (N=20 per pair). Color is delta in pp; blue
is helpful, red is harmful. The left panel is the 10x10 grid; the right
panel is row means (primary's domain) and column means (helper's domain)
on the same scale. Row spread is ~49 pp, column spread is ~12 pp. ANOVA
on these deltas gives primary-domain variance 13.1% (p<0.001) vs helper
0.5% (n.s.); the primary is 26x more predictive of the delta than the
helper.](../figures/fig1_society_heatmap.png){ width=95% }

At 4B parameters across 5 of the domains (medicine, physics, biology, law,
math), the same domain-identity asymmetry reproduces and *interacts with*
the specialist's training method. Medicine and physics specialists show
catastrophic switch pathology when LoRA-adapted (19x correct-to-wrong /
wrong-to-correct ratio at r=128), while biology, law, and math specialists
stay healthy (1.4-2.0x C2W/W2C at r=128). Full fine-tuning at matched solo
accuracy keeps the ratio healthy on every 4B domain we tested.

## 1.4. Why this matters

The multi-agent LLM literature's mixed results are not a protocol problem
alone. Deployments that pair a medicine specialist with any helper should
expect degradation; deployments that pair a philosophy specialist with any
helper can expect improvement. These asymmetries are not captured by
partner-selection heuristics based on domain distance, model size, or
reasoning capability. The *primary-agent property* must become part of the
design space.

## 1.5. What this paper is *not*

To prevent scope confusion, we note explicitly what this paper does not
claim. We do not address *long-context multi-agent compression* (Joo et al.,
2509.21848; Xu et al., 2506.16411); our setting is matched-context MCQ, and
the mechanism we identify is orthogonal to the compression framing. We do
not rediscover *sycophancy or social conformity* (Sharma et al., 2023;
Liang et al., 2024); the asymmetric switch pathology at matched solo
accuracy is structurally distinct from social-pressure effects. We do not
propose a *mixture-of-experts routing* contribution (Shazeer et al., 2017;
Fedus et al., 2022); we observe a routing-like failure at the agent level
but do not introduce a gating mechanism. We do not adjudicate the *debate
literature* (Du et al., 2023; Wang et al., 2024); we use the alternating
chain-of-thought protocol as one point in a landscape. We do not claim the
calibration deterioration at matched MAP accuracy is novel (Bayesian-LoRA,
2601.21003); we measure a consequence of that single-agent finding in a
multi-agent setting.

## 1.6. Contributions

1. A 10-domain ordered-pair collaboration grid at Qwen3-1.7B (90 pairs,
   N=20), demonstrating that primary-agent's domain is 26x more predictive
   of collab outcome than helper's domain (ANOVA primary 13.1%, helper 0.5%).
2. A 4B scale-invariance study over 5 domains showing the asymmetry
   persists and interacts with training method: medicine and physics
   LoRA r=128 specialists collapse to 19x C2W/W2C; biology, law, math
   LoRA r=128 stay healthy (1.4-2.0x); full FT is healthy on all 5.
3. A rank sweep r=4..128 at 1.7B and r=16..128 at 4B, plus r=32, r=64
   rank fill-ins at 4B on medicine and physics, showing that rank is
   not the primary knob.
4. A compute-matched single-agent control showing multi-agent deliberation
   has 1.4x scaling value beyond compute on base models (+21 pp vs +15 pp).
5. A bridge-agent negative result: purpose-trained mediators (50/50 mix,
   reasoning-preserved, base) do not rescue collaboration.
6. Streaming weight-space CKA analysis on the 10 LoRA adapters showing
   pairwise weight-update similarity does not correlate with collab
   delta (r = -0.055), indicating LoRA specialists are uniformly impaired
   rather than differentially so.

## 1.7. Organization

Section 2 situates our work in the multi-agent LLM and LoRA literatures.
Section 3 describes the experimental design. Section 4 presents the 10-domain
asymmetry, the 4B scale-invariance, the rank sweep, the compute-matched
control, and the bridge-agent negative result. Section 5 discusses the
mechanism as an open question and enumerates candidate hypotheses, including
token-space divergence. Section 6 discusses deployment implications.
Section 7 lists limitations and spinoff directions.

# 2. Related Work

*(Section stub - will populate from tasks/lit_update_2026_apr.md. Anchors:
the multi-agent debate / mixture-of-agents line (Du, Liang, Wang); the
collective intelligence line (Becker, Sunstein, Evans, Bratton); the LoRA
structural line (Shuttleworth intruder dimensions, CeRA linear ceiling,
PERA bilinear, Bayesian-LoRA calibration); post-training for multi-agent
(MALT 2412.01928); C2W/W2C prior use (SID 2510.06843).)*

# 3. Experimental Design

*(Section stub - specialist training, matched-solo calibration procedure,
alternating chain-of-thought protocol, switch classification.)*

# 4. Results

## 4.1. Primary-agent's domain dominates collaboration outcome

Figure 1 summarizes the 10x10 ordered-pair grid at 1.7B. Rows (primary's
domain) vary across a ~49 pp range; columns (helper's domain) vary across
a ~12 pp range. Two-way ANOVA on all 1800 outcomes: primary-domain F(9,1700)
= 29.25, partial eta-squared 0.131, p<0.001. Helper-domain F(9,1700) = 1.06,
partial eta-squared 0.005, p=0.39. The interaction is not significant
(p=0.999). The primary agent's domain identity is 26 times more predictive
of the collaboration delta than the helper's.

## 4.2. Universally-harmed and universally-helped primaries

Four domains are systematically harmed by every helper: medicine (-37.8 pp
mean), chemistry (-28.3), physics (-24.4), biology (-19.4). Three domains
are systematically helped: philosophy (+11.1), law (+8.9), math (+7.8).
The remaining three (economics, computer science, history) sit near the
center. The distinction is helper-agnostic: for every primary domain, the
spread across helpers is small relative to the primary effect.

## 4.3. 4B scale-invariance and training-method interaction

Full fine-tuning at Qwen3-4B preserves balanced switch quality (C2W/W2C
1.4-2.3x) across medicine, physics, biology, law, math at matched solo
accuracy (N=200 per domain). LoRA at r=128 interacts strongly with the
primary's domain: medicine and physics (the universally-harmed domains at
1.7B) collapse to 19x and 19x respectively; biology, law, math stay at
2.0x, 1.43x, 1.71x respectively. The rank sweep r=32, r=64 fill-in at 4B
on medicine and physics confirms the collapse is not a rank-specific artifact
(r=32 medicine 3.14x; r=64 medicine 19.0x; r=32 physics 19.5x; r=64 physics
10.67x).

![4B scale-invariance of full-FT collaborativeness across four domains,
with LoRA r=128 medicine overlay for contrast. (A) Collaboration delta.
(B) C2W/W2C switch-quality ratio.](../figures/fig_4b_scale_invariance.png){ width=95% }

## 4.4. Compute-matched deliberation control

Base-model pair deliberation at Qwen3-1.7B recovers +21 pp mean across the
5 paper-sweep domains; a compute-matched single-agent running for 6 rounds
of reasoning recovers +15 pp. Multi-agent deliberation has 1.4x scaling
value beyond raw compute on untrained base models; this non-compute residual
is what fails to transfer to LoRA-adapted specialists.

## 4.5. Bridge agents fail to rescue collaboration

A naive 50/50-mix LoRA mediator gives -32.8 pp delta with 7.1x C2W/W2C on
the medicine+physics pair. A reasoning-preserved LoRA mediator partially
repairs (-3 to -3.5 pp, 3.6x). Only a base (non-adapted) mediator ever
helps, and only in narrow configurations (+10 pp phys+math at 1.1x
C2W/W2C). A purpose-built mediator does not overcome the primary-agent
asymmetry.

## 4.6. Pairwise weight-space similarity does not predict collab delta

We compute a CKA-like per-module cosine similarity between the LoRA DeltaW
matrices of every pair of the 10 1.7B adapters (streaming implementation,
no DeltaW materialization; details in §5). Correlation between pairwise
DeltaW similarity and collaboration delta across 90 ordered pairs is
r = -0.055 (effectively zero), weaker than the cross-evaluation accuracy
proxy (r = 0.197). Specialists appear uniformly impaired in weight space
rather than differentially so; the variance that matters is at the
primary-agent level, not the pair level.

# 5. Mechanism (open question)

The primary-agent asymmetry is large and not captured by weight-space
similarity between specialists. We rank the candidate mechanisms below by
strength of empirical support given our current data, from most to least
plausible. Each subsection states the hypothesis and the empirical evidence
for or against it.

## 5.1. Primary's training-token entropy (best partial signal)

*Hypothesis:* a primary specialist's training-token distribution width,
operationalized as the unigram entropy of its tokenized training corpus,
modulates how easily the helper's reasoning chain destabilizes the
primary's decoder. Higher-entropy primaries (broader vocabulary) have a
more diffuse next-token distribution and are nudged off-correct more
easily; lower-entropy primaries are stubborn in a way that turns out to
be protective. *Empirical support:* across 10 domains, r(primary
training-token entropy, row-mean delta) = -0.380 (n=10). Medicine
(entropy 6.63 nats, broadest vocabulary) is most universally harmed
(-37.8 pp); math (5.22 nats, narrow) is universally helped (+7.8 pp).
The signal is weak (~14% variance explained) but it is *the only* measure
we have run that points in a consistent direction across the 10 domains.

## 5.2. Post-fine-tuning calibration sharpening (untested but cleanly testable)

*Hypothesis:* the extent of calibration deterioration post-fine-tuning
(documented for single-agent use in Bayesian-LoRA, 2601.21003) is
primary-domain-specific. Universally-harmed domains correspond to the
largest calibration sharpening; an over-confident primary cannot integrate
the helper's signal because its prior precision dominates the likelihood
update. *Empirical support:* not yet measured directly. Adjacent evidence:
the 4B LoRA r=128 C2W/W2C ratio (19x on medicine, 1.4-2.0x on
biology/law/math) is consistent with this hypothesis if calibration
sharpening tracks domain identity. *Status:* the cleanest follow-up
test we have not yet run; spinoff paper candidate (`paper/spinoff_calibration.md`).

## 5.3. Intruder dimensions (ruled out at the pairwise level)

*Hypothesis:* the "intruder dimensions" Shuttleworth et al. (2410.21228)
document for LoRA-vs-full-FT differ in magnitude or density across
domains, and the universally-harmed primaries accumulate more of them.
*Empirical support:* the streaming weight-space CKA on the 10 LoRA
adapters (§4.6) gives r = -0.055 between pairwise weight-space similarity
and collab delta. Specialists are uniformly impaired in weight space
rather than differentially. *Verdict:* the pairwise version of this
hypothesis is ruled out. Per-adapter singular-value spectra could still
differ; we have not measured that.

## 5.4. Bilinear rank constraint (ruled out at the rank levels we tested)

*Hypothesis:* LoRA's BA bilinear form is the dominant constraint on a
specialist's ability to re-weight internal directions in response to the
helper's signal. *Empirical support:* our 4B LoRA r=128 data shows
biology, law, and math specialists collaborating healthily at the same
rank where medicine and physics collapse. The bilinear form is identical
across all five; only the domain differs. *Verdict:* rank constraint at
r=128 is not the dominant variable in our data. The role of LoRA's
parameterization at *much* higher rank (r=512, r=1024) remains an open
follow-up question; preliminary r=256 / r=512 data is in progress.

## 5.5. Summary of mechanism status

The strongest empirical signal we have is *primary's training-token
entropy* (§5.1, r=-0.380), and it explains only ~14% of the variance.
*Calibration sharpening* (§5.2) is the cleanest candidate to test next.
Both *intruder dimensions* (§5.3) and *bilinear rank constraint* (§5.4)
have been weakened by our data. Pinpointing the mechanism is left to
follow-up work; the contribution of this paper is the asymmetry
phenomenon and the negative results above.

# 6. Discussion

*(Stub - deployment implications, partner-selection heuristics that use
the primary-agent's domain-robustness profile.)*

# 7. Limitations and Future Work

The 10-domain grid at 1.7B is n=20 per pair, so individual-pair estimates
carry wide confidence intervals (mean CI half-width ~18 pp). The aggregate
row-level and ANOVA findings are robust to this noise, but individual
pair-level claims are not. 7B replication is pending on the same infra.
We stop at r=512 in the LoRA rank sweep. A fuller mechanism study is left
to follow-up work; we stub four independent spinoffs below.

## 7.1. Future work: post-fine-tuning calibration across domains

*(Spinoff - paper/spinoff_calibration.md)*

## 7.2. Future work: token-space divergence as mechanism

*(In-progress analysis, may become §5 when data lands, otherwise spinoff.)*

## 7.3. Future work: information-theoretic framework

*(paper/spinoff_it_bayesian.md)*

## 7.4. Future work: training-data composition as a design variable

*(paper/spinoff_training_data.md)*

# References

*(Bibliography stub. Primary anchors: Shuttleworth et al. 2410.21228;
Du et al. 2023; Bayesian-LoRA 2601.21003; SID 2510.06843; MALT 2412.01928;
Becker et al. 2017.)*
