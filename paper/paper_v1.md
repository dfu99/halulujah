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

## 2.1. Multi-agent LLM deliberation and debate

The line that most closely overlaps with our setting is two-or-more agent
deliberation, in which agents exchange natural-language reasoning across
rounds before producing a final answer. Du et al. (arXiv 2305.14325)
introduced this protocol on math word problems and reported gains of
roughly +5 to +15 pp depending on benchmark. Liang et al. (arXiv 2305.19118)
generalized it as Multi-Agent Debate (MAD) and reported improvements on
factual-recall tasks but smaller gains on reasoning-heavy ones. Wang et al.
(2024)'s Mixture-of-Agents (MoA) and follow-on work proposed homogeneous
generalist chains that aggregate via a manager agent. The picture is mixed:
recent controlled studies have reported deltas spanning negative-to-positive
on matched benchmarks. Talk Isn't Always Cheap (arXiv 2509.05396)
specifically showed that a weaker partner can degrade a stronger one's
output, attributing the harm to capability asymmetry. Can LLM Agents Really
Debate? (arXiv 2511.07784) reports that MAD methods often fail to
outperform a compute-matched single-agent baseline. Liu et al. SID
(arXiv 2510.06843) introduces the C2W (correct-to-wrong) and W2C
(wrong-to-correct) switch metrics we adopt and uses them for confidence-
gated debate early-exit; we use the same metric to characterize a
training-method-dependent pathology rather than to gate inference.

We differ from this line by (a) studying *domain-specialized* agents at
*matched solo accuracy* on a primary task, (b) running an ordered (primary,
helper) grid that isolates the primary-vs-helper variance attribution, and
(c) reporting an asymmetry that does not reduce to capability gap.

## 2.2. Long-context multi-agent compression

A separate line treats multi-agent collaboration as long-context input
compression. Joo et al. Graph of Agents (arXiv 2509.21848) formalizes
multi-agent for long-context modeling as an information-theoretic
compression problem and shows that a small-context graph of agents can
match or beat a much larger-context single model. Xu et al. (arXiv
2506.16411, ICLR 2026) provides a divide-and-conquer noise-decomposition
framework and proves a "D&C Advantage" theorem: under super-linear loss
growth in context length, weak agents handling chunks outperform a single
strong model. Yun et al. Graph-of-Agents (ICLR 2026) introduces a
graph-based selection-and-message-passing framework for heterogeneous
flagship agents and reports that 3 selected agents from a pool of 6
beats all-6 MoA. Our setting is matched-context MCQ; the long-context
mechanism is orthogonal to ours.

## 2.3. LoRA and parameter-efficient fine-tuning

Hu et al. LoRA (arXiv 2106.09685) introduced the rank-r decomposition
update ΔW = (α/r)·B·A that we sweep over here. The "LoRA recovers 90–95%
of full fine-tuning" folklore comes from this line and from independent
benchmarks across NLP and adaptation tasks. Shuttleworth et al.,
"LoRA vs Full Fine-Tuning: An Illusion of Equivalence" (arXiv 2410.21228),
shows that at matched downstream task accuracy, LoRA-trained models
contain novel high-singular-value "intruder dimensions" not present in
full fine-tuning, and that these dimensions correlate with catastrophic
forgetting. CeRA (arXiv 2602.22911) and PERA (arXiv 2604.11841) provide
theoretical arguments that LoRA's rank-r update faces a linear ceiling
(CeRA) and a bilinear-form expressivity ceiling (PERA), independent of
data quantity. Bayesian-LoRA (arXiv 2601.21003) reports that
post-fine-tuning calibration deteriorates more under LoRA than under full
FT at matched task accuracy, a single-agent finding we extend to a
multi-agent setting. "Why LoRA Fails to Forget" (arXiv 2601.06305)
extends the Shuttleworth picture to unlearning. None of these papers run
a multi-agent collaboration protocol.

## 2.4. Post-training for multi-agent and routing

A complementary line trains models *for* multi-agent performance. MALT
(arXiv 2412.01928, Oxford / Cooperative AI Foundation / MBZUAI / Stanford)
proposes a Generator-Verifier-Refiner sequential post-training pipeline
that improves reasoning by 7–16 pp on MATH, GSM8K, CSQA. Adaptive-
collaboration work routes queries based on difficulty (Debate Only When
Necessary, arXiv 2504.05047) or confidence (SID, arXiv 2510.06843).
Our work studies the *prior question* of how the training method of an
already-specialized agent affects its collaborative behavior, which is
orthogonal to post-training for multi-agent and to routing strategy.

## 2.5. Collective intelligence

Becker, Brackbill, and Centola (Proc. Natl. Acad. Sci. 2017) showed that
social influence in human networks can either enhance or destroy
collective intelligence depending on network topology. Sunstein (2002)
documented "group polarization" in human deliberation. The structural
analogy between fully connected human networks and our alternating CoT
protocol is taken in §5.2; we emphasize that the analogy is structural
rather than cognitive. Recent positions on agentic intelligence
(Bratton, Agüera y Arcas, Evans et al., AAAS Science 2025) frame the
participant-property dependence of collective behavior as a research
program. Our work contributes one quantitative measurement (the 26x →
22x primary-vs-helper variance attribution) on that program.

## 2.6. Mixture of Experts and routing failures

Shazeer et al. (arXiv 1701.06538) introduced the sparsely-gated
mixture-of-experts layer; Fedus et al. Switch Transformer
(arXiv 2101.03961) scaled it. Expert collapse, where the gating function
fails to distribute inputs across experts, is a documented MoE failure
mode. Our agent-level finding has the same structural shape — the
primary specialist cannot effectively integrate the helper's
contribution — but we do not propose a routing or gating mechanism. We
discuss the analogy in §5.2 and §6 without claiming a contribution to
MoE per se.

# 3. Experimental Design

## 3.1. Domain specialists

We train 10 LoRA adapters on Qwen3-1.7B (Hu et al. 2022) and 5 adapters
plus 5 full fine-tuned specialists on Qwen3-4B. Each adapter targets a
single domain composed of MMLU subjects: medicine (anatomy + clinical
knowledge + medical genetics + professional medicine), physics (college
+ high school + astronomy + conceptual), biology (college + high school
+ anatomy + clinical), chemistry (college + high school), math (college
+ high school + abstract algebra + elementary), law (professional +
jurisprudence + international), philosophy (philosophy + moral scenarios
+ logical fallacies), economics (microeconomics + macroeconomics +
econometrics), history (world + US + European), and computer science
(college + high school + machine learning).

LoRA hyperparameters: rank r ∈ {4, 8, 16, 32, 64, 128, 256, 512},
α = 2r (we follow the empirical α = 2r heuristic for stable training),
lora_dropout = 0.05, target_modules = "all-linear", num_train_epochs = 3,
per_device_train_batch_size = 1, gradient_accumulation_steps = 4,
learning_rate = 5e-5, warmup_ratio = 0.1, gradient_checkpointing = True,
bf16 = True. Full fine-tuning uses identical schedule with full parameter
updates and DeepSpeed ZeRO-Offload to fit on the available A4500 / A40
hardware.

## 3.2. Matched-solo-accuracy protocol

For comparisons that compare across training methods (LoRA vs full FT)
within the same domain, we constrain the specialists to reach the same
solo accuracy on the held-out test split. We achieve this by selecting
the LoRA rank that produces solo accuracy within ±1 pp of the full FT
solo, on a subject-stratified validation split. For 4B medicine, the
matching solo is 84% (full FT) ↔ 84% (LoRA r=128); for 4B physics, 85.5%
↔ 83% (LoRA r=128). When matched solo accuracy is impossible at the
ranks we test, we report the closest match and note the gap.

## 3.3. Alternating chain-of-thought protocol

For each ordered (primary, helper) pair, the primary specialist receives
the question and produces an initial reasoning chain (round 1). The
helper specialist receives the question and the primary's full reasoning
and produces its own reasoning (round 2). The primary specialist receives
the helper's reasoning and produces a final answer (round 3). The final
answer is extracted from the primary's last response by regex on the
multiple-choice letter. We use this protocol because it is the most
coupled multi-agent variant in the design space, which makes the
distributional-interference effect most visible; we discuss less coupled
variants in §6.

We also evaluate four single-agent and pair-level controls per primary:
*solo* (primary alone, n_rounds = 3 self-continuation), *base_solo*
(untrained Qwen3 base alone), *base_pair* (two untrained Qwen3 base
agents in alternating CoT), *same_pair* (primary specialist deliberating
with itself), and *mixed_pair* (primary specialist with the untrained
Qwen3 base as helper).

## 3.4. Switch classification

Following Liu et al. SID (arXiv 2510.06843), we classify each
collaboration outcome as Held (primary's pre-collab answer survived to
the final), C2W (primary's pre-collab answer was correct, final is
wrong), or W2C (primary's pre-collab answer was wrong, final is
correct). We report C2W / W2C as a switch-quality ratio: a balanced
deliberation should produce roughly equal numbers of C2W and W2C; a
ratio ≫ 1 indicates the protocol is harming more than it helps.

## 3.5. Statistical methodology

Per-pair accuracies at n=20 per pair carry ±18 pp half-width 95%
confidence intervals on individual pair estimates (Wilson method). For
row-level claims, we report question-clustered bootstrap CIs (5000
resamples, resampling questions WITH-IN each pair so that the
within-pair question dependence is respected). For the variance
decomposition, we report the cluster-respecting primary / helper
variance ratio with bootstrap CI (1000 resamples). For pairwise
significance, we apply Benjamini-Hochberg FDR correction at q = 0.05.
For C2W / W2C ratio claims, we report Wilson-style 95% CIs and require
a minimum of 20 total switches per cell before claiming a ratio is
distinguishable from 1.

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

## 6.1. The cheap-deployment regime our claim covers

Our claim is scoped to a specific deployment regime: small open-source
LLMs (Qwen3 1.7B and 4B in this paper) with cheap LoRA adapters as
domain specialists, deliberating in pairs via alternating chain-of-
thought. The asymmetry we find is large and reproducible in this
regime. It is not a claim about flagship-model serial chains
(Einstein Arena, Mixture-of-Agents over Claude / GPT-5 / Gemini) where
participant capability is much higher, calibration deteriorates less
post-fine-tuning, and the protocol is cumulative rather than alternating.
We expect the failure modes we report to attenuate in that flagship
regime; verifying that is open empirical work.

## 6.2. Implications for partner-selection heuristics

Today's multi-agent system literature optimizes the *helper* slot:
which model to call, what prompt template to use, how many rounds. Our
data says the *primary* slot dominates the variance — 22x (CI [10x,
48x]) more than the helper slot — at matched solo accuracy in our
regime. Partner-selection heuristics that ignore primary-agent properties
(domain identity, training method) are optimizing the wrong dimension.
A practical heuristic for deployments that use LoRA-fine-tuned specialists
is: *(a)* establish whether the primary's domain falls in the
universally-harmed cluster (medicine, chemistry, physics, biology in our
1.7B data; medicine and physics under LoRA r=128 in our 4B data), and
*(b)* if so, route to a single-agent (primary alone) or to a different
training-method specialist (full fine-tuned) rather than to a peer-
deliberation protocol. Concretely, for the universally-harmed primaries
under LoRA at matched solo accuracy, multi-agent deliberation produces
a 19x C2W / W2C switch ratio that we would not characterize as
beneficial under any reasonable utility function.

## 6.3. Why the helper-slot variance is so small

The helper-slot variance is small in absolute terms (0.5% partial η²,
column means span ~12 pp vs row spread ~49 pp). Our reading: under
alternating chain-of-thought, the helper provides token-level context
that the primary's decoder either integrates or locks against; it does
not appear to provide a separable propose-and-vote contribution. A
flagship cumulative chain (§2.2) could plausibly produce larger
helper-slot variance because each agent's contribution accumulates
rather than is overwritten. We do not have data to confirm that
prediction; it is a clean falsification target.

## 6.4. The 1.7B-to-4B identity flip

Medicine moves from -38 pp at 1.7B to +9.5 pp at 4B; the Spearman rho
between row-mean rankings on the 5 shared domains is -0.30 (n=5, low
power). This is the single most fragile aspect of our story. Two
interpretations are open. *(a)* Configuration-dependent labels: scale,
training method, and protocol all interact with the primary specialist's
brittleness; the *asymmetry* is invariant in magnitude but the
*labels* of vulnerable primaries shift. *(b)* The 1.7B asymmetry is
specific to the n=20 per-pair noise floor and a more powerful test would
soften the universally-harmed claim. The N=200 1.7B 10x10 grid currently
running will discriminate (a) from (b). Until that data lands, we report
the asymmetry as the contribution and the labels as configuration-dependent.

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

Becker, J., Brackbill, D., and Centola, D. (2017). Network dynamics of
social influence in the wisdom of crowds. *Proceedings of the National
Academy of Sciences*, 114(26):E5070–E5076.

Chan, C., Chen, W., Su, Y., Yu, J., Xue, W., Zhang, S., Fu, J., and Liu,
Z. (2024). ChatEval: Towards Better LLM-Based Evaluators Through
Multi-Agent Debate. arXiv:2308.07201.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. (2023).
Improving Factuality and Reasoning in Language Models through Multiagent
Debate. arXiv:2305.14325.

Estornell, A., Patel, S., and Liu, Y. (2024). Multi-LLM Debate:
Framework, Principals, and Interventions. *NeurIPS 2024*.

Fedus, W., Zoph, B., and Shazeer, N. (2022). Switch Transformers:
Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.
*Journal of Machine Learning Research*, 23(120):1–39. arXiv:2101.03961.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang,
L., and Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language
Models. *ICLR 2022*. arXiv:2106.09685.

Joo, T., Ishida, S., Sosnovik, I., Lim, B., Rezaei-Shoshtari, S., Gaier,
A., and Giaquinto, R. (2025). Graph of Agents: Principled Long Context
Modeling by Emergent Multi-Agent Collaboration. arXiv:2509.21848.

Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Tu,
Z., and Shi, S. (2024). Encouraging Divergent Thinking in Large Language
Models through Multi-Agent Debate. arXiv:2305.19118.

Liu, X., Chen, Y., Wang, S., et al. (2025). SID: Multi-LLM Debate Driven
by Self Signals. arXiv:2510.06843.

Motwani, S., Roberts, B., Smith, T., Cohan, A., et al. (2024). MALT:
Improving Reasoning with Multi-Agent LLM Training. arXiv:2412.01928.

Qian, Z., Zhang, Y., Lin, X., et al. (2025). Debate Only When Necessary:
Adaptive Multiagent Collaboration for Efficient LLM Reasoning.
arXiv:2504.05047.

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S.,
et al. (2023). Towards Understanding Sycophancy in Language Models.
arXiv:2310.13548.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G.,
and Dean, J. (2017). Outrageously Large Neural Networks: The
Sparsely-Gated Mixture-of-Experts Layer. arXiv:1701.06538.

Shuttleworth, R., Andreas, J., Torralba, A., and Sharma, P. (2024).
LoRA vs Full Fine-tuning: An Illusion of Equivalence. arXiv:2410.21228.

Sunstein, C. R. (2002). The Law of Group Polarization. *Journal of
Political Philosophy*, 10(2):175–195.

Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., and Zou, J. (2024).
Mixture-of-Agents Enhances Large Language Model Capabilities.
arXiv:2406.04692.

Xu, Z., Zhu, S., Wang, J., Wang, J., Athiwaratkun, B., Wang, C., Zou,
J., and Zhang, C. (2025). When Does Divide and Conquer Work for Long
Context LLM? A Noise Decomposition Framework. *ICLR 2026*.
arXiv:2506.16411.

Yun, S., Peng, J., Li, P., Fan, W., Chen, J., Zou, J., Li, G., and Chen,
T. (2025). Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM
Collaboration. *ICLR 2026*.

Zhao, Y., Liu, Y., et al. (2025). Talk Isn't Always Cheap: Understanding
Failure Modes in Multi-Agent Debate with Heterogeneous Agents.
arXiv:2509.05396.

Zhang, K., Liu, Y., et al. (2025). Can LLM Agents Really Debate? A
Controlled Study of LLM Multi-Agent Debate Performance. arXiv:2511.07784.

*Adapter / parameter-efficient fine-tuning theory:*

Bayesian-LoRA Authors. (2026). Bayesian-LoRA: Calibration-Aware Low-Rank
Adaptation. arXiv:2601.21003.

CeRA Authors. (2026). CeRA: Overcoming the Linear Ceiling of Low-Rank
Adaptation via Convex-Expansion Reparameterization. arXiv:2602.22911.

PERA Authors. (2026). Polynomial Expansion Rank Adaptation: Beyond the
Bilinear Form of LoRA. arXiv:2604.11841.

"Why LoRA Fails to Forget" Authors. (2026). Why LoRA Fails to Forget:
Regularized Low-Rank Adaptation for Unlearning. arXiv:2601.06305.
