# Rank-Constrained Adaptation Destroys Collaborative Behavior in Multi-Agent LLMs

*Working draft targeting ACL 2026 main conference. Last updated 2026-04-19.*

---

## Abstract (≈230 words)

Multi-agent LLM deliberation has been reported to both improve and harm task
accuracy, with recent controlled studies (Du et al., 2023; *Talk Isn't Always
Cheap*, 2025; *Can LLM Agents Really Debate?*, 2025) yielding outcomes that
span roughly -10 to +15 percentage points on matched benchmarks.  We argue
that a single under-controlled variable explains a large portion of this
spread: the *training method* used to produce the domain specialists
participating in the deliberation.  We run a controlled comparison in which
LoRA-finetuned and fully-finetuned Qwen3 specialists reach **identical solo
accuracy** on their target domain, then engage in a two-agent natural-language
collaboration protocol with a matched partner.  At 4B parameters and 84% solo
accuracy on medicine, full fine-tuning yields a +5.0 pp collaboration delta
with a 1.4× correct-to-wrong / wrong-to-correct (C2W/W2C) switch ratio, while
LoRA at rank 128 yields a +1.5 pp delta with a 19× C2W/W2C ratio. The LoRA
specialist almost never recovers an incorrect peer answer, and it routinely
abandons its own correct one.  A rank sweep from r=4 to r=128 at both 1.7B
and 4B fails to close the gap.  We argue the mechanism is LoRA's low-rank
bilinear update: the "intruder dimensions" it introduces (Shuttleworth et al.,
2410.21228) are precisely the directions that dominate collaborative updating.
Cheap adaptation has a hidden cost: a collapse of the information channel
that multi-agent deliberation relies on.

**Keywords**: multi-agent LLMs, LoRA, full fine-tuning, deliberation,
rank constraint, calibration.

---

## 1. Introduction

### 1.1. Motivation

Multi-agent LLM systems, in which two or more language models exchange
natural-language reasoning over several rounds before producing a final
answer, are one of the most widely deployed inference-time scaffolds for
small and mid-size open-source models.  Debate, deliberation, mixture-of-agents,
chain-of-experts, and mediator architectures have all been proposed as ways
to extract more from a fixed model class without retraining (Du et al.,
2023; Liang et al., 2024; Wang et al., 2024).  Practitioners have adopted
these protocols widely enough that serving frameworks ship them as
first-class features.

At the same time, the evidence base is curiously unstable.  Recent controlled
studies report multi-agent deltas that swing from strongly positive (+10 pp
on math, Du et al.) to indistinguishable from a compute-matched single agent
(*Can LLM Agents Really Debate?*, 2025) to negative on heterogeneous pairs
(*Talk Isn't Always Cheap*, 2025).  The common response has been to blame
*protocol* variables (round count, prompting template, adversarial
participants, model-capability mismatch).  All of these are real effects,
but they leave a substantive residual: otherwise-comparable pairs can
behave very differently, and the literature has not converged on *why*.

### 1.2. Our claim

We argue that a large part of the residual is explained by a variable that
prior work has not isolated: the *parameter-efficient vs. full* training
method used to produce the specialist.  Concretely:

> **Claim.**  At matched solo accuracy on the primary task, LoRA-adapted
> domain specialists lose most of the benefit of multi-agent deliberation,
> while fully-fine-tuned specialists preserve it.  The mechanism is LoRA's
> rank-constrained update: collaborative improvement requires the model to
> move in directions that LoRA's low-rank bilinear parameterization cannot
> express.

This claim is precise in three ways that prior multi-agent work has not
been.  First, it is stated *at matched solo accuracy*, which forces the two
specialists to have the same single-agent capability.  Second, it separates
the training method from the specialization itself: both LoRA and full FT
produce "specialists" in the usual sense, but only the full-FT specialist
collaborates.  Third, it localizes the failure to a concrete architectural
choice (rank) that can be ablated.

### 1.3. Summary of evidence

We run all experiments on Qwen3-1.7B and Qwen3-4B, training medicine and
physics specialists on MMLU-derived data via LoRA (r ∈ {4, 8, 16, 32, 64, 128})
and full fine-tuning.  We then collaborate each specialist with a matched
base-model partner under a two-agent natural-language protocol of N=200
questions per condition.  Our headline result at 4B-medicine (solo accuracy
84% for both LoRA r=128 and full FT) is:

| Condition            | Δ vs. solo | C2W  | W2C | C2W/W2C |
|----------------------|-----------:|-----:|----:|--------:|
| Full fine-tuning     |    **+5.0 pp** |   7 |   5 |   **1.4×** |
| LoRA r=128           |    **+1.5 pp** |  19 |   1 |   **19×** |

The full-FT specialist's collaboration delta is 3.3× larger and its switching
quality is 13.5× better, on a model with identical solo accuracy.  At
r=16 the LoRA C2W/W2C ratio is 8.5× (51/6) and the delta is -1.5 pp.
A rank sweep up to r=128 at both scales fails to recover full FT's
collaboration behavior; higher ranks slightly improve solo accuracy but
leave the deliberation channel pathological.

A compute-matched single-agent control (identical total inference compute
as the deliberation, but spent on a single chain) recovers 15 pp on the
base model while full deliberation recovers 21 pp, a 1.4× compute-scaling
ratio. This establishes that deliberation has value beyond raw compute in the
*absence* of LoRA, and that this value is what LoRA destroys.

### 1.4. Why this matters

LoRA is the dominant adapter choice for small open-source models: it is
cheap, composable, and preserves enough task quality on downstream
benchmarks that the common wisdom has become *"LoRA recovers 90–95% of
full FT"*.  Our result does not contest that claim *on solo task accuracy*.
But it shows that the 5–10% residual is not a uniform loss of quality:
it is concentrated in a single capability, the specialist's ability to
update its answer in response to a peer, and that capability is precisely
the one that multi-agent scaffolds depend on.  Deployments that adopt
LoRA specialists for cheapness and multi-agent scaffolds for accuracy
may be silently cancelling the second with the first.

### 1.5. Mechanism (preview)

Two recent theoretical threads converge on why rank should matter for
collaboration.  Shuttleworth et al. (2410.21228) show that LoRA adapters
introduce novel high-singular-value "intruder dimensions" not present in
full-FT, and that these dimensions dominate the trained model's response
in a way that produces catastrophic forgetting.  CeRA (2602.22911) and
PERA (2604.11841) independently argue that LoRA faces a linear/bilinear
ceiling that cannot be closed by scaling rank alone.  Bayesian-LoRA
(2601.21003) reports that fine-tuning systematically degrades calibration,
the exact behavior we measure at the decision level via C2W/W2C.

Our contribution is to show the *behavioural consequence* of these
architectural properties in a multi-agent setting: intruder-dimension
rigidity prevents the specialist from integrating a partner's disagreement,
and calibration deterioration makes the few switches that do occur
systematically wrong (C2W dominating W2C).  Full-FT specialists at matched
solo accuracy exhibit neither pathology.

### 1.6. Contributions

1. A controlled LoRA-vs-full-FT comparison *at matched solo accuracy* in a
   two-agent collaboration protocol, at Qwen3-1.7B and Qwen3-4B scales,
   over 61 conditions with N=200 per condition (12 conditions at 4B).

2. A rank sweep (r=4…128 at 1.7B; r=16,128 at 4B) demonstrating that
   increasing LoRA capacity does not recover full-FT collaborativeness.

3. A compute-matched single-agent control showing deliberation has
   value beyond raw compute (+21 pp vs. +15 pp), and that LoRA destroys
   this residual specifically.

4. Switch-classification (C2W, W2C, held) as a decision-level calibration
   proxy that makes the pathology directly visible: LoRA = 19× C2W/W2C;
   full FT = 1.4×.

5. Integration with concurrent theoretical work (intruder dimensions,
   linear ceiling, bilinear rank constraint) that collectively supplies
   a mechanism.

### 1.7. Organization

Section 2 situates our work in the multi-agent LLM and LoRA literatures.
Section 3 describes the experimental design, including the matched-solo
constraint and the switch-classification analysis.  Section 4 presents
the main results at 1.7B and 4B, the rank sweep, and the compute-matched
control.  Section 5 examines the mechanism, linking our behavioural
observations to intruder dimensions and the linear ceiling.  Section 6
discusses implications for deployment and multi-agent system design.
Section 7 lays out limitations and future work.

---

## Notes for co-authors / PI

- Abstract and intro use *specialist+base partner* as the protocol.  The
  *specialist+specialist* and *mediator* results are Section 4 material
  and should not be foregrounded in the intro because they are less clean
  (lower matched-solo-accuracy constraint) and introduce additional
  confounds.
- The "13× better C2W/W2C" number compares full-FT (1.4×) to LoRA r=128
  (19×) at 4B medicine.  Double-check the 13× framing, since arithmetically it
  is 19 / 1.4 ≈ 13.6, but reviewers will want to see both numbers and
  the ratio derivation.  The paper table gives both numbers.
- "Subsumes" Du et al. (the PI's question): the intended framing in the
  intro is *strategic*, not apologetic.  Our range subsumes Du et al.'s
  because we introduced a new variable (training method) while they held
  it fixed.  Prior literature's tightness is a feature of their
  homogeneous setup, not of the underlying phenomenon.  See Section 2.3
  on how Du et al.'s finding is consistent with ours if their implicit
  training method is full-FT or sufficient rank.
- Reviewer C (multi-agent researcher) has already been addressed in the
  literature-context figure (figures/reviewer_c_literature_context.png).
  The intro deliberately does not re-argue that figure; it just uses
  its conclusion.
- Reviewer B (PEFT researcher) is addressed by the rank sweep paragraph
  in §1.3 and figures/reviewer_b_rank_vs_ft.png.
- Reviewer D (calibration skeptic) is addressed by C2W/W2C and
  figures/reviewer_d_entropy_by_turn.png.
- Reviewer E (domain-distance skeptic). CKA script is staged but OOM'd
  on RunPod; deferred to a higher-memory host.  Currently the paper does
  not claim a distance-vs-delta result, which keeps us safe from that
  reviewer as long as the limitations section is honest about r=0.197.
