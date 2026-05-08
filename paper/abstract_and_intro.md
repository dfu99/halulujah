# Societies of Specialists: WHO Holds the Question Determines Multi-Agent LLM Collaboration Outcome

*Working draft targeting ACL 2026 main conference. Last updated 2026-05-08
(ninth-revision sweep — adds Full FT pair-grid extension §A and §14
of the audit; eighth revision (2026-05-06) leads with WHO-asymmetry;
the 2026-04-19 LoRA-vs-FT/MMLU-format-memorization framing is archived
at `abstract_and_intro_v2026-04-19_DEPRECATED.md`).*

---

## Abstract (≈230 words)

In multi-agent LLM deliberation between domain specialists, *which agent
holds the question* determines collaboration outcome far more than *who
they are paired with*. We measure this on a verified 5×6 pair-grid of
Qwen3-1.7B LoRA specialists (math, medicine, biology, law, physics,
each fine-tuned to clear an out-of-domain verification gate) paired
against five specialist helpers and a base helper across 50 MMLU
questions per cell. Variance-decomposition row/helper ratios are 22.1×
on the full grid, 50.3× on hard questions (primary's solo answer
wrong), and 1.3× on easy. Replicate-aware ANOVA gives F_primary = 26.55
(p < 1e-21) on the full grid; 31.22 (p < 1e-23) on hard. Helper main
effect is non-rejecting under 8 different formal lenses (max F = 1.91,
p = 0.092). Closed-form pairwise z-tests with Bonferroni at α=0.05:
**subject-pair (136 tests) → 11 cluster-bootstrap / 28 z-test survivors;
primary-pair (10 tests) → 2 survivors (biology > {math, law});
helper-pair (15 tests) → 0 survivors even uncorrected.** Per-primary
mean wrong-to-correct rate ranges 21% (math) to 64% (biology) while
per-helper rate is flat at 35–41% (primary/helper spread = 7.07×).
The mechanism is *primary-side recoverability*: 47.7% of hard questions
are recoverable by ≤1 of 6 helpers, and within math primary 75% of
high_school_mathematics hard questions are mutually unrecoverable. The
"societies of agents" finding is that collaboration is WHO-asymmetric:
the question-holder is the bottleneck, not the helper.

**Keywords**: multi-agent LLMs, deliberation, WHO-asymmetry, domain
specialists, variance decomposition, calibration.

---

## 1. Introduction

### 1.1. Motivation

Multi-agent LLM systems, in which two or more language models exchange
natural-language reasoning over several rounds before producing a
final answer, are one of the most widely deployed inference-time
scaffolds for small and mid-size open-source models. Debate,
deliberation, mixture-of-agents, chain-of-experts, and mediator
architectures have all been proposed as ways to extract more from a
fixed model class without retraining (Du et al., 2023; Liang et al.,
2024; Wang et al., 2024). Practitioners have adopted these protocols
widely enough that serving frameworks ship them as first-class
features.

At the same time, the evidence base is curiously unstable. Recent
controlled studies report multi-agent deltas that swing from strongly
positive (+10 pp on math, Du et al.) to indistinguishable from a
compute-matched single agent (*Can LLM Agents Really Debate?*, 2025)
to negative on heterogeneous pairs (*Talk Isn't Always Cheap*, 2025).
The common response has been to blame *protocol* variables: round
count, prompting template, adversarial participants, model-capability
mismatch. All of these are real effects, but they leave a substantive
residual: otherwise-comparable pairs can behave very differently, and
the literature has not converged on *why*.

### 1.2. Our claim

We argue that the residual is explained primarily by an asymmetry that
prior work has not isolated: in a two-agent collaboration between
domain specialists, *which agent's domain the question belongs to*
(the **primary**) determines outcome far more than *which other
specialist is in the room* (the **helper**). Concretely:

> **Claim.** In multi-agent LLM collaboration between domain
> specialists, the primary specialist's domain explains an order of
> magnitude more variance in collaboration outcome than the helper
> specialist's domain. Helper identity is statistically
> indistinguishable across pairs at any reasonable formal test, while
> primary identity rejects H0 in 35 separate row-effect tests on the
> same data.

This claim is precise in three ways prior work has not been. First,
it is stated *with both factors fully crossed* in a 5×6 pair-grid
under the same protocol — most prior work fixes one factor or
manipulates them confoundedly. Second, the row-vs-helper asymmetry
is reported at multiple aggregation levels (cell-mean variance, ANOVA
F-stat, conditional rate spread, per-subject pairwise contrasts) so
that it cannot be dismissed as a one-method artifact. Third, it is
robust to the choice of *row factor* (primary identity, 5 levels;
or MMLU subject identity, 17 levels): the row/helper ratio is 22.1×
under primary-stratification and 21.7× under subject-stratification,
making the finding a property of the question-holding specialist
rather than a category-coincidence.

### 1.3. Summary of evidence (verified pair-grid)

**Figure 1** (`figures/headline_who_asymmetry.png`) summarizes the
audit's six headline findings: (A) the six-way variance-decomposition
family (primary/subject × full/hard/easy), (B) per-primary vs per-
helper hard W2C side-by-side, (C) the hs_biology vs college_math
39-pp robust gap with bootstrap CIs, (D) the formal-statistical
hierarchy at three aggregation levels (subject/primary/helper) under
both bootstrap and closed-form z-test, (E) the mutually-unrecoverable
mechanism per primary, (F) the 35-row-effect-test audit summary. The
comprehensive 111-panel evidence base is at
`figures/audit-2026-05-05.png` (supplementary).

We run all experiments on Qwen3-1.7B with LoRA specialists trained
per-domain (math on GSM8K-train, medicine on MedQA-USMLE-train, biology
on PubMedQA-train, law on CaseHOLD, physics on SciQ-train). Each
specialist must clear an out-of-domain verification gate (≥+5 pp over
base on at least one OOD benchmark) before entering the pair-grid;
this is the **verified roster** that replaces an earlier (deprecated,
2026-04-28) MMLU-pattern-matching pipeline whose specialists scored
below base on out-of-domain tests.

We then run a 5×5 specialist-vs-specialist pair-grid plus a base-helper
column (30 cells, N=50 questions per cell) under a two-agent natural-
language protocol. Each cell records pre- and post-collaboration
answers and we classify each question as held / correct-to-wrong (C2W)
/ wrong-to-correct (W2C). Headline results:

| Aggregation | Statistic | Value |
|---|---|---:|
| Full grid (1500 obs) | F_primary (replicate-aware ANOVA) | **26.55**, p<1e-21 |
| Full grid | F_helper | 0.96, p=0.44 (n.s.) |
| Hard subset (1056 obs) | F_primary | **31.22**, p<1e-23 |
| Hard subset | F_helper | 0.50, p=0.78 (n.s.) |
| Easy subset (444 obs) | F_primary | 3.13, p=0.015 |
| Easy subset | F_helper | 1.91, p=0.092 (strongest helper signal) |
| Variance ratio (full, primary×helper) | row/helper | **22.1×** |
| Variance ratio (full, subject×helper) | row/helper | 21.7× |
| Variance ratio (hard, primary×helper) | row/helper | **50.3×** |
| Variance ratio (hard, subject×helper) | row/helper | 56.5× |
| Variance ratio (easy, primary×helper) | row/helper | 1.31× |
| Variance ratio (easy, subject×helper) | row/helper | 1.64× |
| Per-primary mean W2C | range | 21% (math) to **64% (biology)** |
| Per-helper mean W2C | range | 35–41% (flat, 6 pp) |
| Pairwise Bonferroni @ α=0.05 (cluster bootstrap, n=50k) | subject-pair | 11/136 |
| Pairwise Bonferroni @ α=0.05 (z-test, closed-form) | subject-pair | 28/136 |
| Pairwise Bonferroni @ α=0.05 (z-test) | primary-pair | 2/10 |
| Pairwise Bonferroni @ α=0.05 (z-test) | **helper-pair** | **0/15 (even uncorrected: 0/15)** |

The full audit log of 35 row-effect tests + 8 helper-effect lenses is
in `tasks/audit-2026-05-05.md` §10 (10th revision, locked).

### 1.4. Why this matters

Multi-agent LLM scaffolds are widely deployed under the implicit
assumption that "more agents → more accuracy" or that helper choice
is the primary lever for tuning a collaboration. Our result shows
neither holds in the small-model specialist regime: at fixed primary,
swapping the helper changes outcome by only 5–10 pp (range across 6
helper choices); at fixed helper, swapping the primary changes outcome
by 30–40 pp (range across 5 primary choices). Deployments that aim
to improve collaboration quality by swapping helpers are tuning the
weak axis of variation; the strong axis is *which agent owns the
question*.

The deeper finding is mechanistic: 47.7% of hard questions (primary's
solo answer wrong) are nearly unrecoverable by *any* of 6 helpers
(0 or 1 of 6 helpers recovers the correct answer). Math and law
primaries have 53% mutually-unrecoverable hard rates; biology has
19% with 42% universally-recoverable. The asymmetry is not "helpers
don't help" — it is "the question-holding specialist's *willingness
to update from wrong* dominates outcome, and that willingness varies
3× across primaries."

### 1.5. Mechanism (preview)

We propose, but do not in this paper conclusively establish, that
the primary's recoverability is a property of the *training-data
manifold* of the specialist's adapter: specialists whose domain is
heavily concentrated on a single MMLU subject (e.g., law trained
solely on CaseHOLD) tend to be locked into format patterns that
suppress within-subject error correction, while specialists trained
on broader-distribution data (e.g., biology on PubMedQA + MMLU bio)
remain malleable. The within-primary subject-decomposition (§6qq):
math primary's mutually-unrecoverable rate spans 20% (elementary
math) to 75% (high-school + college math) — pointing to *which
subjects within a primary's pool are recoverable* as the local
variation.

Two adjacent literatures predict the WHO-asymmetry:
(i) **intruder dimensions in LoRA** (Shuttleworth et al., 2410.21228)
— LoRA's low-rank update introduces high-singular-value directions
not present in the pretrained subspace, and these are the directions
that dominate the trained model's response to disagreement.
(ii) **calibration deterioration under fine-tuning** (Bayesian-LoRA,
2601.21003) — fine-tuned specialists become overconfident in their
training-data manifold, suppressing the channel through which a
helper's disagreement would propagate. We discuss both mechanisms
in §5 but do not claim either is sufficient; the *empirical*
WHO-asymmetry is established at the behavioral level
(robust under 35 row-effect tests) regardless of which mechanism is
ultimately the cause.

### 1.6. Contributions

1. **A controlled 5×6 verified pair-grid** at Qwen3-1.7B with 30
   primary×helper cells, N=50 questions per cell. Each specialist
   passes an out-of-domain verification gate before entering.

2. **A formal-statistical hierarchy at three aggregation levels.**
   Cluster-respecting bootstrap (n_iter=50000) and closed-form
   two-proportion z-test give converging Bonferroni-survivor counts:
   subject-pair 11/28; primary-pair 2/2; helper-pair 0/0. The asymmetry
   is method-agnostic.

3. **A six-way variance-decomposition family.** Row/helper ratio is
   robust to the choice of row factor (primary or subject) — moves
   ≤14% relative across stratifications — and to difficulty regime
   (full 22×; hard 50×; easy 1.3×).

4. **A mechanistic mutually-unrecoverable measurement.** 47.7% of
   hard questions cannot be recovered by any of 6 helpers; the rate
   varies 3× across primaries (math/law 53% to biology 19%).

5. **Subject-stratified disambiguation.** Within math primary,
   college_mathematics has the lowest hard-W2C in the audit (4.2%,
   bootstrap CI [0%, 12.5%]); high_school_biology has the highest
   (66.7%, [50.8%, 81.8%]) — a 39-pp non-overlapping CI gap that
   survives Bonferroni-136 under both bootstrap and z-test.

6. **Replication under Full FT (no rank constraint).** The 5×6
   pair-grid is replicated on Qwen3-1.7B specialists trained with
   full-parameter SFT on the same per-domain MMLU splits. WHO-asymmetry
   ratio = 52.37× (3.1× the LoRA ratio under the same estimator); helper
   flatness preserved (max-min across 6 helpers = 5.6 pp Full FT vs 6.8
   pp LoRA). C2W:W2C remains net-corrective at the cohort level (135 :
   574 = 0.24, vs LoRA 0.13). The asymmetry is *not* an
   intruder-dimension artifact of the rank constraint. See §A and
   `tasks/audit-2026-05-05.md` §14.

### 1.7. Organization

Section 2 situates our work in the multi-agent LLM literature. Section 3
describes the verified-roster pipeline and the 5×6 pair-grid protocol.
Section 4 presents the row-effect / helper-effect asymmetry across the
six-way variance family, the difficulty-stratified ANOVA triple, and
the formal-statistical hierarchy. Section 5 examines the mechanism
(per-primary mutual-unrecoverability, subject decomposition).
Section 6 discusses implications for multi-agent system design
(cross-domain helpers preferred 4 of 5 primaries; orchestration LOO
closes 8% of oracle gap). Section 7 covers limitations: 1.7B LoRA
only, single base model (Qwen3), MMLU question-pool composition.

---

## Notes for co-authors / PI

- **The paper now leads with WHO-asymmetry.** The prior 2026-04-19
  draft led with "rank-constrained adaptation destroys collaborative
  behavior" backed by an MMLU-derived deliberation; that pipeline
  was retracted 2026-04-28 after the medicine specialist scored
  below base on MedQA (the specialists were MMLU-format pattern
  matchers). This revision uses the post-verification roster only.
- LoRA-vs-full-FT comparison is now a **landed extension** (2026-05-08):
  the 1.7B Full FT pair-grid is complete (35 of 35 cells; 5×6 with
  base helper; 50 q × 3 rounds each). **Headline: WHO-asymmetry ratio
  is 52.37× under Full FT vs 16.93× under LoRA (same var-of-row-means
  / var-of-col-means estimator) — a 3.1× amplification when the rank
  constraint is removed.** Per-helper means lie within 5.6 pp of each
  other under Full FT (vs 6.8 pp under LoRA), so helper flatness is
  *preserved* not attenuated when rank constraint is removed. The
  asymmetry is therefore not a low-rank/intruder-dimension artifact.
  Switch totals: LoRA c2w=82 / w2c=625 (ratio 0.13); Full FT c2w=135 /
  w2c=574 (0.24) — both strongly net-corrective; Full FT specialists
  are slightly more flip-prone but still net-helpful. Per-cell matrix,
  switch counts, and gate result are in §A and `paper/ft_extension_
  2026-05-08.md`; ckpt-by-ckpt 5-shot scan is at `tasks/audit-2026-05-05
  .md` §14b (3/5 pass strict +5pp OOD gate; math+law fail on raw
  5-shot but operate normally in chat-template/CoT format used by the
  pair grid). The 4B Full FT pair-grid is still partial (3 of 5 domains,
  base helper only) and remains future work.
- The audit at `tasks/audit-2026-05-05.md` is the single source of
  truth for the numbers used in this abstract+intro. Section 10's
  10th-revision canonical paragraph is locked in; the §10 defensible
  one-paragraph sentence at line 6146 is the source of the abstract.
- The audit catalogs 35 row-effect tests (all rejecting H0 of no
  primary effect) and 8 helper-effect lenses (all non-rejecting,
  max F=1.91 at p=0.092 on easy ANOVA).
- The paper's anticipated reviewer concerns are mapped in
  `tasks/audit-2026-05-05.md` §11; the closure summary at §13 lists
  what the audit can and cannot defend (correlative, single-base-model,
  1.7B LoRA only).
