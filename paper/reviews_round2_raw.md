# Round 2 Reviews: Revised Paper

Reviews generated from 5 expert reviewer personas on the revised paper.
The paper was revised to address all 5 unanimous critiques from Round 1.

---



# Review: Round 2 — "When Helping Hurts: Systematic Accuracy Degradation in Alternating Chain-of-Thought Collaboration Between Domain-Specialized LLMs"

**Reviewer:** James Evans, University of Chicago
**Review Round:** 2 (Revision)

---

## 1. Summary of Revisions

The authors have made several responsive changes since Round 1:

- **Statistical confidence:** Bootstrap 95% CIs are now reported for the mean effect ([-12.1%, -6.3%]) and per-pair CI width is acknowledged (±17.9pp). This partially addresses my concern about n=20 being underpowered.
- **Variance decomposition:** Two-way ANOVA is now included with eta-squared values (primary domain 13.1%, helper 0.5%, interaction 2.0%, residual 84.6%). This was a central request and it is delivered.
- **Training set size confound:** Addressed in a dedicated subsection (4.5) with a correlation analysis (r=0.092) and qualitative argument that the relationship is non-monotonic.
- **Mechanistic analysis:** Section 5.1 now offers "distributional interference under sequential conditioning" as a framework, though it remains qualitative.
- **Collective intelligence connection:** Section 5.2 now explicitly cites Becker et al. (2017) and draws a structural analogy between alternating CoT and fully connected influence networks.
- **Failure mode taxonomy:** Expanded to 1,101 incorrect and 699 correct chains with percentage breakdowns.

The paper is substantially improved in structure and statistical rigor. The title change from any prior framing to "accuracy degradation" is also appropriate and more precise.

---

## 2. Strengths of the Revision

**The ANOVA is the strongest addition.** The variance decomposition tells a clean story: primary domain identity explains 26x more variance than helper identity. This is the kind of result that changes how people think about multi-agent system design. The near-zero helper effect (p=0.39) is arguably more interesting than the main negative finding, because it says the problem is structural to the primary agent, not relational between agents.

**The failure mode taxonomy is now empirically grounded.** The asymmetry between answer-switching (21.2% of failures) and correction (17.9% of successes) is a useful quantitative finding. It gives the "helping hurts" claim a specific mechanism: collaboration is a net destroyer of correct answers because switching outpaces correction.

**The training size analysis is honest.** Rather than dismissing the confound, the authors show r=0.092 and acknowledge it as a partial confound while demonstrating non-monotonicity with specific examples. This is the right approach.

**The scope claims are now better calibrated.** The limitations section explicitly flags single protocol, single model, MCQ-only, and the absence of KL divergence analysis. The terminological shift to "accuracy degradation" rather than "reasoning collapse" is welcome — it says exactly what is measured.

---

## 3. Remaining Weaknesses

**3.1 The residual variance problem (84.6%) is not discussed.** This is my most significant remaining concern. The ANOVA tells us that primary domain explains 13.1% and everything else you measured explains about 2.5% more. That leaves 84.6% unexplained. The paper does not address what this residual likely contains. Is it question-level difficulty? Individual chain stochasticity from sampling? Something about specific question-domain interactions? A paper that foregrounds variance decomposition as a contribution needs to grapple with where most of the variance actually lives. At minimum, you need a paragraph in the Discussion acknowledging this and offering hypotheses. Ideally, you would decompose further — nest questions within domains, or report question-level random effects.

**3.2 The mechanistic analysis remains promissory.** Section 5.1 proposes "distributional interference under sequential conditioning" but provides no empirical evidence for it. My Round 1 critique asked for mechanistic analysis — token-level KL divergence, attention pattern shifts, or representation-space measurements. The revision acknowledges this gap in limitations but does not attempt even a lightweight version. You have the model weights and the chains. Computing KL divergence between solo and collaborative token distributions for a subset of examples is feasible and would transform Section 5.1 from speculation into evidence. As it stands, the mechanistic story is a hypothesis, not a finding.

**3.3 The collective intelligence analogy is still surface-level.** Citing Becker et al. and noting that alternating CoT resembles a fully connected network is a start, but the analogy needs to do more work. In Becker et al., the key finding is that *network structure* modulates whether social influence helps or hurts — decentralized networks preserve diversity while centralized ones destroy it. Your prediction that "loosely coupled protocols should be less harmful" follows directly, but you could sharpen this considerably. What is the LLM analog of opinion diversity? Token distribution entropy? If you measured entropy of the primary agent's answer distribution before and after helper input, you could test whether collaboration reduces distributional diversity in the same way social influence reduces opinion diversity. Without this, the connection to collective intelligence remains an analogy rather than a theoretical contribution.

**3.4 Per-pair CIs are still wide.** You now report them honestly (±17.9pp), which is good. But this means that for any individual pair, you cannot distinguish a +8pp benefit from a -27pp harm at 95% confidence. The aggregate effect is well-estimated (CI excludes zero), but pair-level claims — "medicine degrades with all partners" — rest on point estimates with wide uncertainty. The paper sometimes shifts between aggregate claims (well-supported) and pair-level claims (underpowered) without flagging the difference. Be more careful about this.

**3.5 No effect size interpretation for the ANOVA.** η²=13.1% for primary domain — is this large or small in the context of LLM evaluation studies? You provide no benchmarks. In social science, this would be considered a medium-to-large effect. In ML, where dataset and architecture effects often dominate, it might be modest. A single sentence contextualizing the effect size would help readers calibrate.

---

## 4. Questions for Authors

1. **Residual variance:** Have you examined whether question-level difficulty (e.g., solo accuracy per question) explains a substantial portion of the 84.6% residual? A random-effects model with questions nested within domains would be informative.

2. **Medicine's catastrophic failure:** Medicine drops 25-45% with every partner. Is this an artifact of the evaluation questions being unusually hard, or does the medicine LoRA also perform poorly solo relative to the base model? What is medicine's solo accuracy, and how does it compare to other domains?

3. **Philosophy's universal benefit:** You report philosophy benefits from all partners (+5% to +20%). Is philosophy's solo baseline unusually low, creating more room for improvement? Or does something about philosophical reasoning chains — perhaps their generality or abstractness — make them robust to distributional interference?

4. **Base model contamination:** Qwen3-1.7B was likely trained on MMLU or similar benchmarks. How do you rule out that domain-level effects reflect the base model's pre-existing domain strengths/weaknesses rather than properties of the LoRA fine-tuning?

5. **KL divergence feasibility:** You acknowledge this gap. Is there a specific reason you did not compute even a pilot KL analysis? If it is computational, a sample of 50-100 chains would suffice for a proof of concept.

---

## 5. Recommendation

**Minor Revision.**

The paper has improved substantially. The ANOVA, bootstrap CIs, training size analysis, and failure mode taxonomy collectively make the empirical contribution solid. The core finding — that collaboration is net harmful and that the harm is primarily a function of the primary agent's domain identity — is well-supported at the aggregate level and genuinely useful for the multi-agent systems community.

However, three issues prevent acceptance in the current form:

1. The 84.6% residual variance must be discussed and, if possible, partially decomposed. This is not optional — it is the majority of your outcome.
2. The mechanistic section needs at minimum a pilot empirical analysis (KL divergence on a subset) or must be explicitly reframed as future work rather than presented as an interpretation of your results.
3. Per-pair claims need to be hedged appropriately given the CI width, or the sample size needs to increase for the domains where you make strong individual-pair claims (especially medicine).

These are all addressable without new experiments (items 1 and 3 are analytical/editorial; item 2 requires modest computation on existing data). I expect the next revision to be the final one.

---

I now have full context on both the Round 1 review and the revised paper. Let me compose the Round 2 review.

---

## Round 2 Review

**Reviewer: Blaise Aguera y Arcas, Google DeepMind**

**Paper: "Systematic Accuracy Degradation in Alternating Chain-of-Thought Collaboration Between Domain-Specialized LLMs"**

---

### 1. Summary of Revisions

The authors have made substantive revisions in response to Round 1 critiques from all five reviewers. The key changes:

- **Title scoped** from "Multi-Agent" to "Alternating Chain-of-Thought Collaboration" -- directly addressing the protocol specificity concern raised by four reviewers.
- **Two-way ANOVA** added (F(9,1700)=29.25, p<0.001 for primary domain; F(9,1700)=1.06, p=0.39 for helper). This was the Evans critique primarily, but I also asked for it implicitly when I wanted the "primary determines outcome" claim substantiated statistically. This is a real improvement -- the variance decomposition is now properly done, with effect sizes and interaction terms.
- **Failure mode analysis** of 1,101 incorrect and 699 correct responses, categorized into confident incorrect (49.6%), extraction failure (29.2%), and answer switching (21.2%). This directly addresses the universal reviewer complaint about missing mechanistic analysis.
- **Bootstrap 95% CIs** on all reported numbers, with honest acknowledgment that per-pair CIs are wide (mean ±17.9pp).
- **Training size confound** addressed with r=0.092, non-monotonic relationship (CS benefits at 500, medicine collapses at 824). This was my concern and Evans's.
- **MoE connection** added in Section 5.3 -- the no-gating interpretation I specifically requested.
- **Mechanistic language** cleaned up: "distributional interference under sequential conditioning" replaces anthropomorphic framing. This was Bratton's critique primarily, but it also strengthens the paper's precision.

**Not done:** base-model-as-helper control, conclusions-only protocol test, expansion to 50+ questions per pair. All acknowledged as future work.

---

### 2. Strengths

- **The ANOVA is the single biggest improvement.** The claim that primary domain identity explains 26x more variance than helper identity (13.1% vs 0.5%) is now properly quantified. The non-significant interaction term (p=0.999) is particularly informative -- it means specific pairings genuinely do not matter beyond domain main effects. This transforms an eyeball claim into a statistical result. Well done.

- **The failure mode taxonomy is valuable.** The answer-switching rate (21.2% of failures) versus correction rate (17.9% of successes) gives a concrete asymmetry that explains the net negative effect. The observation that collaboration is slightly more likely to flip a correct answer to incorrect than to fix an incorrect one is a clean, quotable finding. I would have liked to see this broken down by domain (is medicine's collapse driven by answer switching or confident incorrect?), but even the aggregate analysis is a genuine contribution.

- **The training size confound is effectively neutralized.** r=0.092 is negligible. The non-monotonic examples (medicine at 824 examples collapsing while CS at 500 benefits) are convincing. This was one of my primary concerns in Round 1, and I am satisfied it has been addressed.

- **The MoE connection is appropriate in scope.** Section 5.3 draws the parallel I asked for -- the specialist cannot route between its own knowledge and the helper's reasoning, there is no gating mechanism -- without overclaiming. The analogy is structural and clearly framed as such.

- **The scoped title and claims are honest.** The paper no longer overgeneralizes to "multi-agent collaboration" broadly. Section 5.5 explicitly enumerates six limitations. This is the right posture for a paper with these constraints.

---

### 3. Remaining Weaknesses

- **The base-model-as-helper control remains the most important missing experiment.** I asked for this in Round 1 (Question 4), and it is acknowledged as future work but not done. This matters because the paper's central claim is about domain specialization creating vulnerability. But if the base model (no LoRA) as helper produces the same collapse pattern in medicine, then the vulnerability is not about specialist-to-specialist interference -- it is about ANY foreign reasoning disrupting a fragile adapter. Conversely, if the base model as helper is less harmful, that would implicate LoRA-specific distributional divergence as the mechanism. This is computationally trivial (one inference run per domain, reusing existing infrastructure) and would take less time than writing Section 5.3. Its absence weakens the mechanistic claims.

- **The conclusions-only protocol remains untested.** This is the paper's own theoretical prediction (Section 5.2): that reducing coupling should reduce harm. The authors predict it but do not test it. In the biological framing I think in: you have identified a pathological symbiosis, hypothesized that the coupling mechanism (full reasoning chain injection) is the cause, but not tested the obvious intervention (reduce the coupling). A paper that predicts and tests is substantially stronger than one that predicts and defers.

- **n=20 per pair is still the elephant in the room.** The bootstrap CIs are a genuine improvement -- at least the uncertainty is now visible. But ±17.9pp means that an individual pair showing +5% could easily be -13% or +23%. The domain-level aggregates (9 pairs per domain) have narrower intervals and are defensible. But the heatmap -- which is visually striking and will be the figure people remember -- shows pair-level data at a resolution the statistics cannot support. I would recommend either (a) expanding to 50+ questions for at least the extreme cases (medicine, philosophy) as a robustness check, or (b) presenting only domain-level aggregates in the main text and moving pair-level data to supplementary material with explicit caveats.

- **The 84.6% residual variance deserves more discussion.** The ANOVA shows primary domain explains 13.1% and helper explains 0.5%. That leaves 84.6% unexplained. The paper attributes this to "within-pair question-level variance" but does not investigate further. What drives question-level susceptibility to collaboration interference? Are there question characteristics (difficulty, ambiguity, answer position) that predict whether collaboration helps or hurts on that specific question? Even a brief exploratory analysis would signal that the authors understand the residual is not noise to be dismissed.

- **The failure mode analysis lacks domain-level breakdown.** The aggregate taxonomy is useful, but the most interesting question is: does medicine collapse through answer switching (the helper's reasoning hijacks the primary) or through confident incorrect answering (the primary generates wrong answers independently after exposure)? If medicine's dominant failure mode is answer switching while philosophy's rare failures are extraction errors, that tells a mechanistic story. The data exist to do this analysis.

---

### 4. Questions

1. **Is the answer-switching rate higher for medicine than for philosophy?** If medicine collapses primarily because it switches away from correct answers after receiving helper reasoning, while philosophy maintains its answers, this would be direct evidence for differential susceptibility to distributional conditioning. You have 1,101 classified failures -- breaking them down by primary domain should take minutes.

2. **What is the base model's solo accuracy on these same questions, and how does it compare to the specialists?** If the base model scores 35% on medicine questions while the medicine LoRA scores 45%, the LoRA is only adding 10 percentage points of domain knowledge. Is that thin margin what makes it "fragile"? Conversely, if philosophy's LoRA does not improve much over base (base already decent at philosophy), the "robustness" may reflect the base model's pre-existing strength rather than the adapter's quality.

3. **What is the extraction failure rate for solo versus collaboration?** If extraction failures (29.2% of collaboration errors) are also common in solo performance, then nearly a third of your "collaboration harm" is actually an output parsing problem, not a reasoning problem. This distinction matters for the mechanistic interpretation.

4. **Have you considered that the three failure modes may be protocol artifacts rather than collaboration artifacts?** Confident incorrect could happen in solo. Extraction failure could happen in solo. Only answer switching is unambiguously caused by collaboration. If you reported only answer switching as the "collaboration-specific" failure mode, what would the net delta look like?

---

### 5. Recommendation

**Minor Revision.**

The paper has improved substantially from Round 1. The ANOVA, failure mode analysis, training-size confound analysis, MoE connection, and scoped claims collectively address the majority of concerns raised by all five reviewers. The central finding -- that vulnerability to accuracy degradation under alternating CoT is domain-intrinsic, with primary domain explaining 26x more variance than helper identity -- is now properly quantified and I find it credible.

I am downgrading my recommendation from Major to Minor Revision because the revisions demonstrate that the authors understand the weaknesses and are making genuine improvements. The two missing experiments (base-model-as-helper, conclusions-only protocol) would elevate this from a solid empirical contribution to a paper with mechanistic depth, but their absence does not invalidate the existing results. They are clearly scoped as future work.

For the minor revision, I would ask for:

1. Domain-level breakdown of failure modes (especially answer-switching rates by domain).
2. Brief analysis of the 84.6% residual -- even exploratory.
3. Solo extraction failure rate reported for comparison.
4. Consider presenting pair-level heatmap with explicit CI overlay or moving to supplement.

These are all analyses on existing data and should require no new experiments. If the authors also manage to run the base-model-as-helper control before final submission, I would consider that grounds for Accept without further review.

---



# Review: Round 2

**"Alternating Chain-of-Thought Collaboration Between Domain-Specialized LLMs"**

---

## 1. Summary of Revisions

The authors have made substantial structural and terminological revisions in response to Round 1 feedback. The title is now correctly scoped to the specific protocol under study rather than generalizing to "multi-agent collaboration" as a category. The anthropomorphic vocabulary has been systematically replaced: "reasoning collapse" becomes "accuracy degradation," and the new term "distributional interference under sequential conditioning" provides a mechanistic description grounded in what actually happens to token distributions rather than in folk-psychological attributions of epistemic states to matrix multiplications. The ANOVA provides the statistical machinery that was missing for the intrinsic difficulty claim. The limitations section is now honest about scope. The Sunstein citation is appropriately hedged.

These are real improvements. The paper is measurably more disciplined than the prior draft.

---

## 2. Strengths

**The terminological correction is more than cosmetic.** "Distributional interference under sequential conditioning" does genuine theoretical work. It names the actual computational event -- one model's output distribution reshaping the conditioning context for the next model in ways that degrade task-relevant signal. This is falsifiable, decomposable, and does not require anyone to believe that a language model "reasons" or experiences "fragility." The explicit disclaimer about not making philosophical claims about reasoning capacity is exactly the kind of epistemic hygiene this literature needs more of.

**The ANOVA result is clarifying.** Primary domain explains 13.1% of variance; helper domain explains 0.5% and is not significant. This is a clean result. It tells us that the difficulty structure is largely intrinsic to the domain material and its relationship to the model's training distribution, not an artifact of which other specialist is injected into the pipeline. The collaboration protocol is, statistically speaking, approximately domain-agnostic in its degradation effects. That is worth knowing.

**Answer switching at 21.2% is the most important number in the paper.** It provides direct, non-inferential evidence that the sequential conditioning mechanism actively overwrites correct answers. This is not a model failing to arrive at correctness; it is a model that had the correct answer and then lost it through the collaboration protocol. This distinction matters for anyone designing multi-agent systems.

**The MoE connection is productive.** Expert collapse and routing failures in Mixture-of-Experts architectures provide a well-studied computational analogy that does not require importing concepts from political theory or social epistemology. This is the right neighborhood for theoretical grounding.

---

## 3. Remaining Weaknesses

**The philosophy finding remains the most interesting and least explained result in the paper.** You acknowledge this. But the problem is not merely that the logit entropy analysis is missing -- it is that without it, the philosophy result is doing double duty as both an empirical anomaly and a vague gesture toward something about "abstract reasoning" that the paper cannot cash out. Either investigate it properly or demote it. As it stands, the reader is invited to speculate, which is precisely what a paper this methodologically careful should not encourage. One concrete suggestion: if you cannot do the logit entropy comparison, at minimum report the answer-switching rate for philosophy specifically, broken out from the aggregate 21.2%. If philosophy shows a different switching pattern, that is informative even without the distributional analysis.

**The training data quality confound is acknowledged but not bounded.** Listing it in limitations is necessary but not sufficient. The concern is not abstract: LoRA fine-tuning on small, domain-specific corpora of varying quality is a direct confounder for cross-domain accuracy comparisons. You have ten domains. You could report basic corpus statistics -- token count, vocabulary diversity, average passage length, proportion of technical terminology -- and test whether any of these correlate with the accuracy degradation magnitude. This would not eliminate the confound, but it would indicate whether the effect sizes you observe track trivially with data quality or whether something else is operating.

**"Single protocol" is still underweighted as a limitation.** The paper now acknowledges it, but the framing throughout still periodically slips into generalizations about "collaboration" rather than "this specific alternating CoT protocol." The abstract, introduction, and conclusion should be audited line by line for any sentence that implies the findings generalize beyond sequential two-agent alternating prompting. They do not, and the paper cannot claim they do.

**The Sunstein reference, while now hedged, still pulls in the wrong direction.** Marking the analogy as "structural rather than cognitive" is better, but the analogy itself remains strained. Group polarization in deliberating humans involves preference amplification through social signaling, status dynamics, and argument availability -- none of which have analogues in sequential token generation. A "structural" analogy that shares no structural features with the target phenomenon is not structural; it is decorative. I would recommend removing it entirely and letting the MoE comparison do the theoretical work. The paper is stronger without the social epistemology frame.

---

## 4. Questions for the Authors

1. What is the answer-switching rate for philosophy specifically? If it deviates substantially from the 21.2% aggregate, that alone would advance understanding of the anomaly more than any speculative explanation.

2. You report that helper domain effect is not significant (eta-squared 0.5%). Did you test for specific pairwise interactions? It is possible that the aggregate helper effect is null while specific domain pairings produce significant interference. A non-significant main effect can mask meaningful interactions.

3. The "distributional interference" framing implies a testable prediction: interference should scale with the distributional distance between domains (measurable as, e.g., KL divergence between domain-conditioned output distributions). Have you considered testing this? It would elevate the term from descriptive label to explanatory mechanism.

4. What is the baseline accuracy of each domain specialist on its own domain without any collaboration protocol? The paper reports collaboration results, but the degradation is only meaningful relative to a clearly reported solo baseline for each of the ten domains individually.

---

## 5. Recommendation

**Accept with minor revisions.**

The paper has addressed the most serious problems from Round 1. The terminological discipline is genuine, the ANOVA is well-executed, and the answer-switching analysis provides concrete empirical grounding. The remaining issues -- the unexplained philosophy anomaly, the unbounded training data confound, the residual Sunstein analogy -- are addressable without restructuring the paper.

Required revisions: (a) Either investigate the philosophy finding with at minimum a domain-specific answer-switching breakdown, or explicitly demote it from a highlighted result to a reported observation in supplementary material. (b) Report basic corpus statistics for the ten training sets and test for correlation with degradation magnitude. (c) Remove the Sunstein analogy or replace it with a citation that shares actual structural features with the phenomenon. (d) Audit abstract, introduction, and conclusion for residual over-generalization beyond the specific protocol studied.

The paper contributes a useful negative result about a specific multi-agent protocol, supported by appropriate statistical analysis and an honest limitations section. That is sufficient for publication. What it should not do is gesture toward broader implications it has not earned.

---



# Round 2 Review — "Alternating Chain-of-Thought Collaboration"

## 1. Summary of Revisions

The authors have addressed three of my six Round 1 concerns substantively:

**Claim scoping (addressed).** The title and framing now correctly distinguish alternating CoT from debate. The explicit acknowledgment that this design point sacrifices epistemic independence is exactly right — this was my primary conceptual concern and it is resolved.

**Solo baseline (addressed).** The 3-round self-continuation control is the correct design. This ensures the comparison isolates the effect of cross-agent information, not simply extended generation length. Good.

**Statistical characterization (partially addressed).** The bootstrap CIs and the ANOVA on 1,800 raw observations are meaningful improvements over the original presentation. The aggregate CI of [-12.1%, -6.3%] not crossing zero is informative. The honest acknowledgment of ±17.9pp pair-level uncertainty is appropriate.

Three concerns remain unaddressed: position bias, seed replication, and round ablation. One concern (sample size) is acknowledged but not fixed.

---

## 2. Strengths

**The failure mode taxonomy is genuinely useful.** The 1,101-failure analysis with the answer-switching rate (21.2%) is the most valuable new contribution. This is not just showing that collaboration hurts — it is showing *how* it hurts. The asymmetry between correction rate (17.9%) and switching rate (21.2%) gives a mechanistic account of the net negative effect. This is the kind of analysis that makes a negative result publishable.

**The scoped claims are now defensible.** The paper no longer overgeneralizes to "debate" or "multi-agent reasoning." It makes a precise claim about a precise intervention, which I can evaluate on its own terms.

**The ANOVA design is sound.** Using the 1,800 individual question-level observations rather than collapsing to 90 pair means is statistically correct and extracts much more information from the existing data.

---

## 3. Remaining Weaknesses

### 3a. Answer position bias — still a confound (severity: moderate-high)

With multiple-choice questions and alternating generation, the helper's reasoning appears *after* the specialist's initial answer. If the model has any recency bias — and small models notoriously do — the helper's content may be disproportionately weighted simply because it appears later in the context window, not because of its semantic content. The 21.2% answer-switching rate could partly reflect position bias rather than genuine cross-domain interference.

**What I need:** A control where the helper's reasoning is replaced with a randomly sampled reasoning trace from an unrelated domain (or even shuffled tokens of equivalent length). If answer-switching persists at similar rates, the effect is positional. If it drops, the effect is semantic. This is a clean, inexpensive experiment.

### 3b. Single random seed (severity: moderate)

The entire result rests on one decoding trajectory per question per configuration. With temperature-based sampling (which I assume is nonzero given these are generation tasks), a single seed gives you one draw from the output distribution. The aggregate CI helps, but it cannot distinguish "robust finding" from "this particular seed happened to produce a consistent pattern."

**What I need:** Three seeds minimum. Report seed-level variance. If the aggregate CI still excludes zero across seeds, I am satisfied.

### 3c. No round count ablation (severity: moderate)

You compare 3-round collaboration against 3-round solo. But you do not show what happens at 1 round or 2 rounds. The failure taxonomy suggests answer-switching accumulates over rounds — if so, there may be a round count where collaboration is net-positive before the switching effect dominates. Without this ablation, you cannot determine whether the problem is collaboration itself or *extended* collaboration.

**What I need:** Report accuracy at rounds 1, 2, and 3 separately for both solo and collaborative conditions. This requires no new inference — just evaluate intermediate outputs.

### 3d. Sample size (severity: low-moderate, acknowledged)

n=20 with ±17.9pp CIs per pair means individual domain-pair effects are not reliably estimated. The aggregate analysis compensates partially, but pair-level claims (e.g., "domain X hurts domain Y most") remain underpowered. The authors acknowledge this honestly. I flag it but do not block on it if the other concerns are addressed.

---

## 4. Questions for the Authors

1. **What temperature and sampling parameters were used for the 3-round generation?** If temperature > 0, the single-seed concern is acute. If temperature = 0 (greedy), seed replication is moot but you should state this explicitly.

2. **For the answer-switching failures, can you report the round at which the switch occurs?** If most switches happen at round 2 (first helper injection), that strengthens your mechanistic story. If they accumulate uniformly, the story is different.

3. **The correction rate of 17.9% — is this concentrated in specific domain pairs, or distributed uniformly?** If certain pairs show net-positive collaboration while others are strongly negative, the conclusion should be "collaboration is harmful *on average* but beneficial for specific knowledge-distance configurations," which is a more nuanced and more interesting finding.

4. **Have you examined whether the helper's confidence (e.g., hedging language, explicit uncertainty markers) correlates with switching rate?** A practical intervention might be: filter or downweight uncertain helper reasoning before injection.

---

## 5. Recommendation

**Not yet acceptable. Revise and resubmit (minor-to-moderate revision).**

The conceptual framing and statistical analysis have improved substantially. The failure taxonomy is strong new work. But the three unaddressed experimental controls — position bias, seed replication, and round ablation — are not unreasonable requests and at least two of them (position bias control, round ablation from existing outputs) should be inexpensive to run.

**Priority ordering of what I need to see:**

1. **Position bias control** — highest priority, directly threatens the causal interpretation of answer-switching.
2. **Round ablation from existing data** — essentially free, substantially enriches the analysis.
3. **Seed replication OR explicit statement of greedy decoding** — one or the other resolves this.
4. Larger n is desirable but I will not block publication on it if 1-3 are addressed.

If the authors address items 1-3, I expect to recommend acceptance. The scoped claims, the solo baseline design, and the failure taxonomy are already at a publishable standard — the paper just needs these remaining controls to close the loop on internal validity.

---

## Reviewer 5: AI Safety Perspective (Round 2)


## Round 2 Review: AI Safety and Alignment Perspective

### 1. Summary of Revisions

The revision addresses the most critical of my Round 1 concerns -- the absence of qualitative chain analysis. New Section 3.5 and expanded Section 4.4 classify 1,101 failures into three modes (confident incorrect: 49.6%, extraction failure: 29.2%, answer switching: 21.2%) and 699 successes into three modes (confirmation: 72.5%, correction: 17.9%, elaboration: 9.6%). The mechanistic framing has shifted from "reasoning collapse" to "distributional interference under sequential conditioning," which is more precise. Limitations have been expanded from implicit to an explicit 6-item list. The paper now scopes its claims more carefully to the alternating CoT protocol.

Three of my five Round 1 requests remain unaddressed: confidence/calibration analysis, same-domain collaboration control, and connection to the sycophancy literature.

### 2. Strengths

**The answer-switch asymmetry is the paper's strongest new result from a safety perspective.** The finding that 21.2% of failures involve switching from correct to incorrect, while only 17.9% of successes involve correction from incorrect to correct, is precisely the kind of directional analysis I asked for. This asymmetry is the mechanistic core of why collaboration is net harmful, and it is now clearly stated. For safety practitioners, this is actionable: it tells you that under this protocol, the expected value of "getting help" is negative because the damage channel (switching away from correct answers) is wider than the repair channel (fixing wrong answers).

**The failure mode taxonomy is useful for mitigation design.** Confident incorrect (49.6%) and extraction failure (29.2%) suggest different intervention points. Confident incorrect implies the model's distribution has shifted but remains coherent -- a monitoring system could potentially detect this through distributional comparison. Extraction failure implies the model's generation has become structurally malformed -- detectable by simple format validation.

**The mechanistic reframing is an improvement.** "Distributional drift caused by conditioning on out-of-distribution tokens" is more honest and more useful than "reasoning collapse." It correctly identifies the mechanism as a distributional phenomenon rather than a cognitive one, and it avoids importing claims about internal reasoning processes that the evidence does not support.

**The limitations section is now honest about what this paper cannot claim.** Listing n=20, single protocol, single scale, MCQ-only, missing KL divergence, and training set confound as explicit limitations is a significant improvement.

### 3. Remaining Weaknesses

**3.1. Confidence/calibration analysis remains absent, and this is the most safety-critical gap.** My Round 1 question was: when the medicine specialist gets a question right solo but wrong in collaboration, does the collaborative answer express more, less, or equal confidence? The paper now tells me that 49.6% of failures are "confident incorrect" -- but this is a subjective label based on examining reasoning chains, not a quantitative measurement. Logit entropy, token-level probability of the chosen answer letter, or even a simple textual confidence classifier would transform this from a qualitative observation into a measurable safety signal. The distinction matters: if collaboration produces *high-confidence* wrong answers, it defeats the primary defense against hallucination in deployed systems (using model uncertainty as a safety signal). If confidence drops commensurately with accuracy, the system is at least self-consistent in its degradation and monitoring can catch it.

**3.2. Same-domain control is still missing, and it matters for the mechanistic claim.** The paper now claims the mechanism is "distributional drift caused by conditioning on out-of-distribution tokens." This is a testable claim: if you run medicine+medicine collaboration (where the helper's tokens are *in-distribution* for the primary), the mechanism predicts no degradation or reduced degradation. Without this control, you cannot distinguish "conditioning on any extended reasoning chain causes drift" from "conditioning on foreign-domain tokens causes drift." This is one experiment, on one domain, requiring no new infrastructure -- it should have been done.

**3.3. The sycophancy literature is still absent.** Sharma et al. (2023, "Towards Understanding Sycophancy in Language Models") directly study the phenomenon of models changing correct answers after receiving disagreeing input. The answer-switching failure mode (21.2%) is structurally identical to what the sycophancy literature calls "sycophantic behavior" -- the model abandons a correct position after receiving contrary input. The paper explicitly claims this is "not sycophantic deference in a social sense" but "distributional drift." This is a reasonable distinction, but it needs to be argued against the existing literature, not asserted in a vacuum. Without citing Sharma et al. and distinguishing the phenomenon, the paper appears unaware of directly relevant prior work.

**3.4. The answer-switching analysis is underdeveloped.** The 21.2% rate is reported as an aggregate. But the safety-relevant question is: *which domains* have the highest answer-switching rates? If medicine has a 40% switch rate while philosophy has a 5% switch rate, that would directly connect the failure mode taxonomy to the domain-level patterns and would strengthen the paper's central argument enormously. The data exists -- it just needs to be broken out.

**3.5. No discussion of cascading failure in multi-hop chains.** The paper tests 3-round collaboration. In deployed agentic systems, chains can be 10, 50, or 100 steps. Does the degradation plateau, or does it compound? Even a brief analysis of Round 1 vs. Round 2 vs. Round 3 accuracy within the existing data would address whether degradation is a one-time shock or a compounding process. For safety assessment of real systems, this distinction is critical.

### 4. Questions for the Authors

1. Can you break out the answer-switching rate by primary domain? I would predict medicine has the highest switch rate and philosophy the lowest, which would close the explanatory loop between failure modes and domain-level patterns.

2. You have 3 rounds of reasoning. Can you report accuracy at each round? Specifically: if you extract Agent A's answer after Round 1 (before any collaboration), after Round 2 (after one helper turn), and after Round 3 (final), do you see monotonic degradation or a single-step drop? This is trivially extractable from your existing data and would tell us whether additional rounds of collaboration would make things worse.

3. The "extraction failure" category (29.2%) is concerning from a deployment perspective. Are these failures concentrated in specific domains or specific helpers? If a particular helper consistently causes the primary to produce malformed output, that is a different safety concern than random format failures.

4. Have you considered that the base model (Qwen3-1.7B) may have been RLHF-tuned in ways that increase susceptibility to answer-switching? Bai et al. (2022) and subsequent Anthropic work show that RLHF can increase agreeableness. If Qwen3's alignment training makes it more likely to defer to input context, the answer-switching rate may be inflated relative to a base (non-instruct) model. This connects directly to the safety literature on RLHF-induced sycophancy.

### 5. Recommendation

**Minor Revision** (unchanged from Round 1, but the bar has risen).

The chain analysis addition is substantial and addresses my primary Round 1 concern. The answer-switch asymmetry finding is genuinely important for the safety community. However, three of my original five critiques remain unaddressed:

- Confidence/calibration (feasible, high safety relevance, no new experiments needed -- just compute logit entropy over existing runs)
- Same-domain control (one experiment, directly tests the mechanistic claim)
- Sycophancy literature (no experiments needed, just proper citation and positioning)

Additionally, breaking out answer-switching rates by domain (item 3.4 above) requires only re-analysis of existing data and would significantly strengthen the paper.

I would accept the paper with these additions. Without them, the failure mode analysis remains descriptive rather than mechanistic, and the paper's positioning relative to the sycophancy literature -- which studies an almost identical phenomenon -- is a gap that reviewers at safety-focused venues will notice immediately.

