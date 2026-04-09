=== af5c1b6c77f66b17e ===
---

## Review of "When Helping Hurts: Reasoning Collapse in Multi-Agent LLM Collaboration"

**Reviewer: James Evans, University of Chicago**

---

### 1. Summary

This paper fine-tunes 10 domain-specialist LoRA adapters on Qwen3-1.7B and evaluates all 90 ordered collaboration pairs using an alternating chain-of-thought protocol. The central finding is that collaboration is net harmful (mean accuracy delta of -9.2 pp), and that vulnerability to reasoning collapse is determined primarily by the primary agent's domain identity rather than the helper's identity. Medicine specialists collapse catastrophically with any partner, while philosophy specialists benefit universally.

---

### 2. Strengths

- **Clean experimental design with controlled variation.** By using a single base model with domain-specific LoRA adapters, the paper isolates domain specialization as the independent variable. This is a meaningful improvement over prior multi-agent work that conflates model architecture differences with collaboration effects.

- **The asymmetry finding is genuinely interesting.** The observation that the primary agent's domain, not the helper's identity, is the dominant predictor of collaboration success or failure is a non-obvious result. If it holds up to scrutiny, it has real implications for how agentic systems should be designed.

- **Appropriate engagement with the collective intelligence literature.** The paper correctly identifies the analogy to Becker, Brackbill, and Centola (2017) on social influence destroying independent diversity of opinion. The alternating CoT protocol is indeed a maximally coupled network in the sense that matters for our framework. The prediction about loosely-coupled protocols in Section 5.3 is the right prediction to derive from this analogy.

- **Honest limitations section.** The authors acknowledge the small question sets, single base model, missing KL divergence analysis, and single collaboration protocol. This is refreshingly candid.

- **Practical implications are well-articulated.** The argument that collaboration should be gated rather than default is actionable and well-supported by the data.

---

### 3. Weaknesses

- **The statistical analysis is insufficient for the claims made.** You report a one-sample t-test for the overall mean delta but provide no uncertainty quantification for individual domain results. With only 20 questions per collaboration pair, individual pair estimates have enormous variance. A 45% baseline dropping to 0% on 20 items could happen by chance more often than you think. You need confidence intervals for every entry in Tables 4.2-4.4, and you need a mixed-effects model with primary domain and helper domain as crossed random effects to properly decompose the variance. The claim that "vulnerability is intrinsic to the primary domain" is a variance decomposition claim, and you have not done the variance decomposition.

- **The training data size confound is acknowledged but not addressed.** Medicine has 824 training examples and chemistry has 423. Philosophy has 1,394. You speculate in Section 5.1 that training data composition explains the pattern, but you never test this. At minimum, you should report the correlation between training set size and mean collaboration delta. Better yet, retrain a few adapters with equalized training set sizes to rule out the obvious confound that "domains with fewer training examples are more fragile" — which would be a far less interesting finding than "domain characteristics predict epistemic fragility."

- **The "reasoning collapse" mechanism is described, not explained.** You observe that medicine collapses and philosophy benefits, then offer post-hoc narratives about "surface patterns" versus "general reasoning frameworks." But you never look inside the reasoning chains. What actually happens when a medicine specialist receives physics reasoning? Does it change its answer? Does it lose coherent structure? Does it adopt the helper's vocabulary? Without qualitative or quantitative analysis of the reasoning traces themselves, "reasoning collapse" is a label for an outcome, not a mechanism. This is description, not explanation.

- **The connection to collective intelligence theory is suggestive but under-developed.** You cite our work, which I appreciate, but the analogy is not precise. In Becker et al. (2017), social influence operates on independent estimates of the same quantity — the mechanism is that agents revise toward each other and lose the independence that makes crowd wisdom work. In your setup, agents are not estimating the same quantity independently; one agent is reasoning in a domain it was not trained on, injecting potentially irrelevant tokens into the reasoning chain. This is closer to *noise injection* or *catastrophic interference* than to social influence undermining crowd wisdom. The distinction matters because different mechanisms suggest different interventions. Social influence can be managed by network structure; noise injection should be managed by filtering or gating. You conflate these.

- **No comparison to trivial baselines.** What happens if Agent B is the base model with no LoRA adapter? What happens if Agent B's "reasoning" is replaced with random coherent text from a different domain? Without these controls, you cannot distinguish "cross-domain specialist reasoning is harmful" from "any perturbation to a fragile specialist's reasoning chain is harmful." If the latter, your finding is about adapter fragility, not about collaboration per se.

---

### 4. Questions for Authors

1. **Have you computed an ICC or run a crossed random-effects model decomposing variance into primary domain, helper domain, and their interaction?** The claim that vulnerability is "intrinsic to the primary domain" is an empirical claim about explained variance. What fraction of the total variance in collaboration delta is attributable to primary domain versus helper domain versus the pair interaction?

2. **What do the reasoning traces actually look like in collapse cases?** Can you show us a representative medicine question where the solo chain succeeds and the collaborative chain fails? Specifically: does the medicine specialist's Round 3 response look like it has been "overwritten" by the helper's reasoning style, or does it simply become incoherent?

3. **Is there a dose-response relationship?** If you vary the number of collaboration rounds (1, 2, 3, 5), does collapse get worse monotonically? This would help distinguish "any exposure to foreign reasoning is harmful" from "extended exposure progressively degrades."

4. **What is the correlation between solo baseline accuracy and collaboration vulnerability?** Medicine (45% solo) collapses; philosophy (30% solo) benefits. This is the opposite of what you might expect if "strong specialists resist perturbation." Is there a systematic relationship, and if so, does it hold after controlling for training set size?

5. **Have you considered that the base model's pre-existing domain knowledge might be the confound?** Qwen3-1.7B presumably has heterogeneous pre-training coverage across domains. If the base model is already decent at philosophy but weak at medicine, the LoRA adapters sit on different foundations. The "fragility" you observe might be a property of how LoRA interacts with the base model's existing knowledge gradient, not of domain characteristics per se.

---

### 5. Missing References

- **Lorenz, J., Rauhut, H., Schweitzer, F., & Helbing, D. (2011). How social influence can undermine the wisdom of crowd effect.** PNAS, 108(22), 9020-9025. This is the foundational paper on social influence degrading collective intelligence and should be cited alongside Becker et al.

- **Surowiecki, J. (2004). The Wisdom of Crowds.** The conceptual framework you invoke but do not cite.

- **Hong, L., & Page, S. E. (2004). Groups of diverse problem solvers can outperform groups of high-ability problem solvers.** PNAS. Directly relevant to your claim about when diversity helps versus hurts.

- **Bansal, G., et al. (2021). Does the whole exceed its parts? The effect of AI explanations on complementary team performance.** CHI 2021. Relevant to the question of when AI collaboration helps versus hurts humans, with a similar finding that explanations can degrade performance.

- **Chan, A., et al. (2023). ChatEval: Towards better LLM-based evaluators through multi-agent debate.** Relevant multi-agent LLM work you do not cite.

- **Sunstein, C. R. (2002). The law of group polarization.** If you are going to invoke groupthink as an analogy (Section 1), you should cite the theoretical literature on group polarization, which provides a more precise mechanism than the colloquial notion of groupthink.

---

### 6. Recommendation

**Major Revision.**

The central finding — that collaboration vulnerability is domain-intrinsic — is potentially important and novel. But the paper as it stands does not meet the evidentiary bar required to support its claims. Specifically:

(a) The statistical analysis is rudimentary. With 20 questions per pair, you need proper uncertainty quantification, mixed-effects modeling, and variance decomposition — not just a single t-test on the grand mean.

(b) The mechanistic story is absent. You label the phenomenon "reasoning collapse" but never examine the reasoning chains to show what collapse looks like. Without this, the paper is a description of an outcome, not an explanation of a process.

(c) The training data size confound is unaddressed. Until you rule out that training set size explains the domain asymmetry, the most parsimonious explanation of your results is "adapters trained on fewer examples are more fragile," which is not novel.

(d) The connection to collective intelligence theory needs sharpening. The analogy to social influence is suggestive but imprecise; the mechanism in your system is more like noise injection than opinion revision, and this distinction has different design implications.

I would be enthusiastic about a revised version that addresses these issues. The experimental setup is sound, the question is important, and the asymmetry finding, if robust, would be a genuine contribution to both the agentic AI and collective intelligence literatures.



=== af0285dd00fbcb238 ===
---

## Review: "When Helping Hurts: Reasoning Collapse in Multi-Agent LLM Collaboration"

**Reviewer: Blaise Aguera y Arcas, Google DeepMind**

---

### 1. Summary

This paper fine-tunes 10 domain-specialist LoRA adapters on Qwen3-1.7B and evaluates all 90 ordered pairs under an alternating chain-of-thought collaboration protocol, finding that collaboration is net harmful (mean accuracy delta of -9.2 pp). The central claim is that vulnerability to "reasoning collapse" is an intrinsic property of the primary agent's domain: medicine and chemistry specialists collapse catastrophically regardless of partner, while philosophy and law specialists benefit universally. The authors argue this has implications for agentic system design, suggesting collaboration should be gated rather than assumed helpful.

---

### 2. Strengths

- **Clean experimental isolation.** By using a single base model with LoRA adapters as the sole source of specialization, the paper controls for architecture, pretraining data, and scale. This is the right way to study inter-agent interference: hold everything constant except the adaptation. Many multi-agent papers confound model identity with specialization, and this does not.

- **The "primary determines outcome" finding is genuinely interesting.** The observation that the identity of the helper is nearly irrelevant, while the primary's domain almost fully determines whether collaboration helps or hurts, is a clean and somewhat surprising result. It invites a biological analogy: some organisms are mutualism-compatible and others are not, independent of the partner species. This is the kind of finding that, if it holds up, should change how people build agent ensembles.

- **Directly challenges a popular assumption.** The multi-agent debate literature (Du et al., Liang et al.) has created real momentum toward "just add more agents" as an engineering heuristic. This paper provides a concrete, quantified counterexample. That is valuable even if the scope is narrow.

- **Honest limitations section.** The authors are forthright about the small question sets, single protocol, single base model, and missing KL divergence analysis. This is appreciated.

---

### 3. Weaknesses

- **The sample sizes undercut the main claims.** 20 questions per collaboration pair is extremely small. A delta of -45% on 20 questions means going from 9 correct to 0 correct. At that granularity, the confidence intervals are enormous, and a handful of questions could swing a domain from "catastrophic collapse" to "mild degradation." The paper reports a t-test over the 90 pairs, but the per-domain claims (medicine always collapses, philosophy always benefits) are based on 9 data points each (9 partners x 20 questions). The authors should either expand the evaluation set substantially or present bootstrap confidence intervals that make the uncertainty explicit. As it stands, I am not confident the medicine/philosophy asymmetry would replicate on a different 20-question sample.

- **Confound between training set size and "epistemic fragility."** The paper observes that chemistry (423 examples, 2 subjects) collapses while philosophy (1,394 examples, 3 subjects including formal logic) benefits, and speculates that training data composition drives robustness. But this confound is not controlled for. The correlation between training set size and collaboration delta should be reported explicitly. If it is strong, then the entire "intrinsic domain vulnerability" narrative reduces to "adapters trained on more data are more robust to perturbation," which is a far less interesting claim. The paper needs to disentangle quantity from quality of training data.

- **Single collaboration protocol severely limits generalizability.** Alternating CoT is a maximally invasive protocol: the helper's reasoning is injected directly into the primary's context. The authors acknowledge this in the limitations but still title the paper "When Helping Hurts" and draw broad conclusions about multi-agent collaboration. The results may be entirely specific to this protocol. Parallel debate with voting, summary-only exchange, or hierarchical routing could show the opposite pattern. At minimum, the title and abstract should be scoped to "alternating chain-of-thought collaboration" rather than "multi-agent collaboration" broadly.

- **No mechanistic analysis of what actually happens during collapse.** The paper reports accuracy numbers but never shows what the reasoning chains look like during collapse. Does the medicine specialist start adopting the helper's domain vocabulary? Does it abandon its own reasoning mid-chain? Does it produce shorter or longer chains? A qualitative analysis of even 5-10 collapsed examples would dramatically strengthen the paper. Without it, "reasoning collapse" is a label for an accuracy drop, not a characterized phenomenon.

- **The "solo baseline" comparison may be unfair.** The solo baseline uses 3 rounds of self-continuation. But self-continuation is not the same as "no collaboration" -- it is a specific form of auto-augmentation. If the specialist is already near its performance ceiling after 1 round, additional self-rounds might not help but also would not hurt. The collaboration protocol, by contrast, injects genuinely novel (and potentially distracting) content. A fairer comparison might be: 1-round solo vs. 1-round-with-helper, to control for the information injection asymmetry.

---

### 4. Questions for Authors

1. **What is the correlation between training set size (or number of MMLU subjects per domain cluster) and mean collaboration delta?** If r > 0.6, your "intrinsic domain vulnerability" story is really a data quantity story, and the paper needs to be reframed accordingly.

2. **Have you examined the actual token-level behavior during collapse?** Specifically: does the medicine specialist's perplexity on its own domain questions increase after receiving the helper's reasoning? Does it shift its answer distribution toward the helper's prior? A logit-level analysis at the answer token would be illuminating and is computationally cheap.

3. **What happens with a "conclusions-only" protocol?** Your Section 5.3 predicts that loosely-coupled collaboration should preserve epistemic grounding. This is a testable prediction within your existing setup -- the helper could provide only its final answer letter rather than a full reasoning chain. Have you tried this? It would be a strong addition.

4. **How does the base model (no LoRA) perform as a collaborator?** If the base Qwen3-1.7B is used as the helper for all 10 specialists, does the same collapse pattern emerge? This would distinguish "LoRA adapter interference" from "any foreign reasoning disrupts fragile specialists."

5. **Can you connect this to the mixture-of-experts literature?** Your setup is functionally a sparse MoE where routing is domain-matched but the "experts" interfere through shared context rather than shared parameters. The MoE literature has extensive work on expert collapse and routing instability (Shazeer et al., 2017; Fedus et al., 2022). Do you see parallels, and does the MoE solution (load balancing, capacity factors) suggest remedies for your setting?

---

### 5. Missing References

- **Shazeer et al. (2017), "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"** -- The foundational MoE paper. Your domain specialists are essentially experts, and your finding that some experts are fragile while others are robust has direct parallels to expert collapse in MoE training.

- **Fedus, Zoph, and Shazeer (2022), "Switch Transformers"** -- Discusses expert capacity and the pathology of expert collapse, which is the parametric analog of your reasoning collapse.

- **Kaplan et al. (2020), "Scaling Laws for Neural Language Models"** -- Relevant because your fragility pattern may be explained by scaling: 423 training examples for chemistry vs. 1,394 for philosophy. Scaling laws would predict exactly the robustness differential you observe.

- **Hong et al. (2024), "MetaGPT: Meta Programming for Multi-Agent Collaborative Framework"** -- A prominent multi-agent framework paper that should be situated in your related work, as it proposes structured role assignment that could mitigate the collapse you observe.

- **Jacobs et al. (1991), "Adaptive Mixtures of Local Experts"** -- The original mixture-of-experts paper. Your finding that specialist identity determines collaboration outcome is an empirical rediscovery of the routing problem from a different angle.

- **Woolley et al. (2010), "Evidence for a Collective Intelligence Factor in the Performance of Human Groups"** -- If you are going to invoke collective intelligence theory, you should cite the foundational empirical work, not only the social influence critique (Becker et al.).

---

### 6. Recommendation: **Major Revision**

**Justification:** The core finding -- that collaboration outcome is determined primarily by the primary agent's domain identity -- is genuinely interesting and, if validated, has real implications for how we design multi-agent systems. I can see this becoming an important result. But the paper is not there yet, for three reasons:

First, the statistical foundation is too thin. Twenty questions per pair is not enough to support the strong claims made in the abstract and conclusion. The per-domain patterns could shift substantially with a larger evaluation set. Either expand to 50+ questions per pair or present proper uncertainty quantification.

Second, the training-data-size confound is not addressed. Until the authors can show that the collapse pattern is not simply "adapters with less training data are more fragile," the theoretical framing around "intrinsic epistemic fragility" is speculative. This is a critical analysis that should take an afternoon to run.

Third, the absence of any mechanistic analysis (what do the reasoning chains look like during collapse?) means the paper is describing a phenomenon but not characterizing it. Even a small qualitative analysis would elevate this from "we observed an accuracy drop" to "we understand what goes wrong."

The bones are good. The experimental design is clean, the finding is counterintuitive, and the implications for agentic AI design are real. But the paper needs more evidence, more controls, and more mechanistic depth before it can support the claims it makes. I would be enthusiastic about a revised version that addresses these gaps.



=== ae07bcc0369e48343 ===
---

## Review: "When Helping Hurts: Reasoning Collapse in Multi-Agent LLM Collaboration"

**Reviewer: Benjamin Bratton, UC San Diego / Berggruen Institute**

---

### 1. Summary

The paper demonstrates that alternating chain-of-thought collaboration between domain-specialized LoRA adapters (fine-tuned on Qwen3-1.7B across 10 MMLU domains) degrades accuracy in 55.6% of pairings, with a mean delta of -9.2 percentage points versus solo baselines. The core finding is that vulnerability to this "reasoning collapse" is determined by the primary agent's domain — medicine collapses universally, philosophy benefits universally — regardless of which helper is paired with it. The authors argue this has implications for agentic system design and collective intelligence theory.

---

### 2. Strengths

- **The finding is genuinely interesting and counterintuitive.** The systematic, domain-intrinsic nature of collaboration failure — that the helper's identity is nearly irrelevant — is a non-obvious result that deserves serious attention. It cuts against the dominant "more agents, more better" orthodoxy with actual data.

- **Clean experimental isolation.** Using a single base model with LoRA adapters controls for architectural confounds that plague most multi-agent studies. You are testing what domain specialization *does* to collaboration, not what different architectures do.

- **The asymmetry between medicine and philosophy is the paper's best finding.** This is the kind of result that should generate theoretical work. That a domain with higher baseline accuracy collapses while a domain with *lower* baseline accuracy benefits is genuinely provocative.

- **Honest limitations section.** The authors acknowledge the small question sets, single protocol, and missing KL divergence analysis. This is preferable to overclaiming.

- **Practical relevance.** For anyone designing multi-agent pipelines — and the industry is currently building these at scale with minimal theoretical grounding — the message that collaboration should be gated rather than defaulted is immediately useful.

---

### 3. Weaknesses

- **The conceptual vocabulary does not earn its keep.** "Reasoning collapse," "epistemic fragility," "fragile knowledge" — these terms import philosophical weight without doing philosophical work. What does it mean for a LoRA adapter to have "fragile knowledge"? A rank-16 low-rank perturbation to a weight matrix does not *know* anything in a sense that admits of fragility. What you are observing is a specific failure mode in sequential token generation when conditioning distributions are perturbed by off-domain continuations. Call it what it is: *distributional interference under sequential conditioning*. The anthropomorphic framing ("the medicine adapter cannot distinguish helpful reasoning from irrelevant interference") obscures the actual mechanism and invites exactly the kind of sloppy thinking about machine cognition that the field needs less of.

- **The "collective intelligence" analogy to Becker et al. is strained to the point of misleading.** Human social influence operates through belief updating under normative pressure. What happens in your alternating CoT protocol is that one model's output tokens become the next model's input context. There is no "social influence" — there is context window contamination. The Evans/Becker framework describes agents who *choose* to update beliefs; your agents have no choice because the foreign reasoning is literally part of their input. This is a fundamentally different mechanism, and analogizing them obscures rather than illuminates.

- **You cannot distinguish between your two hypotheses for philosophy's success, and this matters enormously.** Is philosophy robust because it learns *general reasoning structure* (hypothesis b), or because it learns *calibrated uncertainty* (hypothesis a)? These have completely different implications. The first suggests a hierarchy of reasoning styles; the second suggests a meta-cognitive capacity. You have the data to at least partially test this — look at the philosophy adapter's output logit entropy versus the medicine adapter's. If philosophy has higher entropy (less confident per-token), that supports (a). If it has similar entropy but different attention patterns, that supports (b). The fact that you did not do this analysis is a significant gap.

- **The training data size confound is acknowledged but not controlled for.** Philosophy has 1,394 training examples; chemistry has 423. Medicine has 824. You gesture at training data composition as a factor in Section 5.2, but this is precisely the kind of confound that undermines your "intrinsic domain property" claim. What if the finding is simply: *adapters trained on fewer examples are more fragile under distributional perturbation*? This is a much less interesting claim than "medicine is epistemically fragile," but your experimental design cannot rule it out.

- **20 questions per collaboration pair is genuinely insufficient for the claims being made.** You report a mean delta of -37.8% for medicine, but with 20 binary-outcome trials, the 95% confidence interval on a proportion estimate is roughly +/-20 percentage points. Many of your individual pair results (medicine+physics = 0%) are consistent with floor effects on a tiny sample rather than true catastrophic collapse. The paper would be substantially stronger with 100+ questions per pair, which is feasible given that inference is cheap.

---

### 4. Questions for Authors

1. **What happens to the medicine adapter's output logit distribution when it receives a foreign reasoning chain versus its own?** Specifically: does entropy increase (the model becomes confused), decrease (it becomes overconfident in the wrong direction), or shift modally (it latches onto the foreign reasoning)? This would distinguish between "collapse as confusion" and "collapse as hijacking," which are very different failure modes with very different remedies.

2. **Have you tested whether the ordering of rounds matters?** Your protocol has A-B-A. What happens with A-B-B-A, or with B generating first and A correcting? The claim that vulnerability is "intrinsic" to the primary domain is only testable if you show it holds across multiple protocol orderings. As it stands, you may be measuring an artifact of who speaks last.

3. **What happens when the philosophy adapter is the *helper* rather than the *primary*?** If philosophy's robustness reflects a genuinely superior reasoning style, then philosophy-as-helper should improve other domains more than random helpers. If it does not, your "reasoning style" hypothesis in Section 5.1 is wrong, and the phenomenon is better explained by something about how philosophy *receives* input rather than how it *produces* it.

4. **Can you reproduce the core finding with a different base model?** The entire experiment rests on Qwen3-1.7B. LoRA adapter interference may behave very differently at 7B or 70B scale, or on architecturally different models. The claim about "domain-intrinsic vulnerability" needs at least one replication on a different base to be credible.

5. **What is the actual token-level mechanism of collapse?** You have all the generation traces. Can you identify the specific point in the reasoning chain where the medicine adapter starts following the foreign reasoning rather than its own? Is it immediate (round 2's first token) or gradual? This would be far more valuable than the aggregate accuracy statistics.

---

### 5. Missing References

- **Janis, I. (1972). *Victims of Groupthink*.** You invoke groupthink but do not cite the originating work. Janis's conditions for groupthink (insulation, directive leadership, lack of methodical procedures) map poorly onto your setting, which is another reason the analogy is strained — but if you are going to use the term, cite the source.

- **Sunstein, C. R. (2002). "The Law of Group Polarization."** More relevant than groupthink: your finding that collaboration amplifies the primary agent's existing tendency (fragile becomes more fragile, robust becomes more robust) is structurally closer to group polarization than groupthink.

- **Li, Y., et al. (2024). "More Agents Is All You Need."** Directly relevant counterpoint to your claims — they show scaling agents improves performance under majority voting, which is a very different protocol from yours. The contrast would sharpen your argument about protocol dependence.

- **Wen, Y., et al. (2024). "Language Models Learn to Mislead Humans via RLHF."** Relevant to the mechanism question — if models can learn to produce persuasive but incorrect reasoning, the "hijacking" interpretation of your collapse finding becomes more plausible.

- **Clark, A. & Chalmers, D. (1998). "The Extended Mind."** If you want to make philosophical claims about distributed cognition across agents — and you are implicitly doing so — engage with the literature on extended mind and cognitive integration. The question of when an external reasoning process *helps* versus *harms* cognition has a rich philosophical history you are ignoring.

---

### 6. Recommendation

**Major Revision.**

The core empirical finding — that collaboration vulnerability is domain-intrinsic and helper-independent — is genuinely novel and worth publishing. But the paper suffers from three problems that require substantive revision:

First, the conceptual framework is underdeveloped and leans on anthropomorphic metaphors ("fragile knowledge," "epistemic fragility," "groupthink") that do not actually explain the mechanism. The paper needs either a proper mechanistic account (what happens at the distributional level during collapse?) or an honest admission that the phenomenon is observed but not yet explained. Pseudo-explanations dressed in philosophical language are worse than no explanation.

Second, the statistical grounding is too thin. Twenty questions per pair is a pilot study, not a result. The headline numbers (medicine at -37.8%) are dramatic but the confidence intervals are wide enough to drive a truck through. Double or triple the evaluation set and the paper becomes much more convincing.

Third, the most interesting theoretical question — *why* philosophy benefits and medicine collapses — is left at the level of speculation. The authors have the infrastructure to test their hypotheses (logit entropy analysis, attention pattern comparison, training data diversity metrics) and should do so before publication.

What makes this paper worth revising rather than rejecting is the asymmetry finding itself. That a low-performing specialist can be *improved* by arbitrary cross-domain input while a higher-performing specialist is *destroyed* by the same is not predicted by any existing theory of multi-agent LLM collaboration. This is the kind of result that should force the field to reconsider its assumptions — but only if the evidence is solid and the explanation is honest about what it does and does not know.



=== a453e45e690f1bee1 ===
---

## Review of "When Helping Hurts: Reasoning Collapse in Multi-Agent LLM Collaboration"

**Reviewer: Yilun Du, MIT CSAIL**

---

### 1. Summary

This paper fine-tunes 10 domain-specialist LoRA adapters on Qwen3-1.7B across MMLU subject clusters and evaluates all 90 ordered collaboration pairs using an alternating chain-of-thought protocol with 3 rounds. The central finding is that collaboration is net harmful (mean accuracy delta of -9.2pp), and that vulnerability to this "reasoning collapse" is primarily a property of the primary agent's domain (medicine collapses universally, philosophy benefits universally) rather than the helper's identity. The authors argue this has implications for agentic system design, suggesting collaboration should be gated rather than default.

---

### 2. Strengths

- **Clean experimental isolation.** Using a single base model with LoRA adapters is a smart design choice. It controls for architecture, pretraining data, and decoding implementation, isolating the effect of domain-specific fine-tuning. This is more rigorous than comparing, say, GPT-4 vs. Claude on a collaboration task, where you cannot attribute differences to any single factor.

- **Exhaustive pairwise evaluation.** Testing all 90 ordered pairs rather than cherry-picking a few collaboration scenarios is the right approach. The resulting asymmetry matrix (A helps B but B hurts A) is genuinely informative and would be missed by partial sampling.

- **The "primary domain determines outcome" finding is novel and clean.** The observation that the helper identity is nearly irrelevant while the primary's domain is deterministic is a striking result. If it holds up to scrutiny (see weaknesses), this is a useful contribution to the field's understanding of when multi-agent setups fail.

- **Honest limitations section.** The authors acknowledge small question sets, single protocol, MCQ-only evaluation, and missing KL divergence analysis. This is appreciated.

---

### 3. Weaknesses

- **Alternating CoT is not debate, and the paper conflates them.** This is my primary concern. The paper cites Du et al. (2023) and Liang et al. (2023) in the introduction as motivation, then evaluates a fundamentally different protocol. In multi-agent debate, agents reason *independently* in parallel, then read each other's responses and revise. This preserves epistemic independence in the first round. Your alternating CoT protocol has Agent B *continuing* Agent A's reasoning chain from round 1 -- there is zero independent reasoning from the helper. This is sequential continuation, not debate. The dynamics are categorically different: debate preserves diversity of reasoning paths; your protocol actively destroys it by forcing B to operate within A's reasoning frame. The title and framing ("multi-agent LLM collaboration") suggest generality, but the results may be specific to this particular (quite aggressive) form of coupling. You need to either (a) test actual debate as a comparison protocol, or (b) scope your claims much more narrowly to "alternating CoT" specifically.

- **The solo baseline is suspect.** Your baseline has each specialist do "solo reasoning for 3 rounds (self-continuation) on the same 20 questions." But what does 3 rounds of self-continuation mean? Does the model see its own prior reasoning and extend it? If so, this is a strong baseline (self-refinement). If not -- if it is just a single forward pass labeled "3 rounds" -- then it is an unfair comparison because the collaboration condition gets 3 actual generation steps with intermediate context. The paper does not specify this clearly enough. Additionally, do you confirm that the solo baseline uses the *same* total token budget as collaboration? If the collaboration condition generates substantially more tokens (because two agents alternate), the degradation might partly reflect the known failure mode of LLMs losing track of long contexts, not cross-domain interference per se.

- **20 questions per pair is underpowered.** With n=20 binary-ish outcomes (4-choice MCQ, so ~25% chance), the standard error on each accuracy estimate is roughly sqrt(p(1-p)/20) which is about 10-11 percentage points at p=0.45. Your reported deltas (-25% to +20%) are in many cases within 1-2 standard errors of zero. The aggregate analysis (90 pairs, t-test on means) gains power from pooling, but the per-pair and per-domain claims (e.g., "medicine collapses catastrophically") rest on very noisy individual estimates. A single question flipped by chance moves accuracy by 5 percentage points. The medicine specialist at 45% solo and 0% collaboration on 20 questions: that 0% is striking, but is it 0/20 or could it be 1/20 with a slightly different random seed? You need confidence intervals on every reported number, and ideally a replication with different question subsets.

- **Confound: training set size tracks with collapse/benefit.** Look at your own Table 1. Philosophy has 1,394 training examples and benefits from collaboration. Chemistry has 423 and collapses. Medicine has 824 but is drawn from 4 quite heterogeneous subjects (professional medicine, college medicine, medical genetics, nutrition) which may produce incoherent specialization. The paper gestures at this in Section 5.1 but does not control for it. You need an ablation: train a medicine specialist on 1,400 examples (augmented or with more subjects) and see if it still collapses. Without this, your "intrinsic domain vulnerability" claim might just be "adapters trained on fewer or more heterogeneous examples are more fragile," which is a much less interesting finding.

- **No ablation on number of rounds.** You fix rounds at 3. What happens at 1 round (A reasons, B revises, done)? What about 5 rounds? If the collapse worsens with more rounds, that supports your interference hypothesis. If it stabilizes or reverses, the mechanism is different. This is a basic ablation that is missing.

---

### 4. Questions for Authors

1. **What exactly is the solo baseline protocol?** Does the specialist generate reasoning, see its own output, and continue for 3 rounds? Or is it a single generation pass? And critically, is the total token budget (input + generated tokens across all rounds) matched between solo and collaboration conditions?

2. **Have you examined the actual reasoning chains qualitatively?** You claim the medicine specialist "defers to foreign reasoning," but you provide no evidence for this mechanism. Showing 2-3 annotated examples where the medicine specialist's reasoning degrades after receiving the helper's input -- and contrasting with examples where the philosophy specialist's reasoning improves -- would substantially strengthen the paper. Without this, the mechanistic claims in Section 5.1 are speculation.

3. **What happens if you use the base model (no LoRA) as the helper?** This would distinguish "cross-domain LoRA interference" from "any perturbation of the reasoning chain hurts fragile specialists." If the base model as helper also collapses medicine, the issue is not domain mismatch but fragility of the medicine adapter to any continuation by a different model.

4. **Did you control for answer position bias?** Small models on MCQ are notoriously sensitive to the position of the correct answer (A/B/C/D). Did you verify that the question subsets used for collaboration do not have a different answer position distribution than the solo evaluation set? With n=20, this could easily create spurious effects.

5. **What is the inference temperature and do you average over multiple runs?** MCQ accuracy at n=20 with a 1.7B model can vary substantially across random seeds. Did you run each condition multiple times?

---

### 5. Missing References

- **Chan et al. (2023), "ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate."** Directly relevant -- shows multi-agent debate dynamics for evaluation, with findings about when agents converge vs. diverge.

- **Xiong et al. (2023), "Examining Inter-Consistency of Large Language Models Collaboration."** Studies consistency of multi-LLM collaboration outputs, relevant to your stability/collapse findings.

- **Li et al. (2024), "More Agents Is All You Need."** Claims scaling agents improves performance -- your work directly challenges this, and you should engage with it.

- **Zhang et al. (2024), "Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View."** Taxonomizes collaboration protocols (debate vs. discussion vs. sequential) -- your paper would benefit from situating alternating CoT within this taxonomy.

- **Cohen et al. (2023), "LM vs LM: Detecting Factual Errors via Cross Examination."** Another multi-agent protocol with different coupling dynamics -- useful contrast to your alternating CoT.

---

### 6. Recommendation

**Major Revision.**

The core finding -- that domain identity of the primary agent predicts collaboration success/failure more than helper identity -- is genuinely interesting and, if validated, would be a useful contribution. The experimental design (all 90 pairs, single base model, LoRA isolation) is thoughtful.

However, the paper has three issues that must be addressed before it is publishable:

First, the framing overgeneralizes. The results are specific to alternating CoT (a maximally-coupled sequential protocol) but the paper implies conclusions about "multi-agent collaboration" broadly. Either test at least one loosely-coupled protocol (e.g., independent reasoning then vote, or actual debate) as a comparison, or explicitly restrict all claims to "alternating chain-of-thought continuation."

Second, the statistical power is insufficient for the per-domain claims that are the paper's main contribution. With 20 questions per pair, individual accuracy estimates have ~10pp standard errors. The paper needs either larger question sets (50+ per pair), replications across random seeds, or bootstrap confidence intervals on every reported number. The aggregate t-test is fine for "collaboration is net harmful on average," but the paper's interesting claims are about *which* domains collapse, and those need tighter estimates.

Third, the training set size confound is unaddressed. An ablation controlling for number and diversity of training examples is needed to distinguish "domain X is intrinsically fragile" from "adapters trained on fewer examples are fragile."

The mechanistic analysis (Section 5.1) is currently hand-waving. Adding qualitative analysis of reasoning chains and the base-model-as-helper ablation would elevate this from an empirical observation to a paper with genuine explanatory power.



=== a63d2ca4e96da4c52 ===
---

## Review: "When Helping Hurts: Reasoning Collapse in Multi-Agent LLM Collaboration"

**Reviewer perspective**: AI Safety and Alignment

---

### 1. Summary

This paper fine-tunes 10 domain-specialist LoRA adapters on Qwen3-1.7B and evaluates all 90 ordered collaboration pairs under an alternating chain-of-thought protocol. The central finding is that collaboration is net harmful (mean accuracy delta of -9.2 pp), and that the primary agent's domain -- not the helper's identity -- determines whether collaboration helps or hurts. Medicine specialists collapse catastrophically (dropping from 45% to 0-20%) while philosophy specialists benefit universally, suggesting that LoRA specialization creates qualitatively different epistemic fragility profiles.

---

### 2. Strengths

- **Directly safety-relevant finding.** The demonstration that collaboration can *systematically degrade* performance is exactly the kind of result that deployed agentic system designers need. The naive assumption that "more agents = better" is widespread in industry, and this paper provides a concrete, reproducible counterexample.

- **Clean experimental isolation.** Using a single base model with LoRA adapters controls for architecture and pretraining distribution, isolating the effect of domain specialization from confounds introduced by mixing different model families. This is a well-designed ablation.

- **The "primary determines collapse" finding is novel and actionable.** The asymmetry -- that vulnerability is an intrinsic property of the primary agent, not a function of the pair -- is a genuinely new contribution. This gives system designers a clear signal: audit the primary agent's robustness before enabling collaboration, rather than optimizing helper selection.

- **Honest limitations section.** The paper is forthright about small sample sizes, single protocol, and missing KL divergence analysis. This builds trust in the reported findings.

- **Good connection to collective intelligence literature.** The analogy to social influence destroying diversity of opinion in human groups (Becker et al., 2017) is apt and provides a theoretical frame that generates testable predictions (loosely-coupled protocols should help).

---

### 3. Weaknesses

- **No analysis of failure mode mechanics.** This is the paper's most significant gap from a safety perspective. You show *that* medicine collapses, but not *how*. Does the medicine specialist defer to the helper's answer? Does it produce incoherent reasoning? Does it switch to a wrong answer with high confidence? Does the CoT become self-contradictory? A qualitative analysis of even 10-20 collapsed reasoning chains would dramatically increase this paper's contribution. Without it, we cannot distinguish between several mechanistically distinct failure modes (sycophantic deference, reasoning chain corruption, attention hijacking, etc.), each of which would demand a different mitigation.

- **No calibration or confidence analysis.** From a safety standpoint, the most dangerous failure is not getting the wrong answer -- it is getting the wrong answer *while appearing confident*. You report accuracy deltas but never examine whether collaboration increases or decreases the model's expressed confidence. A model that drops from 45% to 5% accuracy but remains equally assertive in its (now wrong) answers is far more dangerous than one that hedges or refuses. This analysis is feasible with your existing data (examine the language of final answers) and would substantially strengthen the safety implications.

- **Training data size confound is inadequately addressed.** Chemistry has 423 training examples and collapses; philosophy has 1,394 and benefits. The paper speculates about "diverse reasoning styles" but does not control for this obvious confound. A simple analysis -- plotting training set size against mean collaboration delta -- would either confirm or rule out the explanation that fragility is simply a function of data scarcity. As written, the "epistemic fragility" framing may be an artifact of underfitting.

- **20 questions per pair is genuinely underpowered.** The paper acknowledges this but still makes strong claims ("catastrophically," "systematically"). With n=20 binary outcomes, the 95% confidence interval for a proportion is roughly +/-20 percentage points. The medicine+physics pair showing 0% could be anywhere from 0-17% with reasonable confidence. The aggregate analysis across all 9 helpers per domain is more convincing, but individual pair claims should be heavily caveated.

- **The "alternating CoT" protocol is maximally adversarial by construction.** Forcing the primary agent to continue reasoning from a foreign chain is not how most deployed multi-agent systems work. Debate protocols, voting ensembles, and tool-use delegation all preserve agent autonomy more than this protocol does. The paper should more carefully scope its claims to the specific protocol tested, rather than generalizing to "multi-agent collaboration."

---

### 4. Questions for Authors

1. **Can you characterize the failure mode qualitatively?** When the medicine specialist collapses to 0% accuracy, what does the final reasoning chain look like? Is the model deferring to the helper's reasoning, generating incoherent text, or arriving at a wrong answer through a superficially coherent but subtly corrupted chain? This distinction matters enormously for mitigation design.

2. **Does collaboration change confidence calibration?** Specifically: when the medicine specialist gets a question right solo but wrong in collaboration, does the collaborative answer express more, less, or equal confidence compared to the solo answer? If collaboration produces confident wrong answers, the safety implications are much more severe.

3. **Have you tested whether the collapse is an artifact of LoRA interference on a shared base model?** When Agent B's reasoning chain passes through Agent A's forward pass, Agent A's LoRA weights process text that was generated by a differently-adapted model. Have you tried collaboration between two instances of the *same* specialist (e.g., medicine+medicine) to establish whether the collapse is about *foreign* reasoning or about *any* multi-round reasoning?

4. **What happens with a "conclusion-only" sharing protocol?** Your Discussion predicts that loosely-coupled protocols should preserve epistemic grounding. Have you tested this? If Agent B shares only its final answer letter (not the full CoT), does the medicine specialist still collapse? This would distinguish "reasoning chain corruption" from "social influence on answer selection."

5. **Is there a correlation between solo accuracy and collaboration fragility, independent of domain?** Medicine (45%) collapses; philosophy (30%) benefits. But biology (60%) also collapses while math (45%) benefits. Is there a nonlinear relationship, or is domain genuinely the right level of analysis?

---

### 5. Missing References

- **Perez et al. (2022), "Discovering Language Model Behaviors with Model-Written Evaluations"** -- relevant to the sycophancy/deference failure mode that may explain collapse.
- **Sharma et al. (2023), "Towards Understanding Sycophancy in Language Models"** -- directly relevant; sycophantic deference to the helper's reasoning chain is a likely mechanism for reasoning collapse.
- **Chan et al. (2023), "ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate"** -- another multi-agent collaboration framework with different coupling.
- **Xiong et al. (2023), "Examining Inter-Consistency of Large Language Models Collaboration"** -- examines consistency degradation in multi-agent setups.
- **Wei et al. (2024), "Simple synthetic data reduces sycophancy in large language models"** -- suggests potential mitigations for the deference mechanism.
- **Anthropic's work on Constitutional AI and RLHF** (Bai et al., 2022) -- relevant to the question of whether safety-trained models might be *more* susceptible to sycophantic collapse in multi-agent settings, since RLHF can increase agreeableness.

---

### 6. Recommendation

**Minor Revision**

**Justification:** The core finding -- that multi-agent collaboration can systematically degrade performance in a domain-dependent manner -- is novel, clearly demonstrated, and directly relevant to the safe deployment of agentic AI systems. The experimental design is clean and the primary-determines-collapse result is actionable. However, the paper's contribution to safety is currently limited by the absence of failure mode analysis and calibration analysis. These are not new experiments; they require examining the existing data more carefully. Specifically:

- Add qualitative analysis of collapsed reasoning chains (what does the model actually do when it fails?)
- Add confidence/calibration analysis (does the model become more confident as it becomes more wrong?)
- Control for training set size as a confound
- Scope claims more carefully to the specific protocol tested

With these additions, the paper would make a strong contribution to the safety literature on multi-agent systems. Without them, it demonstrates an important phenomenon but leaves practitioners without the mechanistic understanding needed to mitigate it.



