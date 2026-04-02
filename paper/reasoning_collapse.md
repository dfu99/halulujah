# When Helping Hurts: Systematic Accuracy Degradation in Alternating Chain-of-Thought Collaboration Between Domain-Specialized LLMs

## Abstract

Multi-agent collaboration between large language models is widely assumed to improve reasoning through diverse perspectives. We test this assumption under controlled conditions using alternating chain-of-thought (CoT) collaboration between domain-specialized LLM agents and find systematic *accuracy degradation* — a drop in performance below solo baselines. We fine-tune 10 domain-specialist LoRA adapters on Qwen3-1.7B using MMLU subsets spanning physics, law, biology, computer science, history, mathematics, chemistry, economics, philosophy, and medicine. Across 90 ordered collaboration pairs (20 questions each, 3 reasoning rounds), collaboration is **net harmful**: 55.6% of pairs show accuracy degradation, with a mean delta of −9.2 percentage points (95% CI: ±17.9pp per pair). Two-way ANOVA reveals that the primary agent's domain identity explains 13.1% of outcome variance (F(9,1700)=29.25, p<0.001), while helper identity explains only 0.5% (F(9,1700)=1.06, p=0.39). Qualitative chain analysis identifies three failure modes: confident incorrect answers (49.6%), answer extraction failures (29.2%), and answer switching after helper input (21.2%). Medicine specialists degrade catastrophically with all partners (−25% to −45%), while philosophy specialists benefit universally (+5% to +20%). These results demonstrate that under alternating CoT, collaboration amplifies domain-specific accuracy vulnerabilities rather than compensating for them.

## 1. Introduction

The rise of agentic AI systems has renewed interest in multi-agent collaboration as a path to more capable reasoning. Inspired by ensemble methods and collective intelligence research, recent work proposes that multiple LLM agents can debate, discuss, or co-reason their way to better answers than any individual agent (Du et al., 2023; Liang et al., 2023; Wang et al., 2024). The implicit assumption is that diversity of perspective — whether from different model architectures, prompting strategies, or fine-tuning — provides complementary information that improves collective output.

We test this assumption under controlled conditions using a specific collaboration protocol — alternating chain-of-thought — and find it fails in a systematic and theoretically informative way.

Our experimental setup isolates the effect of domain specialization on collaboration outcomes. We fine-tune 10 domain-specialist LoRA adapters on a single base model (Qwen3-1.7B), each trained on a distinct MMLU subject cluster. We then evaluate all 90 ordered pairs in an alternating CoT protocol: Agent A (the primary, domain-matched specialist) begins reasoning, then Agent B (the helper, from a different domain) continues the reasoning chain, and they alternate for 3 rounds before a final answer is extracted.

The results are striking. Collaboration is net harmful under this protocol: the average accuracy change is −9.2 percentage points relative to solo reasoning. More importantly, the pattern is not random — two-way ANOVA shows that the primary agent's domain identity explains 26 times more variance than the helper's identity (13.1% vs. 0.5%, p<0.001 vs. p=0.39). Certain domains — medicine, chemistry, physics — are systematically harmed by collaboration with every partner. Others — philosophy, law, mathematics — systematically benefit from every partner. Analysis of 1,101 failed reasoning chains reveals three distinct failure modes: confident incorrect answering (49.6%), answer extraction failure (29.2%), and answer switching after helper input (21.2%).

These findings have implications for the design of agentic AI systems. Under alternating CoT:

1. **Agent composition requires domain-aware gating.** The decision of whether to collaborate should be informed by the primary agent's domain characteristics, not assumed to benefit all agents equally.

2. **Accuracy degradation is domain-intrinsic.** When a specialist's accuracy drops during collaboration, this reflects a property of that specialist's learned representations, not the quality of help received. We use "accuracy degradation" rather than "reasoning collapse" to describe this phenomenon precisely: it is a measurable drop in correct-answer rate under sequential cross-domain conditioning, not a philosophical claim about reasoning capacity.

3. **Protocol design constrains outcomes.** Alternating CoT is a maximally coupled protocol that destroys the epistemic independence between agents. Whether looser protocols (conclusion-sharing, debate, voting) exhibit similar patterns is an open question we discuss but do not test.

## 2. Related Work

### Multi-Agent LLM Collaboration

Du et al. (2023) introduced multi-agent debate, showing that multiple ChatGPT instances debating can improve factuality and mathematical reasoning. Liang et al. (2023) proposed encouraging divergent thinking in multi-agent systems. Wang et al. (2024) present a more nuanced view, finding that multi-agent discussions are not uniformly beneficial. Critically, most prior work evaluates collaboration between identical or near-identical models with different prompts, not between genuinely specialized agents with different parametric knowledge. Our work fills this gap by testing collaboration between agents whose distributional differences arise from LoRA fine-tuning rather than prompting variation.

### Domain Specialization in LLMs

LoRA (Hu et al., 2022) enables efficient domain adaptation by training low-rank updates to pretrained weights. Recent work shows domain specialists can achieve competitive performance with significantly less compute than general models (arXiv 2501.02068). However, the interaction between separately-trained specialists — particularly how one specialist's reasoning chain affects another's accuracy — has received little attention.

### Collective Intelligence and Social Influence

Becker, Brackbill, and Centola (2017) showed that social influence in human groups can either enhance or destroy collective intelligence depending on network structure. Fully connected networks tend to collapse opinion diversity, while decentralized networks preserve it. Sunstein (2002) documented "group polarization" — the tendency for group deliberation to push individuals toward more extreme positions. Our alternating CoT protocol is analogous to a fully connected network: each agent directly receives the other's complete reasoning chain, maximizing the potential for distributional interference. This suggests that the accuracy degradation we observe may share mechanisms with social influence effects in human collective intelligence, though we emphasize that the analogy is structural rather than cognitive.

### Mixture of Experts and Routing Failures

The mixture-of-experts (MoE) architecture (Shazeer et al., 2017; Fedus et al., 2022) routes inputs to specialized sub-networks. Expert collapse — where the router fails to diversify across experts — is a known failure mode. Our finding that certain domain specialists cannot maintain accuracy when receiving foreign reasoning chains is conceptually related: the specialist cannot effectively "route" between its own parametric knowledge and the helper's reasoning contribution.

## 3. Methods

### 3.1 Domain Specialist Training

We selected 10 domains from the MMLU benchmark (Hendrycks et al., 2021), each defined by a cluster of related subjects:

| Domain | MMLU Subjects | Training Examples |
|--------|--------------|-------------------|
| Physics | college_physics, high_school_physics, astronomy, conceptual_physics | 783 |
| Law | professional_law, jurisprudence, international_law | 1,907 |
| Biology | college_biology, high_school_biology, anatomy, clinical_knowledge | 794 |
| Computer Science | computer_security, machine_learning, college_computer_science, high_school_computer_science | 500 |
| History | high_school_us_history, high_school_world_history, high_school_european_history, prehistory | 837 |
| Mathematics | high_school_mathematics, college_mathematics, abstract_algebra, elementary_mathematics | 677 |
| Chemistry | high_school_chemistry, college_chemistry | 423 |
| Economics | high_school_microeconomics, high_school_macroeconomics, econometrics | 679 |
| Philosophy | philosophy, formal_logic, moral_scenarios | 1,394 |
| Medicine | professional_medicine, college_medicine, medical_genetics, nutrition | 824 |

For each domain, we fine-tuned a LoRA adapter (r=16, α=32, targeting all linear layers) on Qwen3-1.7B using SFTTrainer with gradient checkpointing. Training configuration: 3 epochs, batch size 2, learning rate 2e-4, cosine schedule. Each adapter adds ~27M trainable parameters (~1.6% of the 1.7B base model).

Training set sizes range from 423 (chemistry) to 1,907 (law). We address the potential confound between training set size and collaboration outcomes in Section 4.5.

### 3.2 Cross-Domain Evaluation

We evaluated all 11 models (base + 10 specialists) on 50 held-out questions per domain (500 total). Questions were formatted as 4-choice MCQ with chain-of-thought prompting. This produces an 11×10 accuracy matrix (model × question_domain).

### 3.3 Alternating Chain-of-Thought Collaboration Protocol

For each ordered pair (A, B) where A ≠ B, we evaluated collaboration on 20 questions from A's domain:

**Protocol:**
1. **Round 1**: Agent A receives the question and generates an initial reasoning chain
2. **Round 2**: Agent B receives the question and Agent A's full reasoning chain, then generates its own reasoning
3. **Round 3**: Agent A receives Agent B's reasoning and produces a final answer

The final answer is extracted from Agent A's last response. This alternating protocol forces agents to integrate foreign reasoning through sequential conditioning on the other agent's token-level output, unlike parallel debate where agents reason independently then aggregate. We emphasize that this is a specific point in the design space of multi-agent collaboration protocols — it is maximally coupled and does not preserve the epistemic independence between agents that characterizes debate protocols (Du et al., 2023).

**Solo baselines**: Each specialist performs the same 3-round protocol as self-continuation (Agent A reasons for all 3 rounds) on the same 20 questions, controlling for the effect of extended reasoning.

### 3.4 Statistical Methods

Given the sample size (n=20 per collaboration pair), we compute 95% bootstrap confidence intervals (10,000 resamples, percentile method) for all reported accuracy values. We perform two-way ANOVA with primary domain and helper domain as fixed factors to decompose variance in collaboration accuracy. Training set size is tested as a covariate. All p-values are reported alongside effect sizes (η²).

### 3.5 Failure Mode Classification

We classified all 1,101 incorrect collaboration responses into failure modes by examining the reasoning chain:

- **Confident incorrect**: Agent A produces a clear but wrong answer letter (49.6%)
- **Extraction failure**: Final response does not contain an extractable answer letter, scored as incorrect (29.2%)
- **Answer switching**: Agent A's initial reasoning contained the correct answer, but the final response after collaboration switched to an incorrect answer (21.2%)

For comparison, we also classified 699 correct collaboration responses:
- **Confirmation**: Agent A initially had the correct answer and maintained it (72.5%)
- **Correction**: Agent A initially had an incorrect or unclear answer, and collaboration led to the correct answer (17.9%)
- **Elaboration**: Extended reasoning chain led to the correct answer through additional detail (9.6%)

## 4. Results

### 4.1 Collaboration is Net Harmful

Across 90 ordered collaboration pairs:
- **50 pairs (55.6%)** showed accuracy degradation (negative delta)
- **34 pairs (37.8%)** showed improvement (positive delta)
- **6 pairs (6.7%)** showed no change
- **Mean delta: −9.2 percentage points** (bootstrap 95% CI of the mean: [−12.1%, −6.3%])

The mean CI width for individual pair accuracies is ±17.9 percentage points, reflecting the limited statistical power at n=20 per pair. While individual pair-level estimates carry substantial uncertainty, the aggregate pattern — that more pairs are harmed than helped, and that harm is systematically concentrated in specific domains — is robust.

### 4.2 Variance Decomposition: Domain Identity Dominates

Two-way ANOVA on collaboration accuracy (1,800 individual question outcomes) reveals:

| Source | SS | df | F | p | η² |
|--------|-----|-----|-------|---------|------|
| Primary domain | 56.03 | 9 | 29.25 | <0.001 | 13.1% |
| Helper domain | 2.03 | 9 | 1.06 | 0.391 | 0.5% |
| Interaction | 8.76 | 81 | 0.51 | 0.999 | 2.0% |
| Residual | 361.85 | 1700 | — | — | 84.6% |

The primary agent's domain identity explains **26 times more variance** than the helper's identity (13.1% vs. 0.5%). The interaction term is not significant (p=0.999), meaning specific pairings do not matter beyond the main effects. This is the statistical basis for our central claim: vulnerability to accuracy degradation under alternating CoT is an intrinsic property of the primary domain, not a function of which helper is paired with it.

The large residual (84.6%) reflects within-pair question-level variance — individual questions vary substantially in difficulty and susceptibility to collaboration effects.

### 4.3 Domain-Level Patterns

**Domains systematically harmed by all helpers:**

| Primary Domain | Solo Baseline | Mean Collab Accuracy | Mean Delta | 95% CI of Delta |
|---------------|--------------|---------------------|-----------|----------------|
| Medicine | 45% | 7.2% | −37.8% | [−42.8%, −32.8%] |
| Chemistry | 45% | 16.7% | −28.3% | [−34.7%, −22.0%] |
| Physics | 40% | 15.6% | −24.4% | [−29.4%, −19.4%] |
| Biology | 60% | 40.6% | −19.4% | [−26.7%, −12.2%] |

**Domains systematically helped by all helpers:**

| Primary Domain | Solo Baseline | Mean Collab Accuracy | Mean Delta | 95% CI of Delta |
|---------------|--------------|---------------------|-----------|----------------|
| Philosophy | 30% | 41.1% | +11.1% | [+4.4%, +17.8%] |
| Math | 45% | 52.8% | +7.8% | [+1.1%, +14.4%] |
| Law | 45% | 53.9% | +8.9% | [+2.2%, +15.6%] |
| Economics | 50% | 55.0% | +5.0% | [−1.7%, +11.7%] |

### 4.4 Failure Mode Analysis

Of 1,800 collaboration responses, 1,101 (61.2%) were incorrect. Analysis of failure modes reveals:

**Failure modes (incorrect responses):**
- **Confident incorrect** (546, 49.6%): Agent A produces a clear answer that is wrong. The reasoning chain shows no hesitation — the agent is not confused, it is simply wrong.
- **Extraction failure** (322, 29.2%): The final response cannot be parsed into a valid answer letter. This often manifests as the model generating additional reasoning without ever committing to an answer, or producing malformed output (e.g., starting with `<think>` tags without a final answer).
- **Answer switching** (233, 21.2%): Agent A initially had the correct answer in Round 1, but switched to an incorrect answer after receiving Agent B's reasoning in Round 2-3. This is the most theoretically interesting failure mode — it represents direct evidence that the helper's reasoning chain caused the primary to abandon a correct position.

**Success modes (correct responses):**
- **Confirmation** (507, 72.5%): Agent A had the correct answer throughout and the helper's input did not disrupt it.
- **Correction** (125, 17.9%): The helper's reasoning led Agent A to the correct answer when it initially had an incorrect one. This demonstrates that collaboration *can* help — but the net effect is negative because corrections are outnumbered by answer switches and other failures.
- **Elaboration** (67, 9.6%): Extended reasoning led to the correct answer.

The answer-switch rate of 21.2% among failures — compared to a correction rate of 17.9% among successes — suggests the collaboration protocol is slightly more likely to cause a correct answer to become incorrect than to fix an incorrect answer. This asymmetry drives the net negative effect.

### 4.5 Training Set Size Confound

Reviewers may note that training set sizes vary substantially (423 to 1,907 examples) and that domains with more training data (philosophy: 1,394; law: 1,907) tend to benefit from collaboration, while domains with less data (chemistry: 423) tend to be harmed. We test this directly:

- Point-biserial correlation between training set size and per-question collaboration accuracy: r = 0.092, p < 0.001
- When training set size is added as a covariate to the ANOVA, the primary domain effect remains significant (p < 0.001), and training size explains minimal additional variance

However, the relationship is not monotonic. Medicine (824 training examples) is the most severely harmed domain, while computer science (500 examples) benefits from collaboration. Economics (679 examples) benefits while physics (783 examples, comparable size) is harmed. Training set size is a partial confound that explains some between-domain variance, but does not account for the full pattern. We discuss possible mechanistic explanations in Section 5.1.

A definitive test would train a medicine specialist on 1,400 examples (matching philosophy's training set size) and retest collaboration. We leave this ablation for future work.

### 4.6 Cross-Evaluation Distance

Using the cross-evaluation accuracy gap as a proxy for domain distance, we find a weak positive correlation with collaboration delta (r = 0.197, n = 90). This metric captures how differently two specialists perform on each other's domains but is an imperfect proxy for distributional distance. KL divergence between specialist logit distributions — a more direct measure — requires additional computation (currently pending). The weak correlation reinforces the ANOVA finding: domain distance is less predictive than domain identity.

## 5. Discussion

### 5.1 Mechanistic Interpretation

We propose that the accuracy degradation observed under alternating CoT results from **distributional interference under sequential conditioning**: when Agent B's token-level output is fed into Agent A's context, it shifts Agent A's conditional distribution away from the distribution that would produce correct answers for A's domain.

This framing avoids anthropomorphic language ("reasoning collapse," "epistemic fragility") while capturing the key mechanism. The severity of interference depends on how sensitive the primary specialist's output distribution is to perturbation of its input distribution. Medicine's extreme sensitivity (−37.8% mean delta) suggests its LoRA adapter learned narrow, high-confidence patterns that are easily disrupted. Philosophy's robustness (+11.1% mean delta) suggests its adapter learned broader distributional patterns — potentially because philosophy training data (formal logic, ethics, moral scenarios) exercises more diverse reasoning modes.

The answer-switching failure mode (21.2%) provides direct evidence for this mechanism: the primary agent had the correct answer, received the helper's reasoning, and switched to an incorrect answer. This is not sycophantic deference in a social sense — it is distributional drift caused by conditioning on out-of-distribution tokens.

### 5.2 Connection to Collective Intelligence

Becker et al. (2017) showed that social influence in fully connected human networks destroys the opinion diversity needed for collective intelligence. Our alternating CoT protocol creates an analogous fully connected structure: each agent receives the other's complete reasoning chain. The structural prediction from this analogy — that loosely coupled protocols should be less harmful — remains untested. A conclusion-only protocol (where the helper shares only its final answer, not its full reasoning chain) would test whether reducing the coupling between agents preserves the primary's distributional integrity while still providing useful signal.

We stress that this is a structural analogy, not a claim about shared cognitive mechanisms between human groups and LLM agent systems.

### 5.3 Connection to Mixture-of-Experts

In MoE architectures (Shazeer et al., 2017; Fedus et al., 2022), expert collapse occurs when the routing function fails to distribute inputs across experts effectively. Our finding that certain specialists cannot integrate foreign reasoning can be viewed as a routing problem at the agent level: the primary specialist lacks the capacity to selectively attend to relevant portions of the helper's reasoning while ignoring irrelevant or harmful portions. Unlike MoE where the router is a learned gating function, our alternating CoT protocol provides no gating — all of the helper's reasoning enters the primary's context.

### 5.4 Implications for Agentic AI Design

Under alternating CoT:

1. **Collaboration should be gated, not default.** Systems should estimate whether a primary agent's domain is robust to collaboration before routing to multi-agent workflows. Our ANOVA results suggest this can be predicted from domain identity alone.

2. **Protocol coupling matters.** The maximally coupled nature of alternating CoT (sharing full reasoning chains) may be responsible for the harmful effects. Designing protocols that share less information (conclusions only) or that preserve agents' independent reasoning (parallel debate with voting) could mitigate accuracy degradation.

3. **Training data composition may predict robustness.** Domains trained on diverse subtopics (philosophy: formal logic + ethics + scenarios) may produce specialists that are more robust to collaboration than narrowly-trained specialists (chemistry: 2 subjects). This hypothesis requires further investigation with controlled training set ablations.

### 5.5 Limitations

1. **Statistical power**: With n=20 questions per collaboration pair, individual pair-level estimates have wide confidence intervals (mean CI width: ±17.9pp). While domain-level aggregate patterns are robust, specific pair comparisons should be interpreted with caution. Expanding to 50+ questions per pair would substantially narrow these intervals.

2. **Single collaboration protocol**: Our results apply specifically to alternating chain-of-thought. Debate protocols (Du et al., 2023), which preserve epistemic independence, or conclusion-sharing protocols, which reduce distributional coupling, may show different patterns. We do not claim that all multi-agent collaboration is harmful — only that this specific protocol, under these conditions, is.

3. **Single base model and scale**: All specialists share Qwen3-1.7B as a base. Cross-architecture collaboration, or collaboration between larger models (7B, 70B), might show different patterns. The sensitivity to alternating CoT interference may be a property of small models that diminishes with scale.

4. **Training set size confound**: Training set sizes range from 423 to 1,907 examples. While our analysis shows this is a partial but not complete explanation, we cannot fully rule out that training data quantity drives the observed vulnerability patterns. Controlled ablation (training all specialists on equal-sized datasets) is needed.

5. **MCQ evaluation only**: Multiple-choice questions are a narrow proxy for reasoning capability. Open-ended generation tasks might show different collaboration dynamics.

6. **No KL divergence distances**: Token-level KL divergence between specialist output distributions — a more direct measure of distributional distance — is pending computation. The cross-evaluation gap is an imperfect proxy.

## 6. Conclusion

We demonstrate that alternating chain-of-thought collaboration between domain-specialized LLMs on Qwen3-1.7B is net harmful, with systematic patterns of accuracy degradation that are determined by the primary agent's domain identity (η²=13.1%, p<0.001) rather than the helper's identity (η²=0.5%, p=0.39). Medicine, chemistry, and physics specialists degrade under collaboration, while philosophy, law, and mathematics specialists benefit. Analysis of 1,101 failed reasoning chains reveals that the dominant failure mode is confident incorrect answering (49.6%), followed by extraction failures (29.2%) and answer switching (21.2%). These findings demonstrate that under maximally coupled collaboration protocols, domain-specialized agents can interfere with each other's accuracy through distributional conditioning effects. The design of multi-agent AI systems should account for domain-specific vulnerability to such interference and consider coupling-aware collaboration protocols.

## References

Becker, J., Brackbill, D., & Centola, D. (2017). Network dynamics of social influence in the wisdom of crowds. PNAS, 114(26), E5070-E5076.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. arXiv:2305.14325.

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. JMLR, 23(120), 1-39.

Gekhman, Z., et al. (2024). Does fine-tuning LLMs on new knowledge encourage hallucinations? arXiv:2405.05904.

Hendrycks, D., et al. (2021). Measuring massive multitask language understanding. ICLR 2021.

Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR 2022.

Kang, H., et al. (2024). Unfamiliar finetuning examples control how language models hallucinate. arXiv:2403.05612.

Liang, T., et al. (2023). Encouraging divergent thinking in large language models through multi-agent debate. arXiv:2305.19118.

Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. ICLR 2017.

Sunstein, C. R. (2002). The law of group polarization. Journal of Political Philosophy, 10(2), 175-195.

Wang, X., et al. (2024). Rethinking the bounds of LLM reasoning: Are multi-agent discussions the silver bullet? arXiv:2402.18272.
