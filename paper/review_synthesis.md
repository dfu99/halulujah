# Review Synthesis: Consensus Critiques & Action Items

## Verdict Summary

| Reviewer | Persona | Recommendation |
|----------|---------|---------------|
| R1 | James Evans (UChicago, Collective Intelligence) | Major Revision |
| R2 | Blaise Agüera y Arcas (DeepMind, Systems) | Major Revision |
| R3 | Benjamin Bratton (UCSD, Philosophy of Computation) | Major Revision |
| R4 | Yilun Du (MIT, Multi-Agent Debate) | Major Revision |
| R5 | AI Safety Perspective | Minor Revision |

**Consensus: Major Revision.** All 5 reviewers find the core finding (domain-intrinsic vulnerability to reasoning collapse) genuinely novel and interesting. All want the paper to survive revision. But 4/5 say it's not ready as-is.

---

## Unanimous Critiques (ALL 5 reviewers raised these)

### 1. Statistical Power (n=20 is too small)
Every reviewer flagged 20 questions per pair as insufficient. CIs are ~±20pp on individual pairs. The dramatic numbers (medicine+physics = 0%) could be floor effects on tiny samples.

**Action**: Expand to 50+ questions per pair. Bootstrap CIs on all reported numbers. Mixed-effects model for variance decomposition.

### 2. Training Set Size Confound
Philosophy (1,394 examples) benefits; chemistry (423) collapses. Is "epistemic fragility" just "undertrained"?

**Action**: Plot training set size vs. mean collaboration delta. If r > 0.6, reframe the narrative. Consider ablation: train medicine on 1,400 examples.

### 3. No Mechanistic Analysis of Reasoning Chains
All reviewers want to SEE what collapsed reasoning looks like. Is it sycophantic deference? Incoherent text? Confident wrong answers?

**Action**: Qualitative analysis of 10-20 collapsed chains vs. 10-20 successful chains. Categorize failure modes.

---

## Strong Critiques (3-4 reviewers)

### 4. Protocol Specificity (R2, R3, R4, R5)
Alternating CoT is one point in a vast design space. Results may not generalize to debate, voting, or conclusion-only protocols. Title/abstract overgeneralize.

**Action**: Test at least one loosely-coupled protocol (conclusion-only sharing). Scope claims to "alternating CoT" in title/abstract.

### 5. Anthropomorphic Language (R3, R4)
"Reasoning collapse," "epistemic fragility," "fragile knowledge" import philosophical weight without doing philosophical work. Bratton especially: call it "distributional interference under sequential conditioning."

**Action**: Keep evocative terms in title/abstract but define them precisely in Section 3. Add distributional-level analysis.

---

## Per-Reviewer Unique Insights

### Evans (R1)
- Variance decomposition: the "intrinsic to primary domain" claim is a statistical claim that needs ANOVA/mixed-effects, not just eyeballing
- The collective intelligence analogy needs sharpening: alternating CoT is noise injection, not social influence

### Agüera y Arcas (R2)
- Connect to MoE literature (Shazeer, Switch Transformers) — domain specialists are experts, collapse is the routing problem
- Predictive framework: can we predict collapse for a new domain BEFORE running the experiment?
- Test base model (no LoRA) as helper to distinguish "LoRA interference" from "any foreign reasoning"

### Bratton (R3)
- Best suggestion: compare philosophy adapter's output logit entropy vs. medicine adapter's — tests calibrated uncertainty (hypothesis a) vs. general reasoning (hypothesis b)
- References: Sunstein "Law of Group Polarization" is better fit than groupthink; Clark & Chalmers "Extended Mind" for philosophical grounding

### Du (R4)
- Critical: alternating CoT ≠ debate. Our protocol destroys epistemic independence; debate preserves it. Must scope claims or test debate
- Solo baseline needs clarification: same token budget? Self-continuation mechanics?
- Answer position bias control needed
- Multiple random seed runs needed

### Safety (R5)
- Most actionable: add calibration analysis (does model become MORE confident while getting MORE wrong?)
- Same-specialist collaboration (medicine+medicine) as control
- Sycophancy literature (Sharma et al., 2023) directly relevant to our failure mode

---

## Priority Action Items (for paper revision)

### Must-Do (blocks acceptance)
1. **Expand question sets to 50+ per pair** — straightforward with existing infrastructure
2. **Plot training set size vs. collaboration delta** — 10 minutes of analysis
3. **Qualitative analysis of collapsed vs. successful chains** — requires reading raw collab_results.json
4. **Bootstrap CIs on all reported numbers** — computational
5. **Scope title/claims to "alternating CoT"** — editorial

### Should-Do (elevates paper significantly)
6. **Test conclusion-only protocol** — new experiment, tests the theoretical prediction
7. **Test base model as helper** — new experiment, distinguishes LoRA interference from general interference
8. **Logit entropy comparison** (philosophy vs. medicine) — uses existing data
9. **Same-domain collaboration control** (medicine+medicine) — new experiment
10. **Mixed-effects model** — primary domain + helper domain as random effects

### Nice-to-Have (future work)
11. Test on second base model (7B or 70B)
12. Round count ablation (1 vs. 3 vs. 5)
13. Actual debate protocol comparison
14. MoE connection paper
