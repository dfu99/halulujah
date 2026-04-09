# Review Synthesis Round 3: Three-Protocol Comparison

## Verdict Summary

| Reviewer | Persona | Round 1 | Round 2 | Round 3 | Change |
|----------|---------|---------|---------|---------|--------|
| R1 | James Evans (UChicago, Collective Intelligence) | Major Revision | Minor Revision | **Minor Revision** | Maintained |
| R2 | Blaise Agüera y Arcas (DeepMind, Systems) | Major Revision | Minor Revision | **Minor → Accept** | Upgraded |
| R3 | Benjamin Bratton (UCSD, Philosophy) | Major Revision | Accept (minor) | **Accept (minor)** | Maintained |
| R4 | Yilun Du (MIT, Multi-Agent Debate) | Major Revision | Minor-Moderate | **Minor Revision** | Upgraded |
| R5 | AI Safety Perspective | Minor Revision | Minor Revision | **Accept (minor)** | Upgraded |

**Consensus: Accept with Minor Revisions.** Three reviewers at Accept (minor), two at Minor Revision. No reviewer requests major changes. Du (R4) is the most cautious but upgraded from Minor-Moderate to Minor.

---

## What Reviewers Universally Praise

### 1. Pre-Collaboration Snapshots — "Methodological advance" (Du)
All 5 reviewers call convergence tracking the strongest new contribution. Du specifically says he plans to adopt the methodology. The 3.4x harmful switching ratio is cited by every reviewer as a clean, protocol-independent finding.

### 2. Same-Domain Controls — "Exactly what was needed" (Evans)
The clean separation (same-domain positive, cross-domain negative) is unanimously praised. This resolves the ambiguity from Round 2 about whether the collaboration *mechanism* is broken vs. domain mismatch being the issue.

### 3. Protocol Comparison — "Transforms descriptive finding into mechanistic one" (Evans)
The asymmetric scoping effect (answer-only best for same-domain, worst for cross-domain) surprises all reviewers. Evans calls it "genuinely novel." Bratton extracts an epistemological principle: "information without context is more corruptive unless the receiver already has the context."

### 4. Medicine Full-CoT Anomaly — Universally flagged as most interesting specific finding
Medicine+medicine under full-cot = -10pp (only negative same-domain pair). Under answer-only = +12pp. Every reviewer wants this investigated further. Safety reviewer calls it "alarming for safety-critical domains."

---

## What the Revision Fixed (from Round 2 critiques)

### Agüera y Arcas: "Conclusions-only protocol still missing" → RESOLVED
Answer-only protocol is exactly this. Results confirm his prediction that protocol design matters, but the direction of the effect (answer-only worst for cross-domain) is surprising.

### Safety: "Same-domain control still missing" → RESOLVED
10 same-domain pairs per protocol across all 10 domains. Medicine same-domain behavior varies dramatically by protocol.

### Evans: "Loosely coupled protocols should be less harmful" → PARTIALLY VALIDATED
True for same-domain (+8.6pp answer-only vs +3.6pp full-cot). False for cross-domain (-13.8pp answer-only vs -8.7pp full-cot). Evans calls the asymmetry "theoretically rich."

### Du: "20 questions per pair is underpowered" → RESOLVED
50 questions per pair × 100 pairs × 3 protocols = 15,000 observations per protocol.

---

## Remaining Critiques (Addressable)

### Methodological (Du, Evans)
1. **Solo baseline variation across protocols** — Physics solo ranges 0.48-0.54 across runs. Need to clarify if same questions used and report paired statistics
2. **Round ablation** — Does degradation worsen round-by-round? Computable from existing data
3. **Ordering effects** — Agent A always primary. Need to test B-then-A
4. **Majority vote baseline** — Independence-preserving comparison, computable from solo data

### Analytical (all reviewers)
5. **Asymmetric scoping theory** — Why does answer-only help same-domain but hurt cross-domain? Evans proposes testable hypothesis: reasoning chains let agents discount off-domain input
6. **c2w/w2c ratio by domain** — Is 3.4x uniform across domains or domain-dependent?
7. **Confidence calibration** (Safety, Du) — Does structured protocol's confidence field predict accuracy?
8. **Cascading failure / ratchet effect** (Safety) — Do wrong switches persist across rounds?

### Scope (Agüera y Arcas, Bratton)
9. **Base-model-as-helper control** — Still missing but accepted as future work
10. **KL divergence for all 10 domains** — Only 3 of 10 computed
11. **Theoretical framing** — Bratton wants stronger epistemological framework; Agüera y Arcas wants ecological framing
12. **Generalization to larger models** — Not blocking

---

## Priority Actions for Next Revision

### Must-do (addressable from existing data)
- [ ] Clarify question sampling: same questions across protocols? If not, discuss confound
- [ ] Compute round-by-round accuracy to test progressive degradation hypothesis
- [ ] Decompose c2w/w2c ratio by domain and by same-vs-cross
- [ ] Analyze structured protocol confidence vs accuracy correlation
- [ ] Compute majority-vote baseline from solo accuracy data
- [ ] Explain medicine full-cot anomaly with dedicated analysis

### Should-do (strengthens but not blocking)
- [ ] Complete KL divergence for all 10 domains (need GPU)
- [ ] Base-model-as-helper control (need GPU)
- [ ] Add ecological/epistemological framing to discussion
- [ ] Add missing references (Janis 1972, Golub & Jackson 2010, Sharma et al. 2023)

### Nice-to-have (future work)
- [ ] Ordering effects ablation (B-then-A)
- [ ] More protocols (debate, voting, hierarchical)
- [ ] Test with larger models (7B, 70B)
- [ ] Multiple random seeds
