# Research — halulujah

## Pivot A: Personality Fingerprinting

### Research Question
Does fine-tuning on individual humans' Q&A responses create a measurable spectral fingerprint in the token distribution, and does high-temperature sampling erode that fingerprint?

### Experiment Plan

**Phase 1: Data Collection & Persona Training**
1. Design a 50-question personality/opinion questionnaire spanning values, preferences, reasoning style (mix of open-ended and constrained)
2. Collect genuine answers from 3-5 distinct humans (diverse backgrounds)
3. Fine-tune separate LoRA adapters on each human's responses using existing `src/halulujah/finetune/sft_lora.py`
4. Result: N persona adapters + 1 base model

**Phase 2: Distribution Measurement**
1. Run each persona model on a held-out probe set (20 questions not in training)
2. Capture full token log-probabilities at each generation step (extend `eval/take_exam.py` to log logits)
3. Compute per-persona metrics:
   - KL divergence from base model (overall distributional shift)
   - Top-k token overlap between personas (vocabulary fingerprint)
   - Stylometric features: sentence length distribution, type-token ratio, POS n-gram frequencies
   - Embedding-space centroid distance between persona outputs (using sentence-transformers, already a dependency)
4. Visualization: PCA/t-SNE of persona output embeddings — do they cluster?

**Phase 3: Hallucination Erosion**
1. Sweep temperature {0.1, 0.5, 1.0, 1.5, 2.0} × top_p {0.5, 0.9} on each persona model (reuse sweep infra)
2. At each setting, measure all Phase 2 metrics
3. Key question: does inter-persona distance collapse as temperature rises? (i.e., do all personas converge to the same high-entropy soup?)
4. Visualization: persona separation (embedding distance or KL) vs temperature curve

**Phase 4: Boundary Analysis**
1. Test cross-persona confusion: feed Person A's questions to Person B's adapter
2. Measure whether the model produces responses closer to B's distribution or reverts toward base
3. Identify which personality dimensions are robust vs fragile under sampling pressure

### Key Metrics
- KL(persona || base) — magnitude of personality imprint
- KL(persona_A || persona_B) — inter-persona separability
- Persona classification accuracy (train a simple classifier on outputs, measure accuracy vs temperature)
- Stylometric consistency score as f(temperature)

### Infrastructure Reuse
- `sft_lora.py` → persona fine-tuning (minimal changes)
- `eval/take_exam.py` → sweep infrastructure (add logit logging)
- `rl/verifier.py` → extend for stylometric scoring
- `config.py` → add PersonaConfig dataclass
- `pipeline/temporal_loo.py` → adapt to persona-leave-one-out

### New Code Needed
- `src/halulujah/persona/questionnaire.py` — question set + answer collection format
- `src/halulujah/persona/fingerprint.py` — KL divergence, stylometric features, embedding distances
- `src/halulujah/persona/erosion.py` — temperature sweep + metric tracking
- `src/scripts/run_persona_experiment.py` — orchestration

### Key Papers
- "The Geometry of Persona" (arXiv 2512.07092) — personality in linear subspaces
- "Risk of Efficient Personalized Text Generation" (arXiv 2502.06560) — LoRA persona fine-tuning
- PERSIST framework (arXiv 2508.04826) — personality measurement instability

---

## Pivot B: Domain-Bounded Ignorance

### Research Question
Can we train an LLM to express calibrated confusion past a domain complexity threshold, and how much shared vocabulary is required for two domain-specialized agents to collaborate effectively?

### Experiment Plan

**Phase 1: Domain Complexity Metric**
1. Define domain complexity scorer using:
   - Jargon density: ratio of domain-specific terms to total tokens (use existing word frequency lists per domain)
   - Concept density: named entity + technical term count per sentence
   - Lexical rarity: inverse document frequency of terms against a general corpus
2. Validate scorer: physics text should score high on physics complexity, low on biology, etc.
3. Calibrate threshold: human-annotated "I wouldn't understand this" boundary for each domain pair

**Phase 2: Confusion Training**
1. Select 3 distinct domains with clear jargon boundaries:
   - Physics (quantum mechanics, thermodynamics)
   - Law (contracts, constitutional)
   - Biology (molecular, ecology)
2. For each domain, create training data:
   - Below-threshold: normal Q&A pairs (model answers competently)
   - Above-threshold: Q&A pairs where target response is calibrated confusion ("I'm not sure what [jargon term] means in this context — could you explain?")
3. Fine-tune 3 domain-specialist adapters using existing LoRA pipeline
4. Each specialist is competent in its domain, confused outside it

**Phase 3: Ignorance Verification**
1. Test each specialist on cross-domain probes
2. Measure: does confusion onset correlate with the complexity score?
3. Key metric: confusion calibration curve — P(confused response) vs domain complexity score
4. Compare against baselines:
   - Base model (answers everything confidently, often wrong)
   - RLHF-aligned model (refuses rather than expressing confusion)
   - Machine unlearning (TOFU-style erasure)

**Phase 4: Collaboration Experiment**
1. Pair specialists in multi-agent dialogue (reuse MPI infra concept from old stubs)
2. Present cross-domain problems requiring both domains (e.g., biophysics, patent law for biotech)
3. Measure collaboration effectiveness:
   - Task accuracy on joint problems
   - Communication efficiency (turns to reach answer)
   - Information transfer quality (does the explainer successfully bridge the jargon gap?)
4. Vary the "shared vocabulary" level: test with 0%, 25%, 50%, 75% domain overlap in training
5. Key finding: minimum shared knowledge for effective collaboration

### Key Metrics
- Confusion calibration: P(confused | complexity > threshold) — should be high
- Domain accuracy: performance within-domain should remain high
- Collaboration accuracy: joint task performance as f(shared vocabulary %)
- Communication cost: turns needed as f(domain distance)

### Infrastructure Reuse
- `sft_lora.py` → domain specialist fine-tuning
- `eval/take_exam.py` → domain probe evaluation
- `rl/verifier.py` → extend for confusion detection
- `oracle/` → domain-specific retrieval for training data generation
- `config.py` → add DomainConfig dataclass

### New Code Needed
- `src/halulujah/domain/complexity.py` — jargon density, concept density, lexical rarity scorer
- `src/halulujah/domain/confusion_data.py` — generate confusion training pairs from domain text
- `src/halulujah/domain/specialist.py` — domain-bounded model wrapper
- `src/halulujah/domain/collaboration.py` — multi-agent dialogue loop with turn tracking
- `src/scripts/run_domain_experiment.py` — orchestration

### Key Papers
- TOFU benchmark (arXiv 2401.06121) — measuring calibrated ignorance
- "Know Your Limits" (TACL 2025) — abstention survey
- "Who's Harry Potter?" (arXiv 2310.02238) — domain knowledge removal
- Lexical complexity estimation (arXiv 2404.01196) — jargon scoring

---

## Recommended Execution Order

*Start with Pivot A (Personality Fingerprinting)* — tighter experimental loop, fewer new components, faster to first result. The questionnaire + fine-tuning + temperature sweep can produce a publishable figure in 1-2 weeks.

*Then Pivot B (Domain-Bounded Ignorance)* — more novel but requires more data engineering (domain corpora, confusion annotation, multi-agent infra). Build on lessons from Pivot A's distribution measurement work.
