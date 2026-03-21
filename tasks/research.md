# Research — halulujah

## Pivot A: Personality Fingerprinting (CONFIRMED)

### Research Question
Does fine-tuning on individual humans' writing create a measurable spectral fingerprint in the token distribution, and does high-temperature sampling erode that fingerprint?

### Confirmed Methodology (2026-03-21)

**Setup**
- Model: Qwen3-1.7B via HuggingFace (full logit access, not Ollama)
- Data: Blog Authorship Corpus, top 5 authors (1600+ posts each), 80/20 train/probe split
- Training: Per-author LoRA adapter via `sft_lora.py`, ~320 posts/author
- Result: 5 persona adapters + 1 base model = 6 models

**Phase 1: Per-Author LoRA Fine-Tuning**
1. Select 5 authors from Blog Authorship Corpus with highest post counts + diverse topics
2. Format blog posts as chat-template training data (system: "Write in this style", user: topic prompt, assistant: blog post)
3. Fine-tune per-author LoRA adapters on Qwen3-1.7B
4. Output: 5 adapters saved to PACE scratch

**Phase 2: Fingerprint Measurement (3 levels)**

*Level 1 — Output Embeddings (coarse)*
- Generate responses to 20 shared probe questions from all 6 models
- Embed with sentence-transformers, compute pairwise cosine distances
- Viz: t-SNE scatter colored by author

*Level 2 — Token-Level KL Divergence (core fingerprint)*
- Run same probe prompts through all 6 models, capture full logit distributions
- Compute KL(persona_i || base) and KL(persona_i || persona_j) at each token position
- Average over positions 1-50 for stability (position 1 is noisiest)
- CRITICAL: KL computed on same input prompt, NOT on generated text — isolates distributional shift from content
- Viz: Heatmap — rows=author pairs, cols=probe questions, cells=mean KL divergence

*Level 3 — Vocabulary Fingerprint (interpretability)*
- For each author, find top-50 tokens where distribution diverges most from base
- These tokens "define" the persona
- Viz: Per-author bar chart of most-divergent tokens

**Phase 3: Temperature Erosion**
- Sweep temperature {0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0} on each persona model
- At each temperature: generate responses, measure Level 1 + Level 2 metrics
- Key plot: persona separation vs temperature (line per author-pair)
- Prediction: KL between personas collapses at high temperature as distributions flatten
- Viz: Line plot — x=temperature, y=mean inter-persona KL

**Phase 4: Cross-Persona Confusion**
- Feed Person A's probe responses through Person B's adapter
- Measure: does output distribution track B or revert to base?

### Topic Control (confirmed)
All authors receive the SAME 20 probe questions. This controls for topic confound —
any divergence in output distributions is attributable to style, not subject matter.

### Key Metrics
- KL(persona || base) — magnitude of personality imprint
- KL(persona_A || persona_B) — inter-persona separability
- Persona classification accuracy vs temperature (logistic regression on embeddings)
- Separation ratio (inter/intra-author embedding distance)

### Visualizations (5 key figures)
1. t-SNE of persona outputs (post-fine-tuning)
2. KL divergence heatmap (author pairs × probe questions)
3. Per-author vocabulary fingerprint (top divergent tokens)
4. Persona separation vs temperature curve
5. Classification accuracy vs temperature curve

### Infrastructure Reuse
- `sft_lora.py` → persona fine-tuning (adapt for blog data)
- `eval/take_exam.py` → probe generation (add logit capture)
- `config.py` → add PersonaConfig dataclass

### New Code Needed
- `src/halulujah/persona/data_prep.py` — format blog posts for SFT, design probe questions
- `src/halulujah/persona/fingerprint.py` — KL divergence, embedding distances, vocab fingerprint
- `src/halulujah/persona/erosion.py` — temperature sweep + metric tracking
- `src/scripts/run_persona_experiment.py` — orchestration
- `bash/persona_finetune.sh` — SLURM job for fine-tuning (A100)
- `bash/persona_measure.sh` — SLURM job for measurement (A100)

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
