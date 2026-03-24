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

## Pivot B: Cross-Domain Hallucination & Collaboration (REVISED 2026-03-23)

### Research Question
Do certain domain knowledge combinations provoke more hallucination than others,
and does domain distance (measured via KL divergence) predict hallucination rate
and collaboration effectiveness?

### Revised Approach
Instead of training "confusion" — directly measure whether domain-specialist
models hallucinate more when asked cross-domain questions, and whether the
hallucination rate correlates with the KL divergence between their domains.

### Experiment Plan

**Phase 1: Domain Specialist Training**
1. Select 3 domains with clear boundaries: physics, law, biology
2. Source training data from MMLU subsets (HuggingFace `cais/mmlu`, "all" config)
   - Physics: `college_physics`, `high_school_physics`, `astronomy`, `conceptual_physics`
   - Law: `professional_law`, `jurisprudence`, `international_law`
   - Biology: `college_biology`, `high_school_biology`, `anatomy`, `clinical_knowledge`
3. Fine-tune 3 domain-specialist LoRA adapters on Qwen3-1.7B (reuse persona pipeline)
4. Result: 3 specialist adapters + 1 base model = 4 models

**Phase 2: Cross-Domain Hallucination Measurement**
1. Construct a cross-domain test set: 50 questions per domain (150 total)
2. Ask each of the 4 models all 150 questions
3. Grade correctness (use MMLU ground-truth answers via verifier)
4. Compute per domain-pair hallucination matrix:
   - In-domain accuracy (physics model on physics Qs)
   - Cross-domain accuracy (physics model on law Qs, biology Qs, etc.)
5. Key metric: accuracy drop from in-domain to cross-domain

**Phase 3: Domain Distance via KL Divergence**
1. Run same 150 questions through all 4 models, capture logit distributions
   (reuse `persona/fingerprint.py`)
2. Compute pairwise KL between specialist adapters (reuse `compute_pairwise_kl`)
3. Correlate: KL(domain_A || domain_B) vs cross-domain hallucination rate
4. Key question: does domain distance predict hallucination severity?

**Phase 4: Temperature × Domain Interaction**
1. Sweep temperature {0.1, 0.5, 1.0, 2.0} on each specialist across all domains
2. Does temperature affect in-domain and cross-domain accuracy differently?
3. Prediction: cross-domain accuracy should degrade faster with temperature
   (less parametric knowledge to anchor the output)

### Key Metrics
- In-domain accuracy per specialist
- Cross-domain accuracy per specialist × domain pair
- Accuracy drop = in_domain - cross_domain (hallucination severity)
- KL(specialist_A || specialist_B) — domain distance
- Correlation: accuracy_drop vs KL distance
- Temperature sensitivity: accuracy vs temperature per domain pair

### Visualizations (4 key figures)
1. Confusion matrix: specialist × question domain → accuracy (heatmap)
2. Domain distance map: KL divergence between all specialist pairs (triangle heatmap)
3. Scatter plot: KL distance vs accuracy drop (the core finding)
4. Temperature × domain interaction: accuracy curves per domain pair

### Infrastructure Reuse
- `persona/data_prep.py` → adapt for MMLU domain data loading
- `persona/fingerprint.py` → KL divergence computation (direct reuse)
- `persona/erosion.py` → temperature sweep (direct reuse)
- `sft_lora.py` → domain specialist fine-tuning
- `rl/verifier.py` → answer grading

### New Code Needed
- `src/halulujah/domain/data_prep.py` — load MMLU subsets, format for SFT
- `src/halulujah/domain/cross_eval.py` — cross-domain evaluation + hallucination scoring
- `src/scripts/run_domain_experiment.py` — orchestration
- `bash/domain_finetune.sh` — SLURM job for domain specialist training
- `bash/domain_measure.sh` — SLURM job for cross-domain measurement

### Key Papers
- HALoGEN (arXiv 2501.08292) — hallucination rates 4-86% by domain across 14 LLMs
- Gekhman et al. (arXiv 2405.05904) — fine-tuning on new knowledge linearly increases hallucination
- Kang et al. (arXiv 2403.05612) — unfamiliar examples shape hallucination type
- Domain specialization efficiency (arXiv 2501.02068) — specialists need 4.3x less compute
