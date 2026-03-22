# Planning — halulujah

## Current Priorities

Both pivots approved. Full experiment plans in `tasks/research.md`.

1. **Pivot A: Personality Fingerprinting** — Start here (tighter loop, faster to first result)
   - Phase 1: Design questionnaire, collect human answers, fine-tune persona LoRA adapters
   - Phase 2: Measure distributional fingerprint (KL divergence, stylometrics, embedding clusters)
   - Phase 3: Temperature erosion sweep — does persona signal collapse under hallucination?
   - Phase 4: Cross-persona boundary analysis

2. **Pivot B: Domain-Bounded Ignorance** — Second priority
   - Phase 1: Build domain complexity scorer (jargon density, concept density, lexical rarity)
   - Phase 2: Train confusion responses above complexity threshold
   - Phase 3: Verify calibrated ignorance vs baselines
   - Phase 4: Multi-agent collaboration with varying shared vocabulary

### Immediate Next Action
- Monitor PACE job 5358810 (Phase 2+3: fingerprint measurement + erosion sweep)
- When done, fetch results and generate 5 key visualizations

### Data & Model Decisions
- Using Blog Authorship Corpus (Kaggle, 681K posts, 19K authors) on PACE scratch
- Model: Qwen3-1.7B (HuggingFace, full logit access for KL divergence)
- All data + models on PACE scratch (`~/scratch/halulujah/`)
- Repo cloned to `~/scratch/halulujah/repo/` on PACE

## Next Steps

- Wait for Phase 2+3 job to complete (5358810)
- Fetch fingerprint + erosion results from PACE
- Generate 5 key visualizations (t-SNE, KL heatmap, vocab fingerprint, erosion curves, accuracy curve)

## Recently Completed

- [2026-03-22] Phase 1 DONE: 5 LoRA adapters trained on Qwen3-1.7B (job 5357133, 24min, A100)
- [2026-03-22] Phase 2+3 submitted (job 5358810: fingerprint measurement + erosion sweep)
- [2026-03-21] Built and submitted persona pipeline: data_prep, fingerprint, erosion, orchestration, SLURM jobs
- [2026-03-21] Feasibility test PASSED: 53.1% accuracy vs 10% chance (5.3x), authors clearly separable
- [2026-03-21] Downloaded Blog Authorship Corpus to PACE scratch, ran feasibility SLURM job
- [2026-03-20] Built blog author feasibility test script + SLURM job (no fine-tuning, embedding-level check)
- [2026-03-20] Pivoted data source: PANDORA (gated) → Blog Authorship Corpus (open access, Kaggle)
- [2026-03-20] Drafted full experiment plans for both pivots (tasks/research.md)
- [2026-03-20] Literature survey on both pivot ideas — both confirmed novel, Slack report delivered
- [2026-03-20] Trimmed README to public-facing result-only summary (removed plans, structure, methodology)
