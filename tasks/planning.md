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
- Submit blog feasibility SLURM job on PACE: `sbatch bash/blog_feasibility.sh`
- Review results: are blog authors separable at embedding level without fine-tuning?
- If YES → proceed to per-author LoRA fine-tuning
- If NO → reconsider approach

### Data Decision
- Using Blog Authorship Corpus (Kaggle, 681K posts, 19K authors) — no request form needed
- All data + models on PACE scratch (`~/scratch/halulujah/`)
- PANDORA dropped for now (gated behind request form)

## Next Steps

- Run feasibility test on PACE
- Based on results, build `src/halulujah/persona/` module for LoRA fine-tuning per author

## Recently Completed

- [2026-03-20] Drafted full experiment plans for both pivots (tasks/research.md)
- [2026-03-20] Literature survey on both pivot ideas — both confirmed novel, Slack report delivered
- [2026-03-20] Trimmed README to public-facing result-only summary (removed plans, structure, methodology)
