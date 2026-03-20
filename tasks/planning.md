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
- Start Pivot A Phase 1: design the personality questionnaire

## Next Steps

- Build `src/halulujah/persona/` module
- Design 50-question personality/opinion questionnaire
- Collect answers from 3-5 humans

## Recently Completed

- [2026-03-20] Drafted full experiment plans for both pivots (tasks/research.md)
- [2026-03-20] Literature survey on both pivot ideas — both confirmed novel, Slack report delivered
- [2026-03-20] Trimmed README to public-facing result-only summary (removed plans, structure, methodology)
