# Planning — halulujah

## Current Priorities

Both pivots approved. Full experiment plans in `tasks/research.md`.

1. **Pivot A: Personality Fingerprinting** — Start here (tighter loop, faster to first result)
   - Phase 1: Design questionnaire, collect human answers, fine-tune persona LoRA adapters
   - Phase 2: Measure distributional fingerprint (KL divergence, stylometrics, embedding clusters)
   - Phase 3: Temperature erosion sweep — does persona signal collapse under hallucination?
   - Phase 4: Cross-persona boundary analysis

2. **Pivot B: Cross-Domain Hallucination** — NOW ACTIVE
   - Phase 1: Fine-tune domain specialists (physics, law, biology) on MMLU subsets
   - Phase 2: Cross-domain evaluation — which domain pairs cause more hallucination?
   - Phase 3: KL divergence distance — does domain distance predict hallucination rate?

### Immediate Next Action
- Monitor PACE job 5511062 (Pivot B Phase 2+3: cross-domain evaluation + KL distance)
- When done, fetch results and generate visualizations

### Data & Model Decisions
- Domain data: MMLU subsets (physics, law, biology) via HuggingFace
- Model: Qwen3-1.7B (same as Pivot A)
- All data + models on PACE scratch (`~/scratch/halulujah/`)

### Key Results (Pivot A)
- KL(persona || base) ≈ 29 nats at T=1.0 — strong persona imprint
- Pairwise KL 7-8 nats — personas distinguishable from each other
- Temperature erosion: KL drops ~3000x from T=0.1 to T=2.0
- KL divergence validated as domain distance metric

### Local Experiment Results (prompt-only, no LoRA)
- Base model outperforms all expert-prompted variants on MCQ accuracy
- Expert prompting causes model to overthink (longer CoT, worse extraction)
- Surprising: law expert → 53% on biology (vs 27% base)
- Implication: prompt-only specialization is NOT equivalent to LoRA fine-tuning

### Prompt vs LoRA Comparison
- Prompt-only personas: sep ratio 1.195, 92% classification (surface style)
- LoRA fine-tuned personas: sep ratio 1.051, but KL ~29 nats (deep distribution shift)
- Prompting = surface style, LoRA = deep distributional shift
- Expert domain prompting hurts MCQ accuracy (model overthinks)

## Next Steps

- Still waiting for PACE job 5511062 (LoRA-based domain Phase 2+3)
- Investigate why expert prompting hurts MCQ accuracy (CoT length? extraction failure?)
- Consider: is the domain prompting failure an answer-extraction artifact or genuine performance drop?

## Recently Completed

- [2026-03-29] Prompt vs LoRA comparison figure: prompting more separable at surface, LoRA deeper (8806a54)
- [2026-03-29] Local persona probe: 92% classification with prompt-only personas (c5ab78e)
- [2026-03-29] Local cross-domain experiment: prompt-only specialization hurts MCQ accuracy (7b71bd8)
- [2026-03-26] Pivot B Phase 1 DONE: 3 domain adapters trained (physics/law/biology, job 5433676, 25min)
- [2026-03-26] Pivot B Phase 2+3 submitted (job 5511062: cross-domain eval + KL distance)
- [2026-03-24] AFK session: vocab fingerprint viz, Pivot A summary figure, Pivot B viz script, smoke tests
- [2026-03-23] Built and submitted Pivot B pipeline: domain data_prep, cross_eval, orchestration, SLURM jobs
- [2026-03-22] Phase 2+3 DONE: fingerprint measured, erosion confirmed (job 5358810, 1h53m, A100)
- [2026-03-22] Phase 1 DONE: 5 LoRA adapters trained on Qwen3-1.7B (job 5357133, 24min, A100)
- [2026-03-21] Built and submitted persona pipeline: data_prep, fingerprint, erosion, orchestration, SLURM jobs
- [2026-03-21] Feasibility test PASSED: 53.1% accuracy vs 10% chance (5.3x), authors clearly separable
- [2026-03-21] Downloaded Blog Authorship Corpus to PACE scratch, ran feasibility SLURM job
- [2026-03-20] Built blog author feasibility test script + SLURM job (no fine-tuning, embedding-level check)
- [2026-03-20] Pivoted data source: PANDORA (gated) → Blog Authorship Corpus (open access, Kaggle)
- [2026-03-20] Drafted full experiment plans for both pivots (tasks/research.md)
- [2026-03-20] Literature survey on both pivot ideas — both confirmed novel, Slack report delivered
- [2026-03-20] Trimmed README to public-facing result-only summary (removed plans, structure, methodology)
