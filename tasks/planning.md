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
- **Reframe paper** around Evans/Bratton/Blaise Science article (10.1126/science.aeg1895):
  - They argue: societies of specialized agents are the scaling path
  - We show: *how* those societies fail when coupling is too tight
  - Connection to vibevelop: our alternating CoT is the anti-pattern; vibevelop's black-box
    interface contracts are the solution (loosely coupled, preserves epistemic independence)
- **Quick wins (CPU-only)**: answer position bias check, round ablation, add references
  (Hong & Page 2004, Clark & Chalmers 1998, Sharma et al. 2023, Evans/Bratton/Blaise 2026)

### RunPod Experiment IN PROGRESS
- **Pod**: RTX A5000 24GB at root@69.30.85.178:22090
- **Phase 1**: Fine-tuning 10 domain adapters (~65 min, ~2.5 it/s per domain)
- **Phase 2**: Cross-domain evaluation
- **Phase 3**: KL divergence (pairwise distance metric)
- **Phase 4**: Protocol comparison × 3 (full-cot, answer-only, structured) × 100 pairs × 50q
- **Estimated total**: ~24-30 hours
- PACE adapters saved to `/media/dan/WD_BLACK/models/halulujah/domain_10_adapters/`

### Base Model Baseline (DONE — 2026-04-11)
- Base Qwen3-1.7B (no LoRA) evaluated on all 10 MMLU domains, 50 questions each
- **Mean accuracy: 15.4%** — below random chance (25% for 4-choice MCQ)
- Specialists average 50.4% — 35pp gap confirms LoRA fine-tuning is essential
- Best: biology/economics 24%. Worst: CS/history/math 8%
- Results: `results/runpod_domain/base_eval/base_solo_eval.json`
- Figure: `results/runpod_domain/figures/base_vs_specialist_solo.png`

### TODO — Remaining
- **Base-model-as-helper collaboration** — script ready (`run_base_collab.py` on RunPod), waiting for GPU to clear from CorticalNN jobs. Will answer: is cross-domain damage from conflicting expertise or any noisy partner?
- **Logit entropy comparison** (philosophy vs medicine adapter output entropy)
- **MMLU-Pro + GSM8K + MedQA pipeline** — planned but not built yet

### Data & Model Decisions
- Domain data: MMLU subsets (3 core + 7 extended domains) via HuggingFace
- Model: Qwen3-1.7B (same as Pivot A)
- Execution: RunPod (primary), local RTX 3060 (backup), PACE (backup)
- RunPod SSH: root@213.173.102.216 -p 19132 -i ~/.ssh/runpod_key

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

### Key Results (Pivot B — 10-Domain Collaboration)
- 90 collaboration pairs tested (10 domains × 9 partners, 20 questions each)
- Collaboration is *net harmful*: mean delta = −9.2%, 50/90 pairs hurt accuracy
- Medicine: catastrophically harmed by ALL helpers (−25% to −45%)
- Chemistry, physics: systematically harmed by all helpers
- Philosophy, law, math: systematically helped by all helpers
- Cross-eval distance proxy: r = 0.197 (weak, need actual KL for stronger signal)
- Best pair: math+biology (+20%), worst: medicine+physics (−45%)

## Next Steps

- Paper rewrite with Evans/Bratton/Blaise framing + vibevelop architecture connection
- Quick wins: position bias check, round ablation, missing references
- GPU experiments blocked on PACE quota — resubmit or use RunPod when ready

## Recently Completed

- [2026-04-11] Base model (no LoRA) eval: 15.4% mean vs 50.4% specialists — 35pp gap (obj-013)
- [2026-04-11] Deployed halulujah to RunPod, uploaded all 10 adapters, base eval scripts committed
- [2026-04-11] Generated base_vs_specialist_solo.png comparison figure
- [2026-04-07] Round 3 reviews: 3 Accept (minor), 2 Minor Revision. Consensus: accept with minor
- [2026-04-07] 3-protocol experiment complete: full-cot, answer-only, structured (100 pairs × 50q each)
- [2026-04-07] 3.4x harmful switching ratio discovered — invariant across all protocols
- [2026-04-02] Round 2 reviews: all 5 reviewers upgrade (4 Major→Minor, 1 Minor→Minor). Consensus: Minor Revision
- [2026-04-02] Paper fully revised: title scoped, anthropomorphic language replaced, ANOVA + CIs + chain analysis integrated
- [2026-04-02] Variance decomposition: primary domain η²=13.1% (p<0.001), helper 0.5% (n.s.) — 26x ratio
- [2026-04-02] Chain analysis: confident wrong (49.6%), extraction failure (29.2%), answer switch (21.2%)
- [2026-04-02] Bootstrap CIs: mean width ±17.9pp (confirms n=20 limitation)
- [2026-04-02] Paper drafted + 5 simulated expert reviews + synthesis. All say Major Revision
- [2026-04-02] Confound analysis: training set size explains ~25% of collaboration variance, not the full story
- [2026-04-02] PACE 10-domain experiment complete: 90 collab pairs, collaboration net harmful (−9.2% mean)
- [2026-04-02] Generated collab heatmap, solo-vs-delta, distance-vs-delta figures
- [2026-03-31] Built Phase 4: alternating CoT collaboration between domain specialists (1cda7b0)
- [2026-03-31] RunPod LoRA experiment complete: KL vs hallucination r=0.715 (obj-009)
- [2026-03-31] Pulled RunPod results via SCP, added 10-domain extended support
- [2026-03-31] Fixed 3 RunPod compatibility bugs (SFTConfig, DOMAINS ref, BatchEncoding)
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
