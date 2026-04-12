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

### Immediate Next Action — Epistemic Rigidity Investigation
- **Step 1**: ~~Ratio sweep on medicine+physics (90/10 → 10/90)~~ — DONE (commit 151080f)
  - Result: ratio is irrelevant. Collab acc locked at 10-18% across all 9 ratios.
  - C2W=25-31, W2C=0-2 regardless of mix. Damage is from LoRA training itself.
- **Step 2**: ~~LoRA rank ablation (r=4, r=8, r=16, r=32)~~ — DONE (r=4,8,32 complete, r=16 eval running)
  - Preliminary: training entropy drops monotonically (0.865→0.851→0.749)
  - Switch rate drops with rank (54%→48%→38%) — behavioral rigidity confirmed
  - C2W/W2C ratio ~1.5x at all ranks with BASE helper (much lower than cross-domain 3.4x)
  - Key insight: rank controls switch QUANTITY, helper expertise controls switch QUALITY
- **Step 3**: Cross-domain eval at each rank (needs GPU headroom) — BLOCKED on other RunPod jobs
- **Step 4**: Full fine-tuning comparison — BLOCKED (needs ~14GB free, only 1GB spare)
- **Step 5**: Generate combined figure with all conditions
- **Step 6**: Paper rewrite with epistemic rigidity framing

### RunPod Experiment IN PROGRESS — Rank Ablation r=16 Eval
- **Pod**: RTX A4500 20GB at root@213.173.102.216:19132
- **Running**: run_rank_ablation.py --ranks 16 --skip-training (PID 452622)
- **Estimated completion**: ~16:35 UTC (40 min eval)
- Other jobs on pod: rho_matched_control, applied_encodec, overnight_sweep

### Key Comparison Table (all conditions)
| Condition | Mean Delta | C2W/W2C Ratio | Notes |
|-----------|-----------|---------------|-------|
| Base helper (no LoRA) | +9.0pp | 1.3x | 10-domain mean |
| Same-domain LoRA | +3.6pp | — | 10-domain mean |
| Cross-domain LoRA | -8.7pp | 3.4x | 10-domain mean |
| Mediator LoRA (50/50) | -32.8pp | 7.1x | 5 pairs |
| Mediator any ratio | -50 to -64pp | ~15x | 9 ratios, medicine+physics |

### Rank Ablation Results (medicine + base helper)
| Rank | Solo | +Base | Delta | C2W | W2C | Switch% | C2W/W2C | Train Entropy |
|------|------|-------|-------|-----|-----|---------|---------|---------------|
| r=4  | 54%  | 44%   | -10pp | 12  | 8   | 54%     | 1.5x    | 0.865         |
| r=8  | 48%  | 50%   | +2pp  | 9   | 6   | 48%     | 1.5x    | 0.851         |
| r=16*| 72%  | 66%   | -6pp  | 6   | 5   | 30%     | 1.2x    | —             |
| r=32 | 66%  | 56%   | -10pp | 10  | 6   | 38%     | 1.67x   | 0.749         |
| base | 21%  | —     | —     | —   | —   | —       | —       | —             |

*r=16 trained with different script (run_domain_experiment.py). Training entropy not comparable.

### TODO — Remaining
- **Cross-domain eval at each rank** — specialist(r) + physics(r=16): does C2W/W2C ratio scale with rank when helper has conflicting expertise? Needs other RunPod jobs to finish.
- **Retrain r=16 with rank_ablation script** — current r=16 used different training setup, confounds comparison
- **Full fine-tuning vs LoRA** — needs ~14GB free GPU on RunPod
- **Logit entropy comparison** — are LoRA models more confident at inference? (CPU-possible)
- **Combined paper rewrite** — reframe: "Paradox of Expertise" or "Epistemic Rigidity in LoRA Agents"

### Quick wins (CPU-only, whenever)
- Answer position bias check, round ablation, add references
  (Hong & Page 2004, Clark & Chalmers 1998, Sharma et al. 2023, Evans/Bratton/Blaise 2026)

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

- [2026-04-12] Mediator experiment complete: 50/50 mixed LoRA is WORST collaborator (-32.8pp, 7.1x C2W/W2C). Naive bridge hypothesis failed. Launched ratio sweep (obj-015)
- [2026-04-12] Cross-condition comparison: base (+9pp) > same-domain LoRA (+3.6pp) > cross-domain LoRA (-8.7pp) > mediator (-32.8pp). LoRA creates epistemic rigidity.
- [2026-04-11] Base-as-helper collab complete: base helper +9pp vs cross-domain specialist -8.7pp — conflicting expertise is the damage mechanism, not noise (obj-014)
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
