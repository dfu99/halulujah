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

### Immediate Next Action — Full Collaboration Matrix (thinking enabled)

Prior work (steps 1-6) was prerequisite cleanup: fixed training format so specialists
actually reason. That work is done and filed. The real experiment starts now.

- ~~**Step 1**: Train RP specialists for 5 domains~~ — DONE
- ~~**Step 2**: Full collaboration matrix~~ — DONE (45 conditions, 453 min)
- ~~**Step 3**: Composite question experiment~~ — DONE (25 conditions, 67 min)
  - Collab helps 3/5 pairs when structurally necessary (phys+math: +40pp)
  - Pattern: both specialists must be weak for collab to help
  - Base pairs still dominate (36% vs specialist 32%)
- ~~**Step 4**: RP mediators with thinking~~ — DONE (9 conditions, 36 min)
  - Old mediator: -32.8pp, 7.1x ratio → RP mediator: +4.0pp, 3.6x ratio
  - Reasoning preservation eliminates catastrophe but switching still biased
  - Best: phys+math (+10pp, 1.1x — genuinely balanced)
- **Step 5**: Paper framing — all experimental data now collected
  - Key decision needed from PI: what framing given these results?

### RunPod Pod Info
- **Pod**: RTX A4500 20GB at root@213.173.102.216:19132
- Other projects on pod: overnight_sweep, sigma_survey_extension

### Key Comparison Table (all conditions, medicine+physics)
| Condition | Solo | +Cross Delta | C2W/W2C | Notes |
|-----------|------|-------------|---------|-------|
| Old specialist (no reasoning) | 66% / 64% | -64pp / -40pp | 34x / 22x | Catastrophic |
| Reasoning-preserved | 42% / 20% | -2pp / +12pp | 1.8x / 1.6x | Collaboration safe |
| Naive CoT (stuffed reasoning) | 14% | +14pp | 0.6x | No domain knowledge |
| Base helper (no LoRA) | — | +9.0pp | 1.3x | 10-domain mean |
| Same-domain LoRA | — | +3.6pp | — | 10-domain mean |
| Cross-domain LoRA | — | -8.7pp | 3.4x | 10-domain mean |
| Mediator any ratio | — | -50 to -64pp | ~15x | 9 ratios |

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
- **Paper framing** — all data in. Need PI direction on framing.
- **75/25 reasoning+domain data mix** — may improve solo accuracy if needed
- **Full fine-tuning vs LoRA** — needs ~14GB free GPU on RunPod
- **Logit entropy comparison** — are LoRA models more confident at inference? (CPU-possible)
- **Paper framing** — decided after all collaboration data collected

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

- [2026-04-19] Paper abstract + intro drafted (paper/abstract_and_intro.md). Refined claim framed strategically: our -10..+28pp range *subsumes* Du et al.'s tight +5..+15pp because we introduced training method (LoRA vs full FT) as a new variable — their tightness is a feature of homogeneous setup, not of the phenomenon. 4B medicine headline: Full FT +5.0pp/1.4× C2W/W2C vs LoRA r=128 +1.5pp/19×, matched solo accuracy 84%. Mechanism grounded in intruder dimensions + bilinear rank constraint + Bayesian-LoRA calibration.
- [2026-04-19] Claim-to-evidence map (paper/claim_evidence_map.md) — 8 claims (C1..C8), each linked to experimental condition, results JSON, and figure. Gap analysis identifies 4 unsupported claims (direct weight-space CKA OOM-deferred, 7B scale not run, r=0.197 distance weak, r>128 not explored). Scaffold for paper write.
- [2026-04-19] RunPod memory-gate protocol added to tasks/lessons.md and AFK seed: `mc runpod check/fits/await/sync/fetch` required before every GPU launch. No scheduler on shared pod — we coordinate ourselves.
- [2026-04-19] Literature sweep April 2026: rank-constraint hypothesis STRENGTHENED by CeRA (linear ceiling), PERA (bilinear LoRA), Shuttleworth "intruder dimensions" (2410.21228 — direct structural LoRA vs full FT difference), Bayesian-LoRA (post-FT overconfidence). No paper renders our result obsolete. Summary in tasks/lit_update_2026_apr.md.
- [2026-04-19] Reviewer C figure: our range -9 to +28pp subsumes Du et al.'s tight +10pp. 6 literature studies as horizontal range bars with our 8 conditions overlaid. Explains heterogeneity (specialist+base vs homogeneous GPT-4) + novel LoRA specialist testing + training method as delta controller. figures/reviewer_c_literature_context.png
- [2026-04-19] Reviewer D figure: switch classification as calibration proxy. Base pair W2C≥C2W (well-calibrated). LoRA r=128 C2W>>W2C at 19× ratio. Full FT at matched solo accuracy keeps 1.4×. figures/reviewer_d_entropy_by_turn.png
- [2026-04-19] Reviewer B figure: rank sweep (r=4..128) does not recover full FT collaborativeness at either 1.7B or 4B scale. Same solo accuracy, 13× better C2W/W2C ratio for full FT. figures/reviewer_b_rank_vs_ft.png
- [2026-04-19] AFK plan written: 8 queue tasks to move toward ACL 2026 submission (intuition update, reviewer-specific ablations, lit sweep, paper draft scaffolding). Intuition.md refined — now focused on rank-constraint mechanism (LoRA vs full FT at matched solo accuracy).
- [2026-04-19] N=200 paper sweep COMPLETE: 61 conditions (49 1.7B + 12 4B). 5 key findings at publication confidence. Deliberation +21pp above compute-matched. LoRA kills collab at both scales (13× worse ratio vs full FT). Pod cleared, all data backed up to WD_BLACK. (obj-022)
- [2026-04-14] RP mediator experiment complete: old -32.8pp → RP +4.0pp, old 7.1x → RP 3.6x ratio. Phys+math best at +10pp/1.1x. (obj-021)
- [2026-04-14] Composite question experiment complete: collab helps 3/5 pairs when structurally necessary (phys+math +40pp). Both specialists must be weak for collab to help. (obj-020)
- [2026-04-13] Full collaboration matrix complete: 45 conditions, base pair +29pp, specialist cross-domain +0.5pp. LoRA constrains deliberation benefit. (obj-019)
- [2026-04-12] Mediator experiment complete: 50/50 mixed LoRA is WORST collaborator (-32.8pp, 7.1x C2W/W2C). Naive bridge hypothesis failed. Launched ratio sweep (obj-015)
- [2026-04-12] Cross-condition comparison: base (+9pp) > same-domain LoRA (+3.6pp) > cross-domain LoRA (-8.7pp) > mediator (-32.8pp). LoRA creates epistemic rigidity.
- [2026-04-11] Base-as-helper collab complete: base helper +9pp vs cross-domain specialist -8.7pp — conflicting expertise is the damage mechanism, not noise (obj-014)
- [2026-04-11] Base model (no LoRA) eval: 15.4% mean vs 50.4% specialists — 35pp gap (obj-013)
- [2026-04-11] Deployed halulujah to RunPod, uploaded all 10 adapters, base eval scripts committed
- [2026-04-11] Generated base_vs_specialist_solo.png comparison figure
- [2026-04-07] Round 3 reviews: 3 Accept (minor), 2 Minor Revision. Consensus: accept with minor
- [2026-04-07] 3-protocol experiment complete: full-cot, answer-only, structured (100 pairs × 50q each)
- [2026-04-07] 3.4x harmful switching ratio invariant across 3 protocols (full-cot 3.4x, answer-only 3.5x, structured 3.4x). Damage is protocol-independent. Documented as obj-010b with viz at results/runpod_domain/protocol_invariance.png
- [2026-04-02] Round 2 reviews: all 5 reviewers upgrade (4 Major→Minor, 1 Minor→Minor). Consensus: Minor Revision
- [2026-04-02] Paper fully revised: title scoped, anthropomorphic language replaced, ANOVA + CIs + chain analysis integrated
- [2026-04-02] Variance decomposition: primary domain η²=13.1% (p<0.001), helper 0.5% (n.s.) — 26x ratio
- [2026-04-02] Chain analysis: confident wrong (49.6%), extraction failure (29.2%), answer switch (21.2%)
- [2026-04-02] Bootstrap CIs: mean half-width ±18.4pp at n=20, effects <36pp indistinguishable from noise — motivated N=200 scale-up. Documented as obj-008b with viz at results/pace_domain_10/bootstrap_cis.png
- [2026-04-02] Paper drafted + 5 simulated expert reviews + synthesis. All say Major Revision
- [2026-04-02] Confound analysis: training set size explains ~25% of collaboration variance, not the full story
- [2026-04-02] PACE 10-domain experiment complete: 90 collab pairs, collaboration net harmful (−9.2% mean)
- [2026-04-02] Three diagnostic figures for PACE 10-domain study: collab heatmap (domain-dependent harm), solo-vs-delta (no inverse relationship), distance-vs-delta (r=0.197 weak). Documented as obj-007b
- [2026-03-31] Built Phase 4: alternating CoT collaboration between domain specialists (1cda7b0)
- [2026-03-31] RunPod LoRA experiment complete: KL vs hallucination r=0.715 (obj-009)
- [2026-03-31] Pulled RunPod results via SCP, added 10-domain extended support
- [2026-03-31] Fixed 3 RunPod compatibility bugs (SFTConfig `overwrite_output_dir` removed in newer TRL, DOMAINS refactor left orphaned refs, BatchEncoding now returned from apply_chat_template). 6.5 hrs total. Documented as obj-009b with timeline viz.
- [2026-03-29] Prompt vs LoRA comparison figure: prompting more separable at surface, LoRA deeper (8806a54)
- [2026-03-29] Local persona probe: 92% classification with prompt-only personas (c5ab78e)
- [2026-03-29] Local cross-domain experiment: prompt-only specialization hurts MCQ accuracy (7b71bd8)
- [2026-03-26] Pivot B Phase 1 DONE: 3 domain adapters trained (physics/law/biology, job 5433676, 25min)
- [2026-03-26] Pivot B Phase 2+3 submitted (job 5511062: cross-domain eval + KL distance)
- [2026-03-24] AFK session: vocab fingerprint viz, Pivot A summary figure, Pivot B viz script, smoke tests
- [2026-03-23] Built and submitted Pivot B pipeline: domain data_prep, cross_eval, orchestration, SLURM jobs
- [2026-03-22] Phase 2+3 DONE: fingerprint measured, erosion confirmed (job 5358810, 1h53m, A100)
- [2026-03-22] Phase 1 DONE: 5 persona LoRA adapters on Qwen3-1.7B (job 5357133, 24min, A100). KL vs base ≈ 29.4 nats, pairwise KL ≈ 7.8 nats. Validated the measurement pipeline. Documented as obj-004b.
- [2026-03-21] Built and submitted persona pipeline: data_prep, fingerprint, erosion, orchestration, SLURM jobs
- [2026-03-21] Feasibility test PASSED: 53.1% accuracy vs 10% chance (5.3x), authors clearly separable
- [2026-03-21] Downloaded Blog Authorship Corpus to PACE scratch, ran feasibility SLURM job
- [2026-03-20] Built blog author feasibility test script + SLURM job (no fine-tuning, embedding-level check)
- [2026-03-20] Pivoted data source: PANDORA (gated) → Blog Authorship Corpus (open access, Kaggle)
- [2026-03-20] Drafted full experiment plans for both pivots (tasks/research.md)
- [2026-03-20] Literature survey on both pivots — both confirmed novel. Pivot A gap: per-human LoRA + KL fingerprint; Pivot B gap: LoRA specialist alternating CoT. Triggered data source pivot PANDORA→Blog Corpus. Documented as obj-001b with novelty map viz.
- [2026-03-20] Trimmed README to public-facing result-only summary (removed plans, structure, methodology)
