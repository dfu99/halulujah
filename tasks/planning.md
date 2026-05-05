# Planning — halulujah

## Current State (2026-05-05)

**Active research direction**: WHO-asymmetry in multi-agent LLM collaboration with
*verified* domain specialists. Earlier MMLU-cluster results were retracted after
the 2026-04-27 audit (empty-think-tag chains) and the 2026-04-28 OOD test (medicine
specialist underperformed base on MedQA, indicating MMLU-format pattern matching).

**2026-05-05 audit (`tasks/audit-2026-05-05.md`, figure `figures/audit-2026-05-05.png`)**
re-derived the verified-LoRA pair-grid numbers. Three findings drive the
paper-rewrite gates:

1. **WHO-asymmetry ratio = 4.47× (95% CI 2.20–7.45 question-clustered).**
   Canonical aggregator: delta cells, full 5×5 + base helper, mean-row
   vs mean-col (audit §6b). Roster-sensitivity envelope **3.05×–6.27×**
   (audit §6c). Question-clustered bootstrap on existing data using
   the deterministic seed=42 question alignment (audit §6f revised,
   §12 resolved): 95% CI **[2.20, 7.45]**, P(ratio > 1) = 100%,
   P(ratio > 2) = 99.1%. The earlier prediction "clustering widens
   the CI" was wrong; clustering tightens it.

2. **§6a "rate ratio 0.70 net-helpful" is mostly an X-parsing artifact
   (audit §6g — critical correction).** 51.6% of pre_a records are
   parsing failures (X), and 67.5% of all W2C events are X→letter
   parsing recoveries, not genuine peer-induced updates. Letter-only
   pooled rate ratio is **0.92** (essentially balanced); 3 of 5
   primaries flip > 1.0 once X-parsing is controlled (math 1.54,
   law 2.04, physics 1.24). The "4 of 5 primaries net-helpful" claim
   does not survive. The honest paper sentence is now: *amplitude of
   switching is rank-amplified; direction of switching is roughly
   balanced once parsing artifacts are excluded*. Follow-up #10
   (re-run with `pre_a_full` capture) still pending for exact
   numbers.

3. **Stickiness vs recovery is highly anti-correlated (audit §6i).**
   Held-rate vs W2C|W: Spearman ρ = -0.92 across 30 cells. Held-rate
   vs C2W|C: ρ = +0.08. LoRA specialists that "stick to their guns"
   abandon recovery without preserving correctness — rank-amplification
   is asymmetric in *direction*, not *magnitude*.

**PI directive 2026-04-29**: do not run more collaboration experiments until each
specialist passes a verification gate (>= base + 5 pp on >=1 OOD benchmark).

## Current Priorities (in order)

### 1. Verified specialists pipeline (active)

Per-domain status:

| Domain | Strategy | Status | Pass? |
|---|---|---|---|
| math | off-the-shelf Qwen2.5-Math-1.5B-Instruct | DONE | yes (2/5 at +5pp) |
| math-Qwen3 | in-house Qwen3-1.7B + LoRA r=16 on GSM8K-train | DONE | yes (3/5 at +5pp on MCQ; GSM8K -5.5 pp) |
| CS | off-the-shelf Qwen2.5-Coder-1.5B-Instruct | DONE | NO (0/4, mean -4.5 pp) |
| medicine | in-house HP sweep on Qwen3-1.7B + MedQA-USMLE-train | DONE | yes (r=64 best, 3/7 at +5pp; college_med +9, MedQA-test +6.5) |
| biology | in-house HP sweep on Qwen3-1.7B + PubMedQA-train (10K) | DONE | weakly (r=16 best, 1/3 at +5pp; hs_bio +7) |
| chemistry | TBD (SciBench, ChemBench, MMLU-Pro chem) | queued | — |
| physics | in-house Qwen3-1.7B + LoRA r=16 on SciQ-train | DONE | yes (1/5: college_physics +6, SciQ-test +4) |
| law | in-house Qwen3-1.7B + LoRA r=16 on CaseHOLD | DONE | yes (1/4: CaseHOLD-test +24, but MMLU law -25/-13/-7; severe overfit) |
| philosophy | TBD (SEP, MoralChoice — careful overlap) | queued | — |
| history | TBD (Wikipedia history, HistorySocialScienceQA) | queued | — |
| economics | TBD (FiQA, econ textbooks) | queued | — |

### 2. Medicine HP sweep (DONE 2026-04-30)

All 4 ranks pass the verification gate. Best rank = r=64 (3/7 benchmarks at +5pp).

| benchmark | base | r=64 spec | delta |
|---|---|---|---|
| anatomy | 56.0 | 53.0 | -3.0 |
| clinical_knowledge | 66.0 | 60.0 | -6.0 |
| college_medicine | 62.0 | 71.0 | **+9.0** |
| medical_genetics | 72.0 | 71.0 | -1.0 |
| professional_medicine | 59.0 | 64.0 | **+5.0** |
| virology | 51.0 | 47.0 | -4.0 |
| MedQA-test | 45.0 | 51.5 | **+6.5** |

Pattern: training on real MedQA-USMLE-train transfers to MedQA-test (held out)
and clinical-format MMLU subjects (college_med, professional_med). Trades
breadth for clinical depth. Adapter at
`/workspace/adapters_1p7b_ood/medicine_sweep/r64/adapter_medicine_medqa`.

### 3. Biology HP sweep (running, launched 2026-04-30 14:31 UTC, ~10 h)

- Pod: same A4500 (medicine sweep finished, freed GPU)
- Script: `src/scripts/run_biology_hp_sweep.py`
- Training: Qwen3-1.7B + LoRA on PubMedQA pqa_artificial (10K subsample),
  3 epochs, lr=5e-5, ranks {8, 16, 32, 64} sequential
- Verifier: `src/scripts/verify_biology_adapter.py` — MMLU college_biology
  + high_school_biology + PubMedQA pqa_labeled held-out
- Output: `results/specialist_verification/biology_sweep/biology_r{rank}.json`

### 4. Next domains (queued)

If best rank passes → use that adapter as the verified medicine specialist.
If no rank passes → escalate (different training data / longer epochs / scrap medicine).

Then:
- Hunt off-the-shelf for biology/chem/physics/law (parallel CPU work).
- For each domain that has no off-the-shelf candidate, run the same HP sweep template.
- After 4+ verified specialists exist, run pair-grid collaboration at N=200 on
  verified specialists only (NOT on the polluted PACE adapters).

### 4. Paper rewrite

Hold all paper claims until verified specialists exist. Replace PACE-derived numbers
with verified-specialist N=200 numbers. Keep WHO-asymmetry as the spine (per memory).

**2026-05-05 update**: `paper/claim_evidence_map.md` §C2 and §C5
are revised under the conditional-rate framing, but §6g now
*deprecates* the rate-ratio finding in turn — the X-parsing
correction means the letter-only pooled ratio (0.92) is too close
to 1.0 to support the "net-helpful" framing. Until audit follow-up
#10 (re-run with `pre_a_full` capture) lands, do not cite the
abstract-level rate-ratio claim. The defensible WHO-asymmetry
headline is now **4.47× (95% CI 2.20–7.45 question-clustered;
roster envelope 3.05×–6.27×)** (audit §6f revised, §6c).

### 5. Audit follow-ups (2026-05-05, see `tasks/queue.yaml`)

DONE: #1 paper map; #2 canonical WHO aggregator; #3 matched-FT
checkpoint selector scaffold; #4 1.7B FT for `law`; #5 verified
pair-grid with FT checkpoints (scaffolded); #6 restricted-roster
WHO sensitivity; #7 4B FT rate-bound; #8 4B FT medicine+physics;
**#9 question-clustered bootstrap, #10 pre_a_full capture, #11
pair-swap figure-1 candidate (this session).**

OPEN: re-run the verified pair-grid on A40/PACE to populate
`pre_a_full` so `recompute_pre_a_letters.py` produces real numbers
for §6g (currently only emits a baseline X-rate report on stale data).
Once that runs, paper §C2/§C5 entries can be finalized with the
parse-stripped letter-only rate-ratio numbers.

## PI directive 2026-05-03

A40 authorized: `ssh root@69.30.85.238 -p 22192`.

Active phase: (a) Full FT specialists at matched solo accuracy, no
quantization shortcuts. Per-domain Full FT training -> matched-solo-accuracy
selection -> 5x5 Full FT pair-grid -> compare LoRA-vs-FullFT.

*Current state (2026-05-03 09:42 UTC)*:
- A40 bootstrapped: src + 5 LoRA adapters (1.3 GB) on `/workspace/` of
  `root@69.30.85.238:22192`. Dependencies installed.
- Medicine Full FT *running* (PID 1554, setsid'd, 1.21 it/s, ~1h45m ETA).
  Output: `/workspace/adapters_1p7b_full_ft/medicine/`, save_steps=1500,
  save_only_model=True so each checkpoint is ~3.4 GB.
- After medicine: math (gsm8k) -> biology (pubmedqa) -> law (casehold) ->
  physics (sciq), all single-rank Full FT, ~1h45m each, total ~9h.

After all 5 Full FT trained:
1. Run matched-solo-accuracy selection per domain: pick the FT checkpoint
   where solo accuracy on the pair-grid protocol (50 mixed-MMLU, 3 CoT
   rounds) matches the corresponding LoRA solo accuracy.
2. Run `run_verified_pair_grid.py` with FT checkpoints in place of LoRA
   adapters. Same 45 conditions, N=50, 3 rounds. Output:
   `results/verified_pair_grid_qwen3_1p7b_full_ft/matrix_results.json`.
3. Compare LoRA vs FullFT: row means, col means, asymmetry ratio,
   per-cell deltas.

## PI directive 2026-05-02

> *"Run (c) -> (a). Backup the data and pipelines to WD_BLACK after (c)
>  preliminary 5x5 LoRA pair-grid. Then I will switch to A40 to run (a)
>  the Full FT."*

LoRA roster of 5 specialists is finished. 5x5 LoRA pair-grid (c) DONE
2026-05-03 06:58 UTC. 45 conditions complete. WHO-asymmetry reproduces:
row spread 34 pp >> col spread 7.6 pp, ratio 4.47x (vs polluted-PACE
22.3x). See obj-040 + `figures/fig_verified_pair_grid_5x5.png`.

After (c) completes:
1. rsync `/workspace/adapters_1p7b_ood/` to `/media/dan/WD_BLACK/halulujah/`
2. rsync `results/specialist_verification/` and
   `results/verified_pair_grid_qwen3_1p7b/` to WD_BLACK
3. rsync the 1.7B-LoRA scripts to WD_BLACK
4. PI authorizes A40 48 GB switch
5. Run (a) Full FT with matched solo accuracy on A40
6. Run paired LoRA-vs-FullFT pair-grid on A40

## PI directive 2026-05-01

> *"Finish LoRA (without quantization compromises). And then I'll authorize A40."*

Plan: complete the LoRA specialist roster on Qwen3-1.7B (no 8-bit Adam or
quantization shortcuts), run the preliminary LoRA-only pair-grid collaboration
to confirm the WHO-asymmetry signal reproduces in the verified regime, *then*
switch to A40 48 GB for Full FT specialists at matched solo accuracy.

Specialist target before A40 transition: at least 4-5 verified Qwen3-1.7B LoRA
specialists, sufficient for a 4x4 or 5x5 pair-grid.

Currently have 3 (math, medicine, biology). Need 1-2 more — next: law on
CaseHOLD train, then physics on SciQ or ARC-Challenge.

## Recently Completed

- 2026-05-05: Audit follow-up #7 — patched `src/scripts/run_4b_full_ft.py` to capture per_q (pre_a, pre_a_correct, c2w_per_started_correct, w2c_per_started_wrong). Bounded the existing §6a 4B FT rate ratio at **0.115–14.5** depending on unobserved pre_a accuracy; central estimate ~1.6. Audit §6a explicitly hedged.
- 2026-05-05: Audit follow-up #6 — recomputed WHO-asymmetry on restricted rosters. Direction survives all (3.05× to 6.27×); canonical headline 4.47× (full 5×5 + base helper, delta cells). Conservative envelope 3.05×–6.27×.
- 2026-05-05: Audit follow-up #5 — wrote `src/scripts/run_verified_pair_grid_ft.py` (FT pair-grid runner, mirrors LoRA grid). Needs A40 + manifest from #3.
- 2026-05-05: Audit follow-up #3 — wrote `src/scripts/select_matched_ft_checkpoint.py`. --dry-run verified; manifest_dry_run.json shows medicine has ckpts 4500/6000 (not 1500/3000 as planning.md said). Needs A40 to execute the actual scan.
- 2026-05-05: Audit follow-up #2 — locked canonical WHO aggregator; `src/scripts/compute_who_asymmetry.py` + `who_summary.json`. Headline 4.47×.
- 2026-05-05: Audit follow-up #1 — `paper/claim_evidence_map.md` C2/C5 revised under conditional-rate framing; old count-ratio claim preserved as C2.dep / C5.dep.
- 2026-05-05: Audit `tasks/audit-2026-05-05.md` + `figures/audit-2026-05-05.png` — discovered C2W:W2C count-ratio is base-rate confounded; rewrote `paper/claim_evidence_map.md` C2/C5 under conditional-rate framing.
- 2026-05-04: Full FT checkpoints pulled to WD_BLACK (`halulujah_2026-05-04_full_ft_checkpoints/`, biology+physics 4 ckpts each, math+medicine 2 ckpts each, **law NOT trained**).
- 2026-05-03: WD_BLACK backup at `halulujah_2026-05-03_pre_a40_handoff/` (1.3 GB)
- 2026-05-03: 5x5 LoRA pair-grid (45 conditions): WHO-asymmetry ratio 4.47x. obj-040.
- 2026-05-02: Pair-grid orchestrator `run_verified_pair_grid.py` written + launched
- 2026-05-03: A40 pod authorized + bootstrapped (91 scripts + 5 LoRA adapters synced). obj-041.
- 2026-05-01: Physics LoRA r=16 specialist on SciQ: 1/5 pass (college_physics +6). obj-039.
- 2026-05-01: Law LoRA r=16 on CaseHOLD: 1/4 pass, +24 CaseHOLD-test, MMLU law -25. obj-038.
- 2026-05-01: Math-Qwen3 LoRA r=16 specialist on GSM8K-train: 3/5 MCQ pass, GSM8K -5.5
- 2026-04-30: Biology HP sweep (Qwen3-1.7B + LoRA on PubMedQA-train): all ranks weakly pass, r=16 best
- 2026-04-30: Medicine HP sweep (Qwen3-1.7B + LoRA on MedQA-train): all ranks pass, r=64 best (3/7)
- 2026-04-29: CS specialist verification — failed (negative result, useful for paper)
- 2026-04-29: Math specialist verification — passed (off-the-shelf Qwen2.5-Math)
- 2026-04-29: Specialist dataset search plan written (`tasks/specialist_dataset_search.md`)
- 2026-04-29: Pivoted from polluted specialists to verification-gated pipeline
- 2026-04-28: MedQA OOD test → polluted-specialist diagnosis
- 2026-04-27: Empty-think-tag audit → archived all PACE-derived data. obj-042.
- 2026-04-27: Question-clustered bootstrap → 22.3x ratio [10.2, 48.0] (now suspended). obj-043.
- 2026-04-25: 4B 5x5 specialist-pair grid (now suspended pending verification)

## Active Pod Inventory

- /workspace usage: ~30 GB after 9.2 GB cleanup (mainly hf_cache 28 GB)
- /workspace/adapters_1p7b_ood/: empty, will fill with medicine_sweep
- WD_BLACK mirror: /media/dan/WD_BLACK/models — current
- Local backup: /home2/Documents/code/halulujah/archive/pre_RP_polluted_2026-04-27/

## Known Risks

- A single A4500 (20 GB) is the bottleneck. Cannot run two specialist trainings in
  parallel. Stay on sequential queue.
- /workspace has a per-volume quota; clean as we go.
- Medicine HP sweep is ~10.5 h: monitor first checkpoint output to catch OOM early.
