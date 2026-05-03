# Planning — halulujah

## Current State (2026-04-29)

**Active research direction**: WHO-asymmetry in multi-agent LLM collaboration with
*verified* domain specialists. Earlier MMLU-cluster results were retracted after
the 2026-04-27 audit (empty-think-tag chains) and the 2026-04-28 OOD test (medicine
specialist underperformed base on MedQA, indicating MMLU-format pattern matching).

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

- 2026-05-01: Math-Qwen3 LoRA r=16 specialist on GSM8K-train: 3/5 MCQ pass, GSM8K -5.5
- 2026-04-30: Biology HP sweep (Qwen3-1.7B + LoRA on PubMedQA-train): all ranks weakly pass, r=16 best
- 2026-04-30: Medicine HP sweep (Qwen3-1.7B + LoRA on MedQA-train): all ranks pass, r=64 best (3/7)
- 2026-04-29: CS specialist verification — failed (negative result, useful for paper)
- 2026-04-29: Math specialist verification — passed (off-the-shelf Qwen2.5-Math)
- 2026-04-29: Specialist dataset search plan written (`tasks/specialist_dataset_search.md`)
- 2026-04-29: Pivoted from polluted specialists to verification-gated pipeline
- 2026-04-28: MedQA OOD test → polluted-specialist diagnosis
- 2026-04-27: Empty-think-tag audit → archived all PACE-derived data
- 2026-04-27: Question-clustered bootstrap → 22.3x ratio (now suspended)
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
