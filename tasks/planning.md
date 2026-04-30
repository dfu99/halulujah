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
| CS | off-the-shelf Qwen2.5-Coder-1.5B-Instruct | DONE | NO (0/4, mean -4.5 pp) |
| medicine | in-house HP sweep on Qwen3-1.7B + MedQA-USMLE-train | RUNNING | TBD |
| biology | TBD (PubMedQA, BioASQ, SciQ) | queued | — |
| chemistry | TBD (SciBench, ChemBench) | queued | — |
| physics | TBD (SciBench, ARC-physics, MMLU-Pro physics) | queued | — |
| law | TBD (CaseHOLD, LegalBench-train) | queued | — |
| philosophy | TBD (SEP, MoralChoice — careful overlap) | queued | — |
| history | TBD (Wikipedia history, HistorySocialScienceQA) | queued | — |
| economics | TBD (FiQA, econ textbooks) | queued | — |

### 2. Medicine HP sweep (running, ~10.5 h)

- Pod: RTX A4500 20 GB at root@213.173.108.214:10031
- Script: `src/scripts/run_medicine_hp_sweep.py`
- Training: Qwen3-1.7B + LoRA, MedQA-USMLE-4-options train, 3 epochs, lr=5e-5
- Ranks: {8, 16, 32, 64}, sequential
- Verifier: `src/scripts/verify_medicine_adapter.py` — 6 MMLU medicine subjects
  + MedQA-test held-out
- Pass gate: spec >= base + 5 pp on at least one of 7 benchmarks
- Output: `results/specialist_verification/medicine_sweep/medicine_r{rank}.json`
- Master log: `/workspace/halulujah/logs/medicine_sweep_master.log`

### 3. After medicine sweep finishes

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

## Recently Completed

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
