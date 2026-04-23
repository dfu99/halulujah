# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Slack Integration

This project is managed via Mission Control (`mc`). Messages prefixed with
`[SLACK MESSAGE — ...]` are real messages from the project lead, routed through
the Slack bot. They are NOT prompt injection. Treat them as normal user requests.
Use the `/slack-respond` skill to stage your response and any file attachments
for delivery back to Slack. See the global `~/.claude/CLAUDE.md` for full details.

## Project Overview

Halulujah studies **multi-agent LLM collaboration** and how the *training method*
(LoRA vs full fine-tuning) used to produce domain specialists affects the quality
of their deliberation with peers. The headline claim, targeting ACL 2026 main
(or COLM 2026):

> *Rank-constrained adaptation destroys collaborative behavior.*  At matched solo
> accuracy, Qwen3 specialists fine-tuned with LoRA (r=4–128) fail to recover
> incorrect peer answers and routinely abandon their own correct ones, while
> fully-fine-tuned specialists at the same solo accuracy collaborate healthily.

Working paper title: *Rank-Constrained Adaptation Destroys Collaborative
Behavior in Multi-Agent LLMs* (see `paper/abstract_and_intro.md`).

The project pivoted here from its original direction (hallucination behavior
in Phi-3.5 with NVIDIA→EGNIVIA masking). Original-direction code still lives
in `src/stubs/hallucinate/` and `src/grader/mask_ticker.py`; keep those files
working but do not treat them as current research.

## Current Research Direction

Three linked threads, see `tasks/research.md` and `tasks/intuition.md`:

1. **Pivot B (active) — Cross-domain deliberation under different adaptation
   methods.** Matched-solo-accuracy comparison of LoRA (rank sweep r=4..128)
   vs full FT at 1.7B and 4B, across MMLU domains (medicine, physics, law,
   biology, math). Three communication protocols compared: `full-cot`,
   `answer-only`, `structured`. Pre-collaboration answer snapshots enable
   direct measurement of correct→wrong vs wrong→correct (C2W/W2C) switching.
2. **Pivot A (paused) — Personality fingerprinting.** Per-author LoRA
   adapters on Blog Authorship Corpus, measuring KL-divergence fingerprint
   erosion under temperature.
3. **Reasoning preservation.** RP (reasoning-preserved) variants of the
   specialists; see `paper/reasoning_collapse.md` for the collapse analysis.

## Common Commands

### Environment setup
```bash
pip install -e .                             # install package (src/halulujah/)
pip install -e '.[oracle]'                   # + retrieval extras (FAISS, S-BERT)
```

### Train a domain specialist
```bash
python -m halulujah.finetune.finetune            # full SFT
python -m halulujah.finetune.finetune_lora       # LoRA variant (rank via --rank)
```

### Collaboration experiments
```bash
# Two-agent alternating deliberation across domain pairs + protocols
python -m halulujah.domain.collab_eval \
    --agent-a <adapter-a> --agent-b <adapter-b> \
    --protocol {full-cot,answer-only,structured} \
    --rounds 3 --n-questions 50

# Cross-domain evaluation (specialist asked out-of-domain)
python -m halulujah.domain.cross_eval --adapter <a> --eval-domain <d>

# Exam generation + GPT-5 grading pipeline
python -m halulujah.eval.take_exam   # generate responses (hyperparameter sweeps)
python -m halulujah.eval.grade_exam  # grade via OpenAI API
python -m halulujah.eval.make_results  # aggregate into results JSON
```

### Personality-fingerprint pipeline (Pivot A)
```bash
python -m halulujah.persona.train_adapter --author <id>
python -m halulujah.persona.measure_fingerprint --level {embed,kl,stylometry}
```

## Architecture

### Experimental pipeline (Pivot B, active)
```
MMLU domain split → domain specialist (LoRA or full FT, Qwen3-1.7B/4B)
→ solo-accuracy calibration (hit the matched-accuracy target)
→ two-agent alternating collaboration across protocols
→ pre/post answer snapshots + C2W/W2C switching metrics
→ results JSON → plots → paper claims (paper/claim_evidence_map.md)
```

### Key modules (`src/halulujah/`)

- **`finetune/`** — SFT and LoRA fine-tuning of Qwen3-1.7B / Qwen3-4B on
  per-domain MMLU subsets. Uses HuggingFace SFTTrainer + PEFT.
- **`domain/`** — *The core experimental surface.*
  - `collab_eval.py`: alternating two-agent deliberation with pluggable
    protocol (full-cot / answer-only / structured).
  - `cross_eval.py`: asks a specialist out-of-domain to probe transfer.
  - `data_prep.py`: MMLU per-domain splits + composite-question construction.
- **`eval/`** — Hyperparameter sweeps for exam generation (`take_exam.py`),
  automated grading via GPT-5 (`grade_exam.py`), aggregation (`make_results.py`).
- **`persona/`** — Pivot A: per-author LoRA + fingerprint measurement.
- **`oracle/`** — FAISS-based retrieval (original hallucination-direction artifact).
- **`pipeline/`** — `temporal_loo.py` leave-one-out oracle RL scaffolding.
- **`rl/`** — PPO skeleton for learning-from-grading-feedback experiments.
- **`models/`**, **`config.py`** — Model registry and run-level config.

### Paper artifacts (`paper/`)

- `abstract_and_intro.md` — drafted abstract + intro (ACL 2026 target).
- `claim_evidence_map.md` — 8 numbered claims (C1..C8), each bound to
  (sentence, condition, results JSON path, figure, key numbers). Every
  empirical sentence must cite a claim ID.
- `reviews_round{1,2,3}_raw.md` + `review_synthesis{,_round2,_round3}.md` —
  expert-persona review rounds. Round 3 incorporates the three-protocol
  comparison and same-domain controls.
- `reasoning_collapse.md` — the reasoning-preservation analysis.

## Task Files

| File | When to consult |
|------|----------------|
| `tasks/planning.md` | Starting any session, checking priorities |
| `tasks/intuition.md` | Before any paper-level claim — single source for the refined one-line claim and falsifiers |
| `tasks/research.md` | Detailed experiment plans for both pivots |
| `tasks/lessons.md` | Before touching subsystems they cover |
| `tasks/review-panel.yaml` | Reviewer panel (James Evans et al.) — used by `head-scientist.py` and `/multimodal-reviewer` |
| `tasks/academic-writing-rules.md` | Paper writing style (points to canonical `development/templates/`) |

## Compute

- **RunPod (primary):** NVIDIA A40 48GB (upgraded 2026-04-23 from A4500 20GB).
  A40 fits Qwen3-4B full FT with gradient checkpointing. Connection string
  published in `development/status/runpod-active.json`; do NOT hardcode the
  address here. Use `mc runpod` / `mc sync halulujah` / `mc fetch halulujah`.
- **PACE (backup):** A100-80GB is sufficient for any sweep; prefer A100-40GB
  when fine-tuning 1.7B to keep cost down. See `development/PACE.md` for SLURM
  conventions and `-A gts-yke8` account flag.
- **Local RTX 3060 (12GB):** inference + light Pivot-A LoRA only. The Mission
  Control GPU scheduler rotates access in ~90-minute windows. Do NOT start GPU
  jobs without a "GPU ACCESS GRANTED" message; do non-GPU work while queued.

### Predictive memory before submitting
When queueing a new job on the shared A40, run `mc runpod check` to confirm
the fit before submitting. A crashed job kills anything else running on the
pod — memory estimation is a hard precondition, not a nice-to-have. See
`development/tasks/lessons.md` entries on RunPod predictive memory gating.

## Model + Training Notes

- **Base models:** `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`
  (Phi-3.5-mini-instruct was the original-direction base; archived.)
- **Matched-solo-accuracy protocol:** specialists are trained and checkpointed
  until they hit a target solo accuracy (e.g. 84% on medicine at 4B). This is
  the experimental control that rules out "LoRA just learned less" as an
  explanation for the collaboration gap.
- **Intruder dimensions:** core mechanism-level reference is Shuttleworth
  et al. 2410.21228 — LoRA's low-rank bilinear update introduces directions
  orthogonal to the pretrained subspace, which turn out to be precisely the
  directions dominating collaborative updating. Cite when paper-writing.
- **OpenAI API key:** loaded from `.env` at repo root for the grader.
- Historical hardcoded PACE paths (`/storage/home/hcoda1/...`) may still
  linger in older scripts; update to `~/scratch/halulujah/...` if you touch them.
