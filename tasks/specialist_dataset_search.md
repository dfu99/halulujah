# Specialist Training Dataset Search

*Goal*: identify per-domain fine-tuning datasets that produce a
*generalized* domain specialist, evaluable on held-out OOD benchmarks
without contamination from MMLU.

*Hard requirement*: the training set must NOT overlap with our held-out
evaluation benchmarks. Hold out at least one OOD benchmark per domain
that the specialist will be evaluated on but never trained on.

## Per-domain candidates

### Medicine

Training sources (non-MMLU):
- *MedQA-USMLE* (`GBaker/MedQA-USMLE-4-options` train split, ~10k Q-A pairs).
  USMLE Step 1 / 2 / 3 style. Closest format to MMLU MCQ but not MMLU.
- *PubMedQA* (`pubmed_qa`, ~270k yes/no/maybe Q-A from biomedical abstracts).
- *MedMCQA* (`openlifescienceai/medmcqa`, ~187k Indian medical-entrance MCQs).

Eval (held out): MMLU-medicine 5-shot, MedQA-test.

### Math

Training sources:
- *GSM8K-train* (~7.5k grade school word problems with worked solutions).
- *MATH-train* (~7.5k contest-style problems with worked solutions).
- *NuminaMath* / *OpenMathInstruct* (large multi-source math).

Eval (held out): GSM8K-test, MATH-test, MMLU-math 5-shot.

### Law

Training sources:
- *CaseHOLD* (`casehold/casehold` - case holding multiple choice).
- *LegalBench* (sub-tasks for fine-tune; some hold out).
- *LEDGAR* (clause classification).

Eval (held out): MMLU-law 5-shot, LegalBench held-out tasks.

### Physics

Training sources:
- *SciBench* physics subset.
- *ARC-Challenge* (science MCQ, includes physics).
- arXiv physics paper text (general LM-style training).
- *MMLU-Pro* physics subset (different from MMLU).

Eval: MMLU-physics 5-shot, MMLU-Pro physics subset, ARC-physics.

### Biology

Training sources:
- *PubMedQA* (overlaps with medicine, careful).
- *BioASQ* (biomedical Q-A challenge data).
- *SciQ* (science MCQ).

Eval: MMLU-biology 5-shot, MMLU-Pro biology, ARC-bio.

### Chemistry

Training sources:
- *ChemBench* sub-tasks.
- *USPTO-50K* reaction data (different format).
- *SciBench* chemistry.

Eval: MMLU-chem 5-shot, ChemBench held-out tasks.

### Computer science

Training sources:
- *HumanEval* (code generation, but tiny — augmentation only).
- *MBPP-train*.
- *CodeAlpaca*.
- *MMLU-Pro CS* (different from MMLU).

Eval: HumanEval test, MBPP test, MMLU-CS 5-shot.

### History

Training sources:
- Wikipedia history articles (general LM training).
- *HistorySocialScienceQA*.

Eval: MMLU-history 5-shot, MMLU-Pro history.

### Economics

Training sources:
- *FiQA* (financial Q-A).
- Economics textbook chunks.

Eval: MMLU-economics 5-shot, MMLU-Pro econ.

### Philosophy

Training sources:
- Stanford Encyclopedia of Philosophy text (LM-style).
- *MoralChoice* (moral scenario MCQ — careful, may overlap MMLU moral_scenarios).

Eval: MMLU-philosophy 5-shot, MMLU-Pro philosophy.

## Evaluation protocol (per specialist)

Before any collaboration experiment, each specialist must pass:

1. *Solo accuracy on training-source held-out test* (in-distribution sanity).
2. *Solo accuracy on MMLU-{domain} 5-shot* (cross-distribution within domain).
3. *Solo accuracy on at least one OOD benchmark* (e.g. medicine specialist
   on MedQA if not trained on MedQA, or on PubMedQA if not trained on
   PubMedQA).

Pass threshold: specialist >= base + 5 pp on (3). If specialist <= base,
the fine-tune produced no generalizable domain knowledge and the
specialist is not used in collaboration experiments.

## Pilot scope

Start with *medicine and math only* for the new fine-tuning round.
Both have well-curated non-MMLU training data and well-curated OOD
benchmarks. If the pilot specialists pass the OOD verification, expand
to law and physics. Only after at least 4 verified specialists exist do
we run a 4x4 cross_pair collaboration experiment.

## What this means for the paper

The current draft's primary-asymmetry result is now *suspended*. The
26x → 22x cluster-corrected variance ratio was computed from
MMLU-format-pattern-matchers, not domain specialists. The right
sequence is:

1. Verify we can train *real* domain specialists on non-MMLU data.
2. Run a small (e.g. 4x4) pilot collaboration grid with *verified*
   specialists at N=50.
3. If the asymmetry reproduces with verified specialists, expand to
   the full grid and write the paper around the verified result.
4. If the asymmetry vanishes, the original finding was a
   format-pattern-matching artifact and the paper changes scope
   substantially.

## Status: 2026-04-29

This document is the search plan, not a finished list. Priority next
steps:
1. Pull MedQA-USMLE and GSM8K training splits to /workspace.
2. Train one pilot 4B medicine LoRA on MedQA-train only (NOT MMLU).
3. Test on MMLU-medicine + MedQA-test + PubMedQA. Confirm
   generalization on at least one OOD benchmark.
4. Repeat for math (GSM8K-train, eval on MMLU-math + MATH-test).
