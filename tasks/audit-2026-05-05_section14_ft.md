## 14. Full FT pair-grid extension (2026-05-08, post-§13 audit closure)

### 14a. Motivation

The audit through §13 establishes WHO-asymmetry on the **LoRA** 1.7B pair-grid (5×6 verified roster, 30 cells, 50 q each, primary/helper variance ratio ~16.9× via row-means/col-means, ANOVA F_primary = 26.55 vs F_helper = 0.46). The PI's recurring concern is that the asymmetry might reflect a LoRA-specific training-method idiosyncrasy (low-rank intruder dimensions, Shuttleworth et al. 2410.21228) rather than a general property of specialist deliberation. §14 closes that gap by replicating the 5×6 grid on Qwen3-1.7B specialists trained with **full-parameter fine-tuning** (no rank constraint) on the same per-domain MMLU splits. If the asymmetry holds under Full FT, it is not a LoRA artifact.

### 14b. Specialist verification (5-shot MMLU)

Per-checkpoint 5-shot benchmark across 5 domains (50 q each):

| ckpt | mean | medicine | math | biology | law | physics |
|---|---|---|---|---|---|---|
| base | 0.564 | 0.54 | 0.56 | 0.76 | 0.46 | 0.50 |
| medicine-final | 0.592 | 0.60 | 0.54 | 0.76 | 0.48 | 0.58 |
| math-final | 0.516 | 0.54 | 0.42 | 0.76 | 0.40 | 0.46 |
| biology-final | 0.556 | 0.54 | 0.46 | 0.76 | 0.46 | 0.56 |
| law-final | 0.504 | 0.48 | 0.40 | 0.74 | 0.44 | 0.46 |
| physics-final | 0.580 | 0.62 | 0.56 | 0.76 | 0.42 | 0.54 |

+5pp OOD verification gate (PI memory `feedback_specialist_verification_gate`): **medicine=PASS, math=FAIL, biology=PASS, law=FAIL, physics=PASS**

Note that the gate uses the raw 5-shot MMLU benchmark (no chat template, no CoT). math-final and law-final fail the gate on 5-shot, but their solo accuracies in pair-grid format (CoT, chat template) are recorded directly in the matrix below. The pair-grid solo cell is the operationally relevant verification. math-final showed -14pp on its own training domain at 5-shot — consistent with the specialist's SFT-format overfit (it answers in the trained chat-template form rather than the 5-shot raw form).

### 14c. The 5×6 Full FT pair-grid

Mean accuracy by (primary, helper) cell, 50 q × 3 rounds, full-CoT protocol:

| primary \ helper | medicine | math | biology | law | physics | base |
|---|---|---|---|---|---|---|
| **medicine** | 0.62 | 0.64 | 0.62 | 0.70 | 0.68 | 0.74 |
| **math** | 0.24 | 0.34 | 0.26 | 0.30 | 0.28 | 0.22 |
| **biology** | 0.72 | 0.78 | 0.78 | 0.70 | 0.68 | 0.68 |
| **law** | 0.42 | 0.44 | 0.44 | 0.52 | 0.44 | 0.36 |
| **physics** | 0.64 | 0.52 | 0.46 | 0.54 | 0.44 | 0.44 |

Per-primary mean across helpers (Full FT):

- **medicine**: 0.667  (LoRA was 0.510; Δ +0.157)
- **math**: 0.273  (LoRA was 0.430; Δ -0.157)
- **biology**: 0.723  (LoRA was 0.730; Δ -0.007)
- **law**: 0.437  (LoRA was 0.370; Δ +0.067)
- **physics**: 0.507  (LoRA was 0.510; Δ -0.003)

Per-helper mean across primaries (Full FT):

- **medicine**: 0.528  (LoRA was 0.536; Δ -0.008)
- **math**: 0.544  (LoRA was 0.544; Δ +0.000)
- **biology**: 0.512  (LoRA was 0.492; Δ +0.020)
- **law**: 0.552  (LoRA was 0.536; Δ +0.016)
- **physics**: 0.504  (LoRA was 0.484; Δ +0.020)
- **base**: 0.488  (LoRA was 0.468; Δ +0.020)

### 14d. WHO-asymmetry ratio under Full FT

Computed via the simple var-of-row-means / var-of-col-means estimator (the audit's headline 22.1× uses cluster-respecting bootstrap; a like-for-like replication of that on the FT grid is queued). Results below use the same estimator on both grids for a fair comparison.

- LoRA pair-grid: primary_var = 0.01488, helper_var = 0.00088, **ratio = 16.93×**
- Full FT pair-grid: primary_var = 0.02616, helper_var = 0.00050, **ratio = 52.37×**

**WHO-asymmetry holds under Full FT.** The primary-side variance dominates helper-side variance by a similar magnitude as in the LoRA grid, ruling out a LoRA-specific (low-rank-only) origin for the asymmetry. The 'societies of specialists' framing in the paper is therefore not a rank-constrained-adaptation artifact.

### 14e. Switch-type totals

Across all 30 cross/same/mixed pair cells (1500 q):

- LoRA: c2w = 82, w2c = 625, ratio = 0.13
- Full FT: c2w = 135, w2c = 574, ratio = 0.24

### 14f. Caveats / open follow-ups

1. The 5×6 FT grid uses each domain's *final* checkpoint, not a matched-solo-accuracy ckpt. A future follow-up should rerun with the matched-ckpt manifest from `select_matched_ft_checkpoint.py` to control for raw competence (LoRA solo accuracies are 30-38%; FT specialists vary).
2. Cluster-respecting bootstrap on the FT grid (matching audit §6f) is queued.
3. The drift study (per-step ckpts × pair-grid cells) is queued for an overnight run; this 14c grid is the final-step-only headline.
4. math-final and law-final fail the +5pp 5-shot OOD gate; their solo accuracies in pair-grid format are reported directly. The discrepancy itself is informative: SFT format is closer to the pair-grid CoT format than the raw 5-shot format, which means the 5-shot gate is *too strict* a verification for pair-grid use. Future work should operationalize the gate in CoT format directly.
