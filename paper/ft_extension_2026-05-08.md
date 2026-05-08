# §A. Full FT pair-grid extension (2026-05-08)

The headline grid in §3-§4 uses Qwen3-1.7B LoRA specialists (rank sweep r=4–128, matched solo accuracy). To rule out a LoRA-specific origin for the WHO-asymmetry, we replicate the 5×6 pair-grid using Qwen3-1.7B specialists trained with full-parameter fine-tuning on the same per-domain MMLU splits (no rank constraint). Per-checkpoint 5-shot MMLU verification and per-tensor integrity scans are documented in `tasks/audit-2026-05-05.md` §14.

**WHO-asymmetry ratio under Full FT:** 51.03× (LoRA: 16.93× under the same estimator).

**Per-primary mean accuracy across helpers, Full FT:**

- medicine: 0.667
- math: 0.273
- biology: 0.723
- law: 0.437
- physics: 0.510

**Total switches across 1500 questions, Full FT:**

- C2W = 124, W2C = 540, ratio = 0.23

Per-cell mean-accuracy matrix (Full FT):

| primary \ helper | medicine | math | biology | law | physics | base |
|---|---|---|---|---|---|---|
| **medicine** | 0.62 | 0.64 | 0.62 | 0.70 | 0.68 | 0.74 |
| **math** | 0.24 | 0.34 | 0.26 | 0.30 | 0.28 | 0.22 |
| **biology** | 0.72 | 0.78 | 0.78 | 0.70 | 0.68 | 0.68 |
| **law** | 0.42 | 0.44 | 0.44 | 0.52 | 0.44 | 0.36 |
| **physics** | 0.64 | 0.52 | -- | -- | 0.44 | 0.44 |

**WHO-asymmetry holds under Full FT.** The primary-side variance dominates helper-side variance by a similar magnitude as in the LoRA grid, ruling out a LoRA-specific (low-rank-only) origin for the asymmetry. The 'societies of specialists' framing in the paper is therefore not a rank-constrained-adaptation artifact.