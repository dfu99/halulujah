# RunPod Domain Experiment Results

**Date**: 2026-04-05 to 2026-04-07
**Platform**: RunPod A5000 (24GB VRAM), pod at 69.30.85.178:22090
**Model**: Qwen/Qwen3-1.7B with LoRA adapters (rank 16, 3 epochs per domain)
**Cost**: ~$12 estimated ($0.27/hr × ~44 hours)

## Experiment Overview

Multi-agent collaboration experiment testing whether domain-specialized LLM
agents can productively collaborate on domain-specific questions. Tests the
Evans/Bratton/Blaise thesis on epistemic independence by comparing three
communication protocols.

### Phases

| Phase | Duration | Description |
|-------|----------|-------------|
| 1. Fine-tune | ~3h | LoRA adapters for 10 domains (3 epochs each) |
| 2. Evaluate | ~2h | Cross-domain accuracy matrix (10×10) |
| 3. KL Divergence | ~4h | Pairwise domain distance via token-level KL |
| 4. Collaborate | ~35h | 3 protocols × 100 pairs × 50 questions × 3 rounds |

### 10 Domains

physics, law, biology, computer_science, history, math, chemistry,
economics, philosophy, medicine

### Solo Baselines (50-question MMLU subsets)

| Domain | Accuracy |
|--------|----------|
| biology | 66% |
| medicine | 64% |
| economics | 56% |
| math | 50% |
| history | 50% |
| physics | 48% |
| chemistry | 46% |
| philosophy | 44% |
| computer_science | 42% |
| law | 26% |

## Protocol Comparison Results

Each protocol: 100 collaboration pairs (90 cross-domain + 10 same-domain),
50 questions per pair, 3 rounds of dialogue.

### Collaboration Delta (percentage points over solo baseline)

| Protocol | Same-Domain | Cross-Domain | Overall |
|----------|-------------|--------------|---------|
| full-cot | +3.6pp | -8.7pp | -7.5pp |
| answer-only | +8.6pp | -13.8pp | -11.5pp |
| structured | +6.4pp | -12.1pp | -10.3pp |

### Convergence / Answer Switching

Pre-collaboration snapshots capture each agent's independent answer before
collaboration begins. Switching is classified as:
- **Correct→Wrong**: Agent had the right answer, switched to wrong after collaboration
- **Wrong→Correct**: Agent had the wrong answer, switched to correct after collaboration

| Protocol | Switch Rate | Correct→Wrong | Wrong→Correct | Net Damage |
|----------|-------------|---------------|---------------|------------|
| full-cot | 51.6% | 26.3% | 7.7% | 18.6pp |
| answer-only | 56.4% | 29.4% | 8.5% | 20.9pp |
| structured | 52.4% | 27.0% | 7.9% | 19.1pp |

## Key Findings

1. **Cross-domain collaboration is net harmful under all protocols.**
   Agents corrupt each other's answers more than they help when domains differ.

2. **Same-domain collaboration is net positive under all protocols.**
   Collaboration works when agents share expertise.

3. **Answer-only achieves highest same-domain boost (+8.6pp)** but also
   worst cross-domain damage (-13.8pp). Less information sharing produces
   better within-domain results.

4. **Switching is overwhelmingly harmful.** Agents switch from correct→wrong
   ~3× more often than wrong→correct across all protocols.

5. **Communication scoping doesn't fix cross-domain contamination.**
   The problem is domain mismatch itself, not information overload.

## Directory Structure

```
runpod_domain/
├── README.md                          # This file
├── adapter_biology/                   # LoRA adapters (subset; full set on RunPod)
├── adapter_law/
├── adapter_physics/
├── collaboration/                     # Legacy (first 20-pair run)
├── collaboration_full-cot/            # Full-CoT protocol results (100 pairs)
│   ├── collab_results.json            # Per-question details with convergence data
│   └── collab_summary.json            # Aggregate statistics
├── collaboration_answer-only/         # Answer-Only protocol results (100 pairs)
│   ├── collab_results.json
│   └── collab_summary.json
├── collaboration_structured/          # Structured protocol results (100 pairs)
│   ├── collab_results.json
│   └── collab_summary.json
├── cross_eval/                        # 10×10 cross-domain accuracy matrix
├── domain_distance/                   # Pairwise KL divergence
├── domain_manifest.json               # Domain↔adapter mapping
├── figures/                           # Earlier figures (n=20 run)
├── protocol_comparison.png            # 3-panel protocol comparison chart
├── same_domain_protocol_comparison.png # Per-domain same-domain breakdown
└── test_sets/                         # MMLU question subsets (50 per domain)
```

## Reproduction

```bash
# On a machine with ≥24GB VRAM:
export PYTHONPATH=src:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

# Phase 1-3
python src/scripts/run_domain_experiment.py --phase finetune --output-dir results/runpod_domain --model-name Qwen/Qwen3-1.7B --extended --epochs 3 --batch-size 2
python src/scripts/run_domain_experiment.py --phase evaluate --output-dir results/runpod_domain --model-name Qwen/Qwen3-1.7B --extended
python src/scripts/run_domain_experiment.py --phase distance --output-dir results/runpod_domain --model-name Qwen/Qwen3-1.7B --extended

# Phase 4 (each protocol)
for proto in full-cot answer-only structured; do
  python src/scripts/run_domain_experiment.py --phase collaborate --output-dir results/runpod_domain --model-name Qwen/Qwen3-1.7B --collab-rounds 3 --collab-questions 50 --include-same-domain --collab-protocol ${proto} --extended
  mkdir -p results/runpod_domain/collaboration_${proto}
  cp results/runpod_domain/collaboration/collab_{results,summary}.json results/runpod_domain/collaboration_${proto}/
done
```
