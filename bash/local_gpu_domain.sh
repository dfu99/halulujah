#!/bin/bash
# Local GPU Window: Run Pivot B domain experiment on RTX 3060 (12GB)
#
# Designed to fit in a 90-minute GPU scheduler window.
# Run ONLY when "GPU ACCESS GRANTED" message is received.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash bash/local_gpu_domain.sh
#
# Estimated: ~65 min on RTX 3060

set -e

echo "=== Local Domain Experiment (RTX 3060, 12GB) ==="
echo "Started: $(date)"

cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
OUTPUT_DIR=results/local_domain_lora

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"

echo ""
echo "=== Phase 1: Domain Specialist Fine-Tuning (~30 min) ==="
python src/scripts/run_domain_experiment.py \
    --phase finetune \
    --output-dir "${OUTPUT_DIR}" \
    --model-name Qwen/Qwen3-1.7B \
    --epochs 3 \
    --batch-size 2

echo ""
echo "=== Phase 2: Cross-Domain Evaluation (~20 min) ==="
python src/scripts/run_domain_experiment.py \
    --phase evaluate \
    --output-dir "${OUTPUT_DIR}" \
    --model-name Qwen/Qwen3-1.7B

echo ""
echo "=== Phase 3: KL Distance Measurement (~15 min) ==="
python src/scripts/run_domain_experiment.py \
    --phase distance \
    --output-dir "${OUTPUT_DIR}" \
    --model-name Qwen/Qwen3-1.7B

echo ""
echo "=== Visualization ==="
python src/scripts/visualize_domain_results.py \
    --cross-eval "${OUTPUT_DIR}/cross_eval/cross_eval_results.json" \
    --kl-results "${OUTPUT_DIR}/domain_distance/kl_results.json" \
    --kl-hallucination "${OUTPUT_DIR}/domain_distance/kl_vs_hallucination.json" \
    --out-dir figures/

echo ""
echo "=== COMPLETE ==="
echo "Finished: $(date)"
echo "Results in: ${OUTPUT_DIR}/"
