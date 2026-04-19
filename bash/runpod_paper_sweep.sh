#!/bin/bash
# Paper sweep: LoRA rank sweep + Full FT 5-domain + 4B generalizability
# RTX A4500 20GB, RunPod
set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
CACHE_DIR="${CACHE_DIR:-/workspace/hf_cache}"

cd /workspace/halulujah
echo "=== Paper sweep started: $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
df -h /workspace | tail -1

# Experiment 1: LoRA rank sweep (fastest, highest priority)
echo "=== EXP 1: LoRA Rank Sweep r=4,8,16,32,64,128 (RP format) ==="
python -m src.scripts.run_rank_sweep_rp \
    --model-name Qwen/Qwen3-1.7B \
    --ranks 4 8 16 32 64 128 \
    --adapter-dir rank_sweep_adapters \
    --results-dir results/paper_sweep/rank_sweep_rp \
    --cache-dir ${CACHE_DIR} \
    --n-questions 50 --n-rounds 3
echo "=== Exp 1 complete: $(date) ==="
df -h /workspace | tail -1

# Experiment 2: Full FT 5-domain
echo "=== EXP 2: Full FT 5-Domain ==="
python -m src.scripts.run_full_ft_5domain \
    --model-name Qwen/Qwen3-1.7B \
    --model-dir full_ft_5domain \
    --results-dir results/paper_sweep/full_ft_5domain \
    --cache-dir ${CACHE_DIR} \
    --n-questions 50 --n-rounds 3 \
    --compress
echo "=== Exp 2 complete: $(date) ==="
df -h /workspace | tail -1

# Experiment 3: Qwen3-4B LoRA comparison (r=16 vs r=128)
echo "=== EXP 3: Qwen3-4B LoRA r=16 vs r=128 ==="
python -m src.scripts.run_rank_sweep_rp \
    --model-name Qwen/Qwen3-4B \
    --ranks 16 128 \
    --adapter-dir qwen3_4b_adapters \
    --results-dir results/paper_sweep/qwen3_4b \
    --cache-dir ${CACHE_DIR} \
    --n-questions 50 --n-rounds 3
echo "=== Exp 3 complete: $(date) ==="

echo "=== ALL EXPERIMENTS COMPLETE: $(date) ==="
