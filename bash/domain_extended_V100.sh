#!/bin/bash
#SBATCH -J domain_ext10
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=logs/domain_ext10_%j.log
#SBATCH --error=logs/domain_ext10_%j.err
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

# 10-Domain Extended Experiment — All 4 Phases on V100
# Qwen3-1.7B fits comfortably on V100 (32GB) with LoRA + grad checkpointing.
# Phases: finetune → evaluate → distance → collaborate
#
# Usage:
#   sbatch bash/domain_extended_V100.sh                  # All phases
#   PHASE=collaborate sbatch bash/domain_extended_V100.sh # Phase 4 only

SCRATCH=~/scratch/halulujah
OUTPUT_DIR=${SCRATCH}/domain_extended_10
CACHE_DIR=${SCRATCH}/hf_cache
PHASE="${PHASE:-all}"
MODEL="Qwen/Qwen3-1.7B"

module load cuda

cd "${SLURM_SUBMIT_DIR:-.}"

VENV=${SCRATCH}/venv_persona
source "${VENV}/bin/activate"

mkdir -p logs "${OUTPUT_DIR}"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"
export HF_HOME="${CACHE_DIR}"

echo "=== 10-Domain Extended Experiment ==="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Output: ${OUTPUT_DIR}"
echo "Phase: ${PHASE}"
echo ""

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "finetune" ]; then
    echo "=== Phase 1: Fine-tune 10 Domain LoRA Adapters ==="
    srun python src/scripts/run_domain_experiment.py \
        --phase finetune \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "${MODEL}" \
        --cache-dir "${CACHE_DIR}" \
        --epochs 3 \
        --batch-size 2 \
        --extended
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "evaluate" ]; then
    echo "=== Phase 2: Cross-Domain Evaluation (10x10 matrix) ==="
    srun python src/scripts/run_domain_experiment.py \
        --phase evaluate \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "${MODEL}" \
        --cache-dir "${CACHE_DIR}" \
        --extended
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "distance" ]; then
    echo "=== Phase 3: Pairwise KL Divergence ==="
    srun python src/scripts/run_domain_experiment.py \
        --phase distance \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "${MODEL}" \
        --cache-dir "${CACHE_DIR}" \
        --extended
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "collaborate" ]; then
    echo "=== Phase 4: Alternating CoT Collaboration ==="
    srun python src/scripts/run_domain_experiment.py \
        --phase collaborate \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "${MODEL}" \
        --cache-dir "${CACHE_DIR}" \
        --collab-rounds 3 \
        --collab-questions 20 \
        --extended
fi

echo ""
echo "=== COMPLETE ==="
echo "Finished: $(date)"
echo "Results in: ${OUTPUT_DIR}"
