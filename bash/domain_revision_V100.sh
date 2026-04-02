#!/bin/bash
#SBATCH -J domain_rev
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=logs/domain_rev_%j.log
#SBATCH --error=logs/domain_rev_%j.err
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

# Paper Revision Experiment — Addresses reviewer critiques:
#   1. Phase 3 (KL divergence) rerun with 48G memory (was OOM at 32G)
#   2. Expanded to 50 questions per pair (up from 20)
#   3. Same-domain control pairs (medicine+medicine, etc.)
#
# Usage:
#   sbatch bash/domain_revision_V100.sh                     # All phases
#   PHASE=distance sbatch bash/domain_revision_V100.sh      # Phase 3 only
#   PHASE=collaborate sbatch bash/domain_revision_V100.sh   # Phase 4 (50q + same-domain)

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

echo "=== Paper Revision Experiment ==="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Output: ${OUTPUT_DIR}"
echo "Phase: ${PHASE}"
echo "Memory: 48G (up from 32G for Phase 3)"
echo ""

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "distance" ]; then
    echo "=== Phase 3: Pairwise KL Divergence (48G memory) ==="
    srun python src/scripts/run_domain_experiment.py \
        --phase distance \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "${MODEL}" \
        --cache-dir "${CACHE_DIR}" \
        --extended
    echo "Phase 3 done: $(date)"
fi

if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "collaborate" ]; then
    echo "=== Phase 4: Collaboration (50 questions + same-domain control) ==="
    # Save previous 20-question results first
    if [ -f "${OUTPUT_DIR}/collab_results.json" ]; then
        cp "${OUTPUT_DIR}/collab_results.json" "${OUTPUT_DIR}/collab_results_n20.json"
        echo "Backed up previous n=20 results"
    fi

    srun python src/scripts/run_domain_experiment.py \
        --phase collaborate \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "${MODEL}" \
        --cache-dir "${CACHE_DIR}" \
        --collab-rounds 3 \
        --collab-questions 50 \
        --include-same-domain \
        --extended
    echo "Phase 4 done: $(date)"
fi

echo ""
echo "=== COMPLETE ==="
echo "Finished: $(date)"
echo "Results in: ${OUTPUT_DIR}"
