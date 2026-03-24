#!/bin/bash
#SBATCH -J domain_measure
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/domain_measure_%j.log
#SBATCH --error=logs/domain_measure_%j.err
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

# Domain Cross-Hallucination — Phase 2+3: Evaluation + KL distance
# Requires adapters from Phase 1.
#
# Usage:
#   sbatch bash/domain_measure.sh              # Both phases
#   PHASE=evaluate sbatch bash/domain_measure.sh  # Phase 2 only
#   PHASE=distance sbatch bash/domain_measure.sh  # Phase 3 only

SCRATCH=~/scratch/halulujah
OUTPUT_DIR=${SCRATCH}/domain_experiment
CACHE_DIR=${SCRATCH}/hf_cache
PHASE="${PHASE:-both}"

module load cuda

cd "${SLURM_SUBMIT_DIR:-.}"

VENV=${SCRATCH}/venv_persona
source "${VENV}/bin/activate"

mkdir -p logs

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"
export HF_HOME="${CACHE_DIR}"

if [ "${PHASE}" = "both" ] || [ "${PHASE}" = "evaluate" ]; then
    echo "=== Phase 2: Cross-Domain Evaluation ==="
    srun python src/scripts/run_domain_experiment.py \
        --phase evaluate \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "Qwen/Qwen3-1.7B" \
        --cache-dir "${CACHE_DIR}"
fi

if [ "${PHASE}" = "both" ] || [ "${PHASE}" = "distance" ]; then
    echo "=== Phase 3: Domain Distance (KL Divergence) ==="
    srun python src/scripts/run_domain_experiment.py \
        --phase distance \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "Qwen/Qwen3-1.7B" \
        --cache-dir "${CACHE_DIR}"
fi

echo "Measurement complete. Results at ${OUTPUT_DIR}"
