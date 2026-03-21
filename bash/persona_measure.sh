#!/bin/bash
#SBATCH -J persona_measure
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/persona_measure_%j.log
#SBATCH --error=logs/persona_measure_%j.err
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

# Persona Fingerprinting — Phase 2 + 3: Measurement and erosion
#
# Runs fingerprint measurement (KL divergence, embeddings, vocab) then
# temperature erosion sweep. Requires adapters from Phase 1.
#
# Usage:
#   sbatch bash/persona_measure.sh              # Both phases
#   PHASE=measure sbatch bash/persona_measure.sh # Phase 2 only
#   PHASE=erosion sbatch bash/persona_measure.sh # Phase 3 only

SCRATCH=~/scratch/halulujah
OUTPUT_DIR=${SCRATCH}/persona_experiment
CACHE_DIR=${SCRATCH}/hf_cache
PHASE="${PHASE:-both}"

module load cuda

cd "${SLURM_SUBMIT_DIR:-.}"

VENV=${SCRATCH}/venv_persona
source "${VENV}/bin/activate"

mkdir -p logs

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"
export HF_HOME="${CACHE_DIR}"

if [ "${PHASE}" = "both" ] || [ "${PHASE}" = "measure" ]; then
    echo "=== Phase 2: Fingerprint Measurement ==="
    srun python src/scripts/run_persona_experiment.py \
        --phase measure \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "Qwen/Qwen3-1.7B" \
        --cache-dir "${CACHE_DIR}"
fi

if [ "${PHASE}" = "both" ] || [ "${PHASE}" = "erosion" ]; then
    echo "=== Phase 3: Temperature Erosion ==="
    srun python src/scripts/run_persona_experiment.py \
        --phase erosion \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "Qwen/Qwen3-1.7B" \
        --cache-dir "${CACHE_DIR}"
fi

echo "Measurement complete. Results at ${OUTPUT_DIR}"
