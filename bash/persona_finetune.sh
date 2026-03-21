#!/bin/bash
#SBATCH -J persona_ft
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/persona_ft_%j.log
#SBATCH --error=logs/persona_ft_%j.err
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

# Persona Fingerprinting — Phase 1: Per-author LoRA fine-tuning
#
# Fine-tunes 5 separate LoRA adapters on Qwen3-1.7B, one per blog author.
# Data and models stored on scratch.
#
# Usage: sbatch bash/persona_finetune.sh

SCRATCH=~/scratch/halulujah
DATA_CSV=${SCRATCH}/blog_corpus/blogs/blogtext.csv
OUTPUT_DIR=${SCRATCH}/persona_experiment
CACHE_DIR=${SCRATCH}/hf_cache

module load cuda

cd "${SLURM_SUBMIT_DIR:-.}"

# Use persona venv (created by blog_feasibility.sh), add training deps
VENV=${SCRATCH}/venv_persona
if [ ! -d "${VENV}" ]; then
    echo "Creating venv at ${VENV}"
    module load python
    python -m venv "${VENV}"
fi
source "${VENV}/bin/activate"

# Install training dependencies
pip install -q torch transformers peft trl datasets accelerate \
    sentence-transformers scikit-learn matplotlib 2>&1 | tail -1

mkdir -p logs ${OUTPUT_DIR}

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"
export HF_HOME="${CACHE_DIR}"

srun python src/scripts/run_persona_experiment.py \
    --phase finetune \
    --data-csv "${DATA_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-name "Qwen/Qwen3-1.7B" \
    --cache-dir "${CACHE_DIR}" \
    --num-authors 5 \
    --min-posts 200 \
    --epochs 3

echo "Phase 1 complete. Adapters at ${OUTPUT_DIR}"
