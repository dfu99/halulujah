#!/bin/bash
#SBATCH -J domain_ft
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/domain_ft_%j.log
#SBATCH --error=logs/domain_ft_%j.err
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

# Domain Cross-Hallucination — Phase 1: Per-domain LoRA fine-tuning
# Fine-tunes 3 domain specialists (physics, law, biology) on MMLU subsets.

SCRATCH=~/scratch/halulujah
OUTPUT_DIR=${SCRATCH}/domain_experiment
CACHE_DIR=${SCRATCH}/hf_cache

module load cuda

cd "${SLURM_SUBMIT_DIR:-.}"

VENV=${SCRATCH}/venv_persona
source "${VENV}/bin/activate"

mkdir -p logs ${OUTPUT_DIR}

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"
export HF_HOME="${CACHE_DIR}"

srun python src/scripts/run_domain_experiment.py \
    --phase finetune \
    --output-dir "${OUTPUT_DIR}" \
    --model-name "Qwen/Qwen3-1.7B" \
    --cache-dir "${CACHE_DIR}" \
    --epochs 3

echo "Phase 1 complete. Adapters at ${OUTPUT_DIR}"
