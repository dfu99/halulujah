#!/bin/bash
#SBATCH -J qwen3_exam
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --time=48:00:00

HELD_OUT_YEAR="${HELD_OUT_YEAR:-2022}"
MODEL_PATH="${MODEL_PATH:-outputs/sft_lora_holdout_${HELD_OUT_YEAR}}"

module load python cuda

cd "${SLURM_SUBMIT_DIR:-.}"
source "${VENV_PATH:-venv/bin/activate}"

srun python src/scripts/run_eval.py exam \
    --config configs/experiment.yaml \
    --model-path "${MODEL_PATH}" \
    --held-out-year ${HELD_OUT_YEAR}
