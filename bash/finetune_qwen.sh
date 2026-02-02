#!/bin/bash
#SBATCH -J qwen3_sft
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --time=48:00:00

# Configurable held-out year (pass via: sbatch --export=HELD_OUT_YEAR=2022 bash/finetune_qwen.sh)
HELD_OUT_YEAR="${HELD_OUT_YEAR:-2022}"

module load python cuda

cd "${SLURM_SUBMIT_DIR:-.}"
source "${VENV_PATH:-venv/bin/activate}"

srun python src/scripts/run_finetune.py \
    --config configs/experiment.yaml \
    --held-out-year ${HELD_OUT_YEAR}
