#!/bin/bash
#SBATCH -J qwen3_rl
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=80G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --time=12:00:00

HELD_OUT_YEAR="${HELD_OUT_YEAR:-2022}"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/sft_lora_holdout_${HELD_OUT_YEAR}}"
ORACLE_DIR="${ORACLE_DIR:-outputs/oracle_${HELD_OUT_YEAR}}"

module load cuda

cd "${SLURM_SUBMIT_DIR:-.}"
source "${VENV_PATH:-venv/bin/activate}"

srun python src/scripts/run_rl.py \
    --config configs/experiment.yaml \
    --sft-adapter "${SFT_ADAPTER}" \
    --oracle-dir "${ORACLE_DIR}" \
    --held-out-year ${HELD_OUT_YEAR}
