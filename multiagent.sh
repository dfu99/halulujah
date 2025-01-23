#!/bin/bash
#SBATCH -J phi3_ft
#SBATCH -A gts-yke8
#SBATCH -N2 --ntasks=2 --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Load necessary modules
module load python/3.10.10
module load cuda

# Activate virtual environment
source ~/personal/halulujah/venv_ft/bin/activate

export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=0

# Run Python script
srun python src/finetune.py
