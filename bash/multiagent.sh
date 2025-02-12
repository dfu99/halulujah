#!/bin/bash
#SBATCH -J multiagent
#SBATCH -A gts-yke8
#SBATCH -N2 --ntasks=2 --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/output_%j.log  # Log output per job ID
#SBATCH --error=logs/error_%j.log    # Log errors per job ID

# Load necessary modules
module load python/3.10.10
module load cuda

# Activate virtual environment
source ~/personal/halulujah/venv_ft/bin/activate

export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=0

# Run Python script
srun --ntasks=2 python src/multiagent/example2.py
