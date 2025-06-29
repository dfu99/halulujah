#!/bin/bash
#SBATCH -J phi3_ft
#SBATCH -A gts-yke8
#SBATCH -N1 --ntasks=1 --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

# Load necessary modules
module load python/3.10.10
module load cuda

# Change to working directory
cd ~/scratch/halulujah

# Activate virtual environment
source venv_ft/bin/activate

export RANK=0
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=0
export MASTER_PORT=12355
export MASTER_ADDR="localhost"

# Run Python script
srun python src/finetune/finetune.py
