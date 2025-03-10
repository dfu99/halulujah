#!/bin/bash
#SBATCH -J rl
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00

# Load necessary modules
module load cuda

# Activate virtual environment
source ~/scratch/halulujah/venv_RL/bin/activate

# Change to working directory
cd ~/scratch/halulujah

# Run Python script
srun python src/rl/example2.py
