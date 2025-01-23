#!/bin/bash
#SBATCH -J phi3_base
#SBATCH -A gts-yke8
#SBATCH -N1 --gres=gpu:RTX_6000:1
#SBATCH --cpus-per-task=6

# Load necessary modules
module load python/3.10.10
module load cuda

cd ~/scratch/halulujah

# Activate virtual environment
source venv_inference/bin/activate

# Run Python script
srun python src/run_phi3.py
