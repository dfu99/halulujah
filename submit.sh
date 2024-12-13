#!/bin/bash
#SBATCH -J phi3_demo
#SBATCH -A gts-yke8
#SBATCH -N1 --gres=gpu:1
#SBATCH --cpus-per-task=8

# Load necessary modules
module load python/3.10.10
module load cuda

# Activate virtual environment
source ~/personal/halulujah/venv/bin/activate

# Run Python script
python src/run_phi3.py
