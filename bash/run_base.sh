#!/bin/bash
#SBATCH -J phi3_base
#SBATCH -A gts-yke8
#SBATCH -N1 --gres=gpu:RTX_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/storage/home/hcoda1/6/dfu71/scratch/halulujah/logs/halulu/slurm_%j.out

# Load necessary modules
module load python/3.10.10
module load cuda

cd ~/scratch/halulujah

# Activate virtual environment
source venv_inference/bin/activate

# Run Python script
srun python src/finetune/run_phi3.py
