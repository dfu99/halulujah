#!/bin/bash
#SBATCH -J phi3_inference
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=/storage/home/hcoda1/6/dfu71/p-yke8-0/halulujah/logs/slurm-%j.out
#SBATCH --time=48:00:00

# Load necessary modules
module load python
module load cuda

# Change to working directory
cd ~/p-yke8-0/halulujah

# Activate virtual environment
source ~/scratch/venv_ft/bin/activate

# Run Python script
srun python src/finetune/take_exam.py

