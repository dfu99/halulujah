#!/bin/bash
#SBATCH -J multiagent
#SBATCH -A gts-yke8
#SBATCH --nodes=2
#SBATCH --gres=gpu:A100:2
#SBATCH -C A100-80GB
#SBATCH --mem=64G

# Load necessary modules
module load cuda
module load py-mpi4py

# Activate virtual environment
source ~/scratch/halulujah/venv_mpi/bin/activate

# Change to working directory
cd ~/scratch/halulujah

export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=0

# Run Python script
srun --ntasks=2 python src/multiagent/example3.py
