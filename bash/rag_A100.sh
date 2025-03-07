#!/bin/bash
#SBATCH -J rag
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH -C A100-80GB
#SBATCH --mem=80G

# Load necessary modules
module load cuda

# Activate virtual environment
source ~/scratch/halulujah/venv_RAG/bin/activate

# Change to working directory
cd ~/scratch/halulujah

# Run Python script
srun python src/rag/rag_multidoc.py
