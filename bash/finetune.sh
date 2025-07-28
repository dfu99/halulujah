#!/bin/bash
#SBATCH -J phi3_ft
#SBATCH -A gts-yke8
#SBATCH --nodes=2
#SBATCH --gres=gpu:A100:2
#SBATCH -C A100-80GB
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=/storage/home/hcoda1/6/dfu71/scratch/halulujah/logs/slurm-%j.out

# Load necessary modules
module load python/3.10.10
module load cuda

# Change to working directory
cd ~/scratch/halulujah

# Activate virtual environment
source venv_ft/bin/activate

## For single GPU fine tuning
#export RANK=0
#export WORLD_SIZE=$SLURM_NTASKS
#export LOCAL_RANK=0
#export MASTER_PORT=12355
#export MASTER_ADDR="localhost"

# Run Python script
#srun python src/finetune/finetune.py

## Multiple GPUs with DeepSpeed
deepspeed --num_gpus=2 src/finetune/finetune.py
