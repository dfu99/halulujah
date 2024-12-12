#!/bin/bash
#SBATCH --job-name=phi3_demo        # Job name
#SBATCH --output=phi3_output.log    # Output file
#SBATCH --error=phi3_error.log      # Error file
#SBATCH --partition=gpu             # Partition (queue) name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --gres=gpu:1                # Number of GPUs per node
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=32G                   # Memory allocation per node
#SBATCH --time=01:00:00             # Maximum time limit (hh:mm:ss)

# Load necessary modules
module load python/3.9
module load cuda/11.8  # Adjust based on cluster setup

# Activate your virtual environment
source ~/your_virtualenv/bin/activate

# Run your Python script
python run_phi3.py
