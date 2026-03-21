#!/bin/bash
#SBATCH -J blog_feasibility
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/blog_feasibility_%j.log
#SBATCH --error=logs/blog_feasibility_%j.err
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.fu@emory.edu

# Blog Authorship Feasibility Test
# No GPU needed — just embeddings + sklearn
#
# Data already downloaded to ~/scratch/halulujah/blog_corpus/blogs/blogtext.csv
# Usage: sbatch bash/blog_feasibility.sh

SCRATCH=~/scratch/halulujah
DATA_DIR=${SCRATCH}/blog_corpus/blogs
OUT_DIR=${SCRATCH}/results/blog_feasibility

module load python

cd "${SLURM_SUBMIT_DIR:-.}"

# Create venv on scratch if it doesn't exist
VENV=${SCRATCH}/venv_persona
if [ ! -d "${VENV}" ]; then
    echo "Creating venv at ${VENV}"
    python -m venv "${VENV}"
    source "${VENV}/bin/activate"
    pip install --upgrade pip
    pip install sentence-transformers scikit-learn matplotlib numpy
else
    source "${VENV}/bin/activate"
fi

mkdir -p ${OUT_DIR}
mkdir -p logs

srun python src/scripts/run_blog_author_feasibility.py \
    --data-dir "${DATA_DIR}" \
    --out-dir "${OUT_DIR}" \
    --num-authors 10 \
    --min-posts 50 \
    --max-posts 100

echo "Results saved to ${OUT_DIR}"
