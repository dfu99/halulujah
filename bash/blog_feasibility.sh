#!/bin/bash
#SBATCH -J blog_feasibility
#SBATCH -A gts-yke8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --time=2:00:00

# Blog Authorship Feasibility Test
# No GPU needed — just embeddings + sklearn
#
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

# Download Blog Authorship Corpus if not present
if [ ! -d "${DATA_DIR}" ]; then
    echo "Downloading Blog Authorship Corpus..."
    mkdir -p ${SCRATCH}/blog_corpus
    cd ${SCRATCH}/blog_corpus
    # Kaggle dataset: https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus
    # Requires kaggle CLI configured with API token
    kaggle datasets download -d rtatman/blog-authorship-corpus
    unzip -o blog-authorship-corpus.zip -d blogs/
    rm -f blog-authorship-corpus.zip
    cd "${SLURM_SUBMIT_DIR:-.}"
    echo "Blog corpus ready at ${DATA_DIR}"
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
echo "Copy figure: scp dfu71@login-phoenix.pace.gatech.edu:${OUT_DIR}/blog_author_feasibility.png ."
