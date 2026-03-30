#!/bin/bash
# RunPod One-Shot: Complete Pivot B domain experiment
#
# Usage: Paste this into a RunPod terminal (RTX 3090/4090/A6000)
#   curl -sL https://raw.githubusercontent.com/dfu99/halulujah/refactor/qwen3-temporal-rl/bash/runpod_all.sh | bash
#
# Or clone and run:
#   git clone https://github.com/dfu99/halulujah.git && cd halulujah
#   git checkout refactor/qwen3-temporal-rl
#   bash bash/runpod_all.sh
#
# Estimated: ~35 min on RTX 3090, ~$0.20

set -e

echo "=== RunPod Halulujah: Complete Domain Experiment ==="
echo "Started: $(date)"

# Setup
cd /workspace 2>/dev/null || true
if [ ! -d halulujah ]; then
    git clone https://github.com/dfu99/halulujah.git
    cd halulujah
    git checkout refactor/qwen3-temporal-rl
else
    cd halulujah
    git pull
fi

# Install deps
pip install -q torch transformers peft trl datasets accelerate \
    sentence-transformers scikit-learn matplotlib numpy 2>&1 | tail -3

export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
OUTPUT_DIR=/workspace/halulujah/results/runpod_domain

echo ""
echo "=== Phase 1: Domain Specialist Fine-Tuning ==="
echo "Training 3 LoRA adapters (physics, law, biology) on Qwen3-1.7B..."
python src/scripts/run_domain_experiment.py \
    --phase finetune \
    --output-dir "${OUTPUT_DIR}" \
    --model-name Qwen/Qwen3-1.7B \
    --epochs 3 \
    --batch-size 2

echo ""
echo "=== Phase 2: Cross-Domain Evaluation ==="
echo "Testing all models on all domains (4 models × 150 questions)..."
python src/scripts/run_domain_experiment.py \
    --phase evaluate \
    --output-dir "${OUTPUT_DIR}" \
    --model-name Qwen/Qwen3-1.7B

echo ""
echo "=== Phase 3: KL Distance Measurement ==="
echo "Computing pairwise KL divergence between domain specialists..."
python src/scripts/run_domain_experiment.py \
    --phase distance \
    --output-dir "${OUTPUT_DIR}" \
    --model-name Qwen/Qwen3-1.7B

echo ""
echo "=== Phase 4: Visualization ==="
python src/scripts/visualize_domain_results.py \
    --cross-eval "${OUTPUT_DIR}/cross_eval/cross_eval_results.json" \
    --kl-results "${OUTPUT_DIR}/domain_distance/kl_results.json" \
    --kl-hallucination "${OUTPUT_DIR}/domain_distance/kl_vs_hallucination.json" \
    --out-dir "${OUTPUT_DIR}/figures"

echo ""
echo "=== Packaging Results ==="
tar czf /workspace/halulujah_results.tar.gz \
    -C /workspace/halulujah results/runpod_domain/

echo ""
echo "=== COMPLETE ==="
echo "Finished: $(date)"
echo ""
echo "Results in: ${OUTPUT_DIR}/"
echo "Tarball:    /workspace/halulujah_results.tar.gz"
echo ""
echo "Download with: runpodctl recv /workspace/halulujah_results.tar.gz"
echo "Or via pod UI: Files → /workspace/halulujah_results.tar.gz"

# Print summary if cross_eval exists
if [ -f "${OUTPUT_DIR}/cross_eval/cross_eval_results.json" ]; then
    echo ""
    python -c "
import json
with open('${OUTPUT_DIR}/cross_eval/cross_eval_results.json') as f:
    d = json.load(f)
cm = d['confusion_matrix']
print('=== CROSS-DOMAIN ACCURACY MATRIX ===')
header = f'{\"Model\":<25}' + ''.join(f'{d:>12}' for d in cm['domains'])
print(header)
print('-' * len(header))
for m in cm['model_names']:
    row = f'{m:<25}'
    for d in cm['domains']:
        acc = cm['matrix'][m][d]
        row += f'{acc:>11.0%} '
    print(row)
print()
for name, stats in cm['summary'].items():
    if 'accuracy_drop' in stats:
        print(f'  {name}: in={stats[\"in_domain_accuracy\"]:.0%}, cross={stats[\"cross_domain_accuracy\"]:.0%}, drop={stats[\"accuracy_drop\"]:+.0%}')
"
fi

if [ -f "${OUTPUT_DIR}/domain_distance/kl_vs_hallucination.json" ]; then
    echo ""
    python -c "
import json
with open('${OUTPUT_DIR}/domain_distance/kl_vs_hallucination.json') as f:
    d = json.load(f)
print('=== KL DISTANCE vs ACCURACY DROP ===')
for c in d:
    print(f'  {c[\"model\"]} on {c[\"question_domain\"]}: KL={c[\"kl_distance\"]:.2f}, drop={c[\"accuracy_drop\"]:.1%}')
"
fi
