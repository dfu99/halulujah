#!/bin/bash
# RunPod Revision Experiment — Full protocol comparison
#
# Runs on RunPod community cloud (RTX A5000 recommended, ~$0.27/hr)
#
# Protocols tested:
#   1. full-cot:    Share entire chain of thought (maximally coupled)
#   2. answer-only: Share only final answer + 1-sentence rationale
#   3. structured:  Share answer + confidence + key reasoning summary
#
# Workload per protocol:
#   - 10 solo baselines × 50 questions = 500 solo evaluations
#   - 100 pairs (90 cross-domain + 10 same-domain) × 50 questions = 5,000 collab evaluations
#   - Total per protocol: 5,500 evaluations × 3 rounds = 16,500 inference calls
#   - Total across 3 protocols: 49,500 inference calls + KL divergence
#
# Usage:
#   bash bash/runpod_revision.sh                      # All phases + all protocols
#   PHASE=distance bash bash/runpod_revision.sh       # Phase 3 only (KL divergence)
#   PHASE=collaborate bash bash/runpod_revision.sh    # Phase 4 only (3 protocols)
#   PROTOCOL=answer-only bash bash/runpod_revision.sh # Single protocol

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/halulujah/results/revision}"
CACHE_DIR="${CACHE_DIR:-/workspace/hf_cache}"
PHASE="${PHASE:-all}"
PROTOCOL="${PROTOCOL:-all}"
MODEL="Qwen/Qwen3-1.7B"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export HF_HOME="${CACHE_DIR}"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

echo "=== RunPod Revision Experiment ==="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Output: ${OUTPUT_DIR}"
echo "Phase: ${PHASE}"
echo "Protocol: ${PROTOCOL}"
echo ""

# Phase 3: KL divergence
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "distance" ]; then
    echo "=== Phase 3: Pairwise KL Divergence ==="
    python src/scripts/run_domain_experiment.py \
        --phase distance \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "${MODEL}" \
        --cache-dir "${CACHE_DIR}" \
        --extended
    echo "Phase 3 done: $(date)"
fi

# Phase 4: Collaboration with protocol comparison
if [ "${PHASE}" = "all" ] || [ "${PHASE}" = "collaborate" ]; then
    PROTOCOLS=("full-cot" "answer-only" "structured")
    if [ "${PROTOCOL}" != "all" ]; then
        PROTOCOLS=("${PROTOCOL}")
    fi

    for proto in "${PROTOCOLS[@]}"; do
        echo ""
        echo "=== Phase 4: Collaboration — protocol=${proto} ==="
        echo "Started: $(date)"

        PROTO_DIR="${OUTPUT_DIR}/collaboration_${proto}"
        mkdir -p "${PROTO_DIR}"

        python src/scripts/run_domain_experiment.py \
            --phase collaborate \
            --output-dir "${OUTPUT_DIR}" \
            --model-name "${MODEL}" \
            --cache-dir "${CACHE_DIR}" \
            --collab-rounds 3 \
            --collab-questions 50 \
            --include-same-domain \
            --collab-protocol "${proto}" \
            --extended

        # Move results to protocol-specific directory
        if [ -f "${OUTPUT_DIR}/collaboration/collab_results.json" ]; then
            cp "${OUTPUT_DIR}/collaboration/collab_results.json" "${PROTO_DIR}/collab_results.json"
            cp "${OUTPUT_DIR}/collaboration/collab_summary.json" "${PROTO_DIR}/collab_summary.json"
        fi

        echo "Protocol ${proto} done: $(date)"
    done
fi

echo ""
echo "=== COMPLETE ==="
echo "Finished: $(date)"
echo "Results in: ${OUTPUT_DIR}"
echo ""
echo "Protocol results:"
for d in "${OUTPUT_DIR}"/collaboration_*/; do
    [ -d "$d" ] && echo "  $d"
done
