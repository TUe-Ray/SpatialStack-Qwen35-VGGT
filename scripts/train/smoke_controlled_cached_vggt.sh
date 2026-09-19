#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen3.5-4B checkpoint}"
: "${DATASETS:?Set DATASETS to a one-scene dataset alias}"
: "${CACHED_VGGT_MANIFEST:?Set CACHED_VGGT_MANIFEST to the smoke manifest}"
: "${CONTROLLED_FUSION_CANDIDATE:?Set CONTROLLED_FUSION_CANDIDATE}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a temporary smoke checkpoint directory}"

case "$CONTROLLED_FUSION_CANDIDATE" in
    a_premerger_cross_attn|b_llm_add) ;;
    *) echo "Invalid controlled candidate: $CONTROLLED_FUSION_CANDIDATE" >&2; exit 2 ;;
esac

if [[ "${MAX_STEPS:-2}" -lt 1 || "${MAX_STEPS:-2}" -gt 2 ]]; then
    echo "Smoke wrapper permits only one or two optimizer steps" >&2
    exit 2
fi

export USE_GEOMETRY_ENCODER=false
export USE_CACHED_VGGT=true
export TUNE_MM_LLM=false
export LORA_ENABLE=true
export CACHED_VGGT_NUM_FRAMES="${CACHED_VGGT_NUM_FRAMES:-32}"
export MAX_STEPS="${MAX_STEPS:-2}"
export SAVE_STEPS="${SAVE_STEPS:-1}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-1}"

exec bash scripts/train/train.sh
