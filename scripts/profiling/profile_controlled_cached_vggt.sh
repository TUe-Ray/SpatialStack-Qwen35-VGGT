#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to Qwen3.5-4B}"
: "${DATASETS:?Set DATASETS to the profiling annotation alias}"
: "${CACHED_VGGT_MANIFEST:?Set CACHED_VGGT_MANIFEST}"
: "${CONTROLLED_FUSION_CANDIDATE:?Set CONTROLLED_FUSION_CANDIDATE}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to disposable scratch output}"

case "$CONTROLLED_FUSION_CANDIDATE" in
    a_premerger_cross_attn|b_llm_add) ;;
    *) echo "Invalid candidate: $CONTROLLED_FUSION_CANDIDATE" >&2; exit 2 ;;
esac

export USE_GEOMETRY_ENCODER=false
export USE_CACHED_VGGT=true
export TUNE_MM_LLM=false
export LORA_ENABLE=true
export CACHED_VGGT_NUM_FRAMES=32
export MAX_STEPS="${MAX_STEPS:-50}"
export SAVE_STEPS=1000000
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-4}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export CONTROLLED_PROFILE=1
export CONTROLLED_PROFILE_SKIP_SAVE=1

exec bash scripts/train/train.sh
