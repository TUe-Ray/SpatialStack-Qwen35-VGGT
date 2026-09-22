#!/usr/bin/env bash
# Canonical held-out SFT data/batch/checkpoint controls inherited from SpatialFocus.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to the validated Qwen3.5-4B checkpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR on scratch storage}"
: "${CACHED_VGGT_MANIFEST:?Set CACHED_VGGT_MANIFEST for the selected candidate}"
: "${CONTROLLED_FUSION_CANDIDATE:?Set CONTROLLED_FUSION_CANDIDATE}"

VLM3R_DATA_ROOT="${VLM3R_DATA_ROOT:-/scratch-shared/geusdd/VLM3R/data/vlm3r}"
VLM3R_ANNOTATION_ROOT="${VLM3R_ANNOTATION_ROOT:-$VLM3R_DATA_ROOT/VLM-3R-DATA/vsibench_train}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FLA_CACHE_MODE=default
export FLA_CONFIG_DIR="$REPO_ROOT/scripts/profiling/fla_configs_a100"

export VLM3R_SCANNET_ANNOTATION="${VLM3R_SCANNET_ANNOTATION:-$VLM3R_ANNOTATION_ROOT/merged_qa_scannet_train.json}"
export VLM3R_SCANNETPP_ANNOTATION="${VLM3R_SCANNETPP_ANNOTATION:-$VLM3R_ANNOTATION_ROOT/merged_qa_scannetpp_train.json}"
export VLM3R_ROUTEPLAN_ANNOTATION="${VLM3R_ROUTEPLAN_ANNOTATION:-$VLM3R_ANNOTATION_ROOT/merged_qa_route_plan_train.json}"
export VLM3R_MEDIA_ROOT="${VLM3R_MEDIA_ROOT:-$VLM3R_DATA_ROOT}"

case "$CONTROLLED_FUSION_CANDIDATE" in
    a_premerger_cross_attn)
        CACHED_VGGT_ROOT="${CACHED_VGGT_ROOT:-/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23}"
        ;;
    b_llm_add)
        CACHED_VGGT_ROOT="${CACHED_VGGT_ROOT:-/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt}"
        ;;
    *)
        echo "Unsupported controlled candidate: $CONTROLLED_FUSION_CANDIDATE" >&2
        exit 2
        ;;
esac
export CACHED_VGGT_ROOT

# These are fixed experiment controls, not tuning defaults.
export DATASETS="vlm3r_scannet,vlm3r_scannetpp,vlm3r_routeplan"
export EXPECTED_TRAIN_SAMPLES=207658
export TOTAL_BATCH_SIZE=128
export NUM_TRAIN_EPOCHS=1
export MAX_STEPS=-1
export SAVE_STRATEGY=steps
export SAVE_STEPS=100
# drop_last matches the prior SpatialFocus runs: 1,622 optimizer steps and
# therefore 16 periodic checkpoints at interval 100.
export SAVE_TOTAL_LIMIT=20
export CACHED_VGGT_NUM_FRAMES=32
export VIDEO_MAX_FRAMES=32
export VIDEO_MIN_FRAMES=32
export USE_GEOMETRY_ENCODER=false
export USE_CACHED_VGGT=true
export TUNE_MM_LLM=false
export LORA_ENABLE=true
export DATA_FLATTEN=False
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2_opt.json}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
# Preserve the canonical SpatialFocus sample order and last-batch behavior.
export SEED=42
export DATA_SEED=42
export DATALOADER_DROP_LAST=true
# Both candidates exercised every trainable LoRA/fusion parameter in the
# 50-step DDP smokes. Avoid the unnecessary per-step autograd graph traversal.
export DDP_FIND_UNUSED_PARAMETERS=false

if [[ "${CHECK_ONLY:-0}" != "1" ]]; then
    "$PYTHON_BIN" scripts/train/validate_qwen35_runtime.py \
        --attention-implementation "$ATTN_IMPLEMENTATION" \
        --require-fast-linear-attention \
        --deepspeed-config "$DEEPSPEED_CONFIG" \
        --require-cuda
fi

"$PYTHON_BIN" scripts/data/validate_controlled_sft_inputs.py \
    --dataset vlm3r_scannet "$VLM3R_SCANNET_ANNOTATION" "$VLM3R_MEDIA_ROOT" 51779 6cf0368fc34124cd9a3c60077a84704f04a0829bf9ee8296d35bb8242fd9df1e \
    --dataset vlm3r_scannetpp "$VLM3R_SCANNETPP_ANNOTATION" "$VLM3R_MEDIA_ROOT" 151775 00d9de17925ffdf530941d621e70c4855d3f329ad51063ec6648bc8160b135cc \
    --dataset vlm3r_routeplan "$VLM3R_ROUTEPLAN_ANNOTATION" "$VLM3R_MEDIA_ROOT" 4104 dbefc6f768614c10ee839bb35786f9f4df12b92691875e7482fa47ded01ba93b \
    --expected-total "$EXPECTED_TRAIN_SAMPLES" \
    --manifest "$CACHED_VGGT_MANIFEST" \
    --sidecar-root "$CACHED_VGGT_ROOT" \
    --num-frames "$CACHED_VGGT_NUM_FRAMES"

PREVIEW_NNODES="${NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
PREVIEW_NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
WORLD_SIZE_PREVIEW="${WORLD_SIZE:-$((PREVIEW_NNODES * PREVIEW_NPROC_PER_NODE))}"
if (( TOTAL_BATCH_SIZE % WORLD_SIZE_PREVIEW != 0 )); then
    echo "TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE is not divisible by WORLD_SIZE=$WORLD_SIZE_PREVIEW" >&2
    exit 2
fi
echo "CONTROLLED_SFT_CONFIG samples=$EXPECTED_TRAIN_SAMPLES datasets=$DATASETS global_batch=$TOTAL_BATCH_SIZE world_size=$WORLD_SIZE_PREVIEW grad_accum=$((TOTAL_BATCH_SIZE / WORLD_SIZE_PREVIEW)) optimizer_steps_per_epoch=1622 save_steps=$SAVE_STEPS seed=$SEED data_seed=$DATA_SEED drop_last=$DATALOADER_DROP_LAST attention=$ATTN_IMPLEMENTATION"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
    echo "CHECK_ONLY=1: validation complete; training was not launched"
    exit 0
fi

exec bash scripts/train/train.sh
