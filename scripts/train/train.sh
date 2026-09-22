#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation
set -euo pipefail

# ======================
# Distributed Configuration
# ======================
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}               # [Required] Master node IP for multi-GPU training
MASTER_PORT=${MASTER_PORT:-22223}                   # Default rendezvous port

# ======================
# Slurm auto-configuration (overrides defaults when available)
# ======================
if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
fi

NODE_RANK=${NODE_RANK:-${SLURM_PROCID:-0}}
NNODES=${NNODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}}

# Prefer CUDA_VISIBLE_DEVICES to honor manual GPU selection before fallback detection
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VIS_DEVICES="${CUDA_VISIBLE_DEVICES// /}"
    IFS=',' read -r -a __CUDA_DEVICE_LIST <<< "$CUDA_VIS_DEVICES"
    NPROC_PER_NODE=0
    for __dev in "${__CUDA_DEVICE_LIST[@]}"; do
        if [[ -z "$__dev" ]]; then
            continue
        elif [[ "$__dev" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            __start=${BASH_REMATCH[1]}
            __end=${BASH_REMATCH[2]}
            if (( __end >= __start )); then
                NPROC_PER_NODE=$((NPROC_PER_NODE + __end - __start + 1))
            fi
        else
            NPROC_PER_NODE=$((NPROC_PER_NODE + 1))
        fi
    done
fi

if [ -z "${NPROC_PER_NODE:-}" ] || [ "$NPROC_PER_NODE" -le 0 ]; then
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        NPROC_PER_NODE=$SLURM_GPUS_ON_NODE
    else
        NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
    fi
fi

if [ "$NPROC_PER_NODE" -le 0 ]; then
    echo ">>>>> No visible GPUs detected; defaulting to 1 process"
    NPROC_PER_NODE=1
fi

if [ -z "${NNODES:-}" ] || [ "$NNODES" -le 0 ]; then
    NNODES=1
fi

# WORLD_SIZE is used to compute gradient accumulation; helpful to export for torchrun as well
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
export WORLD_SIZE
export NODE_RANK

# ======================
# Path Configuration
# ======================

# For local runs you may switch these to HF ids, e.g.:
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-facebook/VGGT-1B}"

###################################### ENV DIVIDER
OUTPUT_DIR="${OUTPUT_DIR:-./output/spatialstack_train}"              # Directory for saving checkpoints
CACHE_DIR="${CACHE_DIR:-./cache}"                                    # [TrainingArguments] Cache directory for models
mkdir -p "$OUTPUT_DIR"

# ======================
# Training Hyperparameters
# ======================
LR="${LR:-1e-5}"
MAX_STEPS="${MAX_STEPS:--1}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-12800}"
MAX_PIXELS="${MAX_PIXELS:-$((576*28*28))}"
MIN_PIXELS="${MIN_PIXELS:-$((16*28*28))}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_DROP_LAST="${DATALOADER_DROP_LAST:-false}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG-scripts/zero2_opt.json}"
REMOVE_UNUSED_COLUMNS="${REMOVE_UNUSED_COLUMNS:-false}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-0}"
DATA_SEED="${DATA_SEED:-$SEED}"
TF32="${TF32:-false}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-8}"
VIDEO_MIN_FRAMES="${VIDEO_MIN_FRAMES:-4}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-}"
total_batch_size="${TOTAL_BATCH_SIZE:-64}"
EXPECTED_TRAIN_SAMPLES="${EXPECTED_TRAIN_SAMPLES:-}"

if [ "$WORLD_SIZE" -gt 0 ]; then
    if (( total_batch_size % WORLD_SIZE != 0 )); then
        echo ">>>>> TOTAL_BATCH_SIZE=$total_batch_size must be divisible by WORLD_SIZE=$WORLD_SIZE" >&2
        exit 2
    fi
    GRADIENT_ACCUMULATION_STEPS=$(( total_batch_size / WORLD_SIZE ))
else
    GRADIENT_ACCUMULATION_STEPS=$total_batch_size
fi
if [ "$GRADIENT_ACCUMULATION_STEPS" -le 0 ]; then
    echo ">>>>> gradient accumulation would be <1; forcing to 1 (total_batch_size=$total_batch_size, world_size=$WORLD_SIZE)"
    GRADIENT_ACCUMULATION_STEPS=1
fi
effective_global_batch=$((NPROC_PER_NODE * NNODES * GRADIENT_ACCUMULATION_STEPS))
echo ">>>>> world size = $WORLD_SIZE"
echo ">>>>> per-device batch = 1"
echo ">>>>> grad accum = $GRADIENT_ACCUMULATION_STEPS"
echo ">>>>> effective global batch = $effective_global_batch"
echo ">>>>> attention implementation = $ATTN_IMPLEMENTATION"
echo ">>>>> seed/data_seed = $SEED/$DATA_SEED"
echo ">>>>> dataloader drop_last = $DATALOADER_DROP_LAST"

# ======================
# Model Configuration
# ======================
DATASETS="${DATASETS:-spar_234k%60,llava_hound_64k%60,vlm3r_scannet%60,vsi_appr_order%50}"             # [DataArguments] Dataset list
GEOMETRY_ENCODER_TYPE="${GEOMETRY_ENCODER_TYPE:-vggt}"
USE_GEOMETRY_ENCODER="${USE_GEOMETRY_ENCODER:-true}"
USE_CACHED_VGGT="${USE_CACHED_VGGT:-false}"
CONTROLLED_FUSION_CANDIDATE="${CONTROLLED_FUSION_CANDIDATE:-}"
CACHED_VGGT_MANIFEST="${CACHED_VGGT_MANIFEST:-}"
CACHED_VGGT_NUM_FRAMES="${CACHED_VGGT_NUM_FRAMES:-32}"
TUNE_MM_LLM="${TUNE_MM_LLM:-true}"
LORA_ENABLE="${LORA_ENABLE:-false}"
DATA_FLATTEN="${DATA_FLATTEN:-False}"
FEATURE_FUSION_METHOD="${FEATURE_FUSION_METHOD:-deepstack_language_add}"
GEOMETRY_FUSION_LAYERS="${GEOMETRY_FUSION_LAYERS:-0 1 2}"
GEOMETRY_ENCODER_LAYERS="${GEOMETRY_ENCODER_LAYERS:-11 17 23}"
VISION_LANGUAGE_FUSION_LAYERS="${VISION_LANGUAGE_FUSION_LAYERS:-}"

train_args=(
         --model_name_or_path "$MODEL_PATH"
         --tune_mm_llm "$TUNE_MM_LLM"
         --tune_mm_vision False
         --tune_mm_mlp False
         --dataset_use "$DATASETS"
         --output_dir "$OUTPUT_DIR"
         --cache_dir "$CACHE_DIR"
         --bf16
         --per_device_train_batch_size 1
         --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
         --learning_rate "$LR"
         --mm_projector_lr 1e-5
         --vision_tower_lr 1e-6
         --optim adamw_torch
         --model_max_length "$MODEL_MAX_LENGTH"
         --data_flatten "$DATA_FLATTEN"
         --max_pixels "$MAX_PIXELS"
         --min_pixels "$MIN_PIXELS"
         --base_interval 2
         --video_max_frames "$VIDEO_MAX_FRAMES"
         --video_min_frames "$VIDEO_MIN_FRAMES"
         --video_max_frame_pixels $((1664*28*28))
         --video_min_frame_pixels $((256*28*28))
         --num_train_epochs "$NUM_TRAIN_EPOCHS"
         --max_steps "$MAX_STEPS"
         --warmup_ratio 0.03
         --lr_scheduler_type cosine
         --weight_decay 0.01
         --logging_steps 10
         --save_strategy "$SAVE_STRATEGY"
         --save_steps "$SAVE_STEPS"
         --save_total_limit "$SAVE_TOTAL_LIMIT"
         --gradient_checkpointing
         --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
         --dataloader_drop_last "$DATALOADER_DROP_LAST"
         --remove_unused_columns "$REMOVE_UNUSED_COLUMNS"
         --group_by_modality_length true
         --seed "$SEED"
         --data_seed "$DATA_SEED"
         --tf32 "$TF32"
         --report_to none
         --use_geometry_encoder "$USE_GEOMETRY_ENCODER"
         --use_cached_vggt "$USE_CACHED_VGGT"
         --lora_enable "$LORA_ENABLE"
)

if [[ -n "$EXPECTED_TRAIN_SAMPLES" ]]; then
    train_args+=(--expected_train_samples "$EXPECTED_TRAIN_SAMPLES")
fi

if [[ -n "$DDP_FIND_UNUSED_PARAMETERS" ]]; then
    train_args+=(--ddp_find_unused_parameters "$DDP_FIND_UNUSED_PARAMETERS")
fi

if [[ -n "$DEEPSPEED_CONFIG" ]]; then
    train_args+=(--deepspeed "$DEEPSPEED_CONFIG")
fi

if [[ "${USE_CACHED_VGGT,,}" == "true" ]]; then
    if [[ "${USE_GEOMETRY_ENCODER,,}" == "true" ]]; then
        echo ">>>>> USE_CACHED_VGGT and USE_GEOMETRY_ENCODER are mutually exclusive" >&2
        exit 2
    fi
    if [[ -z "$CONTROLLED_FUSION_CANDIDATE" || -z "$CACHED_VGGT_MANIFEST" ]]; then
        echo ">>>>> cached mode requires CONTROLLED_FUSION_CANDIDATE and CACHED_VGGT_MANIFEST" >&2
        exit 2
    fi
    train_args+=(
         --controlled_fusion_candidate "$CONTROLLED_FUSION_CANDIDATE"
         --cached_vggt_manifest "$CACHED_VGGT_MANIFEST"
         --cached_vggt_num_frames "$CACHED_VGGT_NUM_FRAMES"
    )
fi

if [[ "${USE_GEOMETRY_ENCODER,,}" == "true" ]]; then
    train_args+=(
         --geometry_encoder_type "$GEOMETRY_ENCODER_TYPE"
         --geometry_encoder_path "$GEOMETRY_ENCODER_PATH"
         --feature_fusion_method "$FEATURE_FUSION_METHOD"
         --geometry_fusion_layers ${GEOMETRY_FUSION_LAYERS}
         --geometry_encoder_layers ${GEOMETRY_ENCODER_LAYERS}
    )
    if [[ -n "${VISION_LANGUAGE_FUSION_LAYERS}" ]]; then
        train_args+=(
             --vision_language_fusion_layers ${VISION_LANGUAGE_FUSION_LAYERS}
        )
    fi
fi

ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE \
         --nnodes=$NNODES \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         src/qwen_vl/train/train_qwen.py \
         "${train_args[@]}" \
         2>&1 | tee ${OUTPUT_DIR}/train.rank${NODE_RANK}.log
