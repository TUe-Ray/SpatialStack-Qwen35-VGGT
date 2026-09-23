#!/usr/bin/env bash
#SBATCH --job-name=q35-controlled-vsi
#SBATCH --partition=gpu_a100
#SBATCH --account=tesei2748
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=20:00:00

set -euo pipefail

: "${CONTROLLED_FUSION_CANDIDATE:?Select a_premerger_cross_attn or b_llm_add}"
: "${CHECKPOINT_PATH:?Set the exact trained checkpoint directory; no automatic latest selection}"
: "${CACHED_VGGT_MANIFEST:?Set the exact candidate VSI-Bench manifest}"
: "${EXPECTED_MANIFEST_SHA256:?Pin the candidate VSI-Bench manifest SHA256}"
: "${EXPECTED_GIT_COMMIT:?Pin the approved evaluation source commit}"
: "${CONTROLLED_EVAL_ROOT:?Set a fresh scratch output root}"

case "$CONTROLLED_FUSION_CANDIDATE" in
    a_premerger_cross_attn|b_llm_add) ;;
    *) echo "Unsupported controlled candidate: $CONTROLLED_FUSION_CANDIDATE" >&2; exit 2 ;;
esac

CHECKOUT_ROOT="${CHECKOUT_ROOT:-/gpfs/home4/geusdd/shuang/SpatialStack-heldout}"
ENV_ROOT="${ENV_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/env-fast-overlay}"
EVAL_DEPS_ROOT="${EVAL_DEPS_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/eval-deps}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/scratch-shared/geusdd/SpatialStackQwen35/models/Qwen3.5-4B}"
MEDIA_ROOT="${MEDIA_ROOT:-/scratch-shared/geusdd/VLM3R/hf_cache/vsibench}"
PARQUET_ROOT="${PARQUET_ROOT:-/scratch-shared/geusdd/VLM3R/data/vsibench}"
RUN_ROOT="$CONTROLLED_EVAL_ROOT/${CONTROLLED_FUSION_CANDIDATE}-${SLURM_JOB_ID}"

for value in "$CHECKPOINT_PATH" "$CACHED_VGGT_MANIFEST" "$CONTROLLED_EVAL_ROOT"; do
    if [[ "$value" != /* ]]; then
        echo "Checkpoint, manifest, and evaluation root must be absolute paths: $value" >&2
        exit 2
    fi
done
if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite existing evaluation: $RUN_ROOT" >&2
    exit 2
fi
git -C "$CHECKOUT_ROOT" cat-file -e "$EXPECTED_GIT_COMMIT^{commit}"
actual_manifest_sha=$(sha256sum "$CACHED_VGGT_MANIFEST" | cut -d' ' -f1)
if [[ "$actual_manifest_sha" != "$EXPECTED_MANIFEST_SHA256" ]]; then
    echo "VSI-Bench manifest SHA256 mismatch: $CACHED_VGGT_MANIFEST" >&2
    exit 2
fi
test -s "$BASE_MODEL_PATH/config.json"
test -d "$EVAL_DEPS_ROOT/sacrebleu"
mkdir -p "$RUN_ROOT/source" "$RUN_ROOT/triton"
git -C "$CHECKOUT_ROOT" archive "$EXPECTED_GIT_COMMIT" | tar -xf - -C "$RUN_ROOT/source"

module load 2025
module load CUDA/12.9.1
export PROJECT_ROOT="$RUN_ROOT/source"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$EVAL_DEPS_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PATH="$ENV_ROOT/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export HF_HOME=/scratch-shared/geusdd/VLM3R/hf_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="$RUN_ROOT/triton"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FLA_CACHE_MODE=default
export FLA_CONFIG_DIR="$PROJECT_ROOT/scripts/profiling/fla_configs_a100"
export NCCL_NVLS_ENABLE=0
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

cd "$PROJECT_ROOT"
"$ENV_ROOT/bin/python" scripts/evaluation/validate_controlled_vsibench.py \
    --candidate "$CONTROLLED_FUSION_CANDIDATE" \
    --checkpoint "$CHECKPOINT_PATH" \
    --base-model "$BASE_MODEL_PATH" \
    --manifest "$CACHED_VGGT_MANIFEST" \
    --media-root "$MEDIA_ROOT" \
    --parquet "$PARQUET_ROOT/test_pruned.parquet" \
    --parquet "$PARQUET_ROOT/test_debiased.parquet"
"$ENV_ROOT/bin/python" - <<'PY'
import os
import torch
count = torch.cuda.device_count()
if not torch.cuda.is_available() or count != 4:
    raise SystemExit(f"Controlled VSI-Bench requires four visible GPUs, got {count}; CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
print(f"CONTROLLED_VSI_GPU_PREFLIGHT gpus={count}", flush=True)
PY
"$ENV_ROOT/bin/python" scripts/train/validate_qwen35_runtime.py \
    --attention-implementation flash_attention_2 \
    --require-fast-linear-attention --require-cuda

MODEL_ARGS_BASE="pretrained=$CHECKPOINT_PATH,use_flash_attention_2=true,min_pixels=12544,max_pixels=451584,max_num_frames=32,max_length=12800,disable_thinking=true,cached_vggt_manifest=$CACHED_VGGT_MANIFEST,cached_vggt_dataset=vsibench,cached_vggt_data_root=$MEDIA_ROOT"
echo "CONTROLLED_VSI_CONFIG job=$SLURM_JOB_ID candidate=$CONTROLLED_FUSION_CANDIDATE code=$EXPECTED_GIT_COMMIT checkpoint=$CHECKPOINT_PATH manifest_sha=$actual_manifest_sha samples=5130 videos=288 frames=32 gpus=4 output=$RUN_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_eval() {
    local result_root="$1"
    shift
    "$ENV_ROOT/bin/python" -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=4 \
        --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
        -m lmms_eval \
        --model qwen3_5 --model_args "$MODEL_ARGS_BASE" \
        --tasks vsibench_snellius --batch_size 1 \
        --output_path "$result_root" --verbosity INFO --log_samples "$@"
}

run_eval "$RUN_ROOT/smoke" --limit 4
echo "CONTROLLED_VSI_SMOKE candidate=$CONTROLLED_FUSION_CANDIDATE status=ok"
run_eval "$RUN_ROOT/full"
echo "CONTROLLED_VSI_COMPLETE candidate=$CONTROLLED_FUSION_CANDIDATE status=ok output=$RUN_ROOT/full"
