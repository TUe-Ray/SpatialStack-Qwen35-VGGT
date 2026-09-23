#!/usr/bin/env bash
#SBATCH --job-name=q35-base-vsi
#SBATCH --partition=gpu_a100
#SBATCH --account=tesei2748
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=08:00:00

set -euo pipefail

: "${BASE_EVAL_ROOT:?Set a scratch output root}"
: "${EXPECTED_GIT_COMMIT:?Pin the evaluation source commit}"

CHECKOUT_ROOT="${CHECKOUT_ROOT:-/gpfs/home4/geusdd/shuang/SpatialStack-heldout}"
ENV_ROOT="${ENV_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/env-fast-overlay}"
EVAL_DEPS_ROOT="${EVAL_DEPS_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/eval-deps}"
MODEL_PATH="${MODEL_PATH:-/scratch-shared/geusdd/SpatialStackQwen35/models/Qwen3.5-4B}"
RUN_ROOT="$BASE_EVAL_ROOT/job-$SLURM_JOB_ID"

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite existing evaluation: $RUN_ROOT" >&2
    exit 2
fi
if ! git -C "$CHECKOUT_ROOT" cat-file -e "$EXPECTED_GIT_COMMIT^{commit}"; then
    echo "Pinned evaluation code commit is unavailable: $EXPECTED_GIT_COMMIT" >&2
    exit 2
fi
test -s "$MODEL_PATH/config.json"
test -d "$EVAL_DEPS_ROOT/sacrebleu"
test -s /scratch-shared/geusdd/VLM3R/data/vsibench/test_pruned.parquet
test -s /scratch-shared/geusdd/VLM3R/data/vsibench/test_debiased.parquet
test -d /scratch-shared/geusdd/VLM3R/hf_cache/vsibench

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
export MODEL_PATH
export MODEL_IMPL=qwen3_5
export BENCHMARKS=vsibench_snellius
export MODEL_ARGS_BASE="pretrained=$MODEL_PATH,use_flash_attention_2=true,min_pixels=12544,max_pixels=451584,max_num_frames=32,max_length=12800,disable_thinking=true"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export NCCL_NVLS_ENABLE=0

cd "$PROJECT_ROOT"
"$ENV_ROOT/bin/python" - <<'PY'
import os
import torch

count = torch.cuda.device_count()
if not torch.cuda.is_available() or count != 4:
    raise SystemExit(
        f"Qwen3.5 base eval requires four visible GPUs, got {count}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
    )
print(f"BASE_VSI_GPU_PREFLIGHT gpus={count} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
PY
"$ENV_ROOT/bin/python" scripts/train/validate_qwen35_runtime.py \
    --attention-implementation flash_attention_2 \
    --require-fast-linear-attention \
    --require-cuda

echo "BASE_VSI_CONFIG job=$SLURM_JOB_ID code=$EXPECTED_GIT_COMMIT model=$MODEL_PATH samples=5130 frames=32 gpus=4 output=$RUN_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_eval() {
    local result_root="$1"
    shift
    "$ENV_ROOT/bin/python" -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=4 \
        --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
        -m lmms_eval \
        --model "$MODEL_IMPL" \
        --model_args "$MODEL_ARGS_BASE" \
        --tasks "$BENCHMARKS" \
        --batch_size 1 \
        --output_path "$result_root" \
        --verbosity INFO \
        --log_samples \
        "$@"
}

run_eval "$RUN_ROOT/smoke" --limit 4
echo "BASE_VSI_SMOKE status=ok"

run_eval "$RUN_ROOT/full"
echo "BASE_VSI_COMPLETE status=ok output=$RUN_ROOT/full"
