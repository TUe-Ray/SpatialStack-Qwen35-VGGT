#!/usr/bin/env bash
#SBATCH --job-name=q35-fast-env
#SBATCH --account=tesei2748
#SBATCH --partition=rome
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/geusdd/SpatialStackQwen35/build-logs/%x_%j.out
#SBATCH --error=/scratch-shared/geusdd/SpatialStackQwen35/build-logs/%x_%j.err

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/home4/geusdd/shuang/SpatialStack-heldout}"
ENV_ROOT="${ENV_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/env-fast-overlay}"
BUILD_ROOT="${BUILD_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35}"
PYTHON_BIN="$ENV_ROOT/bin/python"

cd "$REPO_ROOT"
module load 2025
module load CUDA/12.9.1

test -x "$PYTHON_BIN"
mkdir -p "$BUILD_ROOT/pip-cache" "$BUILD_ROOT/build-temp" "$BUILD_ROOT/build-logs"

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$BUILD_ROOT/pip-cache"
export TMPDIR="$BUILD_ROOT/build-temp"
export TORCH_CUDA_ARCH_LIST=8.0
# FlashAttention's backward kernels use substantial host RAM under CUDA 12.9.
# Eight parallel NVCC processes exceeded 120 GiB on Snellius job 26984232.
export MAX_JOBS=2
export NVCC_THREADS=2

echo "BUILD_CONFIG job_id=$SLURM_JOB_ID host=$(hostname -f) env=$ENV_ROOT cuda_home=${CUDA_HOME:-unset} arch=$TORCH_CUDA_ARCH_LIST max_jobs=$MAX_JOBS"
"$PYTHON_BIN" -u -c 'import sys, torch; print("python", sys.version); print("torch", torch.__version__, "cuda", torch.version.cuda)'
nvcc --version

constraints="$REPO_ROOT/scripts/train/qwen35_runtime_constraints.txt"
"$PYTHON_BIN" -m pip install \
    --constraint "$constraints" \
    deepspeed==0.16.4 flash_attn==2.8.3 \
    --no-build-isolation
"$PYTHON_BIN" -m pip install \
    --constraint "$constraints" \
    'git+https://github.com/Dao-AILab/causal-conv1d' \
    --no-build-isolation
"$PYTHON_BIN" -m pip install \
    --constraint "$constraints" \
    'git+https://github.com/fla-org/flash-linear-attention'

"$PYTHON_BIN" scripts/train/validate_qwen35_runtime.py \
    --attention-implementation flash_attention_2 \
    --require-fast-linear-attention \
    --deepspeed-config scripts/zero2_opt.json
"$PYTHON_BIN" -m pip freeze > "$BUILD_ROOT/build-logs/qwen35-fast-env-${SLURM_JOB_ID}.freeze.txt"
echo "BUILD_COMPLETE job_id=$SLURM_JOB_ID env=$ENV_ROOT"
