#!/usr/bin/env bash
#SBATCH --job-name=q35-vggt-prof
#SBATCH --partition=gpu_a100
#SBATCH --account=tesei2748
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00

set -euo pipefail

: "${PROFILE_ROOT:?Set PROFILE_ROOT}"
: "${CONTROLLED_FUSION_CANDIDATE:?Set CONTROLLED_FUSION_CANDIDATE}"

REPO_ROOT="${REPO_ROOT:-/gpfs/home4/geusdd/shuang/SpatialStack-heldout}"
ENV_ROOT="${ENV_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/env-fast-overlay}"
RUN_ROOT="$PROFILE_ROOT/${CONTROLLED_FUSION_CANDIDATE}-${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$RUN_ROOT/output" "$RUN_ROOT/triton"
cd "$REPO_ROOT"

# DeepSpeed imports its CUDA extension builder before the first training step.
# The scratch Python overlay has CUDA wheels but does not provide nvcc itself.
module load 2025
module load CUDA/12.9.1
test -x "$CUDA_HOME/bin/nvcc"

export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FLA_CACHE_MODE=default
export FLA_CONFIG_DIR="$REPO_ROOT/scripts/profiling/fla_configs_a100"
export PATH="$ENV_ROOT/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export TRITON_CACHE_DIR="$RUN_ROOT/triton"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/scratch-shared/geusdd/SpatialStackQwen35/torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
export OUTPUT_DIR="$RUN_ROOT/output"
export CACHE_DIR="${CACHE_DIR:-/scratch-shared/geusdd/SpatialStackQwen35/hf-cache}"
export MODEL_PATH="${MODEL_PATH:-/scratch-shared/geusdd/SpatialStackQwen35/models/Qwen3.5-4B}"
export DATASETS="${DATASETS:-vlm3r_scannet}"
export VLM3R_SCANNET_ANNOTATION="${VLM3R_SCANNET_ANNOTATION:?Set profiling annotation}"
export VLM3R_SCANNET_MEDIA_ROOT="${VLM3R_SCANNET_MEDIA_ROOT:-/scratch-shared/geusdd/VLM3R/data/vlm3r}"
export CACHED_VGGT_MANIFEST="${CACHED_VGGT_MANIFEST:?Set profiling manifest}"
export MAX_STEPS="${MAX_STEPS:-20}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

echo "PROFILE_JOB job_id=$SLURM_JOB_ID host=$(hostname) candidate=$CONTROLLED_FUSION_CANDIDATE"
echo "PROFILE_TOPOLOGY nodes=$SLURM_JOB_NUM_NODES gpus=$SLURM_GPUS_ON_NODE cpus=$SLURM_CPUS_PER_TASK total_batch=$TOTAL_BATCH_SIZE"
echo "PROFILE_CUDA_HOME $CUDA_HOME"
echo "PROFILE_FLA_CONFIG mode=$FLA_CACHE_MODE dir=$FLA_CONFIG_DIR"
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms 500 > "$RUN_ROOT/nvidia-smi.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

# Leave cleanup time before the Slurm hard limit; short diagnostics can override.
set +e
timeout --signal=TERM --kill-after=30s "${PROFILE_TIMEOUT:-26m}" \
    bash scripts/profiling/profile_controlled_cached_vggt.sh
STATUS=$?
set -e
if [[ "$STATUS" -ne 0 && "$STATUS" -ne 124 ]]; then
    exit "$STATUS"
fi
echo "PROFILE_EXIT status=$STATUS run_root=$RUN_ROOT"
