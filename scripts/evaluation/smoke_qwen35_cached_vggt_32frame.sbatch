#!/usr/bin/env bash
#SBATCH --job-name=q35-eval-32f
#SBATCH --partition=gpu_a100
#SBATCH --account=tesei2748
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:30:00

set -euo pipefail

: "${EVAL_SMOKE_BASE:?Set a scratch output root}"
: "${EXPECTED_GIT_COMMIT:?Pin the approved evaluation code commit}"

CHECKOUT_ROOT="${CHECKOUT_ROOT:-/gpfs/home4/geusdd/shuang/SpatialStack-heldout}"
ENV_ROOT="${ENV_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/env-fast-overlay}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/formal-manifests}"
MEDIA_ROOT="/scratch-shared/geusdd/VLM3R/data/vlm3r"
CHECKPOINT_ROOT="/scratch-shared/geusdd/SpatialStackQwen35/smoke/checkpoints"
RUN_ROOT="$EVAL_SMOKE_BASE/job-$SLURM_JOB_ID"

if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite existing evaluation smoke: $RUN_ROOT" >&2
    exit 2
fi
if ! git -C "$CHECKOUT_ROOT" cat-file -e "$EXPECTED_GIT_COMMIT^{commit}"; then
    echo "Pinned evaluation code commit is unavailable: $EXPECTED_GIT_COMMIT" >&2
    exit 2
fi
mkdir -p "$RUN_ROOT/source" "$RUN_ROOT/triton"
git -C "$CHECKOUT_ROOT" archive "$EXPECTED_GIT_COMMIT" | tar -xf - -C "$RUN_ROOT/source"

module load 2025
module load CUDA/12.9.1
export REPO_ROOT="$RUN_ROOT/source"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PATH="$ENV_ROOT/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export TRITON_CACHE_DIR="$RUN_ROOT/triton"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FLA_CACHE_MODE=default
export FLA_CONFIG_DIR="$REPO_ROOT/scripts/profiling/fla_configs_a100"
export HF_HOME="${HF_HOME:-/scratch-shared/geusdd/SpatialStackQwen35/hf-cache}"

cd "$REPO_ROOT"
echo "EVAL_SMOKE_CONFIG job=$SLURM_JOB_ID code=$EXPECTED_GIT_COMMIT gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) frames=32 min_pixels=$((16*28*28)) max_pixels=$((576*28*28)) output=$RUN_ROOT"
"$ENV_ROOT/bin/python" scripts/train/validate_qwen35_runtime.py \
    --attention-implementation flash_attention_2 \
    --require-fast-linear-attention \
    --require-cuda

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -lms 500 > "$RUN_ROOT/nvidia-smi.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

set +e
timeout --signal=TERM --kill-after=30s 12m \
    "$ENV_ROOT/bin/python" scripts/train/smoke_lmms_eval_cached_vggt.py \
    --checkpoint "$CHECKPOINT_ROOT/candidate-a-formal" \
    --manifest "$MANIFEST_ROOT/candidate-a-l23-full.json" \
    --media-root "$MEDIA_ROOT" \
    --num-frames 32 \
    --min-pixels "$((16*28*28))" \
    --max-pixels "$((576*28*28))" \
    --flash-attention-2 --require-fast-runtime \
    > "$RUN_ROOT/candidate-a.log" 2>&1
A_STATUS=$?
echo "EVAL_SMOKE_CANDIDATE candidate=A status=$A_STATUS log=$RUN_ROOT/candidate-a.log"

timeout --signal=TERM --kill-after=30s 12m \
    "$ENV_ROOT/bin/python" scripts/train/smoke_lmms_eval_cached_vggt.py \
    --checkpoint "$CHECKPOINT_ROOT/candidate-b-formal" \
    --manifest "$MANIFEST_ROOT/candidate-b-three-layer-full.json" \
    --media-root "$MEDIA_ROOT" \
    --num-frames 32 \
    --min-pixels "$((16*28*28))" \
    --max-pixels "$((576*28*28))" \
    --flash-attention-2 --require-fast-runtime \
    > "$RUN_ROOT/candidate-b.log" 2>&1
B_STATUS=$?
set -e
echo "EVAL_SMOKE_CANDIDATE candidate=B status=$B_STATUS log=$RUN_ROOT/candidate-b.log"

if (( A_STATUS != 0 || B_STATUS != 0 )); then
    echo "EVAL_SMOKE_COMPLETE status=failed A=$A_STATUS B=$B_STATUS" >&2
    exit 1
fi
echo "EVAL_SMOKE_COMPLETE status=ok A=0 B=0"
