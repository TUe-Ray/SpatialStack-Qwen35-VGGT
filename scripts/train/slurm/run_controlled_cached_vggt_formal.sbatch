#!/usr/bin/env bash
#SBATCH --job-name=q35-vggt-formal
#SBATCH --partition=gpu_a100
#SBATCH --account=tesei2748
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00

set -euo pipefail

: "${CONTROLLED_FUSION_CANDIDATE:?Select controlled Candidate A or B}"
: "${TRAIN_BASE:?Set a scratch output root}"
: "${EXPECTED_GIT_COMMIT:?Pin the approved code commit at submission}"

CHECKOUT_ROOT="${CHECKOUT_ROOT:-/gpfs/home4/geusdd/shuang/SpatialStack-heldout}"
ENV_ROOT="${ENV_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/env-fast-overlay}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/formal-manifests}"
RUN_ROOT="$TRAIN_BASE/${CONTROLLED_FUSION_CANDIDATE}-${SLURM_JOB_ID}"

case "$CONTROLLED_FUSION_CANDIDATE" in
    a_premerger_cross_attn)
        CACHED_VGGT_MANIFEST="$MANIFEST_ROOT/candidate-a-l23-full.json"
        EXPECTED_MANIFEST_SHA=30f8d4ae28fb73504f023d89eae558b8b6bdfc830717a3d37ba4cca2a60680a5
        ;;
    b_llm_add)
        CACHED_VGGT_MANIFEST="$MANIFEST_ROOT/candidate-b-three-layer-full.json"
        EXPECTED_MANIFEST_SHA=628fafad0e56791cf7fb8e81b6c2f1dc5f3fa71bc4b6da055a5c465fc6521a50
        ;;
    *)
        echo "Unsupported controlled candidate: $CONTROLLED_FUSION_CANDIDATE" >&2
        exit 2
        ;;
esac

actual_manifest_sha=$(sha256sum "$CACHED_VGGT_MANIFEST" | cut -d' ' -f1)
if [[ "$actual_manifest_sha" != "$EXPECTED_MANIFEST_SHA" ]]; then
    echo "Formal manifest SHA256 mismatch: $CACHED_VGGT_MANIFEST" >&2
    exit 2
fi
if ! git -C "$CHECKOUT_ROOT" cat-file -e "$EXPECTED_GIT_COMMIT^{commit}"; then
    echo "Pinned source commit is unavailable: $EXPECTED_GIT_COMMIT" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite an existing formal run: $RUN_ROOT" >&2
    exit 2
fi
mkdir -p "$RUN_ROOT/source" "$RUN_ROOT/output"
git -C "$CHECKOUT_ROOT" archive "$EXPECTED_GIT_COMMIT" | tar -xf - -C "$RUN_ROOT/source"

module load 2025
module load CUDA/12.9.1
test -x "$CUDA_HOME/bin/nvcc"

export REPO_ROOT="$RUN_ROOT/source"
export OUTPUT_DIR="$RUN_ROOT/output"
export CACHED_VGGT_MANIFEST
export PYTHON_BIN="$ENV_ROOT/bin/python"
export PATH="$ENV_ROOT/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export CACHE_DIR="${CACHE_DIR:-/scratch-shared/geusdd/SpatialStackQwen35/hf-cache}"
export MODEL_PATH="${MODEL_PATH:-/scratch-shared/geusdd/SpatialStackQwen35/models/Qwen3.5-4B}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/scratch-shared/geusdd/SpatialStackQwen35/torch_extensions}"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
mkdir -p "$TORCH_EXTENSIONS_DIR"

# Never inherit short-smoke controls into the official SFT run.
unset CONTROLLED_PROFILE CONTROLLED_PROFILE_SKIP_SAVE
unset CONTROLLED_PROFILE_COMM_STEP CONTROLLED_PROFILE_SAVE_AT_STEP
unset CONTROLLED_PROFILE_STOP_AFTER_STEP CHECK_ONLY

echo "FORMAL_SFT job=$SLURM_JOB_ID candidate=$CONTROLLED_FUSION_CANDIDATE code=$EXPECTED_GIT_COMMIT manifest_sha=$actual_manifest_sha nodes=$SLURM_JOB_NUM_NODES gpus_per_node=4 output=$OUTPUT_DIR"
cd "$REPO_ROOT"
srun --kill-on-bad-exit=1 --ntasks=4 --ntasks-per-node=1 \
    bash scripts/train/slurm/run_controlled_cached_vggt_formal_node.sh
