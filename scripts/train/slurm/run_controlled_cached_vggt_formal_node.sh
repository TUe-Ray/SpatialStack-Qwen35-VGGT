#!/usr/bin/env bash
# One Slurm task per node, with four local torchrun GPU processes.
set -euo pipefail

: "${REPO_ROOT:?}"
: "${OUTPUT_DIR:?}"
: "${SLURM_PROCID:?}"

cd "$REPO_ROOT"
export TRITON_CACHE_DIR="$(dirname "$OUTPUT_DIR")/triton/node-$SLURM_PROCID"
mkdir -p "$TRITON_CACHE_DIR"
echo "FORMAL_SFT_NODE node_rank=$SLURM_PROCID host=$(hostname -f) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
bash scripts/train/train_controlled_cached_vggt_formal.sh
