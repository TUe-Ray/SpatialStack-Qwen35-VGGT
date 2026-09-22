#!/usr/bin/env bash
# One Slurm task per node, with four local torchrun GPU processes.
set -euo pipefail

: "${REPO_ROOT:?}"
: "${OUTPUT_DIR:?}"
: "${SLURM_PROCID:?}"

cd "$REPO_ROOT"
export TRITON_CACHE_DIR="$(dirname "$OUTPUT_DIR")/triton/node-$SLURM_PROCID"
mkdir -p "$TRITON_CACHE_DIR"

"$PYTHON_BIN" - <<'PY'
import os
import socket
import torch

expected = 4
actual = torch.cuda.device_count()
if not torch.cuda.is_available() or actual != expected:
    raise SystemExit(
        f"{socket.gethostname()}: expected {expected} visible GPUs, got {actual}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
    )
print(
    f"FORMAL_SFT_GPU_PREFLIGHT host={socket.gethostname()} "
    f"gpus={actual} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
    flush=True,
)
PY

echo "FORMAL_SFT_NODE node_rank=$SLURM_PROCID host=$(hostname -f) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
bash scripts/train/train_controlled_cached_vggt_formal.sh
