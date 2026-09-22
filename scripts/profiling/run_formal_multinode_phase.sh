#!/usr/bin/env bash
# One Slurm task per node; torchrun starts four local GPU workers.
set -euo pipefail

: "${PROFILE_ROOT:?}"
: "${PROFILE_PHASE:?}"
: "${OUTPUT_DIR:?}"
: "${SLURM_PROCID:?}"

NODE_ROOT="$PROFILE_ROOT/node-$SLURM_PROCID"
mkdir -p "$NODE_ROOT" "$NODE_ROOT/triton-$PROFILE_PHASE"
export TRITON_CACHE_DIR="$NODE_ROOT/triton-$PROFILE_PHASE"

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
    f"FORMAL_PROFILE_GPU_PREFLIGHT host={socket.gethostname()} "
    f"gpus={actual} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
    flush=True,
)
PY

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw \
    --format=csv,noheader,nounits -lms 500 > "$NODE_ROOT/nvidia-smi-$PROFILE_PHASE.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

echo "FORMAL_PROFILE_NODE phase=$PROFILE_PHASE node_rank=$SLURM_PROCID host=$(hostname -f) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
bash scripts/train/train_controlled_cached_vggt_formal.sh
