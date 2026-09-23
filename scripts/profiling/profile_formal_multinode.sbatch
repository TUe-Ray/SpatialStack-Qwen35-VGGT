#!/usr/bin/env bash
#SBATCH --job-name=q35-resume-8gpu
#SBATCH --partition=gpu_a100
#SBATCH --account=tesei2748
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00

set -euo pipefail

: "${CONTROLLED_FUSION_CANDIDATE:?Choose a_premerger_cross_attn or b_llm_add}"
: "${PROFILE_BASE:?Set a scratch profiling output root}"
: "${EXPECTED_GIT_COMMIT:?Pin the approved smoke source commit}"

CHECKOUT_ROOT="${CHECKOUT_ROOT:-/gpfs/home4/geusdd/shuang/SpatialStack-heldout}"
ENV_ROOT="${ENV_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/env-fast-overlay}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/formal-manifests}"
PROFILE_ROOT="$PROFILE_BASE/${CONTROLLED_FUSION_CANDIDATE}-${SLURM_JOB_ID}"
if [[ -e "$PROFILE_ROOT" ]]; then
    echo "Refusing to overwrite existing smoke output: $PROFILE_ROOT" >&2
    exit 2
fi
if ! git -C "$CHECKOUT_ROOT" cat-file -e "$EXPECTED_GIT_COMMIT^{commit}"; then
    echo "Pinned smoke code commit is unavailable: $EXPECTED_GIT_COMMIT" >&2
    exit 2
fi
mkdir -p "$PROFILE_ROOT/source" "$PROFILE_ROOT/output" "$PROFILE_ROOT/phase-logs"
git -C "$CHECKOUT_ROOT" archive "$EXPECTED_GIT_COMMIT" | tar -xf - -C "$PROFILE_ROOT/source"
REPO_ROOT="$PROFILE_ROOT/source"
cd "$REPO_ROOT"

module load 2025
module load CUDA/12.9.1
test -x "$CUDA_HOME/bin/nvcc"

case "$CONTROLLED_FUSION_CANDIDATE" in
    a_premerger_cross_attn)
        CACHED_VGGT_MANIFEST="$MANIFEST_ROOT/candidate-a-l23-full.json"
        ;;
    b_llm_add)
        CACHED_VGGT_MANIFEST="$MANIFEST_ROOT/candidate-b-three-layer-full.json"
        ;;
    *)
        echo "Unsupported candidate: $CONTROLLED_FUSION_CANDIDATE" >&2
        exit 2
        ;;
esac
test -s "$CACHED_VGGT_MANIFEST"

export REPO_ROOT ENV_ROOT PROFILE_ROOT CACHED_VGGT_MANIFEST
export PYTHON_BIN="$ENV_ROOT/bin/python"
export PATH="$ENV_ROOT/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
export OUTPUT_DIR="$PROFILE_ROOT/output"
export CACHE_DIR="${CACHE_DIR:-/scratch-shared/geusdd/SpatialStackQwen35/hf-cache}"
export MODEL_PATH="${MODEL_PATH:-/scratch-shared/geusdd/SpatialStackQwen35/models/Qwen3.5-4B}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/scratch-shared/geusdd/SpatialStackQwen35/torch_extensions}"
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export CONTROLLED_PROFILE=1
export CONTROLLED_PROFILE_SKIP_SAVE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export NNODES="$SLURM_JOB_NUM_NODES"
export NPROC_PER_NODE=4
mkdir -p "$TORCH_EXTENSIONS_DIR"

echo "FORMAL_PROFILE_CONFIG job=$SLURM_JOB_ID candidate=$CONTROLLED_FUSION_CANDIDATE nodes=$SLURM_JOB_NUM_NODES gpus_per_node=4 manifest=$CACHED_VGGT_MANIFEST output=$PROFILE_ROOT"
echo "FORMAL_PROFILE_GIT commit=$EXPECTED_GIT_COMMIT"

# All production hyperparameters, full QA data, and the formal 100-step save
# interval remain fixed in train_controlled_cached_vggt_formal.sh. The only
# smoke-specific change is one forced checkpoint after three optimizer steps.
export PROFILE_PHASE=initial
export CONTROLLED_PROFILE_COMM_STEP=3
export CONTROLLED_PROFILE_SAVE_AT_STEP=3
export CONTROLLED_PROFILE_STOP_AFTER_STEP=3
time srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --gpus-per-task=4 --kill-on-bad-exit=1 --wait=30 --export=ALL \
    bash scripts/profiling/run_formal_multinode_phase.sh

checkpoint="$OUTPUT_DIR/checkpoint-3"
test -s "$checkpoint/trainer_state.json"
test -s "$checkpoint/adapter_model.safetensors"
test -s "$checkpoint/controlled_vggt_fusion.bin"
"$PYTHON_BIN" -c 'import json,sys; state=json.load(open(sys.argv[1])); assert state["global_step"]==3,state["global_step"]; print("FORMAL_PROFILE_CHECKPOINT step=3 verified=true")' "$checkpoint/trainer_state.json"
optim_count=$(find "$checkpoint/global_step3" -maxdepth 1 -name '*optim_states.pt' | wc -l)
test "$optim_count" -eq 8
echo "FORMAL_PROFILE_CHECKPOINT_SHARDS optimizer_shards=$optim_count"
cp "$OUTPUT_DIR"/train.rank*.log "$PROFILE_ROOT/phase-logs/"

export PROFILE_PHASE=resume
export CONTROLLED_PROFILE_COMM_STEP=0
export CONTROLLED_PROFILE_SAVE_AT_STEP=0
export CONTROLLED_PROFILE_STOP_AFTER_STEP=5
time srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --gpus-per-task=4 --kill-on-bad-exit=1 --wait=30 --export=ALL \
    bash scripts/profiling/run_formal_multinode_phase.sh

"$PYTHON_BIN" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); rows=[json.loads(line.split("CONTROLLED_PROFILE_STEP ",1)[1]) for line in p.read_text().splitlines() if "CONTROLLED_PROFILE_STEP " in line]; steps=[row["optimizer_step"] for row in rows]; assert steps==[4,5],steps; print("FORMAL_PROFILE_RESUME verified=true steps=4,5")' "$OUTPUT_DIR/train.rank0.log"
echo "FORMAL_PROFILE_COMPLETE job=$SLURM_JOB_ID candidate=$CONTROLLED_FUSION_CANDIDATE output=$PROFILE_ROOT"
