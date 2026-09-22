#!/usr/bin/env bash
#SBATCH --job-name=q35-full-validate
#SBATCH --partition=rome
#SBATCH --account=tesei2748
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/home4/geusdd/shuang/SpatialStack-heldout}"
ENV_ROOT="${ENV_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/env-fast-overlay}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/formal-manifests}"
cd "$REPO_ROOT"
export PYTHON_BIN="$ENV_ROOT/bin/python"
export PYTHONNOUSERSITE=1
export MODEL_PATH="/scratch-shared/geusdd/SpatialStackQwen35/models/Qwen3.5-4B"
export OUTPUT_DIR="$MANIFEST_ROOT/validation-output-unused"
for CONTROLLED_FUSION_CANDIDATE in a_premerger_cross_attn b_llm_add; do
    export CONTROLLED_FUSION_CANDIDATE
    if [[ "$CONTROLLED_FUSION_CANDIDATE" == a_premerger_cross_attn ]]; then
        export CACHED_VGGT_MANIFEST="$MANIFEST_ROOT/candidate-a-l23-full.json"
    else
        export CACHED_VGGT_MANIFEST="$MANIFEST_ROOT/candidate-b-three-layer-full.json"
    fi
    CHECK_ONLY=1 bash scripts/train/train_controlled_cached_vggt_formal.sh
done
