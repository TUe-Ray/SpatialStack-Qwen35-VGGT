#!/usr/bin/env bash
#SBATCH --job-name=q35-vsi-manifests
#SBATCH --partition=rome
#SBATCH --account=tesei2748
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

: "${EXPECTED_GIT_COMMIT:?Pin the approved source commit}"
: "${MANIFEST_BUILD_ROOT:?Choose a fresh scratch output root}"

CHECKOUT_ROOT="${CHECKOUT_ROOT:-/gpfs/home4/geusdd/shuang/SpatialStack-heldout}"
ENV_ROOT="${ENV_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/env-fast-overlay}"
INVENTORY="${INVENTORY:-/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23_artifacts/cached_vggt_l23_inventory.json}"
MEDIA_ROOT="${MEDIA_ROOT:-/scratch-shared/geusdd/VLM3R/hf_cache/vsibench}"
PARQUET_ROOT="${PARQUET_ROOT:-/scratch-shared/geusdd/VLM3R/data/vsibench}"
RUN_ROOT="$MANIFEST_BUILD_ROOT/job-$SLURM_JOB_ID"

if [[ "$MANIFEST_BUILD_ROOT" != /* ]]; then
    echo "MANIFEST_BUILD_ROOT must be an absolute scratch path" >&2
    exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to overwrite existing manifest build: $RUN_ROOT" >&2
    exit 2
fi
git -C "$CHECKOUT_ROOT" cat-file -e "$EXPECTED_GIT_COMMIT^{commit}"
mkdir -p "$RUN_ROOT/source" "$RUN_ROOT/manifests"
git -C "$CHECKOUT_ROOT" archive "$EXPECTED_GIT_COMMIT" | tar -xf - -C "$RUN_ROOT/source"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
cd "$RUN_ROOT/source"
"$ENV_ROOT/bin/python" scripts/data/build_vsibench_cached_vggt_manifests.py \
    --inventory "$INVENTORY" \
    --parquet "$PARQUET_ROOT/test_pruned.parquet" \
    --parquet "$PARQUET_ROOT/test_debiased.parquet" \
    --media-root "$MEDIA_ROOT" \
    --output-a "$RUN_ROOT/manifests/candidate-a-l23-vsibench.json" \
    --output-b "$RUN_ROOT/manifests/candidate-b-three-layer-vsibench.json" \
    --workers 8
echo "VSI_MANIFEST_BUILD_COMPLETE job=$SLURM_JOB_ID code=$EXPECTED_GIT_COMMIT output=$RUN_ROOT/manifests"
