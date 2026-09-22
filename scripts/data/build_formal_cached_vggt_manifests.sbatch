#!/usr/bin/env bash
#SBATCH --job-name=q35-full-manifest
#SBATCH --partition=rome
#SBATCH --account=tesei2748
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/home4/geusdd/shuang/SpatialStack-heldout}"
ENV_ROOT="${ENV_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/env-fast-overlay}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch-shared/geusdd/SpatialStackQwen35/formal-manifests}"
ANNOTATION_ROOT="/scratch-shared/geusdd/VLM3R/data/vlm3r/VLM-3R-DATA/vsibench_train"
MEDIA_ROOT="/scratch-shared/geusdd/VLM3R/data/vlm3r"
INVENTORY="/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23_artifacts/cached_vggt_l23_inventory.json"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
"$ENV_ROOT/bin/python" scripts/data/build_formal_cached_vggt_manifests.py \
    --inventory "$INVENTORY" \
    --spec vlm3r_scannet "$ANNOTATION_ROOT/merged_qa_scannet_train.json" "$MEDIA_ROOT" \
    --spec vlm3r_scannetpp "$ANNOTATION_ROOT/merged_qa_scannetpp_train.json" "$MEDIA_ROOT" \
    --spec vlm3r_routeplan "$ANNOTATION_ROOT/merged_qa_route_plan_train.json" "$MEDIA_ROOT" \
    --output-a "$OUTPUT_ROOT/candidate-a-l23-full.json" \
    --output-b "$OUTPUT_ROOT/candidate-b-three-layer-full.json" \
    --workers 8
