#!/usr/bin/env bash
#SBATCH --job-name=q35-job-audit
#SBATCH --partition=rome
#SBATCH --account=tesei2748
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

set -euo pipefail
: "${TARGET_JOB_ID:?}"
: "${TARGET_LABEL:?}"
: "${TARGET_SLURM_LOG:?}"
: "${MONITOR_ROOT:?}"

REPO_ROOT="${REPO_ROOT:-/gpfs/home4/geusdd/shuang/SpatialStack-heldout}"
cd "$REPO_ROOT"
args=(
    --job-id "$TARGET_JOB_ID"
    --label "$TARGET_LABEL"
    --slurm-log "$TARGET_SLURM_LOG"
    --output-root "$MONITOR_ROOT"
)
if [[ -n "${TARGET_ARTIFACT_DIR:-}" ]]; then
    args+=(--artifact-dir "$TARGET_ARTIFACT_DIR")
fi
python3 scripts/profiling/report_slurm_job.py "${args[@]}"
