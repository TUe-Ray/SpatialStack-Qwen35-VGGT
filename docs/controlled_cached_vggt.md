# Controlled Qwen3.5 + cached-VGGT experiment

This path implements the two held-out SpatialFocus candidates. It is separate
from the official online-geometry recipe and contains no probe logic.

## Manifest

`--cached_vggt_manifest` is a JSON document with this exact schema:

```json
{
  "schema": "spatialfocus.cached_vggt.v1",
  "records": [
    {
      "dataset": "vlm3r_scannet",
      "video": "relative/path/scene.mp4",
      "sidecar": "relative/or/absolute/scene.pt",
      "sha256": "required-by-default-sidecar-sha256",
      "frame_idx": [0, 3, 6],
      "frame_positions": [0, 2]
    }
  ]
}
```

`dataset` must equal the dataset alias passed in `--dataset_use`, and `video`
must exactly equal the annotation's relative video path. There is no basename
or stem fallback. `frame_idx`, when present, validates the entire sidecar
sequence. `frame_positions`, when present, is the exact increasing subset to
use and must have `cached_vggt_num_frames` elements. Otherwise that many
positions are selected uniformly from the cached sequence. Duplicate,
reordered, missing, non-finite, or shape-inconsistent data raises immediately.
SHA256 verification is enabled by default and a missing digest is an error.

Build this manifest directly from one or more annotation/media/sidecar roots:

```bash
python scripts/data/build_cached_vggt_manifest.py \
  --spec vlm3r_scannet /path/train.json /path/media /path/vggt/scannet \
  --layers 11 17 23 \
  --output /path/cached_vggt_manifest.json
```

Expected sidecar tensors are
`frames.aggregated_tokens.{11,17,23}: [F,1374,2048]`. The loader removes the
first five special tokens before tensors reach the model.

Candidate A training and evaluation require an L23-only sidecar: both
`meta.intermediate_layer_idx` and `frames.aggregated_tokens` must contain
exactly layer 23. Build its manifest with `--layers 23 --exact-layers` and the
L23-only cache root. Candidate B continues to accept the established
multi-layer sidecars and selects layers 11/17/23.

## Candidates

- `a_premerger_cross_attn`: layer 23 only; resize 37x37 patches to each Qwen
  pre-merger grid, reorder to native Qwen 2x2 merger order, frame-local
  cross-attention plus residual at width 1024, then the frozen native merger.
- `b_llm_add`: independent layer 11/17/23 RMSNorm + 2x2 merger/MLP projectors,
  added before language blocks 0/1/2 at width 2560. Terminal linears are
  initialized to zero.

Both runs freeze the vision encoder and native visual merger. The controlled
training entry point requires language LoRA so only LoRA and candidate fusion
parameters are trainable.

Example (do not use this as an official/full launch without reviewing all
shared experiment controls):

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
DATASETS=vlm3r_scannet \
USE_GEOMETRY_ENCODER=false \
USE_CACHED_VGGT=true \
CONTROLLED_FUSION_CANDIDATE=a_premerger_cross_attn \
CACHED_VGGT_MANIFEST=/path/to/manifest.json \
CACHED_VGGT_NUM_FRAMES=32 \
TUNE_MM_LLM=false \
LORA_ENABLE=true \
bash scripts/train/train.sh
```

For an implementation smoke, use
`scripts/train/smoke_controlled_cached_vggt.sh` with a one-scene dataset alias,
batch size one, a temporary output directory, and `MAX_STEPS=1` or `2`. The
wrapper uses SDPA, disables DeepSpeed unless explicitly requested, and performs
an additional backward-time finite check with separate fusion and LoRA gradient
coverage counts. That scan is smoke-only and is not enabled by the normal
training entry point.

`scripts/train/reload_controlled_cached_vggt_smoke.py` verifies direct
checkpoint reload and short generation. For the actual evaluation adapter,
`scripts/train/smoke_lmms_eval_cached_vggt.py` drives the repository's
`lmms_eval.models.qwen3_5.generate_until` path on one manifest-backed video.
Both require a local checkpoint, manifest, and media root.

`scripts/train/create_synthetic_cached_vggt_smoke.py` can create a
schema-valid sidecar solely for engineering tests when real VGGT caches are not
present. Its output is marked `synthetic_smoke_only` and must never be used for
scientific training, probing, ranking, or VSI-Bench reporting.

No formal SFT, pre-SFT probe, or full VSI-Bench evaluation was launched while
implementing and smoke-validating this path. The external
architecture-ranking metric remains validation `delta125` (higher is better).

## SpatialFocus-equivalent formal SFT controls

`scripts/train/train_controlled_cached_vggt_formal.sh` is the guarded launch
wrapper for the eventual full comparison. It fixes the training mixture to
the three un-subsampled SpatialFocus annotations:

| Alias | Annotation | QA samples |
|---|---|---:|
| `vlm3r_scannet` | `merged_qa_scannet_train.json` | 51,779 |
| `vlm3r_scannetpp` | `merged_qa_scannetpp_train.json` | 151,775 |
| `vlm3r_routeplan` | `merged_qa_route_plan_train.json` | 4,104 |
| **Total** | | **207,658** |

The wrapper also fixes one epoch, 32 cached frames, effective global batch
128, and checkpoint interval 100 optimizer steps. It retains 20 checkpoints,
enough for all 16 periodic saves in the 1,623-step epoch. On 4 nodes with 4
GPUs per node and microbatch one, gradient accumulation is 8.

Before starting `torchrun`, the wrapper verifies the three individual counts
and canonical annotation SHA256 identities, the total count, every unique RGB
path, exact dataset/video coverage in the cached-VGGT manifest, 32 strictly
increasing manifest frame IDs, sidecar existence, and that every selected
sidecar is under the candidate-specific cache root. Candidate A defaults to
the L23-only root; Candidate B defaults to the L11/L17/L23 root. Any mismatch
stops before model loading.

Required launch-specific variables are `MODEL_PATH`, `OUTPUT_DIR`,
`CONTROLLED_FUSION_CANDIDATE`, and `CACHED_VGGT_MANIFEST`. The annotation,
media, and sidecar roots may be relocated through the documented environment
variables in the wrapper without changing the fixed aliases or sample counts.
