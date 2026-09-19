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

For an implementation smoke, additionally set `--max_steps 2`, a one-scene
dataset alias, batch size one, and a temporary output directory. Formal SFT,
Pre-SFT probing, official/full SFT, and full VSI-Bench are deliberately outside
this path. The external architecture-ranking metric remains validation
`delta125` (higher is better).
