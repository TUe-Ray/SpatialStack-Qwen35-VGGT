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
enough for all 16 periodic saves. The current Qwen3.5 Transformers 5.3.0
Trainer reported 1,623 update steps in the 8-GPU formal smoke even with
`dataloader_drop_last=True`: the final partial gradient-accumulation window
counts as an update. Earlier SpatialFocus runs reported 1,622 under their
training runtime, so the observed step count must be recorded per run. On 4
nodes with 4 GPUs per node and microbatch one, gradient accumulation is 8.
The wrapper fixes both `seed` and `data_seed` to 42, matching those formal
SpatialFocus runs.

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

## Qwen3.5 runtime and throughput guard

`scripts/train/validate_qwen35_runtime.py` runs before a real formal launch and
before every profiling launch. It requires the versions documented by the
official SpatialStack Qwen3.5 setup, verifies that FlashAttention 2 imports,
and checks that Transformers reports the Qwen3.5 gated-delta linear-attention
fast path available through both `causal-conv1d` and
`flash-linear-attention`. A formal ZeRO-2 launch additionally requires
DeepSpeed 0.16.4. Missing acceleration is an error rather than a silent
fallback to the much slower PyTorch implementation.

On Snellius, `scripts/train/slurm/build_qwen35_fast_env.sbatch` builds these
CUDA extensions on a CPU compute node into the isolated scratch overlay
`/scratch-shared/geusdd/SpatialStackQwen35/env-fast-overlay`; it never mutates
the previously validated base environment. The build is fixed to A100 SM80
and deliberately uses only two parallel NVCC jobs because eight-way
parallelism exceeded 120 GiB of host RAM in build job 26984232.

The controlled wrappers also set `ddp_find_unused_parameters=False`. Both
candidate smokes covered every intended trainable LoRA/fusion parameter, and
PyTorch explicitly reported that the prior `True` setting found no unused
parameters while adding an autograd-graph traversal every iteration. This
change removes distributed bookkeeping only; it does not alter the forward,
loss, gradients, or optimizer.

The Qwen3.5 model loader explicitly forwards `ATTN_IMPLEMENTATION` (default
`flash_attention_2`) for both the controlled and plain model paths. Earlier
profiling did not forward this setting and the profiling environment lacked
all of `flash_attn`, `causal-conv1d`, and `flash-linear-attention`; its epoch
extrapolation must therefore not be treated as a formal runtime estimate.

The profiling-only wrapper now defaults to the formal ZeRO-2 config, 32 exact
frames, one sample per GPU, gradient accumulation 8 on a four-GPU node, and
20 optimizer steps (the first 10 excluded from steady-state statistics).
It does not save checkpoints. For a profile shorter than one data epoch,
`scripts/profiling/select_manifest_qa.py` selects all canonical QA belonging
to the manifest's exact video set; it does not invent or duplicate samples.
The one-node effective batch is 32, not the four-node formal batch of 128,
so any epoch-time projection must disclose the cross-node topology change.

The scratch runtime loads the CUDA toolkit module before DeepSpeed import and
uses PyTorch's expandable-segment allocator to avoid fragmentation at the
32-frame A100 memory limit. The `fla_configs_a100` files pin the launch
configurations selected by the successful Candidate A A100 autotuning run.
Candidate B's unconstrained autotuner exhausted device memory while testing
larger configurations; pinning these kernel launch parameters let both
candidates complete the same 20-step ZeRO-2 profile. These files tune the
Qwen3.5 fast-path kernels only; they do not change fusion or cache semantics.

The custom Qwen3.5 decoder still builds M-RoPE from the full multimodal
position IDs, but passes the text-axis 2-D IDs into decoder-layer attention.
FlashAttention's packed-sequence detection cannot safely interpret the 3-D
M-RoPE IDs as text positions. Both controlled candidates completed a
32-frame forward/backward after this correction.

## Deliberate differences from official SpatialStack

The held-out experiment retains the official Qwen3.5 checkpoint, native image
processor and visual merger, 576-patch image budget, BF16, gradient
checkpointing, AdamW, 1e-5 learning rate, 0.01 weight decay, cosine schedule,
0.03 warmup ratio, four loader workers, model length 12800, and ZeRO-2 config.
The following differences are intentional experiment controls rather than
porting discrepancies:

| Setting | Official SpatialStack | Controlled held-out run | Reason |
|---|---|---|---|
| Frames | 4--8 sampled frames | 32 exact sidecar frame IDs | SpatialFocus control and RGB/VGGT correspondence |
| VGGT | Online VGGT-1B | Precomputed L23-only (A) or L11/L17/L23 (B) | Required cache-first experiment |
| Trainable LLM | Full LLM | LoRA r128/alpha256/dropout0.05 | Match controlled SpatialFocus trainable scope policy |
| Global batch | 64 | 128 | Match prior controlled training |
| Seed | 0 | seed/data_seed 42 | Match prior controlled sample ordering |
| Checkpoints | Every 1000 steps | Every 100 steps | Requested experiment retention |
| Candidate B injection | Post-decoder-block in official implementation | Pre-block L0/L1/L2 | SpatialFocus controlled-variant semantics take priority |
| Candidate A | No matching official recipe | Pre-native-merger frame-local L23 cross-attention | SpatialFocus A-prime semantics |

The official defaults do not enable TF32 or `torch.compile`, whereas the old
SpatialFocus runs enabled both. This branch keeps the official Qwen3.5 choices
for now; changing them is a separate common-configuration decision and must be
applied identically to A and B after a measured smoke comparison.

## Prepared controlled VSI-Bench evaluation (not submitted by this change)

The local VSI-Bench parquet pair contains 5,130 QA rows on 288 distinct
videos. The audited VGGT inventory has exactly the same 288 evaluation videos.
`scripts/data/build_vsibench_cached_vggt_manifests.sbatch` builds paired
manifests with dataset key `vsibench` and exact relative video paths. Candidate
A uses the audited L23-only sidecar SHA256; Candidate B hashes the original
three-layer sidecar. The builder validates 32 identical frame IDs and source
provenance across both sidecars. Keep the two generated manifest files and
their reported SHA256 values together as one comparison input.

After reviewing and committing the evaluation code, build the paired
manifests once (this command is documentation, not an automatic submission):

```bash
commit=$(git rev-parse HEAD)
sbatch --output=/scratch-shared/geusdd/SpatialStackQwen35/vsi-manifests-%j.out \
  --export=ALL,EXPECTED_GIT_COMMIT="$commit",MANIFEST_BUILD_ROOT=/scratch-shared/geusdd/SpatialStackQwen35/vsi-manifest-build \
  scripts/data/build_vsibench_cached_vggt_manifests.sbatch
```

For each trained checkpoint, use
`scripts/evaluation/run_qwen35_controlled_vsibench_snellius.sbatch` with:

- `CONTROLLED_FUSION_CANDIDATE=a_premerger_cross_attn` or `b_llm_add`;
- `CHECKPOINT_PATH` pointing to an explicit completed `checkpoint-N` or final
  output directory, never an automatically chosen latest checkpoint;
- `CACHED_VGGT_MANIFEST` and `EXPECTED_MANIFEST_SHA256` from the matching
  manifest builder output;
- `EXPECTED_GIT_COMMIT` and `CONTROLLED_EVAL_ROOT` on shared scratch.

Submit A and B separately with the same committed evaluator, using the exact
manifest paths and SHA256 values emitted by the builder:

```bash
sbatch --output=/scratch-shared/geusdd/SpatialStackQwen35/controlled-vsi-%j.out \
  --export=ALL,CONTROLLED_FUSION_CANDIDATE=a_premerger_cross_attn,CHECKPOINT_PATH=/path/to/A/output/checkpoint-100,CACHED_VGGT_MANIFEST=/path/to/candidate-a-l23-vsibench.json,EXPECTED_MANIFEST_SHA256=<sha256>,EXPECTED_GIT_COMMIT=<commit>,CONTROLLED_EVAL_ROOT=/scratch-shared/geusdd/SpatialStackQwen35/controlled-vsibench \
  scripts/evaluation/run_qwen35_controlled_vsibench_snellius.sbatch
```

Replace the candidate, checkpoint, manifest, and manifest SHA256 for B. The
evaluation job refuses missing inputs or an existing run directory.

The wrapper uses the same 4-A100, 32-frame, 12,800-token, 12,544--451,584
pixel, BF16/FlashAttention-2/Fast Linear Attention and local parquet
VSI-Bench settings as the base-Qwen evaluation. It checks candidate/base-model
identity and all 288 RGB/sidecar pairs before loading the model, runs a
four-sample evaluation smoke, then the complete 5,130-sample evaluation. The
candidate's cached-VGGT loader checks each sidecar SHA256 and exact original
frame IDs; no online VGGT is constructed. Intermediate `checkpoint-N`
directories can lack `processor_config.json`, so the evaluation adapter uses
the unchanged native processor from the recorded base checkpoint in that case.
It rejects missing or shape-incompatible trained fusion weights rather than
silently evaluating randomly initialized fusion modules.

These scripts only prepare evaluation. Do not submit A/B evaluations until
their training checkpoint and paired VSI manifests exist and their provenance
has been reviewed. This VSI-Bench score is a post-SFT diagnostic; the separate
pre-SFT architecture ranking metric remains validation `delta125`.
