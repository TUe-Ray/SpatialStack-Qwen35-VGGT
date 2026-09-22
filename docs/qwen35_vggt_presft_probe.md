# Qwen3.5/VGGT pre-SFT representation probe

This is a held-out pre-SFT representation study for three frozen forwards:
the official `Qwen/Qwen3.5-4B` comparator, Candidate A (VGGT L23
pre-merger cross-attention), and Candidate B (VGGT L11/17/23 additions before
language blocks 0/1/2). It is separate from candidate SFT, and no adapter,
trained fusion state, optimizer, or optimizer step may enter extraction.

## Fixed data and levels

- The SpatialFocus ScanNet split has 1,199 videos (1,006 train, 193 validation),
  exactly 32 RGB forward frames per video, and two selected depth-target frames.
  The sample-index SHA-256 is
  `d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e`.
- All three forwards use the same Qwen image processor at a fixed maximum of
  200,704 pixels per frame, FP16, SDPA, the same 32 RGB frames, and the real
  human question identified by each fixed sample ID; answer text is ignored.
  Extracted
  features are bilinearly mapped to the existing 14×14 depth target grid.
- The 17 required levels are `visual_output`, `fusion_output`,
  `projected_features`, and decoder L0/1/2/3/6/9/12/15/18/21/24/27/30/31.
  L maps to `hidden_states[L+1]`; L31 is captured after the final language
  norm. The Common-7 LogME primary mean is L1/3/6/9/15/21/27; L31 is a
  separate diagnostic.
- `visual_output` is native Qwen visual pre-merger tokens for every model.
  `projected_features` is the native visual merger output (after A fusion for
  A). `fusion_output` is the A fused pre-merger output, the B visual-token
  stream after L0 addition but before block 0, or a copy of native visual
  output for the base comparator. These architecture-specific stages must be
  labelled rather than treated as one identical physical module.

## C1 and local placement

Fresh A/B affine maps are regenerated from the SpatialFocus
`c1_structured_isometry_v1` matrix family. Native SFT initialization remains
unchanged until C1 is explicitly enabled. B's native zero terminal maps are
replaced by nonzero canonical maps; scalar gains and A Q/K scale require a
separate hashed 32-video, unlabeled C1 calibration artifact. The old VLM3R
`r0` must not be reused. The new Qwen-specific `r0` is the median across the
32 fixed videos and base-model L0/L1/L2 of the visual-token ratio
`RMS(H_after-H_before)/RMS(H_before)`; it is a per-site target, not divided
across A/B injection sites. A/B formal extraction refuses to run without that
artifact and its matching calibration manifest. `--smoke-zero-gain` is limited
to at most two diagnostic videos and must never be scored as a formal model.

On the local two-TITAN-V host, fixed placement is visual tower, embedding,
fusion, and L0–4 on GPU 0; L5–10 on CPU; L11–31 and final norm on GPU 1.
The tied LM head remains with the embedding on GPU 0. This preserves the
32-frame, 200,704-pixel condition without reducing resolution. On a different
accelerator, validate a one-video forward and its layer values before making
any placement-only change.

The extractor is `scripts/probing/extract_qwen35_presft_features.py`. It
accepts a full VGGT manifest plus a relocatable `--sidecar-root` containing
`scannet/<scene>.pt`, verifies each sidecar against the manifest SHA-256, and
checks exact RGB/VGGT original frame IDs. It saves only the two selected
14×14 tensors per level and video. Full A+B sidecars exceed the local SSD, so
stage them in bounded batches and retain verified manifests/checksums; do not
substitute compact two-frame depth targets for 32-frame VGGT inputs.

The extracted tensor layout matches the SpatialFocus depth-probe trainer:
`features/<model_label>/<feature_level>/frame_<frame_sample_id>.pt`. Reuse
the existing 14×14 `gt_depth` and `metadata` for the same fixed split. The
ordinary 50-epoch seed-0 MLP depth probe and streaming common-seven LogME
must be run only after all 1,199 videos and every required layer are complete.
