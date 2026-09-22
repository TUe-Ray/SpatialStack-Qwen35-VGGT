#!/usr/bin/env python3
"""Exercise the lmms-eval Qwen3.5 cached-VGGT hook on one local video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lmms_eval.api.instance import Instance
from lmms_eval.models.qwen3_5 import Qwen3_5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--dataset", default="vlm3r_scannet")
    parser.add_argument("--video", default="scannet/videos/scene0384_00.mp4")
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=65536)
    parser.add_argument("--max-length", type=int, default=12800)
    parser.add_argument("--flash-attention-2", action="store_true")
    parser.add_argument("--require-fast-runtime", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    media_root = args.media_root.resolve()
    video_path = (media_root / args.video).resolve()
    evaluator = Qwen3_5(
        pretrained=str(checkpoint),
        device="cuda:0",
        device_map="cuda:0",
        batch_size=1,
        use_flash_attention_2=args.flash_attention_2,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_num_frames=args.num_frames,
        max_length=args.max_length,
        cached_vggt_manifest=str(args.manifest.resolve()),
        cached_vggt_dataset=args.dataset,
        cached_vggt_data_root=str(media_root),
    )
    text_config = getattr(evaluator.model.config, "text_config", evaluator.model.config)
    attention_implementation = getattr(text_config, "_attn_implementation", None)
    if attention_implementation is None:
        attention_implementation = getattr(evaluator.model.config, "_attn_implementation", None)
    if args.require_fast_runtime:
        if not all(evaluator.fast_path_runtime.values()):
            raise RuntimeError(f"Qwen3.5 eval fast path is unavailable: {evaluator.fast_path_runtime}")
        if attention_implementation != "flash_attention_2":
            raise RuntimeError(f"Expected FlashAttention 2, got {attention_implementation!r}")

    observed_frame_idx = []
    original_load = evaluator.cached_vggt_store.load

    def capture_cached_load(dataset: str, video: str, data_root: str):
        sample = original_load(dataset, video, data_root)
        observed_frame_idx.extend(sample.frame_idx.tolist())
        return sample

    evaluator.cached_vggt_store.load = capture_cached_load
    evaluator.task_dict = {"cached_vggt_smoke": {"validation": [{}]}}
    request = Instance(
        request_type="generate_until",
        arguments=(
            "Describe this indoor scene briefly.",
            {"max_new_tokens": 4, "temperature": 0, "num_beams": 1},
            lambda _document: [str(video_path)],
            0,
            "cached_vggt_smoke",
            "validation",
        ),
        idx=0,
        metadata={"task": "cached_vggt_smoke", "doc_id": 0, "repeats": 1},
    )
    torch.cuda.reset_peak_memory_stats()
    answers = evaluator.generate_until([request])
    torch.cuda.synchronize()
    if len(observed_frame_idx) != args.num_frames:
        raise RuntimeError(f"Expected {args.num_frames} exact cached/RGB frames, got {len(observed_frame_idx)}")
    base_model = evaluator.model.get_base_model()
    if getattr(base_model.model, "geometry_encoder", None) is not None:
        raise RuntimeError("Online VGGT/geometry encoder was unexpectedly constructed")
    print(
        json.dumps(
            {
                "candidate": evaluator.config.controlled_fusion_candidate,
                "answer": answers[0],
                "num_frames": args.num_frames,
                "frame_idx_first_last": [observed_frame_idx[0], observed_frame_idx[-1]],
                "evaluation_hook": "lmms_eval.models.qwen3_5.generate_until",
                "online_vggt_executed": False,
                "fast_path_runtime": evaluator.fast_path_runtime,
                "attention_implementation": attention_implementation,
                "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    )


if __name__ == "__main__":
    main()
