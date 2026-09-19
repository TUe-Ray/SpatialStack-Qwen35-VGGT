#!/usr/bin/env python3
"""Exercise the lmms-eval Qwen3.5 cached-VGGT hook on one local video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    media_root = args.media_root.resolve()
    video_path = (media_root / args.video).resolve()
    evaluator = Qwen3_5(
        pretrained=str(checkpoint),
        device="cuda:0",
        device_map="cuda:0",
        batch_size=1,
        use_flash_attention_2=False,
        min_pixels=65536,
        max_pixels=65536,
        max_num_frames=args.num_frames,
        cached_vggt_manifest=str(args.manifest.resolve()),
        cached_vggt_dataset=args.dataset,
        cached_vggt_data_root=str(media_root),
    )
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
    answers = evaluator.generate_until([request])
    base_model = evaluator.model.get_base_model()
    if getattr(base_model.model, "geometry_encoder", None) is not None:
        raise RuntimeError("Online VGGT/geometry encoder was unexpectedly constructed")
    print(
        json.dumps(
            {
                "candidate": evaluator.config.controlled_fusion_candidate,
                "answer": answers[0],
                "num_frames": args.num_frames,
                "evaluation_hook": "lmms_eval.models.qwen3_5.generate_until",
                "online_vggt_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
