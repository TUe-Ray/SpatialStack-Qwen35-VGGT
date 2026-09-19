#!/usr/bin/env python3
"""Create a schema-valid synthetic VGGT cache for engineering smoke tests only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-relative", required=True)
    parser.add_argument("--dataset", default="vlm3r_scannet")
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cached-frames", type=int, default=32)
    parser.add_argument("--selected-positions", type=int, nargs="+", default=[0, 31])
    args = parser.parse_args()

    video = args.video.expanduser().resolve()
    reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
    if len(reader) < args.cached_frames:
        raise ValueError("Synthetic smoke requires enough video frames to avoid duplicate frame IDs")
    frame_idx = torch.from_numpy(
        np.linspace(0, len(reader) - 1, args.cached_frames, dtype=np.int64)
    )
    if torch.unique(frame_idx).numel() != frame_idx.numel():
        raise ValueError("Generated smoke frame IDs are duplicated")

    generator = torch.Generator(device="cpu").manual_seed(42)
    layers = {
        str(layer): torch.empty(
            args.cached_frames, 1374, 2048, dtype=torch.bfloat16
        ).normal_(mean=0.0, std=0.02, generator=generator)
        for layer in (11, 17, 23)
    }
    payload = {
        "frames": {"frame_idx": frame_idx, "aggregated_tokens": layers},
        "meta": {
            "source_video": str(video),
            "vggt_weights_path": "SYNTHETIC_ENGINEERING_SMOKE_NO_VGGT_EXECUTION",
            "num_frames": args.cached_frames,
            "input_size": 518,
            "model_image_hw": (518, 518),
            "patch_size": 14,
            "patch_start_idx": 5,
            "feature_dim": 2048,
            "intermediate_layer_idx": [11, 17, 23],
            "token_dtype": "bfloat16",
            "schema": "vggt_aggregated_tokens_v1",
            "synthetic_smoke_only": True,
        },
    }
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.sidecar)
    digest = sha256(args.sidecar)
    manifest = {
        "schema": "spatialfocus.cached_vggt.v1",
        "records": [
            {
                "dataset": args.dataset,
                "video": args.video_relative,
                "sidecar": str(args.sidecar.resolve()),
                "sha256": digest,
                "frame_idx": frame_idx.tolist(),
                "frame_positions": args.selected_positions,
                "synthetic_smoke_only": True,
            }
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"sidecar": str(args.sidecar), "sha256": digest, "manifest": str(args.manifest)}))


if __name__ == "__main__":
    main()
