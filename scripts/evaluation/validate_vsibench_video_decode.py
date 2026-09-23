#!/usr/bin/env python3
"""CPU-only exact-index decode gate for the canonical local VSI-Bench videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import decord
import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, action="append", required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    args = parser.parse_args()
    if args.num_frames < 1:
        parser.error("--num-frames must be positive")

    videos = set()
    qa_count = 0
    for parquet in args.parquet:
        table = pq.read_table(parquet, columns=["dataset", "scene_name"])
        qa_count += table.num_rows
        for row in table.to_pylist():
            dataset, scene = row["dataset"], row["scene_name"]
            if dataset not in {"arkitscenes", "scannet", "scannetpp"}:
                raise ValueError(f"Unexpected VSI-Bench dataset: {dataset!r}")
            if not isinstance(scene, str) or not scene or Path(scene).name != scene or ".." in scene:
                raise ValueError(f"Unsafe VSI-Bench scene: {scene!r}")
            videos.add(f"{dataset}/{scene}.mp4")
    if qa_count != 5130 or len(videos) != 288:
        raise ValueError(f"Expected 5130 QA / 288 videos, got {qa_count} / {len(videos)}")

    short_videos = []
    for count, relative in enumerate(sorted(videos), 1):
        path = args.media_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing VSI-Bench video: {path}")
        reader = decord.VideoReader(str(path), num_threads=4)
        frame_count = len(reader)
        if frame_count < 1:
            raise ValueError(f"Empty VSI-Bench video: {path}")
        if frame_count <= args.num_frames:
            indices = np.arange(frame_count)
            short_videos.append(relative)
        else:
            indices = np.linspace(0, frame_count - 1, args.num_frames).astype(int)
        decoded = reader.get_batch(indices.tolist()).asnumpy()
        if len(decoded) != len(indices) or decoded.ndim != 4 or decoded.shape[-1] != 3:
            raise ValueError(f"Wrong decoded frame tensor for {relative}: {decoded.shape}")
        if count % 25 == 0 or count == len(videos):
            print(f"VSI_DECODE_PROGRESS videos={count}/{len(videos)}", flush=True)
    print(
        f"VSI_DECODE_COMPLETE qa={qa_count} videos={len(videos)} "
        f"short_videos={len(short_videos)} names={short_videos}",
        flush=True,
    )


if __name__ == "__main__":
    main()
