#!/usr/bin/env python3
"""Build a deterministic multi-scene QA subset for throughput profiling."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=50)
    parser.add_argument("--samples-per-scene", type=int, default=4)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    grouped = defaultdict(list)
    for annotation in annotations:
        video = annotation.get("video")
        if isinstance(video, str) and len(grouped[video]) < args.samples_per_scene:
            grouped[video].append(annotation)

    selected = []
    selected_scenes = []
    for video, samples in grouped.items():
        if len(samples) != args.samples_per_scene:
            continue
        selected_scenes.append(video)
        selected.extend(samples)
        if len(selected_scenes) == args.scenes:
            break
    if len(selected_scenes) != args.scenes:
        raise RuntimeError(f"Found only {len(selected_scenes)} qualifying scenes")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
        handle.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "samples": len(selected),
        "unique_videos": len(selected_scenes),
        "samples_per_scene": args.samples_per_scene,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
