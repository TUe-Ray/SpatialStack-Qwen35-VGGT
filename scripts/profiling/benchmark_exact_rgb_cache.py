#!/usr/bin/env python3
"""Compare exact RGB decoding and lossless cache reads on formal sidecars."""

from __future__ import annotations

import argparse
import json
import time

import torch

from qwen_vl.data.cached_vggt import CachedVGGTStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--videos", type=int, default=3)
    parser.add_argument("--decord-threads", type=int, default=1)
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as handle:
        records = json.load(handle)["records"]
    videos = [record["video"] for record in records if record["dataset"] == args.dataset][:args.videos]
    if len(videos) != args.videos:
        raise ValueError(f"Found {len(videos)} videos; expected {args.videos}")

    common = dict(
        manifest_path=args.manifest,
        required_layers=args.layers,
        num_frames=32,
        require_exact_layers=(args.layers == [23]),
        decord_threads=args.decord_threads,
    )
    direct = CachedVGGTStore(**common)
    cached = CachedVGGTStore(**common, rgb_cache_root=args.cache_root)
    for video in videos:
        results = {}
        samples = {}
        for label, store in [
            ("direct_first", direct), ("direct_second", direct),
            ("cache_first", cached), ("cache_second", cached),
        ]:
            started = time.perf_counter()
            sample = store.load(args.dataset, video, args.media_root)
            results[label] = round(time.perf_counter() - started, 4)
            samples[label] = sample
        reference = samples["direct_second"]
        for label in ("cache_first", "cache_second"):
            current = samples[label]
            if not torch.equal(current.frame_idx, reference.frame_idx):
                raise RuntimeError(f"Frame IDs differ: {video}, {label}")
            for layer in reference.features:
                if not torch.equal(current.features[layer], reference.features[layer]):
                    raise RuntimeError(f"VGGT features differ: {video}, {label}, {layer}")
            if [image.tobytes() for image in current.images] != [image.tobytes() for image in reference.images]:
                raise RuntimeError(f"RGB pixels differ: {video}, {label}")
        print(json.dumps({"video": video, "seconds": results, "exact_match": True}), flush=True)


if __name__ == "__main__":
    main()
