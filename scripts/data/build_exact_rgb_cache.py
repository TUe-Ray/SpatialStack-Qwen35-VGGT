#!/usr/bin/env python3
"""Prewarm exact 32-frame RGB entries from a validated VGGT manifest."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from qwen_vl.data.cached_vggt import CachedVGGTStore, sample_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--decord-threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 means every unique annotation video")
    args = parser.parse_args()

    rows = json.loads(Path(args.annotation).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Annotation must be a nonempty JSON list")
    videos = list(dict.fromkeys(row["video"] for row in rows))
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.limit:
        videos = videos[:args.limit]

    store = CachedVGGTStore(
        manifest_path=args.manifest,
        required_layers=args.layers,
        num_frames=32,
        require_exact_layers=(args.layers == [23]),
        rgb_cache_root=args.cache_root,
        decord_threads=args.decord_threads,
    )
    missing = [video for video in videos if sample_key(args.dataset, video) not in store.records]
    if missing:
        raise ValueError(f"Annotation videos missing from VGGT manifest: {missing[:8]}")

    started = time.perf_counter()
    total_bytes = 0
    for index, video in enumerate(videos, 1):
        sample = store.load(args.dataset, video, args.media_root)
        cache_path = store.rgb_cache._path(
            (Path(args.media_root) / video).resolve(), sample.frame_idx.tolist()
        )
        if not cache_path.is_file():
            raise RuntimeError(f"RGB cache entry absent after loading {video}")
        total_bytes += cache_path.stat().st_size
        # The sample is loaded for provenance checks; a later pass validates hits.
        if index == 1 or index % 5 == 0 or index == len(videos):
            print(json.dumps({
                "completed": index, "total": len(videos),
                "elapsed_sec": round(time.perf_counter() - started, 2),
                "cache_bytes": total_bytes,
            }), flush=True)
    print(json.dumps({
        "dataset": args.dataset, "videos": len(videos),
        "cache_bytes": total_bytes, "elapsed_sec": round(time.perf_counter() - started, 2),
    }), flush=True)


if __name__ == "__main__":
    main()
