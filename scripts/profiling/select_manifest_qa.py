#!/usr/bin/env python3
"""Select all canonical QA for a profiling-only, validated sidecar manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "spatialfocus.cached_vggt.v1":
        raise ValueError("Unsupported cached-VGGT manifest schema")
    records = [record for record in manifest["records"] if record["dataset"] == args.dataset]
    videos = [record["video"] for record in records]
    if len(videos) != len(set(videos)) or not videos:
        raise ValueError("Manifest must contain unique, non-empty dataset/video records")
    video_set = set(videos)
    annotations = json.loads(args.annotation.read_text(encoding="utf-8"))
    selected = [row for row in annotations if row.get("video") in video_set]
    present = {row["video"] for row in selected}
    if present != video_set:
        raise ValueError(f"No canonical QA for manifest videos: {sorted(video_set - present)}")
    ids = [row["id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Selected annotations contain duplicate QA IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PROFILE_ANNOTATION qa={len(selected)} scenes={len(videos)} output={args.output}")


if __name__ == "__main__":
    main()
