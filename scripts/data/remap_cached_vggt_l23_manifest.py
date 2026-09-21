#!/usr/bin/env python3
"""Remap a strict cached-VGGT manifest through a validated L23 inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def sidecar_relative_path(relative_video: Path) -> Path:
    parts = list(relative_video.parts)
    if len(parts) >= 2 and parts[-2] == "videos":
        del parts[-2]
    return Path(*parts).with_suffix(".pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if manifest.get("schema") != "spatialfocus.cached_vggt.v1":
        raise ValueError(f"Unsupported manifest schema: {manifest.get('schema')!r}")
    if inventory.get("schema") != "vggt_l23_inventory_v1":
        raise ValueError(f"Unsupported inventory schema: {inventory.get('schema')!r}")
    if inventory.get("scope") != "full" or inventory.get("source_tree_unchanged") is not True:
        raise ValueError("L23 inventory must cover the full, unchanged source tree")

    by_relative_path = {}
    for record in inventory.get("records", []):
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str) or relative_path in by_relative_path:
            raise ValueError(f"Invalid or duplicate inventory relative_path: {relative_path!r}")
        if record.get("tensor_shape") != [32, 1374, 2048] or record.get("tensor_dtype") != "bfloat16":
            raise ValueError(f"Invalid L23 tensor declaration for {relative_path}")
        output_path = Path(record.get("output_sidecar", ""))
        if not output_path.is_file() or output_path.stat().st_size != record.get("output_size"):
            raise FileNotFoundError(f"Inventory output is absent or size-mismatched: {output_path}")
        digest = record.get("output_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid output SHA256 for {relative_path}")
        by_relative_path[relative_path] = record

    remapped = []
    for record in manifest.get("records", []):
        video = record.get("video")
        if not isinstance(video, str):
            raise ValueError("Manifest record is missing video")
        relative_path = sidecar_relative_path(Path(video)).as_posix()
        if relative_path not in by_relative_path:
            raise KeyError(f"No L23 inventory record for {relative_path}")
        l23_record = by_relative_path[relative_path]
        updated = dict(record)
        updated["sidecar"] = l23_record["output_sidecar"]
        updated["sha256"] = l23_record["output_sha256"]
        remapped.append(updated)

    if not remapped:
        raise ValueError("Input manifest has no records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"schema": "spatialfocus.cached_vggt.v1", "records": remapped}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Remapped {len(remapped)} records through validated L23 inventory to {args.output}")


if __name__ == "__main__":
    main()
