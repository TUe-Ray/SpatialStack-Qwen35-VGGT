#!/usr/bin/env python3
"""Build full controlled manifests from the audited L23 conversion inventory.

The L23 inventory already contains verified output SHA256 values. The source
three-layer cache does not, so its SHA256 values are computed here. PyTorch's
mmap loader reads frame IDs and metadata without materializing the large VGGT
feature storages while constructing the manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import torch


SCHEMA = "spatialfocus.cached_vggt.v1"


def sidecar_relative_path(video: str) -> str:
    path = Path(video)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe annotation video path: {video!r}")
    parts = list(path.parts)
    if len(parts) >= 2 and parts[-2] == "videos":
        del parts[-2]
    return Path(*parts).with_suffix(".pt").as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(path: Path, required_layers: set[str], exact_layers: bool = False) -> tuple[list[int], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    frames = payload["frames"]
    frame_idx = frames["frame_idx"]
    layers = frames["aggregated_tokens"]
    meta = payload["meta"]
    if frame_idx.dtype != torch.int64 or frame_idx.device.type != "cpu":
        raise ValueError(f"{path}: frame IDs are not CPU int64")
    ids = frame_idx.tolist()
    if len(ids) != 32 or any(right <= left for left, right in zip(ids, ids[1:])):
        raise ValueError(f"{path}: expected 32 unique, increasing frame IDs")
    if meta.get("schema") != "vggt_aggregated_tokens_v1" or meta.get("patch_start_idx") != 5:
        raise ValueError(f"{path}: incompatible VGGT metadata")
    if not required_layers.issubset(layers):
        raise ValueError(f"{path}: required VGGT layers are absent")
    if exact_layers and (
        {str(layer) for layer in layers} != required_layers
        or {str(layer) for layer in meta.get("intermediate_layer_idx", [])} != required_layers
    ):
        raise ValueError(f"{path}: expected exactly VGGT layers {sorted(required_layers)}")
    for layer in required_layers:
        tensor = layers[layer]
        if tuple(tensor.shape) != (32, 1374, 2048) or tensor.dtype != torch.bfloat16:
            raise ValueError(f"{path}: invalid layer {layer} shape/dtype")
    source_video = meta.get("source_video")
    if not isinstance(source_video, str) or not source_video:
        raise ValueError(f"{path}: missing source video provenance")
    return ids, source_video


def inspect_record(record: dict) -> tuple[str, dict, dict]:
    relative = record["relative_path"]
    l23_path = Path(record["output_sidecar"]).resolve()
    source_path = Path(record["source_sidecar"]).resolve()
    if not l23_path.is_file() or l23_path.stat().st_size != record["output_size"]:
        raise FileNotFoundError(f"L23 sidecar missing/size-mismatched: {l23_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source sidecar missing: {source_path}")
    l23_ids, l23_video = metadata(l23_path, {"23"})
    source_ids, source_video = metadata(source_path, {"11", "17", "23"})
    if l23_ids != source_ids or l23_video != source_video:
        raise ValueError(f"Source/L23 provenance mismatch: {relative}")
    digest = record["output_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Invalid audited L23 SHA256: {relative}")
    common = {"frame_idx": l23_ids, "recorded_source_video": source_video}
    a = {**common, "sidecar": str(l23_path), "sha256": digest}
    b = {**common, "sidecar": str(source_path), "sha256": sha256(source_path)}
    return relative, a, b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--spec", nargs=3, action="append", required=True,
                        metavar=("DATASET", "ANNOTATION", "MEDIA_ROOT"))
    parser.add_argument("--output-a", type=Path, required=True)
    parser.add_argument("--output-b", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if inventory.get("schema") != "vggt_l23_inventory_v1" or inventory.get("scope") != "full":
        raise ValueError("Expected the audited full L23 inventory")
    if inventory.get("source_tree_unchanged") is not True:
        raise ValueError("Inventory does not attest to an unchanged source tree")
    indexed = {}
    for record in inventory["records"]:
        relative = record["relative_path"]
        if relative in indexed:
            raise ValueError(f"Duplicate inventory record: {relative}")
        indexed[relative] = record

    requested: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for alias, annotation_value, media_value in args.spec:
        media_root = Path(media_value).resolve()
        annotations = json.loads(Path(annotation_value).read_text(encoding="utf-8"))
        for row in annotations:
            video = row["video"]
            key = f"{alias}::{video}"
            if key in seen:
                continue
            seen.add(key)
            relative = sidecar_relative_path(video)
            if relative not in indexed:
                raise KeyError(f"No validated sidecar for {key}: {relative}")
            if not (media_root / video).is_file():
                raise FileNotFoundError(f"No RGB video for {key}")
            requested.append((alias, video, relative))

    unique = sorted({relative for _, _, relative in requested})
    started = time.monotonic()
    inspected = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, (relative, a, b) in enumerate(pool.map(inspect_record, (indexed[r] for r in unique)), 1):
            inspected[relative] = (a, b)
            if count % 100 == 0 or count == len(unique):
                print(f"FORMAL_MANIFEST_PROGRESS sidecars={count}/{len(unique)} elapsed_sec={time.monotonic()-started:.1f}", flush=True)

    for output, index in ((args.output_a, 0), (args.output_b, 1)):
        records = [{"dataset": alias, "video": video, **inspected[relative][index]}
                   for alias, video, relative in requested]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"schema": SCHEMA, "records": records}, indent=2) + "\n", encoding="utf-8")
        print(f"FORMAL_MANIFEST_WRITTEN output={output} keys={len(records)} sidecars={len(unique)} sha256={sha256(output)}", flush=True)


if __name__ == "__main__":
    main()
