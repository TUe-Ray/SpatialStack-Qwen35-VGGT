#!/usr/bin/env python3
"""Build a strict training/evaluation manifest from annotations and VGGT sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_annotations(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sidecar_relative_path(relative_video: Path) -> Path:
    """Map an exact media-relative path to the established cache layout."""
    parts = list(relative_video.parts)
    if len(parts) >= 2 and parts[-2] == "videos":
        del parts[-2]
    return Path(*parts).with_suffix(".pt")


def validate_sidecar(path: Path, required_layers: set[str], exact_layers: bool = False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    frame_idx = payload["frames"]["frame_idx"]
    layer_map = payload["frames"]["aggregated_tokens"]
    meta = payload["meta"]
    if not isinstance(frame_idx, torch.Tensor) or frame_idx.dtype != torch.int64 or frame_idx.device.type != "cpu":
        raise ValueError(f"{path}: frame_idx must be CPU int64")
    if frame_idx.ndim != 1 or len(frame_idx) == 0:
        raise ValueError(f"{path}: frame_idx must be non-empty and 1-D")
    if torch.unique(frame_idx).numel() != frame_idx.numel() or not torch.all(frame_idx[1:] > frame_idx[:-1]):
        raise ValueError(f"{path}: duplicated or reordered frame IDs are forbidden")
    if meta.get("schema") != "vggt_aggregated_tokens_v1" or meta.get("patch_start_idx") != 5:
        raise ValueError(f"{path}: unsupported sidecar metadata")
    if not required_layers.issubset(layer_map):
        raise ValueError(f"{path}: missing required layers {sorted(required_layers - set(layer_map))}")
    metadata_layers = {str(layer) for layer in meta.get("intermediate_layer_idx", [])}
    payload_layers = {str(layer) for layer in layer_map}
    if exact_layers and (metadata_layers != required_layers or payload_layers != required_layers):
        raise ValueError(
            f"{path}: exact layers {sorted(required_layers)} required; "
            f"metadata has {sorted(metadata_layers)}, payload has {sorted(payload_layers)}"
        )
    for layer in required_layers:
        tensor = layer_map[layer]
        if tuple(tensor.shape) != (len(frame_idx), 1374, 2048) or tensor.dtype != torch.bfloat16:
            raise ValueError(f"{path}: invalid layer {layer} tensor {tuple(tensor.shape)}/{tensor.dtype}")
    source_video = meta.get("source_video")
    if not isinstance(source_video, str) or not source_video:
        raise ValueError(f"{path}: meta.source_video must be a non-empty string")
    return frame_idx.tolist(), source_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        nargs=4,
        action="append",
        required=True,
        metavar=("DATASET", "ANNOTATIONS", "MEDIA_ROOT", "SIDECAR_ROOT"),
        help="Repeat for each dataset alias; roots preserve the annotation's relative video path.",
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[11, 17, 23])
    parser.add_argument(
        "--exact-layers",
        action="store_true",
        help="Require metadata and payload to contain exactly --layers (used by Candidate A L23-only caches).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    required_layers = {str(layer) for layer in args.layers}
    records = []
    seen = set()
    for dataset, annotation_value, media_value, sidecar_value in args.spec:
        annotation_path = Path(annotation_value).expanduser().resolve()
        media_root = Path(media_value).expanduser().resolve()
        sidecar_root = Path(sidecar_value).expanduser().resolve()
        for annotation in read_annotations(annotation_path):
            video = annotation.get("video")
            if not isinstance(video, str):
                continue
            relative_video = Path(video)
            if relative_video.is_absolute() or ".." in relative_video.parts:
                raise ValueError(f"Annotation video path must be relative: {video!r}")
            normalized = relative_video.as_posix()
            key = f"{dataset}::{normalized}"
            if key in seen:
                continue
            seen.add(key)
            media_path = (media_root / normalized).resolve()
            sidecar_path = (sidecar_root / sidecar_relative_path(relative_video)).resolve()
            if not media_path.is_file() or not sidecar_path.is_file():
                raise FileNotFoundError(f"Missing exact media/sidecar pair: {media_path} / {sidecar_path}")
            frame_idx, recorded_source_video = validate_sidecar(
                sidecar_path, required_layers, exact_layers=args.exact_layers
            )
            records.append(
                {
                    "dataset": dataset,
                    "video": normalized,
                    "sidecar": str(sidecar_path),
                    "sha256": sha256(sidecar_path),
                    "frame_idx": frame_idx,
                    "recorded_source_video": recorded_source_video,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump({"schema": "spatialfocus.cached_vggt.v1", "records": records}, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(records)} exact records to {args.output}")


if __name__ == "__main__":
    main()
