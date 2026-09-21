#!/usr/bin/env python3
"""Fail-fast validation for the controlled SpatialFocus-equivalent SFT mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


MANIFEST_SCHEMA = "spatialfocus.cached_vggt.v1"


def _read_annotations(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            result = [json.loads(line) for line in handle if line.strip()]
    else:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    if not isinstance(result, list):
        raise ValueError(f"Annotation file must contain a list: {path}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_video(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}: video must be a non-empty string")
    normalized = os.path.normpath(value).replace(os.sep, "/")
    if os.path.isabs(normalized) or normalized == ".." or normalized.startswith("../"):
        raise ValueError(f"{location}: video must be relative, got {value!r}")
    return normalized


def _parse_dataset_spec(values: list[str]) -> tuple[str, Path, Path, int, str]:
    alias, annotation, media_root, expected_count, expected_sha256 = values
    return (
        alias,
        Path(annotation).expanduser().resolve(),
        Path(media_root).expanduser().resolve(),
        int(expected_count),
        expected_sha256,
    )


def validate(args: argparse.Namespace) -> dict:
    required_keys: set[str] = set()
    dataset_counts: dict[str, int] = {}
    annotation_hashes: dict[str, str] = {}
    unique_media: set[Path] = set()

    for raw_spec in args.dataset:
        alias, annotation_path, media_root, expected_count, expected_sha256 = (
            _parse_dataset_spec(raw_spec)
        )
        if alias in dataset_counts:
            raise ValueError(f"Duplicate dataset alias: {alias}")
        annotations = _read_annotations(annotation_path)
        if len(annotations) != expected_count:
            raise ValueError(
                f"{alias}: loaded {len(annotations)} annotations, expected {expected_count}"
            )
        for index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                raise ValueError(f"{annotation_path}:{index}: annotation must be an object")
            video = _relative_video(
                annotation.get("video"), location=f"{annotation_path}:{index}"
            )
            required_keys.add(f"{alias}::{video}")
            unique_media.add((media_root / video).resolve())
        dataset_counts[alias] = len(annotations)
        annotation_hash = _sha256(annotation_path)
        if annotation_hash != expected_sha256:
            raise ValueError(
                f"{alias}: annotation SHA256 is {annotation_hash}, "
                f"expected {expected_sha256}"
            )
        annotation_hashes[alias] = annotation_hash

    total_samples = sum(dataset_counts.values())
    if total_samples != args.expected_total:
        raise ValueError(
            f"Total annotation count is {total_samples}, expected {args.expected_total}"
        )

    missing_media = [str(path) for path in unique_media if not path.is_file()]
    if missing_media:
        raise FileNotFoundError(
            f"Missing {len(missing_media)} unique RGB videos; first: {missing_media[:5]}"
        )

    manifest_path = args.manifest.expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"Manifest schema is {manifest.get('schema')!r}, expected {MANIFEST_SCHEMA!r}"
        )

    sidecar_root = args.sidecar_root.expanduser().resolve()
    manifest_records: dict[str, dict] = {}
    sidecars: set[Path] = set()
    for index, record in enumerate(manifest.get("records", [])):
        if not isinstance(record, dict):
            raise ValueError(f"Manifest record {index} must be an object")
        video = _relative_video(record.get("video"), location=f"manifest record {index}")
        alias = record.get("dataset")
        if not isinstance(alias, str) or not alias:
            raise ValueError(f"Manifest record {index} has no dataset alias")
        key = f"{alias}::{video}"
        if key in manifest_records:
            raise ValueError(f"Duplicate manifest key: {key}")
        manifest_records[key] = record
        if key not in required_keys:
            continue

        raw_sidecar = Path(str(record.get("sidecar", ""))).expanduser()
        if not raw_sidecar.is_absolute():
            raw_sidecar = manifest_path.parent / raw_sidecar
        sidecar = raw_sidecar.resolve()
        if not sidecar.is_relative_to(sidecar_root):
            raise ValueError(f"{key}: sidecar is outside required root {sidecar_root}: {sidecar}")
        if not sidecar.is_file():
            raise FileNotFoundError(f"{key}: sidecar not found: {sidecar}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{key}: manifest must contain a SHA256 digest")
        frame_idx = record.get("frame_idx")
        if not isinstance(frame_idx, list) or len(frame_idx) != args.num_frames:
            raise ValueError(
                f"{key}: frame_idx must contain exactly {args.num_frames} entries"
            )
        if any(not isinstance(value, int) for value in frame_idx):
            raise ValueError(f"{key}: frame_idx entries must be integers")
        if any(right <= left for left, right in zip(frame_idx, frame_idx[1:])):
            raise ValueError(f"{key}: frame_idx must be unique and strictly increasing")
        sidecars.add(sidecar)

    missing_records = sorted(required_keys - manifest_records.keys())
    if missing_records:
        raise ValueError(
            f"Manifest lacks {len(missing_records)} required dataset/video keys; "
            f"first: {missing_records[:5]}"
        )

    return {
        "dataset_counts": dataset_counts,
        "total_samples": total_samples,
        "unique_dataset_video_keys": len(required_keys),
        "unique_rgb_videos": len(unique_media),
        "unique_sidecars": len(sidecars),
        "manifest_records": len(manifest_records),
        "annotation_sha256": annotation_hashes,
        "manifest": str(manifest_path),
        "sidecar_root": str(sidecar_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        nargs=5,
        action="append",
        required=True,
        metavar=(
            "ALIAS",
            "ANNOTATIONS",
            "MEDIA_ROOT",
            "EXPECTED_COUNT",
            "EXPECTED_SHA256",
        ),
    )
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(validate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
