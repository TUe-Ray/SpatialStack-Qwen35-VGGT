#!/usr/bin/env python3
"""Read-only gate for a trained controlled checkpoint and exact VSI-Bench cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


EXPECTED_QA = 5130
EXPECTED_VIDEOS = 288
SIDECAR_ROOTS = {
    "a_premerger_cross_attn": Path("/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt_l23"),
    "b_llm_add": Path("/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/vggt"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_video_keys(parquets: list[Path]) -> set[str]:
    count = 0
    keys = set()
    for parquet in parquets:
        table = pq.read_table(parquet, columns=["dataset", "scene_name"])
        count += table.num_rows
        for row in table.to_pylist():
            dataset, scene = row["dataset"], row["scene_name"]
            if dataset not in {"arkitscenes", "scannet", "scannetpp"}:
                raise ValueError(f"Unexpected VSI-Bench dataset: {dataset!r}")
            if not isinstance(scene, str) or not scene or Path(scene).name != scene or ".." in scene:
                raise ValueError(f"Unsafe VSI-Bench scene name: {scene!r}")
            keys.add(f"{dataset}/{scene}.mp4")
    if count != EXPECTED_QA or len(keys) != EXPECTED_VIDEOS:
        raise ValueError(f"VSI-Bench expected {EXPECTED_QA} QA / {EXPECTED_VIDEOS} videos, got {count} / {len(keys)}")
    return keys


def validate_checkpoint(checkpoint: Path, candidate: str, base_model: Path) -> None:
    required = ["config.json", "adapter_config.json", "controlled_vggt_fusion.bin", "trainer_state.json"]
    missing = [name for name in required if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0]
    if not any(
        (checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        missing.append("adapter_model.safetensors|adapter_model.bin")
    if missing:
        raise FileNotFoundError(f"Incomplete trained checkpoint {checkpoint}: {missing}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    adapter = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5" or config.get("use_cached_vggt") is not True:
        raise ValueError("Checkpoint is not a controlled cached-VGGT Qwen3.5 model")
    if config.get("controlled_fusion_candidate") != candidate:
        raise ValueError(f"Wrong checkpoint candidate: {config.get('controlled_fusion_candidate')!r}")
    if config.get("use_geometry_encoder") is not False:
        raise ValueError("Checkpoint unexpectedly enables online geometry execution")
    adapter_base = adapter.get("base_model_name_or_path")
    if not adapter_base or Path(adapter_base).resolve() != base_model.resolve():
        raise ValueError(f"Adapter base-model identity mismatch: {adapter_base!r}")
    if int(state.get("global_step", 0)) < 1:
        raise ValueError("Checkpoint has no completed optimizer step")
    if checkpoint.name.startswith("checkpoint-"):
        step = int(checkpoint.name.removeprefix("checkpoint-"))
        if int(state["global_step"]) != step:
            raise ValueError(f"Checkpoint directory step {step} differs from trainer_state")


def validate_manifest(manifest: Path, keys: set[str], media_root: Path, sidecar_root: Path) -> None:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if document.get("schema") != "spatialfocus.cached_vggt.v1":
        raise ValueError("Wrong cached-VGGT manifest schema")
    records = document.get("records", [])
    if len(records) != EXPECTED_VIDEOS:
        raise ValueError(f"Expected {EXPECTED_VIDEOS} manifest records, got {len(records)}")
    seen = set()
    for record in records:
        video = record.get("video")
        if record.get("dataset") != "vsibench" or video not in keys or video in seen:
            raise ValueError(f"Unexpected/duplicate manifest video: {record.get('dataset')}::{video}")
        seen.add(video)
        if not (media_root / video).is_file():
            raise FileNotFoundError(f"Missing exact RGB video: {media_root / video}")
        sidecar = Path(record["sidecar"]).resolve()
        if not sidecar.is_relative_to(sidecar_root.resolve()) or not sidecar.is_file():
            raise FileNotFoundError(f"Wrong or missing candidate sidecar: {sidecar}")
        expected_sidecar = sidecar_root / Path(video).with_suffix(".pt")
        if sidecar != expected_sidecar.resolve():
            raise ValueError(f"Sidecar/video path mismatch: {video} -> {sidecar}")
        ids = record.get("frame_idx")
        if not isinstance(ids, list) or len(ids) != 32 or any(
            not isinstance(value, int) or isinstance(value, bool) for value in ids
        ) or any(right <= left for left, right in zip(ids, ids[1:])):
            raise ValueError(f"Invalid 32-frame provenance in manifest: {video}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Missing/invalid sidecar SHA256: {video}")
    if seen != keys:
        raise ValueError(f"Manifest video coverage mismatch: {len(keys - seen)} missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(SIDECAR_ROOTS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, action="append", required=True)
    args = parser.parse_args()
    keys = expected_video_keys(args.parquet)
    validate_checkpoint(args.checkpoint, args.candidate, args.base_model)
    validate_manifest(args.manifest, keys, args.media_root, SIDECAR_ROOTS[args.candidate])
    print(
        f"CONTROLLED_VSI_PREFLIGHT candidate={args.candidate} checkpoint={args.checkpoint.resolve()} "
        f"manifest_sha256={sha256(args.manifest)} qa={EXPECTED_QA} videos={EXPECTED_VIDEOS} frames=32",
        flush=True,
    )


if __name__ == "__main__":
    main()
