"""Strict, cache-first VGGT sidecar loading for controlled Qwen3.5 experiments.

The manifest is the provenance boundary.  A record is selected only by the
exact ``<dataset tag>::<normalized video path>`` key; stem/basename fallback is
deliberately unsupported.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from decord import VideoReader
from PIL import Image


MANIFEST_SCHEMA = "spatialfocus.cached_vggt.v1"
VGGT_SPECIAL_TOKENS = 5
VGGT_PATCH_TOKENS = 37 * 37
VGGT_FEATURE_DIM = 2048


class CachedVGGTError(RuntimeError):
    """Raised when cache provenance or tensor invariants are violated."""


def _normalized_relative_path(value: str) -> str:
    normalized = os.path.normpath(value).replace(os.sep, "/")
    if os.path.isabs(normalized) or normalized == ".." or normalized.startswith("../"):
        raise CachedVGGTError(f"Manifest video paths must be relative, got {value!r}")
    return normalized


def sample_key(dataset: str, video: str) -> str:
    if not dataset:
        raise CachedVGGTError("Dataset tag is required for cached VGGT lookup")
    return f"{dataset}::{_normalized_relative_path(video)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CachedVGGTSample:
    images: list[Image.Image]
    frame_positions: torch.LongTensor
    frame_idx: torch.LongTensor
    features: Dict[str, torch.Tensor]
    sidecar_path: str
    profile_timings: Dict[str, float] | None = None


class CachedVGGTStore:
    """Resolve and validate pre-extracted VGGT features and matching RGB."""

    def __init__(
        self,
        manifest_path: str,
        required_layers: Sequence[int],
        num_frames: int,
        verify_sha256: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if required_layers is None:
            raise ValueError("required_layers must be specified")
        self.required_layers = tuple(str(layer) for layer in required_layers)
        self.num_frames = int(num_frames)
        self.verify_sha256 = bool(verify_sha256)
        self._verified_sidecars: set[Path] = set()
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        self.records = self._read_manifest()

    def _read_manifest(self) -> Dict[str, Mapping]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Cached VGGT manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("schema") != MANIFEST_SCHEMA:
            raise CachedVGGTError(
                f"Expected manifest schema {MANIFEST_SCHEMA!r}, got {document.get('schema')!r}"
            )
        records: Dict[str, Mapping] = {}
        for record in document.get("records", []):
            key = sample_key(record.get("dataset", ""), record.get("video", ""))
            if key in records:
                raise CachedVGGTError(f"Duplicate manifest key: {key}")
            if "sidecar" not in record:
                raise CachedVGGTError(f"Manifest record {key} has no sidecar")
            records[key] = record
        if not records:
            raise CachedVGGTError("Cached VGGT manifest contains no records")
        return records

    def _resolve_sidecar(self, record: Mapping) -> Path:
        sidecar = Path(str(record["sidecar"])).expanduser()
        if not sidecar.is_absolute():
            sidecar = self.manifest_path.parent / sidecar
        sidecar = sidecar.resolve()
        if not sidecar.is_file():
            raise FileNotFoundError(f"VGGT sidecar not found: {sidecar}")
        expected_hash = record.get("sha256")
        if self.verify_sha256:
            if not expected_hash:
                raise CachedVGGTError(f"Manifest record has no SHA256 for sidecar: {sidecar}")
            if sidecar not in self._verified_sidecars:
                if _sha256(sidecar) != expected_hash:
                    raise CachedVGGTError(f"SHA256 mismatch for VGGT sidecar: {sidecar}")
                self._verified_sidecars.add(sidecar)
        return sidecar

    def _select_positions(self, frame_count: int, record: Mapping) -> torch.LongTensor:
        explicit = record.get("frame_positions")
        if explicit is None:
            if frame_count < self.num_frames:
                raise CachedVGGTError(
                    f"Sidecar has {frame_count} frames but {self.num_frames} are required"
                )
            positions = np.linspace(0, frame_count - 1, self.num_frames, dtype=np.int64)
        else:
            positions = np.asarray(explicit, dtype=np.int64)
            if len(positions) != self.num_frames:
                raise CachedVGGTError(
                    f"Manifest frame_positions has {len(positions)} entries; expected {self.num_frames}"
                )
        if len(np.unique(positions)) != len(positions):
            raise CachedVGGTError("Selected cached frame positions contain duplicates")
        if np.any(np.diff(positions) <= 0):
            raise CachedVGGTError("Selected cached frame positions must be strictly increasing")
        if positions[0] < 0 or positions[-1] >= frame_count:
            raise CachedVGGTError("Selected cached frame position is out of range")
        return torch.as_tensor(positions, dtype=torch.long)

    @staticmethod
    def _read_exact_frames(video_path: Path, frame_ids: torch.LongTensor) -> list[Image.Image]:
        if video_path.is_dir():
            frame_files = sorted(path for path in video_path.iterdir() if path.is_file())
            if not frame_files:
                raise CachedVGGTError(f"RGB frame directory is empty: {video_path}")
            if int(frame_ids[-1]) >= len(frame_files):
                raise CachedVGGTError(
                    f"Frame ID {int(frame_ids[-1])} exceeds directory length {len(frame_files)}"
                )
            return [Image.open(frame_files[int(idx)]).convert("RGB") for idx in frame_ids]

        if not video_path.is_file():
            raise FileNotFoundError(f"RGB video not found: {video_path}")
        reader = VideoReader(str(video_path), num_threads=4)
        if int(frame_ids[-1]) >= len(reader):
            raise CachedVGGTError(
                f"Frame ID {int(frame_ids[-1])} exceeds video length {len(reader)}"
            )
        decoded = reader.get_batch(frame_ids.tolist()).asnumpy()
        if len(decoded) != len(frame_ids):
            raise CachedVGGTError("Video decoder returned the wrong number of frames")
        return [Image.fromarray(frame).convert("RGB") for frame in decoded]

    def load(self, dataset: str, video: str, data_root: str) -> CachedVGGTSample:
        profile_enabled = os.environ.get("CONTROLLED_PROFILE", "0") == "1"
        load_started = time.perf_counter()
        key = sample_key(dataset, video)
        if key not in self.records:
            raise CachedVGGTError(f"No exact cached VGGT manifest record for {key}")
        record = self.records[key]
        sidecar_path = self._resolve_sidecar(record)
        resolved_at = time.perf_counter()
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        deserialized_at = time.perf_counter()
        try:
            raw_frame_idx = sidecar["frames"]["frame_idx"]
            layer_map = sidecar["frames"]["aggregated_tokens"]
            metadata = sidecar["meta"]
        except (KeyError, TypeError) as exc:
            raise CachedVGGTError(f"Invalid VGGT sidecar schema: {sidecar_path}") from exc

        if not isinstance(raw_frame_idx, torch.Tensor):
            raise CachedVGGTError("frames.frame_idx must be a torch.Tensor")
        if raw_frame_idx.device.type != "cpu" or raw_frame_idx.dtype != torch.int64:
            raise CachedVGGTError(
                f"frames.frame_idx must be CPU int64, got {raw_frame_idx.device}/{raw_frame_idx.dtype}"
            )
        frame_idx = raw_frame_idx
        expected_meta = {
            "schema": "vggt_aggregated_tokens_v1",
            "input_size": 518,
            "model_image_hw": (518, 518),
            "patch_size": 14,
            "patch_start_idx": VGGT_SPECIAL_TOKENS,
            "feature_dim": VGGT_FEATURE_DIM,
            "token_dtype": "bfloat16",
        }
        if not isinstance(metadata, Mapping):
            raise CachedVGGTError("meta must be a mapping")
        for field, expected in expected_meta.items():
            actual = metadata.get(field)
            if field == "model_image_hw" and isinstance(actual, list):
                actual = tuple(actual)
            if actual != expected:
                raise CachedVGGTError(f"meta.{field}={actual!r}; expected {expected!r}")
        if metadata.get("num_frames") != len(frame_idx):
            raise CachedVGGTError("meta.num_frames does not match frames.frame_idx")
        if not isinstance(metadata.get("source_video"), str):
            raise CachedVGGTError("meta.source_video is missing")
        recorded_source = record.get("recorded_source_video")
        if recorded_source is not None and metadata["source_video"] != recorded_source:
            raise CachedVGGTError("meta.source_video differs from the manifest provenance record")
        metadata_layers = {str(layer) for layer in metadata.get("intermediate_layer_idx", [])}
        if not set(self.required_layers).issubset(metadata_layers):
            raise CachedVGGTError(
                f"meta.intermediate_layer_idx lacks required layers {self.required_layers}"
            )

        if frame_idx.ndim != 1 or len(frame_idx) == 0:
            raise CachedVGGTError("frames.frame_idx must be a non-empty 1-D sequence")
        if torch.unique(frame_idx).numel() != frame_idx.numel():
            raise CachedVGGTError("frames.frame_idx contains duplicate original frame IDs")
        if not bool(torch.all(frame_idx[1:] > frame_idx[:-1])):
            raise CachedVGGTError("frames.frame_idx must be strictly increasing")

        positions = self._select_positions(len(frame_idx), record)
        chosen_ids = frame_idx.index_select(0, positions)
        expected_ids = record.get("frame_idx")
        if expected_ids is not None:
            expected = torch.as_tensor(expected_ids, dtype=torch.long)
            if not torch.equal(frame_idx, expected):
                raise CachedVGGTError(f"Sidecar frame_idx differs from manifest for {key}")

        features: Dict[str, torch.Tensor] = {}
        for layer in self.required_layers:
            if layer not in layer_map:
                raise CachedVGGTError(f"VGGT layer {layer} missing in {sidecar_path}")
            tensor = layer_map[layer]
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                raise CachedVGGTError(f"VGGT layer {layer} must be a CPU torch.Tensor")
            expected_shape = (len(frame_idx), VGGT_SPECIAL_TOKENS + VGGT_PATCH_TOKENS, VGGT_FEATURE_DIM)
            if tuple(tensor.shape) != expected_shape:
                raise CachedVGGTError(
                    f"VGGT layer {layer} has shape {tuple(tensor.shape)}, expected {expected_shape}"
                )
            if tensor.dtype != torch.bfloat16:
                raise CachedVGGTError(
                    f"VGGT layer {layer} has dtype {tensor.dtype}; expected torch.bfloat16"
                )
            if not torch.isfinite(tensor.float()).all():
                raise CachedVGGTError(f"VGGT layer {layer} is not finite floating point data")
            # Special/register/camera tokens are removed at this provenance boundary.
            features[layer] = tensor.index_select(0, positions)[:, VGGT_SPECIAL_TOKENS:].contiguous()

        validated_at = time.perf_counter()
        video_path = Path(data_root).expanduser() / _normalized_relative_path(video)
        images = self._read_exact_frames(video_path.resolve(), chosen_ids)
        decoded_at = time.perf_counter()
        if len(images) != len(chosen_ids):
            raise CachedVGGTError("RGB/VGGT frame count mismatch after exact selection")
        return CachedVGGTSample(
            images=images,
            frame_positions=positions,
            frame_idx=chosen_ids,
            features=features,
            sidecar_path=str(sidecar_path),
            profile_timings=(
                {
                    "manifest_resolve_sha_sec": resolved_at - load_started,
                    "sidecar_deserialize_sec": deserialized_at - resolved_at,
                    "sidecar_validate_select_sec": validated_at - deserialized_at,
                    "rgb_decode_sec": decoded_at - validated_at,
                    "cached_vggt_total_sec": decoded_at - load_started,
                }
                if profile_enabled
                else None
            ),
        )


def write_manifest(path: str, records: Iterable[Mapping]) -> None:
    """Small utility used by cache-preparation tooling and tests."""
    output = Path(path)
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"schema": MANIFEST_SCHEMA, "records": list(records)}, handle, indent=2)
        handle.write("\n")
