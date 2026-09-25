"""Lossless, provenance-checked cache of the exact RGB frames selected by VGGT."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import decord
import numpy as np
from PIL import Image


SCHEMA = "spatialfocus.exact_rgb_frames.v1"


class RGBFrameCacheError(RuntimeError):
    """A present RGB cache entry failed its provenance or integrity checks."""


def _file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        # GPFS can report a different st_dev for the same file on another node.
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


class ExactRGBFrameCache:
    """Cache uint8 decoded pixels without changing frame selection or Qwen preprocessing."""

    def __init__(self, root: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, video_path: Path, frame_ids: list[int]) -> Path:
        key = json.dumps(
            [str(video_path.resolve()), frame_ids], separators=(",", ":")
        ).encode("utf-8")
        name = hashlib.sha256(key).hexdigest()
        return self.root / name[:2] / f"{name}.npz"

    @staticmethod
    def _read(path: Path, expected: dict) -> list[Image.Image]:
        try:
            with np.load(path, allow_pickle=False) as cache:
                if set(cache.files) != {"metadata", "rgb"}:
                    raise RGBFrameCacheError(f"RGB cache has unexpected entries: {path}")
                stored = json.loads(cache["metadata"].tobytes().decode("utf-8"))
                if stored.get("schema") != SCHEMA:
                    raise RGBFrameCacheError(f"RGB cache schema mismatch: {path}")
                for key, value in expected.items():
                    actual = stored.get(key)
                    if key == "source" and isinstance(actual, dict):
                        # Accept entries made before st_dev was removed from the
                        # cross-node identity; all stable source fields must match.
                        matches = all(actual.get(field) == field_value for field, field_value in value.items())
                    else:
                        matches = actual == value
                    if not matches:
                        raise RGBFrameCacheError(f"RGB cache {key} mismatch: {path}")
                rgb = cache["rgb"]
            if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
                raise RGBFrameCacheError(f"RGB cache tensor shape/dtype mismatch: {path}")
            if len(rgb) != len(expected["frame_ids"]):
                raise RGBFrameCacheError(f"RGB cache frame count mismatch: {path}")
            if hashlib.sha256(memoryview(rgb)).hexdigest() != stored.get("rgb_sha256"):
                raise RGBFrameCacheError(f"RGB cache pixel digest mismatch: {path}")
            return [Image.fromarray(frame, mode="RGB") for frame in rgb]
        except RGBFrameCacheError:
            raise
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            raise RGBFrameCacheError(f"Cannot read exact RGB cache {path}: {exc}") from exc

    def load_or_create(
        self,
        video_path: Path,
        frame_ids: list[int],
        decode: Callable[[], list[Image.Image]],
    ) -> tuple[list[Image.Image], bool]:
        video_path = video_path.resolve()
        expected = {
            "source": _file_identity(video_path),
            "frame_ids": frame_ids,
            "decoder": {"name": "decord", "version": getattr(decord, "__version__", "unknown")},
        }
        path = self._path(video_path, frame_ids)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return self._read(path, expected), True
        # One writer per video/frame selection across all workers and ranks.
        with (path.parent / f"{path.name}.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                return self._read(path, expected), True

            images = decode()
            if len(images) != len(frame_ids):
                raise RGBFrameCacheError(f"Decoder returned the wrong number of frames: {video_path}")
            rgb = np.stack([np.asarray(image.convert("RGB"), dtype=np.uint8) for image in images])
            if _file_identity(video_path) != expected["source"]:
                raise RGBFrameCacheError(f"Source video changed during RGB cache build: {video_path}")
            metadata = {
                "schema": SCHEMA,
                **expected,
                "rgb_sha256": hashlib.sha256(memoryview(rgb)).hexdigest(),
            }
            descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    np.savez_compressed(
                        output,
                        rgb=rgb,
                        metadata=np.frombuffer(json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8),
                    )
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return images, False
