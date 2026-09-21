import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from qwen_vl.data.cached_vggt import CachedVGGTError, CachedVGGTStore, MANIFEST_SCHEMA
from scripts.data.build_cached_vggt_manifest import sidecar_relative_path


class CachedVGGTStoreTest(unittest.TestCase):
    def test_formal_cache_path_mapping_is_exact(self):
        self.assertEqual(
            sidecar_relative_path(Path("scannet/videos/scene0384_00.mp4")),
            Path("scannet/scene0384_00.pt"),
        )
        self.assertEqual(
            sidecar_relative_path(Path("scannet/scene0384_00.mp4")),
            Path("scannet/scene0384_00.pt"),
        )
        self.assertEqual(
            sidecar_relative_path(Path("scannet/notvideos/scene0384_00.mp4")),
            Path("scannet/notvideos/scene0384_00.pt"),
        )

    def _fixture(self, root: Path, frame_idx=(1, 3, 7), layers=(11, 17, 23)):
        video = root / "media" / "scene"
        video.mkdir(parents=True)
        for index in range(8):
            Image.new("RGB", (2, 2), color=(index, 0, 0)).save(video / f"{index:03d}.png")
        layer_map = {
            str(layer): torch.randn(3, 1374, 2048, dtype=torch.bfloat16)
            for layer in layers
        }
        sidecar = root / "scene.pt"
        torch.save(
            {
                "frames": {"frame_idx": torch.tensor(frame_idx), "aggregated_tokens": layer_map},
                "meta": {
                    "schema": "vggt_aggregated_tokens_v1",
                    "source_video": "/source/scene.mp4",
                    "num_frames": 3,
                    "input_size": 518,
                    "model_image_hw": (518, 518),
                    "patch_size": 14,
                    "patch_start_idx": 5,
                    "feature_dim": 2048,
                    "token_dtype": "bfloat16",
                    "intermediate_layer_idx": list(layers),
                },
            },
            sidecar,
        )
        digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": MANIFEST_SCHEMA,
                    "records": [
                        {
                            "dataset": "fixture",
                            "video": "scene",
                            "sidecar": str(sidecar),
                            "sha256": digest,
                            "frame_idx": list(frame_idx),
                            "frame_positions": [0, 2],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return manifest, layer_map

    def test_exact_frame_correspondence_and_patch_only_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, raw = self._fixture(root)
            store = CachedVGGTStore(str(manifest), [11, 17, 23], num_frames=2)
            sample = store.load("fixture", "scene", str(root / "media"))
            self.assertEqual(sample.frame_idx.tolist(), [1, 7])
            self.assertEqual([image.getpixel((0, 0))[0] for image in sample.images], [1, 7])
            self.assertEqual(set(sample.features), {"11", "17", "23"})
            self.assertEqual(tuple(sample.features["23"].shape), (2, 1369, 2048))
            torch.testing.assert_close(sample.features["23"][0], raw["23"][0, 5:])
            torch.testing.assert_close(sample.features["23"][1], raw["23"][2, 5:])

    def test_duplicate_or_reordered_frame_ids_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._fixture(root, frame_idx=(1, 7, 7))
            store = CachedVGGTStore(str(manifest), [23], num_frames=2)
            with self.assertRaises(CachedVGGTError):
                store.load("fixture", "scene", str(root / "media"))

    def test_no_fuzzy_manifest_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._fixture(root)
            store = CachedVGGTStore(str(manifest), [23], num_frames=2)
            with self.assertRaises(CachedVGGTError):
                store.load("fixture", "other/scene", str(root / "media"))

    def test_exact_l23_rejects_multilayer_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._fixture(root)
            store = CachedVGGTStore(
                str(manifest), [23], num_frames=2, require_exact_layers=True
            )
            with self.assertRaisesRegex(CachedVGGTError, "Exact cached layer set required"):
                store.load("fixture", "scene", str(root / "media"))

    def test_exact_l23_accepts_l23_only_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._fixture(root, layers=(23,))
            store = CachedVGGTStore(
                str(manifest), [23], num_frames=2, require_exact_layers=True
            )
            sample = store.load("fixture", "scene", str(root / "media"))
            self.assertEqual(set(sample.features), {"23"})


if __name__ == "__main__":
    unittest.main()
