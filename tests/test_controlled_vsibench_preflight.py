"""Lightweight filesystem/provenance tests; no model or GPU required."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.evaluation.validate_controlled_vsibench import (
    validate_checkpoint,
    validate_manifest,
)
from scripts.data.build_vsibench_cached_vggt_manifests import inventory_records


class ControlledVSIBenchPreflightTest(unittest.TestCase):
    def test_checkpoint_candidate_and_base_identity(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            base = root / "Qwen3.5-4B"
            base.mkdir()
            checkpoint = root / "checkpoint-100"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(json.dumps({
                "model_type": "qwen3_5", "use_cached_vggt": True,
                "use_geometry_encoder": False, "controlled_fusion_candidate": "a_premerger_cross_attn",
            }))
            (checkpoint / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 100}))
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            (checkpoint / "controlled_vggt_fusion.bin").write_bytes(b"fusion")
            validate_checkpoint(checkpoint, "a_premerger_cross_attn", base)
            with self.assertRaisesRegex(ValueError, "Wrong checkpoint candidate"):
                validate_checkpoint(checkpoint, "b_llm_add", base)
            (checkpoint / "controlled_vggt_fusion.bin").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Incomplete trained checkpoint"):
                validate_checkpoint(checkpoint, "a_premerger_cross_attn", base)

    def test_manifest_exact_pairing_and_candidate_sidecar_root(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            media = root / "media"
            sidecars = root / "sidecars"
            (media / "scannet").mkdir(parents=True)
            (sidecars / "scannet").mkdir(parents=True)
            (media / "scannet" / "scene.mp4").touch()
            (sidecars / "scannet" / "scene.pt").touch()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema": "spatialfocus.cached_vggt.v1",
                "records": [{
                    "dataset": "vsibench", "video": "scannet/scene.mp4",
                    "sidecar": str(sidecars / "scannet" / "scene.pt"),
                    "frame_idx": list(range(32)), "sha256": "a" * 64,
                }],
            }))
            from scripts.evaluation import validate_controlled_vsibench as checks
            original = checks.EXPECTED_VIDEOS
            checks.EXPECTED_VIDEOS = 1
            try:
                validate_manifest(manifest, {"scannet/scene.mp4"}, media, sidecars)
                with self.assertRaisesRegex((FileNotFoundError, ValueError), "sidecar"):
                    validate_manifest(manifest, {"scannet/scene.mp4"}, media, root / "other")
            finally:
                checks.EXPECTED_VIDEOS = original

    def test_inventory_requires_exact_eval_video_set(self):
        inventory = {
            "schema": "vggt_l23_inventory_v1", "scope": "full", "source_tree_unchanged": True,
            "records": [{"scope": "eval", "dataset": "scannet", "sample_id": "scene",
                         "relative_path": "scannet/scene.pt"}],
        }
        self.assertEqual(len(inventory_records(inventory, {("scannet", "scene")})), 1)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            inventory_records(inventory, {("scannet", "another")})


if __name__ == "__main__":
    unittest.main()
