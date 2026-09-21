import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/data/validate_controlled_sft_inputs.py"
SPEC = importlib.util.spec_from_file_location("validate_controlled_sft_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    media = tmp_path / "media"
    cache = tmp_path / "cache"
    media.mkdir()
    cache.mkdir()
    datasets = []
    records = []
    for alias, count in (("vlm3r_scannet", 2), ("vlm3r_scannetpp", 1)):
        annotations = []
        for index in range(count):
            video = f"{alias}/videos/{index}.mp4"
            media_path = media / video
            media_path.parent.mkdir(parents=True, exist_ok=True)
            media_path.write_bytes(b"rgb")
            sidecar = cache / alias / f"{index}.pt"
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_bytes(b"sidecar")
            annotations.append({"video": video})
            records.append(
                {
                    "dataset": alias,
                    "video": video,
                    "sidecar": str(sidecar),
                    "sha256": _sha256(sidecar),
                    "frame_idx": list(range(32)),
                }
            )
        annotation_path = tmp_path / f"{alias}.json"
        annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
        datasets.append(
            [alias, str(annotation_path), str(media), str(count), _sha256(annotation_path)]
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": MODULE.MANIFEST_SCHEMA, "records": records}),
        encoding="utf-8",
    )
    return argparse.Namespace(
        dataset=datasets,
        expected_total=3,
        manifest=manifest,
        sidecar_root=cache,
        num_frames=32,
    )


def test_validate_exact_dataset_mixture_and_manifest(tmp_path):
    result = MODULE.validate(_fixture(tmp_path))
    assert result["dataset_counts"] == {
        "vlm3r_scannet": 2,
        "vlm3r_scannetpp": 1,
    }
    assert result["total_samples"] == 3
    assert result["unique_dataset_video_keys"] == 3
    assert result["unique_sidecars"] == 3


def test_validate_rejects_wrong_dataset_count(tmp_path):
    args = _fixture(tmp_path)
    args.dataset[0][3] = "3"
    with pytest.raises(ValueError, match="loaded 2 annotations, expected 3"):
        MODULE.validate(args)


def test_validate_rejects_wrong_annotation_identity(tmp_path):
    args = _fixture(tmp_path)
    args.dataset[0][4] = "0" * 64
    with pytest.raises(ValueError, match="annotation SHA256"):
        MODULE.validate(args)


def test_validate_rejects_missing_manifest_alias(tmp_path):
    args = _fixture(tmp_path)
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    document["records"] = document["records"][:-1]
    args.manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest lacks 1 required"):
        MODULE.validate(args)
