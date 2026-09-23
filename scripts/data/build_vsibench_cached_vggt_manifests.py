#!/usr/bin/env python3
"""Build exact, paired VSI-Bench manifests from the audited VGGT inventory."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pyarrow.parquet as pq

if __package__:
    from .build_formal_cached_vggt_manifests import metadata, sha256
else:
    from build_formal_cached_vggt_manifests import metadata, sha256


SCHEMA = "spatialfocus.cached_vggt.v1"
EXPECTED_QA = 5130
EXPECTED_VIDEOS = 288
EVAL_INPUT_VIEW_ROOT = Path(
    "/scratch-shared/geusdd/shaoruei/VLM3R/spatial_features/"
    "vggt_missing_staging/_input_views"
)


def video_keys(parquets: list[Path]) -> tuple[int, set[tuple[str, str]]]:
    keys: set[tuple[str, str]] = set()
    count = 0
    for parquet in parquets:
        table = pq.read_table(parquet, columns=["dataset", "scene_name"])
        count += table.num_rows
        for row in table.to_pylist():
            dataset, scene = row["dataset"], row["scene_name"]
            if not isinstance(dataset, str) or not isinstance(scene, str):
                raise ValueError(f"Invalid VSI-Bench dataset/scene in {parquet}: {row}")
            if dataset not in {"arkitscenes", "scannet", "scannetpp"}:
                raise ValueError(f"Unexpected VSI-Bench dataset: {dataset}")
            if not scene or Path(scene).name != scene or "/" in scene or ".." in scene:
                raise ValueError(f"Unsafe VSI-Bench scene name: {scene!r}")
            keys.add((dataset, scene))
    if count != EXPECTED_QA or len(keys) != EXPECTED_VIDEOS:
        raise ValueError(f"Expected {EXPECTED_QA} QA / {EXPECTED_VIDEOS} videos, got {count} / {len(keys)}")
    return count, keys


def inventory_records(inventory: dict, keys: set[tuple[str, str]]) -> list[dict]:
    if inventory.get("schema") != "vggt_l23_inventory_v1" or inventory.get("scope") != "full":
        raise ValueError("Expected the audited full L23 inventory")
    if inventory.get("source_tree_unchanged") is not True:
        raise ValueError("Inventory does not attest to an unchanged source cache")
    records = {}
    for record in inventory["records"]:
        if record.get("scope") != "eval":
            continue
        key = (record["dataset"], record["sample_id"])
        if key in records:
            raise ValueError(f"Duplicate inventory evaluation record: {key}")
        if record["relative_path"] != f"{key[0]}/{key[1]}.pt":
            raise ValueError(f"Unexpected sidecar path for {key}: {record['relative_path']}")
        records[key] = record
    if set(records) != keys:
        raise ValueError(
            f"Inventory/evaluation video mismatch: missing={sorted(keys - set(records))[:5]} "
            f"extra={sorted(set(records) - keys)[:5]}"
        )
    return [records[key] for key in sorted(keys)]


def inspect_record(record: dict, media_root: Path) -> tuple[dict, dict]:
    dataset, scene = record["dataset"], record["sample_id"]
    relative_video = f"{dataset}/{scene}.mp4"
    video = media_root / relative_video
    if not video.is_file():
        raise FileNotFoundError(f"Missing VSI-Bench RGB video: {video}")
    a_sidecar = Path(record["output_sidecar"]).resolve()
    b_sidecar = Path(record["source_sidecar"]).resolve()
    if not a_sidecar.is_file() or a_sidecar.stat().st_size != record["output_size"]:
        raise FileNotFoundError(f"Missing or size-mismatched L23 sidecar: {a_sidecar}")
    if not b_sidecar.is_file():
        raise FileNotFoundError(f"Missing three-layer sidecar: {b_sidecar}")
    expected_a_hash = record["output_sha256"]
    if not isinstance(expected_a_hash, str) or len(expected_a_hash) != 64:
        raise ValueError(f"Invalid audited L23 SHA256: {a_sidecar}")
    a_ids, a_source = metadata(a_sidecar, {"23"}, exact_layers=True)
    b_ids, b_source = metadata(b_sidecar, {"11", "17", "23"})
    if a_ids != b_ids or a_source != b_source:
        raise ValueError(f"L23/three-layer frame or source mismatch: {relative_video}")
    expected_staging_source = EVAL_INPUT_VIEW_ROOT / f"eval_{dataset}" / f"{scene}.mp4"
    expected_original_suffix = f"/{dataset}/videos/{scene}.mp4"
    if a_source != str(expected_staging_source) and not a_source.endswith(expected_original_suffix):
        raise ValueError(f"VGGT source provenance does not match {relative_video}: {a_source}")
    common = {
        "dataset": "vsibench",
        "video": relative_video,
        "frame_idx": a_ids,
        "recorded_source_video": a_source,
    }
    return (
        {**common, "sidecar": str(a_sidecar), "sha256": expected_a_hash},
        {**common, "sidecar": str(b_sidecar), "sha256": sha256(b_sidecar)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, action="append", required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-a", type=Path, required=True)
    parser.add_argument("--output-b", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.output_a.resolve() == args.output_b.resolve():
        parser.error("Workers must be positive and A/B output paths must differ")
    for output in (args.output_a, args.output_b):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    count, keys = video_keys(args.parquet)
    records = inventory_records(inventory, keys)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pairs = list(pool.map(lambda record: inspect_record(record, args.media_root), records))
    for output, index in ((args.output_a, 0), (args.output_b, 1)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"schema": SCHEMA, "records": [pair[index] for pair in pairs]}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"VSI_MANIFEST output={output} qa={count} videos={len(pairs)} sha256={sha256(output)}", flush=True)


if __name__ == "__main__":
    main()
