#!/usr/bin/env python3
"""Extract fixed-ScanNet, 17-level pre-SFT Qwen3.5 depth representations.

This script only performs frozen forward passes.  A/B formal runs require a
verified 32-video C1 artifact; ``--smoke-zero-gain`` is diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry
from qwen_vl.model.qwen35_c1_init import apply_qwen35_c1_artifact, initialize_qwen35_c1


LLM_LAYERS = (0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31)
FEATURE_LEVELS = ("visual_output", "fusion_output", "projected_features") + tuple(
    f"layer_{layer}" for layer in LLM_LAYERS
)
MODEL_LABELS = {
    "base": "qwen35_base_presft",
    "a_premerger_cross_attn": "qwen35_vggt_a_c1_presft",
    "b_llm_add": "qwen35_vggt_b_c1_presft",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_device_map(candidate: str) -> dict[str, int | str]:
    result: dict[str, int | str] = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 1,
        "model.language_model.rotary_emb": 1,
        "lm_head": 0,
    }
    for layer in range(32):
        result[f"model.language_model.layers.{layer}"] = 0 if layer < 5 else ("cpu" if layer < 11 else 1)
    if candidate != "base":
        result["model.controlled_vggt_fusion"] = 0
        # Duplicate state-dict prefix for the model's public module alias.
        result["controlled_vggt_fusion"] = 0
    return result


def grid14(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tokens.ndim != 2 or tokens.shape[0] != height * width:
        raise ValueError(f"Token/grid mismatch: {tuple(tokens.shape)} vs {height}x{width}")
    value = tokens.detach().to(device="cpu", dtype=torch.float32).T.reshape(1, -1, height, width)
    value = F.interpolate(value, size=(14, 14), mode="bilinear", align_corners=False)
    if not torch.isfinite(value).all():
        raise RuntimeError("Non-finite extracted feature")
    return value.squeeze(0).permute(1, 2, 0).contiguous().to(torch.float16)


def load_rgb(frame_path: Path) -> tuple[list[Image.Image], torch.Tensor]:
    payload = torch.load(frame_path, map_location="cpu", weights_only=False)
    frames, indices = payload["frames_rgb_uint8"], payload["source_frame_indices"]
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[0] != 32 or frames.shape[-1] != 3:
        raise ValueError(f"Invalid RGB cache {frame_path}")
    if indices.dtype != torch.int64 or indices.numel() != 32 or not bool((indices[1:] > indices[:-1]).all()):
        raise ValueError(f"Invalid 32-frame ordering in {frame_path}")
    return [Image.fromarray(frame.numpy()).convert("RGB") for frame in frames], indices


def load_vggt(
    *, candidate: str, record: dict[str, Any], frame_idx: torch.Tensor,
    manifest_records: dict[str, dict[str, Any]], manifest_path: Path, sidecar_root: Path | None,
) -> tuple[dict[str, torch.Tensor], str]:
    video = str(record["video_path"])
    manifest_record = manifest_records.get(video)
    if manifest_record is None:
        raise FileNotFoundError(f"VGGT manifest lacks {video}")
    if sidecar_root is None:
        raw = Path(manifest_record["sidecar"])
        sidecar = raw if raw.is_absolute() else manifest_path.parent / raw
    else:
        sidecar = sidecar_root / "scannet" / f"{Path(video).stem}.pt"
    sidecar = sidecar.resolve()
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    actual_sha = sha256_file(sidecar)
    if actual_sha != manifest_record.get("sha256"):
        raise RuntimeError(f"VGGT SHA256 mismatch for {sidecar}")
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    sidecar_idx = payload["frames"]["frame_idx"]
    if not torch.equal(sidecar_idx, frame_idx) or list(manifest_record["frame_idx"]) != frame_idx.tolist():
        raise RuntimeError(f"VGGT/RGB/manifest frame ordering mismatch for {video}")
    meta = payload["meta"]
    if meta.get("schema") != "vggt_aggregated_tokens_v1" or meta.get("patch_start_idx") != 5:
        raise RuntimeError(f"Unexpected VGGT schema for {video}")
    required = ("23",) if candidate == "a_premerger_cross_attn" else ("11", "17", "23")
    layer_map = payload["frames"]["aggregated_tokens"]
    if candidate == "a_premerger_cross_attn" and set(layer_map) != {"23"}:
        raise RuntimeError(f"Candidate A sidecar is not patch-only L23 for {video}")
    features = {}
    for layer in required:
        value = layer_map[layer]
        if value.shape != (32, 1374, 2048) or value.dtype != torch.bfloat16:
            raise RuntimeError(f"Unexpected VGGT L{layer} shape/dtype for {video}")
        features[layer] = value[:, 5:].contiguous()
    return features, actual_sha


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_selected_user_prompts(paths: tuple[Path, Path], videos: list[dict[str, Any]]) -> dict[str, str]:
    """Use only the fixed sample's real SFT human turn, never its answer."""
    required = {str(video["video_sample_id"]): str(video["video_path"]) for video in videos}
    prompts: dict[str, str] = {}
    for path in paths:
        for item in json.loads(path.read_text(encoding="utf-8")):
            sample_id = str(item.get("id", ""))
            if sample_id not in required:
                continue
            if str(item.get("video")) != required[sample_id]:
                raise RuntimeError(f"Annotation/sample video mismatch for {sample_id}")
            human = next(
                (str(turn.get("value", "")) for turn in item.get("conversations", [])
                 if str(turn.get("from", "")).lower() in {"human", "user"}),
                "",
            )
            if not human.strip() or sample_id in prompts:
                raise RuntimeError(f"Missing or duplicate human prompt for {sample_id}")
            prompts[sample_id] = human
    if set(prompts) != set(required):
        raise RuntimeError(f"Missing real user prompts for {len(set(required) - set(prompts))} fixed samples")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(MODEL_LABELS), required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--forward-frames-root", type=Path, required=True)
    parser.add_argument("--annotation-scannet", type=Path, required=True)
    parser.add_argument("--annotation-route-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sidecar-root", type=Path)
    parser.add_argument("--c1-artifact", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--smoke-zero-gain", action="store_true")
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()

    if torch.cuda.device_count() != 2:
        raise RuntimeError("Local extractor requires both validated TITAN V GPUs")
    if args.max_pixels != 200704:
        raise ValueError("Formal Qwen pre-SFT extraction fixes max_pixels=200704")
    if args.candidate != "base":
        if args.manifest is None:
            raise ValueError("A/B extraction requires an exact VGGT manifest")
        if args.smoke_zero_gain:
            if args.c1_artifact or args.limit_videos is None or args.limit_videos > 2:
                raise ValueError("Zero-gain diagnostic is restricted to at most two videos")
        elif args.c1_artifact is None or args.calibration_manifest is None:
            raise ValueError("Formal A/B extraction requires C1 artifact and fixed calibration manifest")
    elif args.manifest or args.c1_artifact or args.smoke_zero_gain:
        raise ValueError("Base comparator must not load VGGT or a fusion artifact")
    if any((args.model / name).exists() for name in ("adapter_model.bin", "adapter_model.safetensors", "non_lora_trainables.bin")):
        raise RuntimeError("Refusing candidate-trained checkpoint")

    sample_sha = sha256_file(args.sample_indices)
    if sample_sha != "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e":
        raise RuntimeError("Fixed ScanNet split checksum differs from the SpatialFocus audit")
    samples = json.loads(args.sample_indices.read_text(encoding="utf-8"))
    videos = [video for video in samples["videos"] if video.get("source_dataset") == "scannet"]
    videos.sort(key=lambda video: (int(video["selected_order"]), str(video["video_path"])))
    if len(videos) != 1199 or sum(video["split"] == "train" for video in videos) != 1006:
        raise RuntimeError("Expected fixed 1,199-video ScanNet split (1,006 train/193 val)")
    annotation_paths = (args.annotation_scannet, args.annotation_route_plan)
    prompt_by_id = load_selected_user_prompts(annotation_paths, videos)
    annotation_sha = {path.name: sha256_file(path) for path in annotation_paths}
    selected_videos = videos[args.start_index:]
    if args.limit_videos is not None:
        selected_videos = selected_videos[:args.limit_videos]
    if not selected_videos:
        raise ValueError("No selected videos")

    manifest_records: dict[str, dict[str, Any]] = {}
    manifest_sha = None
    if args.manifest is not None:
        manifest_sha = sha256_file(args.manifest)
        document = json.loads(args.manifest.read_text(encoding="utf-8"))
        if document.get("schema") != "spatialfocus.cached_vggt.v1":
            raise RuntimeError("Unexpected VGGT manifest schema")
        manifest_records = {str(record["video"]): record for record in document["records"]}

    artifact = None
    artifact_sha = None
    if args.c1_artifact is not None:
        artifact_sha = sha256_file(args.c1_artifact)
        artifact = json.loads(args.c1_artifact.read_text(encoding="utf-8"))
        if artifact.get("calibration_manifest_sha256") != sha256_file(args.calibration_manifest):
            raise RuntimeError("C1 artifact does not match the supplied 32-video calibration manifest")
        if artifact.get("base_weight_index_sha256") != sha256_file(args.model / "model.safetensors.index.json"):
            raise RuntimeError("C1 artifact does not match the supplied base-model weights")
        if artifact.get("vggt_manifest_sha256") != manifest_sha:
            raise RuntimeError("C1 artifact does not match the supplied VGGT manifest")
        if artifact.get("annotation_sha256") != annotation_sha:
            raise RuntimeError("C1 artifact does not match the fixed human prompts")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen3_5" or config.text_config.num_hidden_layers != 32:
        raise RuntimeError("Unexpected base model architecture")
    config.use_geometry_encoder = False
    config.use_cached_vggt = args.candidate != "base"
    if config.use_cached_vggt:
        config.controlled_fusion_candidate = args.candidate
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.size = {"shortest_edge": args.max_pixels, "longest_edge": args.max_pixels}
    device_map = fixed_device_map(args.candidate)
    model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
        args.model, config=config, dtype=torch.float16, attn_implementation="sdpa",
        device_map=device_map, local_files_only=True,
    ).eval()
    if args.candidate != "base":
        fusion = model.model.controlled_vggt_fusion
        if artifact is None:
            initialize_qwen35_c1(fusion)
        else:
            apply_qwen35_c1_artifact(fusion, artifact)
    if any(parameter.requires_grad for parameter in model.parameters()):
        model.requires_grad_(False)
    if any(parameter.device.type == "meta" for parameter in model.parameters()):
        # CPU-offloaded blocks can have meta placeholders under Accelerate;
        # these are frozen and dispatched from the offload weights at forward.
        if any(parameter.requires_grad and parameter.device.type == "meta" for parameter in model.parameters()):
            raise RuntimeError("Unexpected trainable meta parameter")

    label = MODEL_LABELS[args.candidate] + ("_zero_gain_smoke" if args.smoke_zero_gain else "")
    feature_root = args.output_root / "features" / label
    run_meta = {
        "schema": "qwen35_vggt_presft_features_v1",
        "status": "diagnostic_zero_gain" if args.smoke_zero_gain else "formal_or_in_progress",
        "candidate": args.candidate,
        "model_label": label,
        "model_root": str(args.model.resolve()),
        "model_weight_index_sha256": sha256_file(args.model / "model.safetensors.index.json"),
        "sample_indices_sha256": sample_sha,
        "annotation_sha256": annotation_sha,
        "prompt_source": "fixed_sample_id_real_sft_human_turn_only",
        "vggt_manifest_sha256": manifest_sha,
        "c1_artifact_sha256": artifact_sha,
        "feature_levels": FEATURE_LEVELS,
        "llm_layer_indexing": "L -> hidden_states[L+1]; final L31 captured after final norm",
        "frames_per_video": 32,
        "selected_target_frames_per_video": 2,
        "target_grid": [14, 14],
        "max_pixels_per_frame": args.max_pixels,
        "attention_implementation": "sdpa",
        "dtype": "float16",
        "device_map": device_map,
        "optimizer_steps": 0,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    run_meta_path = args.output_root / f"{label}_run_manifest.json"
    if run_meta_path.exists():
        existing = json.loads(run_meta_path.read_text(encoding="utf-8"))
        for key in ("candidate", "model_weight_index_sha256", "sample_indices_sha256", "annotation_sha256", "vggt_manifest_sha256", "c1_artifact_sha256", "max_pixels_per_frame", "git_commit"):
            if existing.get(key) != run_meta.get(key):
                raise RuntimeError(f"Refusing to mix feature runs with different {key}")
    else:
        save_json(run_meta_path, run_meta)

    model_device = model.model.language_model.embed_tokens.weight.device
    for video_number, video in enumerate(selected_videos, start=args.start_index):
        started = time.perf_counter()
        chosen = {int(frame["frame_index"]): str(frame["frame_sample_id"]) for frame in video["frames"]}
        if len(chosen) != 2 or not all(0 <= index < 32 for index in chosen):
            raise RuntimeError("Expected two distinct selected target frame positions")
        expected_paths = [feature_root / level / f"frame_{fsid}.pt" for level in FEATURE_LEVELS for fsid in chosen.values()]
        if all(path.is_file() for path in expected_paths):
            print(json.dumps({"status": "SKIP_COMPLETE", "scene": Path(video["video_path"]).stem}), flush=True)
            continue
        scene = Path(video["video_path"]).stem
        frame_path = args.forward_frames_root / "frames" / "scannet" / f"{scene}.pt"
        images, frame_idx = load_rgb(frame_path)
        features = None
        sidecar_sha = None
        if args.candidate != "base":
            features, sidecar_sha = load_vggt(
                candidate=args.candidate, record=video, frame_idx=frame_idx,
                manifest_records=manifest_records, manifest_path=args.manifest,
                sidecar_root=args.sidecar_root,
            )
        message = [[{
            "role": "user",
            "content": [*({"type": "image", "image": image} for image in images),
                        {"type": "text", "text": prompt_by_id[str(video["video_sample_id"])]}],
        }]]
        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=prompt, images=[images], padding=True, return_tensors="pt")
        grid = inputs["image_grid_thw"]
        if grid.shape != (32, 3) or not bool((grid[:, 0] == 1).all()):
            raise RuntimeError("Expected 32 temporal-1 Qwen image grids")
        merged_shapes = [(int(row[1]) // 2, int(row[2]) // 2) for row in grid]
        raw_shapes = [(int(row[1]), int(row[2])) for row in grid]
        image_positions = torch.nonzero(inputs["input_ids"][0] == config.image_token_id, as_tuple=False).flatten()
        sizes = [height * width for height, width in merged_shapes]
        if image_positions.numel() != sum(sizes):
            raise RuntimeError("Qwen image placeholder count differs from merged visual patches")
        starts = [0]
        for size in sizes:
            starts.append(starts[-1] + size)
        selected_positions = {
            frame: image_positions[starts[frame]:starts[frame + 1]] for frame in chosen
        }
        captured: dict[str, Any] = {"layers": {}, "mergers": []}
        handles = []

        def capture_visual(_module, _args, output):
            captured["visual"] = output.last_hidden_state.detach().cpu()
            pooled = output.pooler_output
            captured["native_projected"] = torch.cat(pooled, dim=0).detach().cpu() if isinstance(pooled, list) else pooled.detach().cpu()

        def capture_merger(_module, _args, output):
            captured["mergers"].append(output.detach().cpu())

        def capture_a_fusion(_module, _args, output):
            captured["fusion"] = output.detach().cpu()

        def capture_b_layer0(_module, layer_args):
            hidden = layer_args[0]
            captured["fusion"] = {
                frame: hidden[0].index_select(0, indices.to(hidden.device)).detach().cpu()
                for frame, indices in selected_positions.items()
            }

        def capture_layer(level: int):
            def hook(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                captured["layers"][level] = {
                    frame: value[0].index_select(0, indices.to(value.device)).detach().cpu()
                    for frame, indices in selected_positions.items()
                }
            return hook

        handles.append(model.model.visual.register_forward_hook(capture_visual))
        handles.append(model.model.visual.merger.register_forward_hook(capture_merger))
        if args.candidate == "a_premerger_cross_attn":
            handles.append(model.model.controlled_vggt_fusion.premerger_cross_attention.register_forward_hook(capture_a_fusion))
        elif args.candidate == "b_llm_add":
            handles.append(model.model.language_model.layers[0].register_forward_pre_hook(capture_b_layer0))
        for layer in LLM_LAYERS:
            module = model.model.language_model.norm if layer == 31 else model.model.language_model.layers[layer]
            handles.append(module.register_forward_hook(capture_layer(layer)))
        inputs = inputs.to(model_device)
        if features is not None:
            inputs["cached_vggt_features"] = [features]
            inputs["cached_vggt_frame_idx"] = [frame_idx]
        try:
            with torch.inference_mode():
                model.model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured["layers"]) != set(LLM_LAYERS):
            raise RuntimeError(f"Missing LLM layer captures for {scene}")
        raw_sizes = [height * width for height, width in raw_shapes]
        raw_starts = [0]
        for size in raw_sizes:
            raw_starts.append(raw_starts[-1] + size)
        visual = captured["visual"]
        projected = captured["mergers"][-1] if args.candidate == "a_premerger_cross_attn" else captured["native_projected"]
        fusion = captured.get("fusion", visual)
        if visual.shape[0] != sum(raw_sizes) or projected.shape[0] != sum(sizes):
            raise RuntimeError(f"Qwen visual/merger token count mismatch for {scene}")
        if args.candidate == "a_premerger_cross_attn" and len(captured["mergers"]) < 2:
            raise RuntimeError("Candidate A did not execute its post-fusion native merger")
        for frame, fsid in chosen.items():
            h, w = raw_shapes[frame]
            mh, mw = merged_shapes[frame]
            raw_slice = slice(raw_starts[frame], raw_starts[frame + 1])
            merged_slice = slice(starts[frame], starts[frame + 1])
            frame_features = {
                "visual_output": grid14(visual[raw_slice], h, w),
                "fusion_output": grid14(fusion[frame] if isinstance(fusion, dict) else fusion[raw_slice],
                                        mh if isinstance(fusion, dict) else h,
                                        mw if isinstance(fusion, dict) else w),
                "projected_features": grid14(projected[merged_slice], mh, mw),
            }
            frame_features.update({
                f"layer_{layer}": grid14(captured["layers"][layer][frame], mh, mw)
                for layer in LLM_LAYERS
            })
            for level, value in frame_features.items():
                target = feature_root / level / f"frame_{fsid}.pt"
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(value, target)
        print(json.dumps({
            "status": "PASS", "video_index": video_number, "scene": scene,
            "candidate": args.candidate, "selected_frames": sorted(chosen),
            "sidecar_sha256": sidecar_sha, "feature_levels": len(FEATURE_LEVELS),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }, sort_keys=True), flush=True)

    if not args.smoke_zero_gain:
        complete_videos = sum(
            all(
                (feature_root / level / f"frame_{frame['frame_sample_id']}.pt").is_file()
                for level in FEATURE_LEVELS for frame in video["frames"]
            )
            for video in videos
        )
        run_meta["complete_videos"] = complete_videos
        run_meta["status"] = "complete" if complete_videos == len(videos) else "in_progress"
        run_meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_json(run_meta_path, run_meta)
        print(json.dumps({"status": run_meta["status"], "candidate": args.candidate,
                          "complete_videos": complete_videos, "expected_videos": len(videos)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
