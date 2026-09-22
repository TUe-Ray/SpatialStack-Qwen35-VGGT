#!/usr/bin/env python3
"""Measure the new Qwen3.5 C1 per-site visual update budget on 32 videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor

from extract_qwen35_presft_features import (
    fixed_device_map,
    load_rgb,
    load_selected_user_prompts,
    save_json,
    sha256_file,
)
from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--forward-frames-root", type=Path, required=True)
    parser.add_argument("--annotation-scannet", type=Path, required=True)
    parser.add_argument("--annotation-route-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-videos", type=int)
    args = parser.parse_args()

    if torch.cuda.device_count() != 2:
        raise RuntimeError("Qwen C1 calibration requires the validated two-TITAN-V placement")
    if any((args.model / name).exists() for name in ("adapter_model.bin", "adapter_model.safetensors", "non_lora_trainables.bin")):
        raise RuntimeError("Refusing to calibrate from a trained candidate checkpoint")
    document = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    if document.get("schema_version") != "c1_calibration_manifest_v1" or document.get("num_samples") != 32:
        raise RuntimeError("Expected the fixed 32-video unlabeled C1 manifest")
    if document.get("source_sample_indices_sha256") != "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e":
        raise RuntimeError("C1 sample split identity differs from the fixed ScanNet audit")
    videos = list(document["videos"])
    if len(videos) != 32 or any(video.get("split") != "train" for video in videos):
        raise RuntimeError("C1 manifest must contain exactly 32 training-split videos")
    if args.limit_videos is not None:
        if not 1 <= args.limit_videos <= 2:
            raise ValueError("Diagnostic calibration limit must be one or two videos")
        videos = videos[:args.limit_videos]
    prompt_paths = (args.annotation_scannet, args.annotation_route_plan)
    prompt_by_id = load_selected_user_prompts(prompt_paths, videos)

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen3_5" or config.text_config.num_hidden_layers != 32:
        raise RuntimeError("Unexpected Qwen base checkpoint")
    config.use_cached_vggt = False
    config.use_geometry_encoder = False
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.size = {"shortest_edge": 200704, "longest_edge": 200704}
    model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
        args.model, config=config, dtype=torch.float16, attn_implementation="sdpa",
        device_map=fixed_device_map("base"), local_files_only=True,
    ).eval()
    model.requires_grad_(False)
    model_device = model.model.language_model.embed_tokens.weight.device

    rows = []
    started = time.perf_counter()
    for index, video in enumerate(videos):
        scene = Path(video["video_path"]).stem
        images, _frame_idx = load_rgb(args.forward_frames_root / "frames" / "scannet" / f"{scene}.pt")
        message = [[{
            "role": "user",
            "content": [*({"type": "image", "image": image} for image in images),
                        {"type": "text", "text": prompt_by_id[str(video["video_sample_id"])]}],
        }]]
        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=prompt, images=[images], padding=True, return_tensors="pt")
        grid = inputs["image_grid_thw"]
        expected = sum(int(row[1]) * int(row[2]) // 4 for row in grid)
        vision_indices = torch.nonzero(inputs["input_ids"][0] == config.image_token_id, as_tuple=False).flatten()
        if len(vision_indices) != expected:
            raise RuntimeError(f"Qwen visual placeholder count mismatch for {scene}")
        before: dict[int, torch.Tensor] = {}
        ratios: dict[int, float] = {}
        handles = []
        for layer in (0, 1, 2):
            def prehook(_module, arguments, layer=layer):
                hidden = arguments[0][0]
                before[layer] = hidden.index_select(0, vision_indices.to(hidden.device)).detach()

            def posthook(_module, _arguments, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                after = hidden[0].index_select(0, vision_indices.to(hidden.device)).detach()
                prior = before.pop(layer)
                if prior.device != after.device:
                    prior = prior.to(after.device)
                ratio = (after.float() - prior.float()).square().mean().sqrt() / prior.float().square().mean().sqrt()
                value = float(ratio.item())
                if not 0 < value < float("inf"):
                    raise RuntimeError(f"Invalid base update ratio at {scene}/L{layer}")
                ratios[layer] = value

            block = model.model.language_model.layers[layer]
            handles.append(block.register_forward_pre_hook(prehook))
            handles.append(block.register_forward_hook(posthook))
        try:
            with torch.inference_mode():
                model.model(**inputs.to(model_device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if set(ratios) != {0, 1, 2}:
            raise RuntimeError(f"Base update ratio capture incomplete for {scene}")
        row = {"scene": scene, "selected_order": int(video["selected_order"]), "ratios": ratios}
        rows.append(row)
        print(json.dumps({"status": "PASS", "index": index, **row}, sort_keys=True), flush=True)

    values = [float(row["ratios"][layer]) for row in rows for layer in (0, 1, 2)]
    artifact = {
        "schema": "qwen35_vggt_base_r0_v1",
        "status": "diagnostic" if args.limit_videos is not None else "formal",
        "base_model": str(args.model.resolve()),
        "base_weight_index_sha256": sha256_file(args.model / "model.safetensors.index.json"),
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
        "annotation_sha256": {path.name: sha256_file(path) for path in prompt_paths},
        "calibration_video_count": len(videos),
        "definition": "median per-video per-L0/1/2 RMS(H_after-H_before)/RMS(H_before), visual placeholders only",
        # Match the audited SpatialFocus C1 convention: torch.median selects
        # the lower middle order statistic for an even 32 x 3 sample count.
        "r0": float(torch.tensor(values, dtype=torch.float32).median().item()),
        "ratios": rows,
        "frames_per_video": 32,
        "max_pixels_per_frame": 200704,
        "optimizer_steps": 0,
        "post_sft_state_loaded": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    save_json(args.output, artifact)
    print(json.dumps({"status": artifact["status"], "r0": artifact["r0"], "videos": len(videos), "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
