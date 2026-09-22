#!/usr/bin/env python3
"""Calibrate deterministic Qwen3.5/VGGT C1 A or B from 32 unlabeled videos."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoProcessor

from extract_qwen35_presft_features import (
    fixed_device_map,
    load_rgb,
    load_selected_user_prompts,
    load_vggt,
    save_json,
    sha256_file,
)
from qwen_vl.model.controlled_vggt_fusion import align_vggt_to_qwen_premerger
from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry
from qwen_vl.model.qwen35_c1_init import SCHEME_VERSION, initialize_qwen35_c1


def rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def lower_median(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError("C1 median received no finite observations")
    return float(torch.tensor(values, dtype=torch.float32).median().item())


def summary(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "median": lower_median(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def prepare_sample(
    *, video: dict[str, Any], args: argparse.Namespace, processor: Any,
    prompt_by_id: dict[str, str], manifest_records: dict[str, dict[str, Any]], model_device: torch.device,
) -> tuple[Any, dict[str, torch.Tensor], torch.Tensor]:
    scene = Path(video["video_path"]).stem
    images, frame_idx = load_rgb(args.forward_frames_root / "frames" / "scannet" / f"{scene}.pt")
    features, _sha = load_vggt(
        candidate=args.candidate, record=video, frame_idx=frame_idx,
        manifest_records=manifest_records, manifest_path=args.manifest, sidecar_root=args.sidecar_root,
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
        raise RuntimeError(f"Invalid Qwen frame grid for {scene}")
    expected = sum(int(row[1]) * int(row[2]) // 4 for row in grid)
    if int((inputs["input_ids"] == 248056).sum()) != expected:
        raise RuntimeError(f"Qwen image-placeholder count mismatch for {scene}")
    return inputs.to(model_device), features, frame_idx


def calibrate_a(
    *, model: Any, videos: list[dict[str, Any]], args: argparse.Namespace,
    processor: Any, prompt_by_id: dict[str, str], manifest_records: dict[str, dict[str, Any]],
    model_device: torch.device, r0: float,
) -> dict[str, Any]:
    block = model.model.controlled_vggt_fusion.premerger_cross_attention
    block.set_c1_state(enabled=True, qk_scale=1.0, residual_gain=0.0)
    heads = block.attention.num_heads
    head_dim = block.attention.head_dim
    count, total, total_sq = 0, 0.0, 0.0
    with torch.inference_mode():
        for video in videos:
            inputs, features, _frame_idx = prepare_sample(
                video=video, args=args, processor=processor, prompt_by_id=prompt_by_id,
                manifest_records=manifest_records, model_device=model_device,
            )
            grid = inputs["image_grid_thw"]
            visual = model.model.get_image_features(
                inputs["pixel_values"], grid, return_dict=True,
            ).last_hidden_state.to(model_device, dtype=torch.float16)
            aligned = align_vggt_to_qwen_premerger(features["23"].to(model_device, dtype=torch.float16), grid)
            offset = 0
            for row in grid:
                size = int(row[1]) * int(row[2])
                v = visual[offset:offset + size]
                g = aligned[offset:offset + size]
                q = block.query(block.visual_norm(v)).reshape(size, heads, head_dim).transpose(0, 1).float()
                k = block.key(block.geometry_norm(g)).reshape(size, heads, head_dim).transpose(0, 1).float()
                logits = (q @ k.transpose(-1, -2)) * (head_dim ** -0.5)
                count += logits.numel()
                total += float(logits.sum().item())
                total_sq += float(logits.square().sum().item())
                offset += size
            print(json.dumps({"stage": "A_qk", "scene": Path(video["video_path"]).stem}), flush=True)
    raw_std = math.sqrt(max(0.0, total_sq / count - (total / count) ** 2))
    if not math.isfinite(raw_std) or raw_std <= 0:
        raise RuntimeError("Candidate A raw Q/K logits have invalid standard deviation")
    qk_scale = 1.0 / math.sqrt(raw_std)
    block.set_c1_state(enabled=True, qk_scale=qk_scale, residual_gain=1.0)

    # Cache only the deterministic visual stream and the gain-one C1 branch;
    # no QA answer, target, gradient, or fitted model state is retained.
    cached: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        for video in videos:
            inputs, features, _frame_idx = prepare_sample(
                video=video, args=args, processor=processor, prompt_by_id=prompt_by_id,
                manifest_records=manifest_records, model_device=model_device,
            )
            grid = inputs["image_grid_thw"]
            visual = model.model.get_image_features(
                inputs["pixel_values"], grid, return_dict=True,
            ).last_hidden_state.to(model_device, dtype=torch.float16)
            fused = block(visual, features["23"].to(model_device, dtype=torch.float16), grid)
            update = fused - visual
            base_embed = model.model.visual.merger(visual)
            cached.append((visual.cpu(), update.cpu(), base_embed.cpu()))
            print(json.dumps({"stage": "A_branch", "scene": Path(video["video_path"]).stem}), flush=True)

    def projected_ratios(gain: float) -> list[float]:
        ratios = []
        with torch.inference_mode():
            for visual_cpu, update_cpu, base_cpu in cached:
                visual = visual_cpu.to(model_device)
                update = update_cpu.to(model_device)
                projected = model.model.visual.merger(visual + float(gain) * update)
                ratios.append(float((rms(projected - base_cpu.to(projected.device)) / rms(base_cpu)).item()))
        return ratios

    upper = 1.0
    bracket = []
    while True:
        observed = lower_median(projected_ratios(upper))
        bracket.append({"gain": upper, "median_ratio": observed})
        if observed >= r0:
            break
        upper *= 2.0
        if upper > 2.0 ** 20:
            raise RuntimeError("Candidate A could not bracket its C1 projected residual gain")
    lower = 0.0
    for _ in range(32):
        middle = (lower + upper) / 2.0
        if lower_median(projected_ratios(middle)) < r0:
            lower = middle
        else:
            upper = middle
    gain = (lower + upper) / 2.0
    observed = summary(projected_ratios(gain))
    if abs(observed["median"] / r0 - 1.0) > 0.05:
        raise RuntimeError("Candidate A calibrated projected ratio differs from r0 by more than 5%")
    block.set_c1_state(enabled=True, qk_scale=qk_scale, residual_gain=gain)
    return {
        "qk_scale": qk_scale,
        "raw_qk_logit_std": raw_std,
        "calibrated_qk_logit_std": raw_std * qk_scale * qk_scale,
        "residual_gain": gain,
        "effective_projected_delta_over_base": observed,
        "bracket": bracket,
    }


def calibrate_b(
    *, model: Any, videos: list[dict[str, Any]], args: argparse.Namespace,
    processor: Any, prompt_by_id: dict[str, str], manifest_records: dict[str, dict[str, Any]],
    model_device: torch.device, r0: float,
) -> dict[str, Any]:
    fusion = model.model.controlled_vggt_fusion
    sources = ("11", "17", "23")
    moments = {source: [0, 0.0] for source in sources}
    with torch.inference_mode():
        for video in videos:
            inputs, features, _frame_idx = prepare_sample(
                video=video, args=args, processor=processor, prompt_by_id=prompt_by_id,
                manifest_records=manifest_records, model_device=model_device,
            )
            for source in sources:
                projector = fusion.language_projectors[source]
                aligned = align_vggt_to_qwen_premerger(features[source].to(model_device, dtype=torch.float16), inputs["image_grid_thw"])
                merged = projector.norm(aligned).reshape(-1, 8192)
                pre_gelu = projector.mlp[0](merged).float()
                moments[source][0] += pre_gelu.numel()
                moments[source][1] += float(pre_gelu.square().sum().item())
            print(json.dumps({"stage": "B_pre_gelu", "scene": Path(video["video_path"]).stem}), flush=True)
    results = {}
    for source in sources:
        count, squared = moments[source]
        raw_rms = math.sqrt(squared / count)
        if not math.isfinite(raw_rms) or raw_rms <= 0:
            raise RuntimeError(f"Invalid Candidate B L{source} pre-GELU RMS")
        scale = 1.0 / raw_rms
        fusion.language_projectors[source].set_c1_state(enabled=True, pre_gelu_scale=scale, residual_gain=0.0)
        results[source] = {"pre_gelu_raw_rms": raw_rms, "pre_gelu_scale": scale}

    def measure_site(site: int) -> list[float]:
        source = sources[site]
        ratios = []
        for video in videos:
            inputs, features, frame_idx = prepare_sample(
                video=video, args=args, processor=processor, prompt_by_id=prompt_by_id,
                manifest_records=manifest_records, model_device=model_device,
            )
            image_positions = torch.nonzero(inputs["input_ids"][0] == model.config.image_token_id, as_tuple=False).flatten()
            captured: dict[str, torch.Tensor] = {}

            def projector_hook(_module, _arguments, output):
                captured["delta"] = output.detach()

            def block_prehook(_module, arguments):
                hidden = arguments[0][0]
                current = hidden.index_select(0, image_positions.to(hidden.device)).detach()
                delta = captured["delta"].to(current.device)
                prior = current.float() - delta.float()
                captured["ratio"] = (rms(delta) / rms(prior)).detach().cpu()

            hooks = [
                fusion.language_projectors[source].register_forward_hook(projector_hook),
                model.model.language_model.layers[site].register_forward_pre_hook(block_prehook),
            ]
            inputs["cached_vggt_features"] = [features]
            inputs["cached_vggt_frame_idx"] = [frame_idx]
            try:
                with torch.inference_mode():
                    model.model(**inputs, use_cache=False)
            finally:
                for handle in hooks:
                    handle.remove()
            value = float(captured["ratio"].item())
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError(f"Invalid Candidate B L{site} delta ratio")
            ratios.append(value)
        return ratios

    for site, source in enumerate(sources):
        projector = fusion.language_projectors[source]
        projector.set_c1_state(enabled=True, pre_gelu_scale=results[source]["pre_gelu_scale"], residual_gain=1.0)
        raw = measure_site(site)
        gain = r0 / lower_median(raw)
        projector.set_c1_state(enabled=True, pre_gelu_scale=results[source]["pre_gelu_scale"], residual_gain=gain)
        results[source]["residual_gain"] = gain
        results[source]["raw_delta_over_hidden"] = summary(raw)
        print(json.dumps({"stage": "B_site", "site": site, "source": source, "gain": gain}, sort_keys=True), flush=True)

    for site, source in enumerate(sources):
        observed = summary(measure_site(site))
        if abs(observed["median"] / r0 - 1.0) > 0.05:
            raise RuntimeError(f"Candidate B final L{site} residual ratio differs from r0 by more than 5%")
        results[source]["calibrated_delta_over_hidden"] = observed
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("a_premerger_cross_attn", "b_llm_add"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-r0", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--forward-frames-root", type=Path, required=True)
    parser.add_argument("--annotation-scannet", type=Path, required=True)
    parser.add_argument("--annotation-route-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-videos", type=int)
    args = parser.parse_args()
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Qwen C1 calibration requires the validated local two-GPU placement")
    if any((args.model / name).exists() for name in ("adapter_model.bin", "adapter_model.safetensors", "non_lora_trainables.bin")):
        raise RuntimeError("Refusing to calibrate from a trained candidate checkpoint")
    calibration_sha = sha256_file(args.calibration_manifest)
    fixed = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    if fixed.get("schema_version") != "c1_calibration_manifest_v1" or fixed.get("num_samples") != 32:
        raise RuntimeError("Expected the audited fixed 32-video C1 manifest")
    videos = list(fixed["videos"])
    if len(videos) != 32 or any(video.get("split") != "train" for video in videos):
        raise RuntimeError("C1 manifest must contain exactly 32 train videos")
    base = json.loads(args.base_r0.read_text(encoding="utf-8"))
    if base.get("schema") != "qwen35_vggt_base_r0_v1" or base.get("calibration_manifest_sha256") != calibration_sha:
        raise RuntimeError("Qwen base r0 artifact/manifest mismatch")
    if base.get("base_weight_index_sha256") != sha256_file(args.model / "model.safetensors.index.json"):
        raise RuntimeError("Qwen base checkpoint identity differs from r0 calibration")
    if args.limit_videos is None:
        if base.get("status") != "formal" or base.get("calibration_video_count") != 32:
            raise RuntimeError("Formal A/B calibration requires formal 32-video Qwen r0")
    else:
        if not 1 <= args.limit_videos <= 2:
            raise ValueError("Diagnostic calibration limit must be one or two videos")
        videos = videos[:args.limit_videos]
    r0 = float(base["r0"])
    if not math.isfinite(r0) or r0 <= 0:
        raise RuntimeError("Invalid Qwen base r0")
    prompt_paths = (args.annotation_scannet, args.annotation_route_plan)
    prompt_by_id = load_selected_user_prompts(prompt_paths, videos)
    annotation_sha = {path.name: sha256_file(path) for path in prompt_paths}
    if base.get("annotation_sha256") != annotation_sha:
        raise RuntimeError("C1 annotation identity differs from Qwen base r0")
    manifest_document = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest_document.get("schema") != "spatialfocus.cached_vggt.v1":
        raise RuntimeError("Unexpected VGGT manifest schema")
    manifest_records = {str(record["video"]): record for record in manifest_document["records"]}
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen3_5" or config.text_config.num_hidden_layers != 32:
        raise RuntimeError("Unexpected Qwen base architecture")
    config.use_cached_vggt = True
    config.use_geometry_encoder = False
    config.controlled_fusion_candidate = args.candidate
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.size = {"shortest_edge": 200704, "longest_edge": 200704}
    model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
        args.model, config=config, dtype=torch.float16, attn_implementation="sdpa",
        device_map=fixed_device_map(args.candidate), local_files_only=True,
    ).eval()
    model.requires_grad_(False)
    initialize_qwen35_c1(model.model.controlled_vggt_fusion)
    model_device = model.model.language_model.embed_tokens.weight.device
    started = time.perf_counter()
    if args.candidate == "a_premerger_cross_attn":
        values = calibrate_a(model=model, videos=videos, args=args, processor=processor,
                             prompt_by_id=prompt_by_id, manifest_records=manifest_records,
                             model_device=model_device, r0=r0)
        site_values = {"A": values}
    else:
        values = calibrate_b(model=model, videos=videos, args=args, processor=processor,
                             prompt_by_id=prompt_by_id, manifest_records=manifest_records,
                             model_device=model_device, r0=r0)
        site_values = {"B": values}
    artifact = {
        "schema": "qwen35_vggt_c1_calibration_v1",
        "status": "diagnostic" if args.limit_videos is not None else "formal",
        "candidate": args.candidate,
        "canonicalization_scheme_version": SCHEME_VERSION,
        "base_model": str(args.model.resolve()),
        "base_weight_index_sha256": sha256_file(args.model / "model.safetensors.index.json"),
        "base_r0_artifact_sha256": sha256_file(args.base_r0),
        "r0": r0,
        "calibration_video_count": len(videos),
        "calibration_manifest_sha256": calibration_sha,
        "annotation_sha256": annotation_sha,
        "vggt_manifest_sha256": sha256_file(args.manifest),
        "frames_per_video": 32,
        "max_pixels_per_frame": 200704,
        "post_sft_state_loaded": False,
        "optimizer_steps": 0,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        **site_values,
    }
    save_json(args.output, artifact)
    print(json.dumps({"status": artifact["status"], "candidate": args.candidate,
                      "r0": r0, "videos": len(videos), "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
