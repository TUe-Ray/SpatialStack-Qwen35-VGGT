#!/usr/bin/env python3
"""One-video local Qwen3.5/VGGT forward smoke; not a formal C1 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry
from qwen_vl.model.qwen35_c1_init import initialize_qwen35_c1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(args: argparse.Namespace) -> tuple[list[Image.Image], torch.Tensor, dict | None, str | None]:
    frame_payload = torch.load(args.frame_cache, map_location="cpu", weights_only=False)
    frames = frame_payload["frames_rgb_uint8"]
    frame_idx = frame_payload["source_frame_indices"]
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[0] != 32:
        raise ValueError("Expected the validated 32-frame RGB cache")
    if frame_idx.dtype != torch.int64 or frame_idx.numel() != 32:
        raise ValueError("Expected 32 original source frame indices")
    images = [Image.fromarray(frame.numpy()).convert("RGB") for frame in frames]
    if args.candidate == "base":
        return images, frame_idx, None, None
    if args.manifest is None:
        raise ValueError("A/B smoke requires --manifest")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "spatialfocus.cached_vggt.v1":
        raise ValueError("Unexpected VGGT manifest schema")
    records = [
        record for record in manifest["records"]
        if record["dataset"] == "vlm3r_scannet" and Path(record["video"]).stem == args.frame_cache.stem
    ]
    if len(records) != 1:
        raise ValueError(f"Expected one exact VGGT manifest record, got {len(records)}")
    record = records[0]
    sidecar = (args.manifest.parent / record["sidecar"]).resolve()
    actual_sha = sha256_file(sidecar)
    if actual_sha != record["sha256"]:
        raise ValueError("VGGT sidecar SHA256 mismatch")
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    if not torch.equal(payload["frames"]["frame_idx"], frame_idx):
        raise ValueError("RGB/VGGT source frame ordering differs")
    if list(record["frame_idx"]) != frame_idx.tolist():
        raise ValueError("Manifest/RGB source frame ordering differs")
    required = ("23",) if args.candidate == "a_premerger_cross_attn" else ("11", "17", "23")
    layer_map = payload["frames"]["aggregated_tokens"]
    features = {}
    for layer in required:
        value = layer_map[layer]
        if value.shape != (32, 1374, 2048) or value.dtype != torch.bfloat16:
            raise ValueError(f"VGGT layer {layer} has unexpected shape/dtype")
        features[layer] = value[:, 5:].contiguous()
    return images, frame_idx, features, actual_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--candidate", choices=("base", "a_premerger_cross_attn", "b_llm_add"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-pixels", type=int, default=65536)
    parser.add_argument("--c1-zero-gain", action="store_true", help="Smoke canonical maps with every residual gain zero")
    parser.add_argument("--offload-lm-head", action="store_true", help="Place only the output head on CPU")
    parser.add_argument("--cpu-middle-layers", type=int, default=0, help="Offload this many blocks ending at L10")
    args = parser.parse_args()
    if args.c1_zero_gain and args.candidate == "base":
        raise ValueError("Base has no new fusion module to initialize with C1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("This local smoke requires both TITAN V GPUs")
    if any((args.model / filename).exists() for filename in ("adapter_model.bin", "adapter_model.safetensors", "non_lora_trainables.bin")):
        raise RuntimeError("Refusing to load a candidate-trained checkpoint")
    images, frame_idx, features, sidecar_sha = load_inputs(args)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen3_5" or config.text_config.num_hidden_layers != 32:
        raise RuntimeError("Unexpected base architecture")
    config.use_geometry_encoder = False
    config.use_cached_vggt = args.candidate != "base"
    if config.use_cached_vggt:
        config.controlled_fusion_candidate = args.candidate
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.image_processor.size = {"shortest_edge": args.max_pixels, "longest_edge": args.max_pixels}
    message = [[{
        "role": "user",
        "content": [*({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": "Describe this indoor scene briefly."}],
    }]]
    prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=prompt, images=[images], padding=True, return_tensors="pt")
    device_map = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        **{f"model.language_model.layers.{layer}": 0 if layer <= 10 else 1 for layer in range(32)},
        "model.language_model.norm": 1,
        "model.language_model.rotary_emb": 1,
        "lm_head": 0,
    }
    if config.use_cached_vggt:
        device_map["model.controlled_vggt_fusion"] = 0
        # The model retains a public alias to the same module; Accelerate
        # checks both state-dict prefixes even though named_modules deduplicates it.
        device_map["controlled_vggt_fusion"] = 0
    if args.offload_lm_head:
        device_map["lm_head"] = "cpu"
    if not 0 <= args.cpu_middle_layers <= 11:
        raise ValueError("--cpu-middle-layers must be between 0 and 11")
    for layer in range(11 - args.cpu_middle_layers, 11):
        device_map[f"model.language_model.layers.{layer}"] = "cpu"
    started = time.perf_counter()
    model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
        args.model,
        config=config,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map=device_map,
        local_files_only=True,
    ).eval()
    model_device = model.model.language_model.embed_tokens.weight.device
    fusion_module = model.model.controlled_vggt_fusion
    if args.c1_zero_gain:
        initialize_qwen35_c1(fusion_module)
    print(json.dumps({
        "stage": "loaded",
        "device_map": model.hf_device_map,
        "embed_device": str(model_device),
        "visual_device": str(next(model.model.visual.parameters()).device),
        "lm_head_device": str(model.lm_head.weight.device),
        "fusion_device": None if fusion_module is None else str(next(fusion_module.parameters()).device),
    }, sort_keys=True), flush=True)
    inputs = inputs.to(model_device)
    if features is not None:
        inputs["cached_vggt_features"] = [features]
        inputs["cached_vggt_frame_idx"] = [frame_idx]
    for gpu in range(2):
        torch.cuda.reset_peak_memory_stats(gpu)
    with torch.inference_mode():
        output = model(**inputs, logits_to_keep=1, use_cache=False)
    for gpu in range(2):
        torch.cuda.synchronize(gpu)
    if not torch.isfinite(output.logits).all():
        raise RuntimeError("Non-finite Qwen output")
    print(json.dumps({
        "status": "PASS",
        "kind": "zero_gain_c1_forward_smoke_not_formal" if args.c1_zero_gain else "native_initialization_forward_smoke_not_formal_c1",
        "candidate": args.candidate,
        "num_frames": len(images),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "image_tokens": int((inputs["input_ids"] == config.image_token_id).sum()),
        "frame_idx_first_last": [int(frame_idx[0]), int(frame_idx[-1])],
        "sidecar_sha256": sidecar_sha,
        "device_map": model.hf_device_map,
        "peak_vram_bytes": [torch.cuda.max_memory_allocated(gpu) for gpu in range(2)],
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "optimizer_steps": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
