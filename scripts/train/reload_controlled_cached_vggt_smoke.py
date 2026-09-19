#!/usr/bin/env python3
"""Reload a controlled PEFT checkpoint and run one cached-VGGT generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor

from qwen_vl.data.cached_vggt import CachedVGGTStore
from qwen_vl.model.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGenerationWithGeometry,
    load_qwen3_5_controlled_submodules,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--dataset", default="vlm3r_scannet")
    parser.add_argument("--video", default="scannet/videos/scene0384_00.mp4")
    parser.add_argument("--num-frames", type=int, default=2)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    with (checkpoint / "adapter_config.json").open("r", encoding="utf-8") as handle:
        base_path = json.load(handle)["base_model_name_or_path"]
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    candidate = config.controlled_fusion_candidate
    layers = [23] if candidate == "a_premerger_cross_attn" else [11, 17, 23]
    store = CachedVGGTStore(args.manifest, layers, args.num_frames, verify_sha256=True)
    sample = store.load(args.dataset, args.video, args.media_root)

    model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
        base_path,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        local_files_only=True,
    )
    loaded = load_qwen3_5_controlled_submodules(model, str(checkpoint))
    if loaded == 0:
        raise RuntimeError("No controlled fusion weights were reloaded")
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False).eval()
    base_model = model.get_base_model()
    if getattr(base_model, "geometry_encoder", None) is not None:
        raise RuntimeError("Online VGGT/geometry encoder was unexpectedly constructed")

    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, padding_side="left")
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in sample.images),
                {"type": "text", "text": "Describe this indoor scene briefly."},
            ],
        },
    ]
    text = processor.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, images=[sample.images], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda:0")
    inputs["cached_vggt_features"] = [
        {layer: tensor.to("cuda:0") for layer, tensor in sample.features.items()}
    ]
    inputs["cached_vggt_frame_idx"] = [sample.frame_idx.to("cuda:0")]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    generated = output[:, inputs["input_ids"].shape[1] :]
    answer = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print(
        json.dumps(
            {
                "candidate": candidate,
                "controlled_keys_loaded": loaded,
                "frame_idx": sample.frame_idx.tolist(),
                "generated_token_count": int(generated.shape[1]),
                "answer": answer,
                "online_vggt_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
