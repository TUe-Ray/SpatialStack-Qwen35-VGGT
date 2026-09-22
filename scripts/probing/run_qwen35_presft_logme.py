#!/usr/bin/env python3
"""Score the fixed common-seven Qwen3.5 pre-SFT depth representations.

The Bayesian linear-regression evidence calculation and target/mask treatment
match SpatialFocus's ``run_pre_sft_logme_proxy.py``.  This runner is separate
because the historical VLM3R candidate registry must not be silently changed.
It requires a complete 1,199-video feature manifest and scores only the 1,006
training videos; validation depth targets never enter LogME.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


SPLIT_SHA256 = "d478cb684958dfc25066821ec83d5216469577c9e282e33bdf87d3c88b200d8e"
COMMON7 = (1, 3, 6, 9, 15, 21, 27)
LABELS = {
    "base": "qwen35_base_presft",
    "a_premerger_cross_attn": "qwen35_vggt_a_c1_presft",
    "b_llm_add": "qwen35_vggt_b_c1_presft",
}
EPS = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def logme_from_statistics(gram: torch.Tensor, cross: torch.Tensor, yy: torch.Tensor, count: int) -> dict[str, Any]:
    """The unchanged SpatialFocus v1 normalized LogME fixed-point equations."""
    gram = (gram + gram.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    largest = max(1.0, float(eigenvalues[-1].abs().item()))
    min_eigenvalue = float(eigenvalues[0].item())
    if min_eigenvalue < -1e-8 * largest:
        raise RuntimeError(f"Materially non-PSD Gram matrix: min={min_eigenvalue}, max={largest}")
    eigenvalues = eigenvalues.clamp_min(0.0)
    projected_cross = eigenvectors.T @ cross
    alpha = beta = 1.0
    converged = False
    gamma = float("nan")
    for iteration in range(1, 101):
        denominator = alpha + beta * eigenvalues
        if not torch.isfinite(denominator).all() or torch.any(denominator <= 0):
            raise RuntimeError("Invalid LogME precision denominator")
        coeff = beta * projected_cross / denominator
        m_norm = torch.dot(coeff, coeff)
        m_cross = torch.dot(coeff, projected_cross)
        m_gram_m = torch.dot(eigenvalues * coeff, coeff)
        gamma_tensor = torch.sum(beta * eigenvalues / denominator)
        residual = yy - 2.0 * m_cross + m_gram_m
        if float(residual.item()) < -EPS:
            raise RuntimeError(f"Materially negative residual square: {float(residual.item())}")
        residual = residual.clamp_min(0.0)
        new_alpha = float((gamma_tensor / (m_norm + EPS)).item())
        new_beta = float(((count - gamma_tensor) / (residual + EPS)).item())
        if not (math.isfinite(new_alpha) and math.isfinite(new_beta) and new_alpha > 0 and new_beta > 0):
            raise RuntimeError("Non-finite LogME precision")
        gamma = float(gamma_tensor.item())
        if max(abs(new_alpha - alpha) / (abs(alpha) + EPS), abs(new_beta - beta) / (abs(beta) + EPS)) < 1e-6:
            alpha, beta, converged = new_alpha, new_beta, True
            break
        alpha, beta = new_alpha, new_beta
    denominator = alpha + beta * eigenvalues
    coeff = beta * projected_cross / denominator
    m_norm = torch.dot(coeff, coeff)
    residual = (yy - 2.0 * torch.dot(coeff, projected_cross) + torch.dot(eigenvalues * coeff, coeff)).clamp_min(0.0)
    score = 0.5 * (
        gram.shape[0] * math.log(alpha)
        + count * math.log(beta)
        - float(torch.log(denominator).sum().item())
        - beta * float(residual.item())
        - alpha * float(m_norm.item())
        - count * math.log(2.0 * math.pi)
    ) / count
    if not math.isfinite(score):
        raise RuntimeError("Non-finite normalized LogME")
    return {
        "logme": score, "alpha": alpha, "beta": beta, "iterations": iteration,
        "converged": converged, "gamma": gamma,
        "residual_sq": float(residual.item()), "m_norm_sq": float(m_norm.item()),
        "minimum_eigenvalue": min_eigenvalue,
    }


def score_layer(
    *, root: Path, label: str, layer: int, frames: list[dict[str, Any]],
    device: torch.device, block_frames: int,
) -> dict[str, Any]:
    gram = cross = None
    yy = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    target_digest = hashlib.sha256()
    pending_x: list[torch.Tensor] = []
    pending_y: list[torch.Tensor] = []

    def flush() -> None:
        nonlocal gram, cross, yy
        if not pending_x:
            return
        x = torch.cat(pending_x, dim=0).to(device=device, dtype=torch.float64)
        y = torch.cat(pending_y, dim=0).to(device=device, dtype=torch.float64)
        if gram is None:
            gram = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float64, device=device)
            cross = torch.zeros(x.shape[1], dtype=torch.float64, device=device)
        assert cross is not None
        gram.addmm_(x.T, x)
        cross.addmv_(x.T, y)
        yy += torch.dot(y, y)
        pending_x.clear()
        pending_y.clear()

    for index, frame in enumerate(frames):
        frame_id = str(frame["frame_sample_id"])
        feature = torch.load(root / "features" / label / f"layer_{layer}" / f"frame_{frame_id}.pt", map_location="cpu", weights_only=True)
        target = torch.load(root / "gt_depth" / f"frame_{frame_id}.pt", map_location="cpu", weights_only=True)
        metadata = torch.load(root / "metadata" / f"frame_{frame_id}.pt", map_location="cpu", weights_only=False)
        if feature.shape[:2] != (14, 14) or target.shape != (14, 14):
            raise RuntimeError(f"Feature/target grid mismatch at {frame_id}/L{layer}")
        valid = metadata.get("gt_valid_mask", torch.isfinite(target) & (target > 0)).reshape(-1).bool()
        x = feature.reshape(-1, feature.shape[-1])
        y = target.reshape(-1)
        valid = valid & torch.isfinite(y) & (y > 0)
        if not torch.isfinite(x[valid]).all():
            raise RuntimeError(f"Non-finite valid features at {frame_id}/L{layer}")
        target_digest.update(frame_id.encode("utf-8"))
        target_digest.update(valid.numpy().tobytes() + y.to(dtype=torch.float32).numpy().tobytes())
        pending_x.append(x[valid])
        pending_y.append(y[valid])
        count += int(valid.sum().item())
        if len(pending_x) >= block_frames:
            flush()
        if (index + 1) % 200 == 0:
            print(json.dumps({"layer": layer, "frames": index + 1, "expected_frames": len(frames)}), flush=True)
    flush()
    if gram is None or cross is None or count == 0:
        raise RuntimeError("No valid LogME observations")
    scores = logme_from_statistics(gram, cross, yy, count)
    return {
        **scores, "layer": layer, "train_frames": len(frames), "valid_tokens": count,
        "feature_dim": gram.shape[0], "target_sha256": target_digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(LABELS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block-frames", type=int, default=16)
    args = parser.parse_args()
    if args.block_frames <= 0:
        raise ValueError("--block-frames must be positive")
    if sha256_file(args.sample_indices) != SPLIT_SHA256:
        raise RuntimeError("Unexpected ScanNet split SHA-256")
    label = LABELS[args.candidate]
    manifest_path = args.output_root / f"{label}_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("complete_videos") != 1199:
        raise RuntimeError("LogME requires a complete, verified 1,199-video extraction")
    if manifest.get("candidate") != args.candidate or manifest.get("sample_indices_sha256") != SPLIT_SHA256:
        raise RuntimeError("Feature manifest candidate/split mismatch")
    videos = json.loads(args.sample_indices.read_text(encoding="utf-8"))["videos"]
    train = [video for video in videos if video["split"] == "train"]
    val = [video for video in videos if video["split"] == "val"]
    if (len(train), len(val)) != (1006, 193):
        raise RuntimeError("Unexpected fixed ScanNet train/validation video counts")
    frames = [frame for video in train for frame in video["frames"]]
    if len(frames) != 2012:
        raise RuntimeError("Expected exactly two train target frames per video")
    result_root = args.output_root / "logme_common7" / label
    result_root.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(manifest_path)
    # ``updated_at`` changes when an idempotent extractor resumes.  Keep the
    # scoring identity tied to immutable model/data/protocol fields instead.
    identity_fields = (
        "candidate", "model_label", "model_weight_index_sha256",
        "sample_indices_sha256", "annotation_sha256", "vggt_manifest_sha256",
        "c1_artifact_sha256", "feature_levels", "frames_per_video",
        "selected_target_frames_per_video", "target_grid", "max_pixels_per_frame",
        "attention_implementation", "dtype", "device_map", "git_commit",
    )
    source_identity = {key: manifest.get(key) for key in identity_fields}
    source_identity_sha = hashlib.sha256(
        json.dumps(source_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    rows = []
    for layer in COMMON7:
        path = result_root / f"layer_{layer}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("feature_identity_sha256") != source_identity_sha:
                raise RuntimeError(f"Refusing to reuse incompatible L{layer} LogME result")
        else:
            started = time.perf_counter()
            row = score_layer(root=args.output_root, label=label, layer=layer, frames=frames,
                              device=torch.device(args.device), block_frames=args.block_frames)
            row.update({
                "schema": "qwen35_vggt_presft_logme_common7_layer_v1",
                "candidate": args.candidate, "model_label": label,
                "sample_indices_sha256": SPLIT_SHA256,
                "feature_manifest_sha256": source_sha,
                "feature_identity_sha256": source_identity_sha,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            write_json(path, row)
        rows.append(row)
        print(json.dumps({"candidate": args.candidate, "layer": layer, "logme": row["logme"]}), flush=True)
    if len({row["target_sha256"] for row in rows}) != 1:
        raise RuntimeError("LogME target identities differ across the fixed common-seven layers")
    aggregate = {
        "schema": "qwen35_vggt_presft_logme_common7_v1",
        "candidate": args.candidate, "model_label": label,
        "layers": list(COMMON7), "mean_logme": sum(row["logme"] for row in rows) / len(rows),
        "per_layer_logme": {str(row["layer"]): row["logme"] for row in rows},
        "train_videos": len(train), "train_frames": len(frames),
        "sample_indices_sha256": SPLIT_SHA256,
        "target_sha256": rows[0]["target_sha256"],
        "feature_manifest_sha256": source_sha,
        "feature_identity_sha256": source_identity_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(result_root / "summary.json", aggregate)
    print(json.dumps({"candidate": args.candidate, "mean_logme": aggregate["mean_logme"], "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
