"""Deterministic C1 initialization for the pre-SFT Qwen3.5/VGGT candidates.

Only newly introduced A/B fusion parameters are changed.  No base-Qwen
weights, optimizer state, random seed, or post-SFT checkpoint is consulted.
The matrix construction is SpatialFocus's c1_structured_isometry_v1 family.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .controlled_vggt_fusion import CachedVGGTControlledFusion


SCHEME_VERSION = "c1_structured_isometry_v1"
_SQUARE_CACHE: dict[int, torch.Tensor] = {}
_RECTANGULAR_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def structured_block_size(dimension: int) -> int:
    if dimension <= 0:
        raise ValueError("C1 dimensions must be positive")
    block = 512
    while block > 1 and dimension % block:
        block //= 2
    return block


def _hadamard_right(values: torch.Tensor, block: int) -> torch.Tensor:
    dimension = values.shape[-1]
    prefix = values.shape[:-1]
    result = values.reshape(*prefix, dimension // block, block)
    width = 1
    while width < block:
        result = result.reshape(*prefix, dimension // block, block // (2 * width), 2, width)
        left, right = result.unbind(dim=-2)
        result = torch.stack((left + right, left - right), dim=-2).reshape(*prefix, dimension // block, block)
        width *= 2
    return result.reshape(*prefix, dimension) * (1.0 / math.sqrt(block))


def canonical_square(dimension: int) -> torch.Tensor:
    """Materialize B @ perfect_shuffle @ B as a CPU FP32 orthogonal map."""
    cached = _SQUARE_CACHE.get(dimension)
    if cached is not None:
        return cached
    block = structured_block_size(dimension)
    matrix = _hadamard_right(torch.eye(dimension, dtype=torch.float32), block)
    matrix = matrix.reshape(dimension, dimension // block, block).transpose(-2, -1).reshape(dimension, dimension)
    matrix = _hadamard_right(matrix, block).contiguous()
    _SQUARE_CACHE[dimension] = matrix
    return matrix


def canonical_linear_weight(d_in: int, d_out: int) -> torch.Tensor:
    """Deterministic semi-isometry in ``nn.Linear.weight`` orientation."""
    key = (d_in, d_out)
    cached = _RECTANGULAR_CACHE.get(key)
    if cached is not None:
        return cached
    if d_in == d_out:
        result = canonical_square(d_in)
    elif d_out > d_in:
        result = canonical_square(d_out)[:, :d_in] @ canonical_square(d_in).T
    else:
        result = canonical_square(d_out) @ canonical_square(d_in)[:d_out, :]
    result = result.contiguous()
    _RECTANGULAR_CACHE[key] = result
    return result


@torch.no_grad()
def _copy_linear(linear: nn.Linear, weight: torch.Tensor) -> None:
    if tuple(linear.weight.shape) != tuple(weight.shape):
        raise ValueError(f"C1 linear shape mismatch: {tuple(linear.weight.shape)} vs {tuple(weight.shape)}")
    linear.weight.copy_(weight.to(device=linear.weight.device, dtype=linear.weight.dtype))
    if linear.bias is not None:
        linear.bias.zero_()


@torch.no_grad()
def _norm_defaults(module: nn.Module) -> None:
    module.weight.fill_(1)
    if getattr(module, "bias", None) is not None:
        module.bias.zero_()


@torch.no_grad()
def _mha_identity(module: nn.MultiheadAttention) -> None:
    if not module._qkv_same_embed_dim:
        raise ValueError("C1 requires packed same-dimension Q/K/V attention")
    dimension = module.embed_dim
    identity = torch.eye(dimension, device=module.in_proj_weight.device, dtype=module.in_proj_weight.dtype)
    module.in_proj_weight.copy_(torch.cat((identity, identity, identity), dim=0))
    if module.in_proj_bias is not None:
        module.in_proj_bias.zero_()
    module.out_proj.weight.copy_(identity)
    if module.out_proj.bias is not None:
        module.out_proj.bias.zero_()


def initialize_qwen35_c1(
    fusion: CachedVGGTControlledFusion,
    *,
    qk_scale: float = 1.0,
    pre_gelu_scales: dict[str, float] | None = None,
    residual_gains: dict[str, float] | None = None,
) -> None:
    """Replace only A/B fresh affine maps with canonical C1 maps.

    Default gains are zero for a safe calibration starting state.  A formal
    probe must subsequently apply a complete, hashed 32-video C1 artifact.
    """
    gains = residual_gains or {}
    if fusion.candidate == fusion.CANDIDATE_A:
        block = fusion.premerger_cross_attention
        if (
            block.query.in_features, block.query.out_features,
            block.key.in_features, block.key.out_features,
            block.attention.num_heads,
        ) != (1024, 1024, 2048, 1024, 16):
            raise ValueError("Candidate A topology does not match audited Qwen3.5 C1 dimensions")
        q_weight = canonical_square(1024)
        kv_weight = canonical_linear_weight(2048, 1024)
        _copy_linear(block.query, q_weight)
        _copy_linear(block.key, kv_weight)
        _copy_linear(block.value, kv_weight)
        _mha_identity(block.attention)
        _copy_linear(block.output, q_weight.T.contiguous())
        for norm in (block.visual_norm, block.geometry_norm, block.output_norm):
            _norm_defaults(norm)
        block.set_c1_state(enabled=True, qk_scale=qk_scale, residual_gain=float(gains.get("A", 0.0)))
        return
    if fusion.candidate != fusion.CANDIDATE_B or tuple(fusion.required_layers) != ("11", "17", "23"):
        raise ValueError("Expected Candidate B with VGGT layers 11/17/23")
    scales = pre_gelu_scales or {}
    in_weight = canonical_linear_weight(8192, 4096)
    out_weight = canonical_linear_weight(4096, 2560)
    for source_layer, projector in fusion.language_projectors.items():
        if (
            projector.mlp[0].in_features, projector.mlp[0].out_features,
            projector.mlp[2].in_features, projector.mlp[2].out_features,
        ) != (8192, 4096, 4096, 2560):
            raise ValueError(f"Candidate B projector {source_layer} has unexpected dimensions")
        _norm_defaults(projector.norm)
        _copy_linear(projector.mlp[0], in_weight)
        _copy_linear(projector.mlp[2], out_weight)
        projector.set_c1_state(
            enabled=True,
            pre_gelu_scale=float(scales.get(source_layer, 1.0)),
            residual_gain=float(gains.get(source_layer, 0.0)),
        )


def apply_qwen35_c1_artifact(fusion: CachedVGGTControlledFusion, artifact: dict) -> None:
    if artifact.get("schema") != "qwen35_vggt_c1_calibration_v1":
        raise ValueError("Expected a formal Qwen3.5/VGGT C1 calibration artifact")
    if artifact.get("status") != "formal":
        raise ValueError("Diagnostic C1 artifacts must not initialize formal probes")
    if artifact.get("canonicalization_scheme_version") != SCHEME_VERSION:
        raise ValueError("C1 matrix scheme mismatch")
    if artifact.get("candidate") != fusion.candidate:
        raise ValueError("C1 artifact candidate mismatch")
    if artifact.get("calibration_video_count") != 32 or not artifact.get("calibration_manifest_sha256"):
        raise ValueError("C1 artifact lacks verified fixed 32-video calibration identity")
    if fusion.candidate == fusion.CANDIDATE_A:
        values = artifact["A"]
        initialize_qwen35_c1(
            fusion,
            qk_scale=float(values["qk_scale"]),
            residual_gains={"A": float(values["residual_gain"])},
        )
    else:
        values = artifact["B"]
        if set(values) != set(fusion.required_layers):
            raise ValueError("C1 artifact lacks one or more Candidate B injection sites")
        initialize_qwen35_c1(
            fusion,
            pre_gelu_scales={layer: float(values[layer]["pre_gelu_scale"]) for layer in fusion.required_layers},
            residual_gains={layer: float(values[layer]["residual_gain"]) for layer in fusion.required_layers},
        )
