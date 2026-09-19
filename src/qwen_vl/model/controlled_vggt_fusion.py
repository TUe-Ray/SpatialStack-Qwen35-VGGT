"""Controlled cached-VGGT fusion modules for the held-out Qwen3.5 experiment."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


VGGT_GRID_SIZE = 37
VGGT_PATCH_TOKENS = VGGT_GRID_SIZE * VGGT_GRID_SIZE


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normalized.to(dtype) * self.weight).to(dtype)


def _validate_grid(grid_thw: torch.Tensor, frame_count: int, merge_size: int) -> list[tuple[int, int]]:
    if grid_thw is None or grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [frames, 3]")
    if len(grid_thw) != frame_count:
        raise ValueError(f"RGB grid has {len(grid_thw)} frames but VGGT has {frame_count}")
    shapes = []
    for index, (temporal, height, width) in enumerate(grid_thw.tolist()):
        if int(temporal) != 1:
            raise ValueError(f"Frame {index} has temporal grid {temporal}; expected 1")
        if int(height) % merge_size or int(width) % merge_size:
            raise ValueError(
                f"Frame {index} grid {height}x{width} is not divisible by merge size {merge_size}"
            )
        shapes.append((int(height), int(width)))
    return shapes


def align_vggt_to_qwen_premerger(
    patch_tokens: torch.Tensor,
    image_grid_thw: torch.Tensor,
    merge_size: int = 2,
) -> torch.Tensor:
    """Resize 37x37 VGGT grids and reorder them to native Qwen merger order."""
    if patch_tokens.ndim != 3 or patch_tokens.shape[1] != VGGT_PATCH_TOKENS:
        raise ValueError(
            f"Expected cached VGGT patches [F,{VGGT_PATCH_TOKENS},D], got {tuple(patch_tokens.shape)}"
        )
    shapes = _validate_grid(image_grid_thw, patch_tokens.shape[0], merge_size)
    aligned = []
    for tokens, (height, width) in zip(patch_tokens, shapes):
        grid = tokens.reshape(VGGT_GRID_SIZE, VGGT_GRID_SIZE, -1).permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(grid.float(), size=(height, width), mode="bilinear", align_corners=False)
        resized = resized.squeeze(0).permute(1, 2, 0).to(tokens.dtype)
        # Qwen's native merger consumes 2x2 neighborhoods contiguously.
        ordered = (
            resized.reshape(height // merge_size, merge_size, width // merge_size, merge_size, -1)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .reshape(height * width, -1)
        )
        aligned.append(ordered)
    return torch.cat(aligned, dim=0)


class PreMergerCrossAttention(nn.Module):
    """SpatialFocus A-prime: frame-local cross-attention plus visual residual."""

    def __init__(
        self,
        visual_dim: int = 1024,
        geometry_dim: int = 2048,
        num_heads: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if visual_dim % num_heads:
            raise ValueError("visual_dim must be divisible by num_heads")
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.geometry_norm = nn.LayerNorm(geometry_dim)
        self.query = nn.Linear(visual_dim, visual_dim)
        self.key = nn.Linear(geometry_dim, visual_dim)
        self.value = nn.Linear(geometry_dim, visual_dim)
        self.attention = nn.MultiheadAttention(visual_dim, num_heads, dropout=0.0, batch_first=True)
        self.output = nn.Linear(visual_dim, visual_dim)
        self.output_norm = nn.LayerNorm(visual_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        geometry_tokens: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        shapes = _validate_grid(image_grid_thw, geometry_tokens.shape[0], merge_size=2)
        split_sizes = [height * width for height, width in shapes]
        if visual_tokens.shape[0] != sum(split_sizes):
            raise ValueError(
                f"Qwen pre-merger token count {visual_tokens.shape[0]} != grid total {sum(split_sizes)}"
            )
        aligned = align_vggt_to_qwen_premerger(geometry_tokens, image_grid_thw)
        fused_frames = []
        for visual, geometry in zip(torch.split(visual_tokens, split_sizes), torch.split(aligned, split_sizes)):
            query = self.query(self.visual_norm(visual)).unsqueeze(0)
            geometry = self.geometry_norm(geometry)
            key = self.key(geometry).unsqueeze(0)
            value = self.value(geometry).unsqueeze(0)
            update, _ = self.attention(query, key, value, need_weights=False)
            update = self.output_norm(self.output(update.squeeze(0)))
            fused_frames.append(self.dropout(visual + update))
        return torch.cat(fused_frames, dim=0)


class LanguageAddProjector(nn.Module):
    """SpatialStack-style 2x2 merger/projector with a zero residual branch."""

    def __init__(
        self,
        geometry_dim: int = 2048,
        hidden_dim: int = 4096,
        language_dim: int = 2560,
        merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.geometry_dim = geometry_dim
        self.merge_size = merge_size
        self.norm = RMSNorm(geometry_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(geometry_dim * merge_size * merge_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, language_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, aligned_premerger: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(aligned_premerger)
        group = self.merge_size * self.merge_size
        if normalized.shape[0] % group:
            raise ValueError("Aligned VGGT token count is not divisible by merge area")
        return self.mlp(normalized.reshape(-1, self.geometry_dim * group))


class CachedVGGTControlledFusion(nn.Module):
    CANDIDATE_A = "a_premerger_cross_attn"
    CANDIDATE_B = "b_llm_add"

    def __init__(self, config) -> None:
        super().__init__()
        self.candidate = str(config.controlled_fusion_candidate).lower()
        visual_dim = int(config.vision_config.hidden_size)
        language_dim = int(config.text_config.hidden_size)
        merge_size = int(config.vision_config.spatial_merge_size)
        if merge_size != 2:
            raise ValueError(f"Controlled VGGT fusion requires Qwen merge size 2, got {merge_size}")
        if self.candidate == self.CANDIDATE_A:
            self.required_layers = ("23",)
            self.premerger_cross_attention = PreMergerCrossAttention(
                visual_dim=visual_dim,
                geometry_dim=2048,
                num_heads=int(getattr(config, "controlled_cross_attention_heads", 16)),
                dropout=float(getattr(config, "controlled_fusion_dropout", 0.1)),
            )
            self.language_projectors = None
        elif self.candidate == self.CANDIDATE_B:
            self.required_layers = ("11", "17", "23")
            self.premerger_cross_attention = None
            self.language_projectors = nn.ModuleDict(
                {
                    layer: LanguageAddProjector(
                        hidden_dim=int(getattr(config, "controlled_projector_hidden_dim", 4096)),
                        language_dim=language_dim,
                        merge_size=merge_size,
                    )
                    for layer in self.required_layers
                }
            )
        else:
            raise ValueError(
                f"Unknown controlled_fusion_candidate {self.candidate!r}; expected "
                f"{self.CANDIDATE_A!r} or {self.CANDIDATE_B!r}"
            )

    def reset_native_initialization(self) -> None:
        """Restore architecture-defined no-op initialization after HF post_init."""
        if self.candidate == self.CANDIDATE_A:
            module = self.premerger_cross_attention
            module.visual_norm.reset_parameters()
            module.geometry_norm.reset_parameters()
            module.query.reset_parameters()
            module.key.reset_parameters()
            module.value.reset_parameters()
            module.attention._reset_parameters()
            module.attention.out_proj.reset_parameters()
            module.output.reset_parameters()
            module.output_norm.reset_parameters()
        elif self.candidate == self.CANDIDATE_B:
            for projector in self.language_projectors.values():
                nn.init.zeros_(projector.mlp[-1].weight)
                nn.init.zeros_(projector.mlp[-1].bias)

    def validate_features(self, features: Mapping[str, torch.Tensor]) -> None:
        if set(features) != set(self.required_layers):
            raise ValueError(
                f"Candidate {self.candidate} requires exactly layers {self.required_layers}, got {tuple(features)}"
            )
        frame_counts = set()
        for layer in self.required_layers:
            tensor = features[layer]
            if tensor.ndim != 3 or tensor.shape[1:] != (VGGT_PATCH_TOKENS, 2048):
                raise ValueError(f"Layer {layer} has invalid cached shape {tuple(tensor.shape)}")
            frame_counts.add(tensor.shape[0])
        if len(frame_counts) != 1:
            raise ValueError("Cached VGGT layers have inconsistent frame counts")

    def fuse_premerger(
        self,
        visual_tokens: torch.Tensor,
        features: Mapping[str, torch.Tensor],
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if self.candidate != self.CANDIDATE_A:
            raise RuntimeError("Pre-merger fusion is only valid for Candidate A")
        self.validate_features(features)
        geometry = features["23"].to(visual_tokens.device, visual_tokens.dtype)
        return self.premerger_cross_attention(visual_tokens, geometry, image_grid_thw)

    def build_language_residuals(
        self,
        features: Mapping[str, torch.Tensor],
        image_grid_thw: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[int, torch.Tensor]:
        if self.candidate != self.CANDIDATE_B:
            raise RuntimeError("Language residuals are only valid for Candidate B")
        self.validate_features(features)
        result = {}
        for llm_layer, vggt_layer in enumerate(self.required_layers):
            geometry = features[vggt_layer].to(device=device, dtype=dtype)
            aligned = align_vggt_to_qwen_premerger(geometry, image_grid_thw)
            result[llm_layer] = self.language_projectors[vggt_layer](aligned)
        return result
