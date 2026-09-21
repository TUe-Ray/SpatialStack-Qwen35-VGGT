import transformers
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # Geometry encoder configuration
    use_geometry_encoder: bool = field(default=False)  # Whether to use 3D geometry encoder
    geometry_encoder_type: str = field(default="vggt")  # Type of geometry encoder ("vggt", "pi3")
    geometry_encoder_path: str = field(default="facebook/VGGT-1B/")  # Path to pre-trained geometry encoder model
    reference_frame: str = field(default="first")  # Reference frame for geometry encoding ("first", "last"), only available for vggt
    feature_fusion_method: str = field(default="add")  # Method to fuse geometry and visual features ("add", "concat", "cross_attention", "gate")
    fusion_num_layers: int = field(default=1)  # Number of layers in the cross-attention module when feature_fusion_method is "cross_attention"
    geometry_merger_type: str = field(default="mlp")  # Type of geometry feature merger ("mlp", "avg")
    geometry_fusion_layers: Optional[List[int]] = field(default=None)  # Vision block indices for layer-wise fusion
    geometry_encoder_layers: Optional[List[int]] = field(default=None)  # Geometry encoder layer indices
    include_camera_token: bool = field(default=False)  # Whether to include camera token
    pos_encoding_type: str = field(default="none")  # Position encoding: "none", "rope2d", or "sincos2d"
    vision_language_fusion_layers: Optional[List[int]] = field(default=None)  # Vision block indices to fuse into decoder

    # Held-out controlled cached-VGGT experiment. This is mutually exclusive
    # with use_geometry_encoder.
    use_cached_vggt: bool = field(default=False)
    controlled_fusion_candidate: Optional[str] = field(default=None)
    controlled_cross_attention_heads: int = field(default=16)
    controlled_fusion_dropout: float = field(default=0.1)
    controlled_projector_hidden_dim: int = field(default=4096)
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=128)
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.05)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)
    cached_vggt_manifest: Optional[str] = field(default=None)
    cached_vggt_num_frames: int = field(default=32)
    cached_vggt_verify_sha256: bool = field(default=True)
    cached_vggt_layers: Optional[List[int]] = field(default=None)
    cached_vggt_require_exact_layers: bool = field(default=False)
    expected_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Fail if the fully materialized training mixture has a different size."
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
