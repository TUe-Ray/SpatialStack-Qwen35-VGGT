import json
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from packaging.version import Version
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_pil
from qwen_vl.data.cached_vggt import CachedVGGTStore


MIN_QWEN3_5_TRANSFORMERS_VERSION = Version("5.3.0")


def require_qwen3_5_support():
    current_version = Version(transformers.__version__)
    if current_version < MIN_QWEN3_5_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Qwen3.5 evaluation requires transformers>="
            f"{MIN_QWEN3_5_TRANSFORMERS_VERSION}, but found {transformers.__version__}."
        )


def patch_qwen3_5_flash_attention():
    try:
        import transformers.modeling_flash_attention_utils as flash_attention_utils
    except ImportError:
        return

    if getattr(flash_attention_utils, "_spatialstack_qwen3_5_mrope_patch", False):
        return

    original_is_packed_sequence = flash_attention_utils._is_packed_sequence

    def patched_is_packed_sequence(position_ids, batch_size):
        if position_ids is not None and getattr(position_ids, "ndim", None) == 3:
            return False
        return original_is_packed_sequence(position_ids, batch_size)

    flash_attention_utils._is_packed_sequence = patched_is_packed_sequence
    flash_attention_utils._spatialstack_qwen3_5_mrope_patch = True


def is_image_path(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def strip_thinking_content(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def detect_qwen3_5_fast_path_runtime():
    runtime = {}
    for module_name in ("fla", "causal_conv1d"):
        try:
            __import__(module_name)
            runtime[module_name] = True
        except ImportError:
            runtime[module_name] = False
    return runtime


def build_qwen3_5_geometry_inputs(images, image_grid_thw, patch_size: int = 14):
    geometry_tensors = []
    max_height = 0
    max_width = 0

    for image, grid in zip(images, image_grid_thw):
        _, grid_h, grid_w = [int(v) for v in grid.tolist()]
        target_height = grid_h * patch_size
        target_width = grid_w * patch_size
        resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255.0
        geometry_tensors.append(tensor)
        max_height = max(max_height, target_height)
        max_width = max(max_width, target_width)

    padded_tensors = []
    for tensor in geometry_tensors:
        h_padding = max_height - tensor.shape[1]
        w_padding = max_width - tensor.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = torch.nn.functional.pad(
                tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
        padded_tensors.append(tensor)

    return padded_tensors


def move_qwen3_5_eval_inputs_to_device(inputs, device):
    inputs = inputs.to(device)
    if "geometry_encoder_inputs" in inputs:
        inputs["geometry_encoder_inputs"] = [tensor.to(device) for tensor in inputs["geometry_encoder_inputs"]]
    if "cached_vggt_features" in inputs:
        inputs["cached_vggt_features"] = [
            {layer: tensor.to(device) for layer, tensor in feature_map.items()}
            for feature_map in inputs["cached_vggt_features"]
        ]
        inputs["cached_vggt_frame_idx"] = [
            tensor.to(device) for tensor in inputs["cached_vggt_frame_idx"]
        ]
    return inputs


@register_model("qwen3_5")
class Qwen3_5(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        add_frame_index: bool = False,
        disable_thinking: bool = True,
        strip_thinking: bool = True,
        max_length: Optional[int] = None,
        geometry_encoder_path: Optional[str] = None,
        cached_vggt_manifest: Optional[str] = None,
        cached_vggt_dataset: str = "vsibench",
        cached_vggt_data_root: Optional[str] = None,
        cached_vggt_verify_sha256: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        require_qwen3_5_support()
        patch_qwen3_5_flash_attention()

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        self.max_num_frames = max_num_frames
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.add_frame_index = add_frame_index
        self.disable_thinking = disable_thinking
        self.strip_thinking = strip_thinking
        self.cached_vggt_dataset = cached_vggt_dataset
        self.cached_vggt_data_root = cached_vggt_data_root
        self.fast_path_runtime = detect_qwen3_5_fast_path_runtime()
        if not all(self.fast_path_runtime.values()):
            missing = ", ".join(name for name, available in self.fast_path_runtime.items() if not available)
            eval_logger.warning(
                f"Qwen3.5 optimized runtime dependencies are missing ({missing}). "
                "Upstream may fall back to slower torch kernels during eval."
            )

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = str(self._device)
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        adapter_config_path = Path(pretrained) / "adapter_config.json"
        adapter_path = str(Path(pretrained).resolve()) if adapter_config_path.is_file() else None
        model_source = pretrained
        if adapter_path is not None:
            with adapter_config_path.open("r", encoding="utf-8") as handle:
                adapter_config = json.load(handle)
            model_source = adapter_config.get("base_model_name_or_path")
            if not model_source:
                raise ValueError(f"PEFT adapter has no base_model_name_or_path: {adapter_config_path}")
        config = AutoConfig.from_pretrained(pretrained)
        model_type = getattr(config, "model_type", None)
        if model_type not in {"qwen3_5", "qwen3_5_vl"}:
            raise ValueError(f"Unsupported model_type '{model_type}' for Qwen3.5 eval adapter.")
        use_cached_vggt = bool(getattr(config, "use_cached_vggt", False))
        use_geometry_model = (
            getattr(config, "use_geometry_encoder", False)
            or getattr(config, "use_vggt_feature", False)
            or use_cached_vggt
        )
        if use_geometry_model and int(batch_size) != 1:
            raise ValueError("Qwen3.5 geometry evaluation currently requires batch_size=1.")

        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Your transformers build does not expose Qwen3_5ForConditionalGeneration. "
                f"Please install transformers>={MIN_QWEN3_5_TRANSFORMERS_VERSION}."
            ) from exc

        geometry_encoder_path = geometry_encoder_path or getattr(config, "geometry_encoder_path", None)
        self.cached_vggt_store = None
        if use_cached_vggt:
            if adapter_path is None:
                raise ValueError("Controlled cached-VGGT evaluation requires a trained PEFT checkpoint")
            fusion_artifact = Path(adapter_path) / "controlled_vggt_fusion.bin"
            if not fusion_artifact.is_file():
                raise FileNotFoundError(f"Controlled fusion checkpoint is missing: {fusion_artifact}")
            if not cached_vggt_manifest or not cached_vggt_data_root:
                raise ValueError(
                    "cached_vggt_manifest and cached_vggt_data_root are required for cached-VGGT evaluation"
                )
            candidate = str(getattr(config, "controlled_fusion_candidate", "")).lower()
            if candidate not in {"a_premerger_cross_attn", "b_llm_add"}:
                raise ValueError(f"Invalid controlled_fusion_candidate in checkpoint config: {candidate!r}")
            dimensions = (
                config.vision_config.hidden_size,
                config.vision_config.out_hidden_size,
                config.text_config.hidden_size,
                config.text_config.num_hidden_layers,
            )
            if dimensions != (1024, 2560, 2560, 32):
                raise ValueError(f"Cached controlled evaluation requires Qwen3.5-4B, got dimensions {dimensions}")
            required_layers = [23] if candidate == "a_premerger_cross_attn" else [11, 17, 23]
            self.cached_vggt_store = CachedVGGTStore(
                cached_vggt_manifest,
                required_layers=required_layers,
                num_frames=max_num_frames,
                verify_sha256=cached_vggt_verify_sha256,
                require_exact_layers=(candidate == "a_premerger_cross_attn"),
            )
        if use_geometry_model:
            from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

            load_class = Qwen3_5ForConditionalGenerationWithGeometry
        else:
            load_class = Qwen3_5ForConditionalGeneration

        load_kwargs = {
            "config": config,
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }
        if getattr(config, "use_geometry_encoder", False):
            load_kwargs["geometry_encoder_path"] = geometry_encoder_path

        if use_flash_attention_2:
            self._model = load_class.from_pretrained(model_source, attn_implementation="flash_attention_2", **load_kwargs).eval()
        else:
            self._model = load_class.from_pretrained(model_source, **load_kwargs).eval()

        if adapter_path is not None:
            from peft import PeftModel
            from qwen_vl.model.modeling_qwen3_5 import load_qwen3_5_controlled_submodules

            if use_cached_vggt:
                fusion_state = torch.load(
                    Path(adapter_path) / "controlled_vggt_fusion.bin",
                    map_location="cpu",
                    weights_only=True,
                )
                expected_fusion_state = self._model.controlled_vggt_fusion.state_dict()
                expected_shapes = {
                    f"model.controlled_vggt_fusion.{key}": tuple(value.shape)
                    for key, value in expected_fusion_state.items()
                }
                observed_shapes = {key: tuple(value.shape) for key, value in fusion_state.items()}
                if observed_shapes != expected_shapes:
                    raise ValueError(
                        f"Controlled fusion artifact keys/shapes do not match {candidate}: "
                        f"{Path(adapter_path) / 'controlled_vggt_fusion.bin'}"
                    )
            loaded = load_qwen3_5_controlled_submodules(self._model, adapter_path)
            if use_cached_vggt:
                expected_keys = len(expected_fusion_state)
                if loaded != expected_keys:
                    raise RuntimeError(
                        f"Controlled fusion checkpoint loaded {loaded}/{expected_keys} tensors: {adapter_path}"
                    )
            self._model = PeftModel.from_pretrained(self._model, adapter_path).eval()

        processor_source = pretrained
        if adapter_path is not None and not (Path(pretrained) / "processor_config.json").is_file():
            # Intermediate Trainer checkpoint-* directories save the tokenizer but
            # only the final output root saves the complete processor.
            processor_source = model_source
        self.processor = AutoProcessor.from_pretrained(
            processor_source,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            padding_side="left",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        if max_length is not None:
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self.model.config
        self._max_length = getattr(self._tokenizer, "model_max_length", None)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def uses_geometry_encoder_for_eval(self):
        return bool(getattr(self.config, "use_geometry_encoder", False))

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3.5")

    def _normalize_visual(self, visual_group):
        if isinstance(visual_group, tuple):
            visual_group = list(visual_group)
        if not isinstance(visual_group, list):
            return visual_group
        if len(visual_group) == 0:
            return None
        if len(visual_group) == 1:
            return visual_group[0]
        return visual_group

    def _sample_video_frames(self, video_path: str):
        if self.cached_vggt_store is not None:
            root = Path(self.cached_vggt_data_root).expanduser().resolve()
            resolved = Path(video_path).expanduser().resolve()
            try:
                relative = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(f"Evaluation video {resolved} is outside cached_vggt_data_root {root}") from exc
            sample = self.cached_vggt_store.load(self.cached_vggt_dataset, relative, str(root))
            return sample.images, sample
        if self.use_custom_video_loader:
            return read_video_pyav_pil(
                video_path,
                num_frm=self.max_num_frames,
                fps=self.fps,
                max_image_size=self.max_image_size,
            ), None

        vr = decord.VideoReader(video_path)
        frame_count = len(vr)
        if frame_count <= self.max_num_frames:
            indices = np.arange(frame_count)
        else:
            indices = np.linspace(0, frame_count - 1, self.max_num_frames).astype(int)
        return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in indices], None

    def _build_sample(self, context, visual):
        sample_images = []
        cached_vggt_sample = None
        user_content = []

        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            frames, cached_vggt_sample = self._sample_video_frames(visual)
            for idx, frame in enumerate(frames):
                if self.add_frame_index:
                    user_content.append({"type": "text", "text": f"Frame-{idx}: "})
                # Keep raw PIL inputs in the chat payload to avoid per-frame base64 encoding.
                user_content.append({"type": "image", "image": frame})
                sample_images.append(frame)
        elif isinstance(visual, str) and is_image_path(visual):
            frame = Image.open(visual).convert("RGB")
            user_content.append({"type": "image", "image": frame})
            sample_images.append(frame)
        elif isinstance(visual, Image.Image):
            frame = visual.convert("RGB")
            user_content.append({"type": "image", "image": frame})
            sample_images.append(frame)
        elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
            for idx, frame in enumerate(visual):
                rgb = frame.convert("RGB")
                if self.add_frame_index:
                    user_content.append({"type": "text", "text": f"Frame-{idx}: "})
                user_content.append({"type": "image", "image": rgb})
                sample_images.append(rgb)
        elif isinstance(visual, (list, tuple)) and all(isinstance(v, str) and is_image_path(v) for v in visual):
            for idx, frame_path in enumerate(visual):
                rgb = Image.open(frame_path).convert("RGB")
                if self.add_frame_index:
                    user_content.append({"type": "text", "text": f"Frame-{idx}: "})
                user_content.append({"type": "image", "image": rgb})
                sample_images.append(rgb)

        user_content.append({"type": "text", "text": context})
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]
        return message, sample_images, cached_vggt_sample

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            split_name = split[0]
            batched_visuals = [doc_to_visual[i](self.task_dict[task_name][split_name][ids]) for i, ids in enumerate(doc_id)]
            batch_start = time.perf_counter()

            gen_kwargs = dict(all_gen_kwargs[0])
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be Union[str, list], got {type(until)}")

            messages = []
            sample_images = []
            cached_vggt_samples = []
            for context, raw_visual in zip(contexts, batched_visuals):
                visual = self._normalize_visual(raw_visual)
                message, images, cached_vggt_sample = self._build_sample(context, visual)
                messages.append(message)
                sample_images.append(images)
                cached_vggt_samples.append(cached_vggt_sample)

            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.disable_thinking:
                chat_template_kwargs["enable_thinking"] = False
            text = self.processor.apply_chat_template(messages, **chat_template_kwargs)
            inputs = self.processor(
                text=text,
                images=sample_images if any(len(images) > 0 for images in sample_images) else None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            if self.uses_geometry_encoder_for_eval():
                if len(sample_images) != 1:
                    raise ValueError("Qwen3.5 geometry eval currently expects per-device batch size 1.")
                geometry_encoder_inputs = build_qwen3_5_geometry_inputs(
                    sample_images[0],
                    inputs["image_grid_thw"],
                )
                inputs["geometry_encoder_inputs"] = [torch.stack(geometry_encoder_inputs)]
            if self.cached_vggt_store is not None:
                if len(cached_vggt_samples) != 1 or cached_vggt_samples[0] is None:
                    raise ValueError("Cached-VGGT evaluation requires one manifest-backed video per batch")
                sample = cached_vggt_samples[0]
                if int(inputs["image_grid_thw"].shape[0]) != len(sample.frame_idx):
                    raise ValueError("Qwen processor frame count differs from cached VGGT frame count")
                inputs["cached_vggt_features"] = [sample.features]
                inputs["cached_vggt_frame_idx"] = [sample.frame_idx]
            preprocess_elapsed = time.perf_counter() - batch_start

            if self.device_map == "auto":
                inputs = move_qwen3_5_eval_inputs_to_device(inputs, "cuda")
            else:
                inputs = move_qwen3_5_eval_inputs_to_device(inputs, self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            generate_start = time.perf_counter()
            output_ids = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=gen_kwargs["temperature"] > 0,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            generate_elapsed = time.perf_counter() - generate_start

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            decode_elapsed = time.perf_counter() - generate_start - generate_elapsed
            input_tokens = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else -1
            output_tokens = (
                sum(int(ids.shape[-1]) for ids in generated_ids_trimmed) if generated_ids_trimmed else 0
            )
            eval_logger.debug(
                f"Qwen3.5 eval batch size={len(contexts)} input_tokens={input_tokens} "
                f"output_tokens={output_tokens} preprocess={preprocess_elapsed:.3f}s "
                f"generate={generate_elapsed:.3f}s decode={decode_elapsed:.3f}s "
                f"fast_path={self.fast_path_runtime}"
            )

            for answer, context in zip(answers, contexts):
                final_answer = strip_thinking_content(answer) if self.strip_thinking else answer
                res.append(final_answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), final_answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
