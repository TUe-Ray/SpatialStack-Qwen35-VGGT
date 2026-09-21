# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import decord # must import decord after torch and before torchvision
import transformers
import json
from typing import Dict
import shutil
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwen_vl.train.trainer
import qwen_vl.train.sampler
from qwen_vl.train.trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
)
from qwen_vl.data.data_qwen import make_supervised_data_module

from qwen_vl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, AutoConfig, set_seed, enable_full_determinism
from transformers.utils.hub import cached_file

local_rank = None
QWEN3_5_MODEL_TYPES = {"qwen3_5", "qwen3_5_vl"}

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def save_controlled_vggt_artifact(model, output_dir: str) -> None:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    controlled_module = getattr(base_model, "controlled_vggt_fusion", None)
    if controlled_module is None:
        return
    controlled_state = {
        f"model.controlled_vggt_fusion.{key}": value.detach().cpu()
        for key, value in controlled_module.state_dict().items()
    }
    torch.save(controlled_state, os.path.join(output_dir, "controlled_vggt_fusion.bin"))
    base_model.config.save_pretrained(output_dir)


class ControlledCheckpointTrainer(Trainer):
    """Keep fusion state beside every PEFT adapter/checkpoint."""

    def __init__(self, *args, **kwargs):
        self._profile_enabled = os.environ.get("CONTROLLED_PROFILE", "0") == "1"
        self._profile_microsteps = []
        self._profile_data_wait_sec = 0.0
        self._profile_last_h2d_sec = 0.0
        self._profile_last_forward_sec = 0.0
        super().__init__(*args, **kwargs)
        if self._profile_enabled:
            self.add_callback(ControlledProfileCallback(self))

    def training_step(self, model, inputs, num_items_in_batch=None):
        profile_timings = inputs.pop("_controlled_profile_timings", None)
        if self._profile_enabled:
            torch.cuda.synchronize()
            started = time.perf_counter()
            self._profile_last_h2d_sec = 0.0
            self._profile_last_forward_sec = 0.0
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self._profile_enabled:
            torch.cuda.synchronize()
            train_sec = time.perf_counter() - started
            self._profile_microsteps.append(
                {
                    "h2d_sec": self._profile_last_h2d_sec,
                    "forward_sec": self._profile_last_forward_sec,
                    "backward_sec": max(
                        0.0,
                        train_sec - self._profile_last_h2d_sec - self._profile_last_forward_sec,
                    ),
                    "train_sec": train_sec,
                    "input": profile_timings or [],
                }
            )
        if os.environ.get("CONTROLLED_SMOKE_VALIDATE_GRADIENTS", "0") != "1":
            return loss

        if not bool(torch.isfinite(loss.detach()).all()):
            raise RuntimeError(f"Controlled smoke produced a non-finite loss: {loss.detach()}")

        audit = {
            "fusion": {"trainable": 0, "gradient_covered": 0},
            "lora": {"trainable": 0, "gradient_covered": 0},
        }
        nonfinite = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            group = "fusion" if "controlled_vggt_fusion" in name else "lora" if "lora_" in name else None
            if group is None:
                continue
            audit[group]["trainable"] += parameter.numel()
            if parameter.grad is None:
                continue
            audit[group]["gradient_covered"] += parameter.numel()
            if not bool(torch.isfinite(parameter.grad.detach()).all()):
                nonfinite.append(name)

        if nonfinite:
            raise RuntimeError(f"Controlled smoke produced non-finite gradients: {nonfinite[:20]}")
        uncovered_groups = [
            group for group, counts in audit.items()
            if counts["trainable"] == 0 or counts["gradient_covered"] == 0
        ]
        if uncovered_groups:
            raise RuntimeError(f"Controlled smoke has no gradient coverage for groups: {uncovered_groups}")
        print(f"Controlled smoke finite-gradient audit: {json.dumps(audit, sort_keys=True)}")
        return loss

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        if not self._profile_enabled:
            return super().get_batch_samples(epoch_iterator, num_batches, device)
        started = time.perf_counter()
        result = super().get_batch_samples(epoch_iterator, num_batches, device)
        self._profile_data_wait_sec = time.perf_counter() - started
        return result

    def _prepare_inputs(self, inputs):
        if not self._profile_enabled:
            return super()._prepare_inputs(inputs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = super()._prepare_inputs(inputs)
        torch.cuda.synchronize()
        self._profile_last_h2d_sec = time.perf_counter() - started
        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self._profile_enabled:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        torch.cuda.synchronize()
        self._profile_last_forward_sec = time.perf_counter() - started
        return result

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        super()._save(output_dir=output_dir, state_dict=state_dict)
        if self.args.should_save:
            save_controlled_vggt_artifact(self.model, output_dir)


class ControlledProfileCallback(transformers.TrainerCallback):
    """Emit compact JSON timings only when CONTROLLED_PROFILE=1."""

    def __init__(self, owner):
        self.owner = owner
        self.step_started = None
        self.optimizer_started = None
        self.optimizer_sec = 0.0

    @staticmethod
    def _rank():
        return int(os.environ.get("RANK", "0"))

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        if self._rank() == 0:
            print("CONTROLLED_PROFILE_CONFIG " + json.dumps({
                "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "bf16": args.bf16,
                "tf32": args.tf32,
                "dataloader_num_workers": args.dataloader_num_workers,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }, sort_keys=True), flush=True)

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.step_started = time.perf_counter()
        self.optimizer_sec = 0.0
        self.owner._profile_microsteps = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.optimizer_started = time.perf_counter()

    def on_optimizer_step(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        if self.optimizer_started is not None:
            self.optimizer_sec = time.perf_counter() - self.optimizer_started

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        compute_sec = time.perf_counter() - self.step_started
        if self._rank() == 0:
            record = {
                "optimizer_step": state.global_step,
                "data_wait_sec": self.owner._profile_data_wait_sec,
                "compute_sec": compute_sec,
                "total_step_sec": self.owner._profile_data_wait_sec + compute_sec,
                "optimizer_sec": self.optimizer_sec,
                "microsteps": self.owner._profile_microsteps,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            }
            print("CONTROLLED_PROFILE_STEP " + json.dumps(record, sort_keys=True), flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        print("CONTROLLED_PROFILE_DEVICE " + json.dumps({
            "rank": self._rank(),
            "optimizer_steps": state.global_step,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        }, sort_keys=True), flush=True)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    if hasattr(trainer.model, "peft_config"):
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

        save_controlled_vggt_artifact(trainer.model, output_dir)


def resolve_model_modules(model):
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    if hasattr(model, "visual") and hasattr(model, "model"):
        return model.visual, getattr(model.visual, "merger", None), model.model, model.lm_head

    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "visual") and hasattr(inner_model, "language_model"):
        return inner_model.visual, getattr(inner_model.visual, "merger", None), inner_model.language_model, model.lm_head

    raise ValueError(f"Unsupported model structure for training: {type(model)}")


def set_model(model_args, model):
    visual_module, merger_module, language_module, lm_head = resolve_model_modules(model)

    if model_args.use_cached_vggt and (model_args.tune_mm_vision or model_args.tune_mm_mlp):
        raise ValueError("Controlled cached-VGGT candidates require the Qwen vision tower and native merger frozen")

    if model_args.tune_mm_vision:
        for n, p in visual_module.named_parameters():
            p.requires_grad = True
    else:
        for n, p in visual_module.named_parameters():
            p.requires_grad = False

    if merger_module is not None:
        if model_args.tune_mm_mlp:
            for n, p in merger_module.named_parameters():
                p.requires_grad = True
        else:
            for n, p in merger_module.named_parameters():
                p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in language_module.named_parameters():
            p.requires_grad = True
        for p in lm_head.parameters():
            p.requires_grad = True
    else:
        for n, p in language_module.named_parameters():
            p.requires_grad = False
        for p in lm_head.parameters():
            p.requires_grad = False

    if model_args.use_geometry_encoder:
        # vggt is frozen
        for n, p in model.geometry_encoder.named_parameters():
            p.requires_grad = False

    controlled_module = getattr(model, "controlled_vggt_fusion", None)
    if controlled_module is not None:
        for parameter in controlled_module.parameters():
            parameter.requires_grad = True


def add_language_lora(model_args, model):
    if not model_args.lora_enable:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("lora_enable requires PEFT; dependencies are not installed automatically") from exc

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    target_modules = [
        name
        for name, module in base_model.named_modules()
        if name.startswith("model.language_model.layers.") and isinstance(module, torch.nn.Linear)
    ]
    if not target_modules:
        raise RuntimeError("No Qwen3.5 language-model Linear modules were found for LoRA")
    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    if model_args.use_cached_vggt:
        base_model = model.get_base_model()
        controlled_module = getattr(base_model, "controlled_vggt_fusion", None)
        if controlled_module is None:
            raise RuntimeError("PEFT wrapping lost the controlled VGGT fusion module")
        for parameter in controlled_module.parameters():
            parameter.requires_grad = True
    return model


def audit_controlled_trainable_parameters(model):
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    if getattr(base_model, "geometry_encoder", None) is not None:
        raise RuntimeError("Online geometry encoder exists in a cached-VGGT controlled run")
    visual_module, merger_module, _, _ = resolve_model_modules(model)
    if any(parameter.requires_grad for parameter in visual_module.parameters()):
        raise RuntimeError("Qwen vision parameters must be frozen in controlled runs")
    if merger_module is not None and any(parameter.requires_grad for parameter in merger_module.parameters()):
        raise RuntimeError("Qwen native visual merger must be frozen in controlled runs")
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters")
    unexpected = [
        name
        for name, _ in trainable
        if "controlled_vggt_fusion" not in name and "lora_" not in name
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters in controlled run: {unexpected[:20]}")
    nonfinite = [name for name, parameter in trainable if not torch.isfinite(parameter.detach().float()).all()]
    if nonfinite:
        raise RuntimeError(f"Non-finite trainable initialization: {nonfinite[:20]}")
    return {
        "total": sum(parameter.numel() for _, parameter in trainable),
        "fusion": sum(parameter.numel() for name, parameter in trainable if "controlled_vggt_fusion" in name),
        "lora": sum(parameter.numel() for name, parameter in trainable if "lora_" in name),
    }

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.use_cached_vggt and model_args.use_geometry_encoder:
        raise ValueError("use_cached_vggt and use_geometry_encoder are mutually exclusive")
    if model_args.use_cached_vggt and not model_args.lora_enable:
        raise ValueError("Controlled SFT requires lora_enable so both candidates share the same trainable LLM scope")
    if model_args.use_cached_vggt and training_args.per_device_train_batch_size != 1:
        raise ValueError("Controlled cached-VGGT SFT currently requires per-device batch size 1")
    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model_type = getattr(config, "model_type", None)

    if model_type == "qwen2_5_vl" or "qwen2.5" in model_args.model_name_or_path.lower():
        if not model_args.use_geometry_encoder:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
            if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
                raise ValueError(
                    "The use_geometry_encoder in config and model_args are not consistent. "
                    "Please check the model config."
                )

            for k in [
                "use_geometry_encoder", 
                "geometry_encoder_type", 
                "reference_frame",
                "feature_fusion_method", 
                "fusion_num_layers",
                "geometry_merger_type",
                "geometry_fusion_layers",
                "geometry_encoder_layers",
                "include_camera_token",
                "pos_encoding_type",
                "vision_language_fusion_layers",
            ]:
                setattr(config, k, getattr(model_args, k))

            assert model_args.geometry_encoder_path is not None, \
                "geometry_encoder_path must be set in the config when use_geometry_encoder is True."
            model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                geometry_encoder_path=model_args.geometry_encoder_path
            )

        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    elif model_type == "qwen2_vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"
        data_args.processor = None
    elif model_type in QWEN3_5_MODEL_TYPES or "qwen3.5" in model_args.model_name_or_path.lower():
        if data_args.data_flatten:
            raise NotImplementedError(
                "Qwen3.5 training does not support data_flatten in this branch."
            )

        from transformers import Qwen3_5ForConditionalGeneration

        if model_args.use_geometry_encoder or model_args.use_cached_vggt:
            from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

            for k in [
                "use_geometry_encoder",
                "geometry_encoder_type",
                "geometry_encoder_path",
                "reference_frame",
                "feature_fusion_method",
                "fusion_num_layers",
                "geometry_merger_type",
                "geometry_fusion_layers",
                "geometry_encoder_layers",
                "include_camera_token",
                "pos_encoding_type",
                "vision_language_fusion_layers",
                "use_cached_vggt",
                "controlled_fusion_candidate",
                "controlled_cross_attention_heads",
                "controlled_fusion_dropout",
                "controlled_projector_hidden_dim",
            ]:
                setattr(config, k, getattr(model_args, k))

            if model_args.use_cached_vggt:
                valid_candidates = {"a_premerger_cross_attn", "b_llm_add"}
                if model_args.controlled_fusion_candidate not in valid_candidates:
                    raise ValueError(
                        f"controlled_fusion_candidate must be one of {sorted(valid_candidates)}"
                    )
                if not data_args.cached_vggt_manifest:
                    raise ValueError("cached_vggt_manifest is required in cached-VGGT mode")
                actual_dimensions = (
                    config.vision_config.hidden_size,
                    config.vision_config.out_hidden_size,
                    config.text_config.hidden_size,
                    config.text_config.num_hidden_layers,
                )
                expected_dimensions = (1024, 2560, 2560, 32)
                if actual_dimensions != expected_dimensions:
                    raise ValueError(
                        f"Controlled held-out runs require Qwen3.5-4B dimensions {expected_dimensions}, "
                        f"got {actual_dimensions}"
                    )
                data_args.cached_vggt_layers = (
                    [23]
                    if model_args.controlled_fusion_candidate == "a_premerger_cross_attn"
                    else [11, 17, 23]
                )
            elif model_args.geometry_encoder_path is None:
                raise ValueError("geometry_encoder_path is required when use_geometry_encoder is true")

            model_load_kwargs = dict(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
            if model_args.use_geometry_encoder:
                model_load_kwargs["geometry_encoder_path"] = model_args.geometry_encoder_path
            model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(**model_load_kwargs)
        else:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            padding_side="right",
        )
        data_args.image_processor = processor.image_processor
        data_args.processor = processor
        data_args.model_type = "qwen3.5"
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}' for training path: {model_args.model_name_or_path}"
        )

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)
    model = add_language_lora(model_args, model)

    import torch.distributed as dist

    def is_rank_zero():
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    if is_rank_zero():
        visual_module, _, language_module, _ = resolve_model_modules(model)
        if hasattr(visual_module, "print_trainable_parameters"):
            visual_module.print_trainable_parameters()
        if hasattr(language_module, "print_trainable_parameters"):
            language_module.print_trainable_parameters()
        if model_args.use_cached_vggt:
            print(f"Controlled trainable audit: {audit_controlled_trainable_parameters(model)}")

    print(model.config)
    if model_args.use_geometry_encoder:
        setattr(data_args, "use_geometry_encoder", model_args.use_geometry_encoder)
    if model_args.use_cached_vggt:
        setattr(data_args, "use_cached_vggt", True)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer_class = ControlledCheckpointTrainer if model_args.use_cached_vggt else Trainer
    trainer = trainer_class(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        if model_args.use_cached_vggt:
            from qwen_vl.model.modeling_qwen3_5 import load_qwen3_5_controlled_submodules

            checkpoint = max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))
            base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
            loaded = load_qwen3_5_controlled_submodules(base_model, str(checkpoint))
            if loaded == 0:
                raise RuntimeError(f"Controlled checkpoint has no fusion artifact: {checkpoint}")
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    if os.environ.get("CONTROLLED_PROFILE_SKIP_SAVE", "0") == "1":
        rank0_print("CONTROLLED_PROFILE_SKIP_SAVE=1: skipping all profiling checkpoint artifacts")
        return
    trainer.save_state()
    if getattr(data_args, "processor", None) is not None:
        data_args.processor.save_pretrained(training_args.output_dir)
    else:
        data_args.image_processor.save_pretrained(training_args.output_dir)

    template_filename = "chat_template.json"
    template_path = os.path.join(training_args.output_dir, template_filename)

    source_path = None
    if os.path.isdir(model_args.model_name_or_path):
        candidate_path = os.path.join(model_args.model_name_or_path, template_filename)
        if os.path.isfile(candidate_path):
            source_path = candidate_path
    else:
        try:
            source_path = cached_file(
                model_args.model_name_or_path,
                template_filename,
                cache_dir=training_args.cache_dir,
            )
        except (OSError, EnvironmentError) as err:
            if getattr(data_args, "processor", None) is None:
                logging.warning("Unable to locate %s for model %s: %s", template_filename, model_args.model_name_or_path, err)

    if source_path:
        if os.path.abspath(source_path) != os.path.abspath(template_path):
            shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))
