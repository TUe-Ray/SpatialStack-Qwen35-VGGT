#!/usr/bin/env python3
"""Fail before Qwen3.5 training when required acceleration is unavailable."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import pathlib
import platform
import sys
from typing import Any


EXPECTED_VERSIONS = {
    "torch": "2.10.0",
    "transformers": "5.3.0",
    "accelerate": "1.13.0",
    "qwen-vl-utils": "0.0.14",
    "flash-attn": "2.8.3",
    "deepspeed": "0.16.4",
}


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def import_module(name: str, errors: list[str]) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception as exc:  # binary-extension import failures matter here
        errors.append(f"cannot import {name}: {type(exc).__name__}: {exc}")
        return None


def check_version(
    distribution: str,
    errors: list[str],
    versions: dict[str, str | None],
) -> None:
    expected = EXPECTED_VERSIONS[distribution]
    actual = distribution_version(distribution)
    versions[distribution] = actual
    if actual is None:
        errors.append(f"missing distribution {distribution}=={expected}")
    elif distribution == "torch" and actual.split("+", 1)[0] == expected:
        pass
    elif actual != expected:
        errors.append(f"{distribution} must be {expected}, found {actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-implementation", default="flash_attention_2")
    parser.add_argument("--require-fast-linear-attention", action="store_true")
    parser.add_argument("--deepspeed-config", default="")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    versions: dict[str, str | None] = {}

    if sys.version_info[:2] != (3, 12):
        errors.append(f"Python must be 3.12, found {platform.python_version()}")

    for distribution in ("torch", "transformers", "accelerate", "qwen-vl-utils"):
        check_version(distribution, errors, versions)

    torch = import_module("torch", errors)
    if torch is not None:
        versions["torch_runtime"] = str(torch.__version__)
        versions["cuda_runtime"] = str(torch.version.cuda)
        versions["cuda_available"] = str(torch.cuda.is_available())
        if not str(torch.__version__).startswith("2.10.0+"):
            errors.append(f"PyTorch CUDA build must be 2.10.0+cu129, found {torch.__version__}")
        if str(torch.version.cuda) != "12.9":
            errors.append(f"PyTorch CUDA runtime must be 12.9, found {torch.version.cuda}")
        if args.require_cuda and not torch.cuda.is_available():
            errors.append("CUDA is required but torch.cuda.is_available() is false")

    if args.attention_implementation == "flash_attention_2":
        check_version("flash-attn", errors, versions)
        import_module("flash_attn", errors)

    if args.require_fast_linear_attention:
        for distribution in ("causal-conv1d", "flash-linear-attention"):
            versions[distribution] = distribution_version(distribution)
            if versions[distribution] is None:
                errors.append(f"missing distribution {distribution}")
        import_module("causal_conv1d", errors)
        import_module("fla", errors)
        qwen35 = import_module("transformers.models.qwen3_5.modeling_qwen3_5", errors)
        # FLA intentionally selects its CPU fallback when a CPU-only build node
        # imports it. The definitive fast-path assertion therefore belongs to
        # the GPU launch preflight (--require-cuda), not the wheel-build job.
        if (
            qwen35 is not None
            and args.require_cuda
            and not bool(getattr(qwen35, "is_fast_path_available", False))
        ):
            errors.append("Transformers reports Qwen3.5 linear-attention fast path unavailable")

    if args.deepspeed_config:
        config = pathlib.Path(args.deepspeed_config)
        if not config.is_file():
            errors.append(f"DeepSpeed config does not exist: {config}")
        check_version("deepspeed", errors, versions)
        cuda_home = os.environ.get("CUDA_HOME")
        versions["cuda_home"] = cuda_home
        if not cuda_home or not (pathlib.Path(cuda_home) / "bin" / "nvcc").is_file():
            errors.append("DeepSpeed requires CUDA_HOME pointing to a toolkit with bin/nvcc")
        else:
            import_module("deepspeed", errors)

    report = {
        "status": "error" if errors else "ok",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "attention_implementation": args.attention_implementation,
        "require_fast_linear_attention": args.require_fast_linear_attention,
        "deepspeed_config": args.deepspeed_config or None,
        "versions": versions,
        "errors": errors,
    }
    print("QWEN35_RUNTIME_VALIDATION " + json.dumps(report, sort_keys=True))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
