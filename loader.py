from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_cuda_module = None


def _kernel_dir() -> Path:
    return Path(__file__).resolve().parent


def _cuda_arch_flags() -> list[str]:
    if not torch.cuda.is_available():
        return []

    major, minor = torch.cuda.get_device_capability()

    return [
        "-gencode",
        f"arch=compute_{major}{minor},"
        f"code=sm_{major}{minor}",
    ]


def _nvcc_flags() -> list[str]:
    flags = [
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    if os.environ.get(
        "LMLM_FAST_MATH",
        "1",
    ) == "1":
        flags.append("--use_fast_math")

    flags.extend(
        _cuda_arch_flags()
    )

    if os.environ.get(
        "LMLM_CUDA_DEBUG",
        "0",
    ) == "1":
        flags.append("-lineinfo")

    return flags


def get_cuda_ops():
    global _cuda_module

    if _cuda_module is not None:
        return _cuda_module

    if not torch.cuda.is_available():
        raise RuntimeError(
            "LMLM CUDA backend requested, "
            "but CUDA is unavailable."
        )

    root = _kernel_dir()

    cu = root / "fused_ops.cu"
    cpp = root / "bindings.cpp"

    if not cu.exists():
        raise FileNotFoundError(cu)

    if not cpp.exists():
        raise FileNotFoundError(cpp)

    verbose = (
        os.environ.get(
            "LMLM_CUDA_VERBOSE",
            "0",
        ) == "1"
    )

    _cuda_module = load(
        name="lmlm_cuda_ops",
        sources=[
            str(cpp),
            str(cu),
        ],
        extra_cflags=[
            "-O3",
        ],
        extra_cuda_cflags=_nvcc_flags(),
        with_cuda=True,
        verbose=verbose,
    )

    return _cuda_module
