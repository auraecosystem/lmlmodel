from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_cuda_module = None


def _kernel_dir() -> Path:
    return Path(__file__).resolve().parent


def _get_cuda_arch_flags() -> list[str]:
    """
    Compile specifically for the currently active GPU.

    This avoids unnecessarily compiling multiple CUDA
    architectures during JIT compilation.
    """
    if not torch.cuda.is_available():
        return []

    major, minor = torch.cuda.get_device_capability()

    return [
        "-gencode",
        f"arch=compute_{major}{minor},"
        f"code=sm_{major}{minor}",
    ]


def _get_cuda_flags() -> list[str]:
    flags = [
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    # Fast math is useful for the approximate GELU path.
    # Disable with:
    #
    # LMLM_FAST_MATH=0
    #
    if os.environ.get(
        "LMLM_FAST_MATH",
        "1",
    ) == "1":
        flags.append("--use_fast_math")

    flags.extend(
        _get_cuda_arch_flags()
    )

    # Optional source-level line information.
    if os.environ.get(
        "LMLM_CUDA_DEBUG",
        "0",
    ) == "1":
        flags.append("-lineinfo")

    return flags


def get_cuda_ops():
    """
    Lazily compile and load the LMLM CUDA extension.

    The resulting module registers:

        torch.ops.lmlm.fused_bias_gelu

    The extension itself is cached by PyTorch/Ninja.
    """

    global _cuda_module

    if _cuda_module is not None:
        return _cuda_module


    # ---------------------------------------------------------
    # CUDA availability
    # ---------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA accelerator requested but no "
            "GPU/CUDA device was found."
        )


    # ---------------------------------------------------------
    # Source discovery
    # ---------------------------------------------------------

    kernel_dir = _kernel_dir()

    cu_file = (
        kernel_dir /
        "fused_ops.cu"
    )

    cpp_file = (
        kernel_dir /
        "bindings.cpp"
    )


    if not cu_file.is_file():
        raise FileNotFoundError(
            f"LMLM CUDA source not found: {cu_file}"
        )

    if not cpp_file.is_file():
        raise FileNotFoundError(
            f"LMLM binding source not found: {cpp_file}"
        )


    # ---------------------------------------------------------
    # Build configuration
    # ---------------------------------------------------------

    extra_cflags = [
        "-O3",
    ]

    extra_cuda_cflags = (
        _get_cuda_flags()
    )


    # ---------------------------------------------------------
    # Optional compiler verbosity
    # ---------------------------------------------------------

    verbose = (
        os.environ.get(
            "LMLM_CUDA_VERBOSE",
            "0",
        ) == "1"
    )


    # ---------------------------------------------------------
    # JIT compilation
    # ---------------------------------------------------------

    _cuda_module = load(
        name="lmlm_cuda_ops",

        sources=[
            str(cpp_file),
            str(cu_file),
        ],

        extra_cflags=extra_cflags,

        extra_cuda_cflags=extra_cuda_cflags,

        with_cuda=True,

        verbose=verbose,
    )

    return _cuda_module
