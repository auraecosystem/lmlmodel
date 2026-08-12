from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F


try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


import lmlm


# ============================================================
# Benchmark utilities
# ============================================================

def cuda_benchmark(
    fn,
    warmup=20,
    iterations=100,
):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()

    for _ in range(iterations):
        fn()

    end.record()

    end.synchronize()

    return (
        start.elapsed_time(end)
        * 1000.0
        / iterations
    )


def correctness(
    reference,
    candidate,
):
    reference = reference.float()
    candidate = candidate.float()

    diff = (
        reference -
        candidate
    ).abs()

    return {
        "max_error": diff.max().item(),
        "mean_error": diff.mean().item(),
        "rmse": torch.sqrt(
            torch.mean(diff * diff)
        ).item(),
    }


# ============================================================
# PyTorch reference
# ============================================================

def pytorch_gelu(
    x,
    bias,
    approximate=True,
):
    mode = (
        "tanh"
        if approximate
        else "none"
    )

    return F.gelu(
        x + bias,
        approximate=mode,
    )


# ============================================================
# torch.compile
# ============================================================

def compiled_gelu(
    approximate=True,
):
    return torch.compile(
        lambda x, bias:
            pytorch_gelu(
                x,
                bias,
                approximate,
            ),
        mode="max-autotune",
        fullgraph=True,
    )


# ============================================================
# Triton implementation
# ============================================================

if TRITON_AVAILABLE:

    @triton.jit
    def fused_bias_gelu_kernel(
        x_ptr,
        b_ptr,
        y_ptr,
        n_elements,
        hidden_dim,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)

        offsets = (
            pid * BLOCK +
            tl.arange(0, BLOCK)
        )

        mask = offsets < n_elements

        x = tl.load(
            x_ptr + offsets,
            mask=mask,
            other=0.0,
        )

        bias_index = (
            offsets % hidden_dim
        )

        b = tl.load(
            b_ptr + bias_index,
            mask=mask,
            other=0.0,
        )

        x = x + b

        inner = (
            0.7978845608028654 *
            (
                x +
                0.044715 *
                x * x * x
            )
        )

        y = (
            0.5 *
            x *
            (
                1.0 +
                tl.math.tanh(inner)
            )
        )

        tl.store(
            y_ptr + offsets,
            y,
            mask=mask,
        )


    def triton_gelu(
        x,
        bias,
    ):
        output = torch.empty_like(x)

        n = x.numel()

        grid = lambda meta: (
            triton.cdiv(
                n,
                meta["BLOCK"],
            ),
        )

        fused_bias_gelu_kernel[
            grid
        ](
            x,
            bias,
            output,
            n,
            x.shape[-1],
            BLOCK=256,
        )

        return output


# ============================================================
# Benchmark suite
# ============================================================

def run_benchmark(
    rows,
    hidden_dim,
    dtype,
    approximate=True,
    warmup=20,
    iterations=100,
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device("cuda")

    x = torch.randn(
        rows,
        hidden_dim,
        device=device,
        dtype=dtype,
    )

    bias = torch.randn(
        hidden_dim,
        device=device,
        dtype=dtype,
    )

    print()
    print("=" * 72)
    print("LMLM FUSED BIAS + GELU BENCHMARK")
    print("=" * 72)

    print(
        f"GPU:      {torch.cuda.get_device_name()}"
    )

    print(
        f"Shape:    [{rows}, {hidden_dim}]"
    )

    print(
        f"Dtype:    {dtype}"
    )

    print(
        f"GELU:     "
        f"{'approximate' if approximate else 'exact'}"
    )

    print("=" * 72)


    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = pytorch_gelu(
        x,
        bias,
        approximate,
    )

    # --------------------------------------------------------
    # LMLM CUDA
    # --------------------------------------------------------

    lmlm_fn = lambda: (
        torch.ops.lmlm.fused_bias_gelu(
            x,
            bias,
            approximate,
        )
    )

    lmlm_output = lmlm_fn()

    lmlm_time = cuda_benchmark(
        lmlm_fn,
        warmup,
        iterations,
    )

    print(
        f"LMLM CUDA:          "
        f"{lmlm_time:.3f} us"
    )

    print(
        "LMLM correctness:",
        correctness(
            reference,
            lmlm_output,
        ),
    )


    # --------------------------------------------------------
    # PyTorch eager
    # --------------------------------------------------------

    eager_fn = lambda: pytorch_gelu(
        x,
        bias,
        approximate,
    )

    eager_time = cuda_benchmark(
        eager_fn,
        warmup,
        iterations,
    )

    print(
        f"PyTorch eager:      "
        f"{eager_time:.3f} us"
    )


    # --------------------------------------------------------
    # torch.compile
    # --------------------------------------------------------

    try:
        compiled = compiled_gelu(
            approximate
        )

        compiled_fn = lambda: compiled(
            x,
            bias,
        )

        # Compilation warmup.
        for _ in range(10):
            compiled_fn()

        torch.cuda.synchronize()

        compiled_time = cuda_benchmark(
            compiled_fn,
            warmup,
            iterations,
        )

        print(
            f"torch.compile:      "
            f"{compiled_time:.3f} us"
        )

    except Exception as exc:
        compiled_time = None

        print(
            f"torch.compile:      unavailable "
            f"({exc})"
        )


    # --------------------------------------------------------
    # Triton
    # --------------------------------------------------------

    triton_time = None

    if TRITON_AVAILABLE:

        try:
            triton_fn = lambda: (
                triton_gelu(
                    x,
                    bias,
                )
            )

            triton_output = triton_fn()

            triton_time = cuda_benchmark(
                triton_fn,
                warmup,
                iterations,
            )

            print(
                f"Triton:             "
                f"{triton_time:.3f} us"
            )

            print(
                "Triton correctness:",
                correctness(
                    reference,
                    triton_output,
                ),
            )

        except Exception as exc:

            print(
                f"Triton:             unavailable "
                f"({exc})"
            )

    else:
        print(
            "Triton:             not installed"
        )


    # --------------------------------------------------------
    # Relative performance
    # --------------------------------------------------------

    print()
    print("-" * 72)

    print(
        f"LMLM speedup vs PyTorch: "
        f"{eager_time / lmlm_time:.2f}x"
    )

    if compiled_time:
        print(
            f"LMLM speedup vs compile: "
            f"{compiled_time / lmlm_time:.2f}x"
        )

    if triton_time:
        print(
            f"LMLM speedup vs Triton:  "
            f"{triton_time / lmlm_time:.2f}x"
        )

    print("-" * 72)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rows",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "fp32",
            "fp16",
            "bf16",
        ],
        default="bf16",
    )

    parser.add_argument(
        "--exact",
        action="store_true",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    run_benchmark(
        rows=args.rows,
        hidden_dim=args.hidden_dim,
        dtype=dtype_map[
            args.dtype
        ],
        approximate=not args.exact,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
