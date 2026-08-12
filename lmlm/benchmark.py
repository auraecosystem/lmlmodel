from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .api import fused_bias_gelu
from .autotune import LMLMAutotuner


try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def benchmark_cuda(
    fn,
    warmup=25,
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


def error_metrics(
    reference,
    output,
):
    diff = (
        reference.float() -
        output.float()
    ).abs()

    return {
        "max": diff.max().item(),
        "mean": diff.mean().item(),
        "rmse": torch.sqrt(
            (diff * diff).mean()
        ).item(),
    }


def pytorch_reference(
    x,
    bias,
    approximate,
):
    return F.gelu(
        x + bias,
        approximate=(
            "tanh"
            if approximate
            else "none"
        ),
    )


def run(
    rows,
    hidden_dim,
    dtype,
    approximate,
    warmup,
    iterations,
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
    print("LMLM FUSED BIAS + GELU")
    print("=" * 72)

    print(
        "GPU:",
        torch.cuda.get_device_name(device),
    )

    print(
        "Shape:",
        tuple(x.shape),
    )

    print(
        "Dtype:",
        dtype,
    )

    print(
        "GELU:",
        "tanh approximation"
        if approximate
        else "exact",
    )

    print("=" * 72)


    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = pytorch_reference(
        x,
        bias,
        approximate,
    )


    # --------------------------------------------------------
    # Autotune
    # --------------------------------------------------------

    tuner = LMLMAutotuner(
        warmup=warmup,
        iterations=iterations,
    )

    tuned = tuner.tune(
        x,
        bias,
        approximate,
    )

    print(
        f"Autotuned block:    "
        f"{tuned.block_size}"
    )

    print(
        f"Autotuned latency:   "
        f"{tuned.latency_us:.3f} us"
    )

    for block, latency in sorted(
        tuned.candidates.items()
    ):
        print(
            f"  BLOCK={block:<4} "
            f"{latency:.3f} us"
        )


    # --------------------------------------------------------
    # LMLM CUDA
    # --------------------------------------------------------

    lmlm_fn = lambda: fused_bias_gelu(
        x,
        bias,
        approximate,
        tuned.block_size,
    )

    lmlm_output = lmlm_fn()

    lmlm_time = benchmark_cuda(
        lmlm_fn,
        warmup,
        iterations,
    )

    print()
    print(
        f"LMLM CUDA:          "
        f"{lmlm_time:.3f} us"
    )

    print(
        "LMLM error:",
        error_metrics(
            reference,
            lmlm_output,
        ),
    )


    # --------------------------------------------------------
    # PyTorch eager
    # --------------------------------------------------------

    eager_fn = lambda: (
        pytorch_reference(
            x,
            bias,
            approximate,
        )
    )

    eager_time = benchmark_cuda(
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

    compile_time = None

    try:
        compiled = torch.compile(
            pytorch_reference,
            mode="max-autotune",
            fullgraph=True,
        )

        for _ in range(10):
            compiled(
                x,
                bias,
                approximate,
            )

        torch.cuda.synchronize()

        compile_fn = lambda: compiled(
            x,
            bias,
            approximate,
        )

        compile_time = benchmark_cuda(
            compile_fn,
            warmup,
            iterations,
        )

        print(
            f"torch.compile:      "
            f"{compile_time:.3f} us"
        )

    except Exception as exc:
        print(
            "torch.compile:      unavailable"
        )
        print(
            "                    ",
            exc,
        )


    # --------------------------------------------------------
    # Triton
    # --------------------------------------------------------

    triton_time = None

    if TRITON_AVAILABLE:

        try:

            @triton.jit
            def kernel(
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

                mask = (
                    offsets <
                    n_elements
                )

                x_value = tl.load(
                    x_ptr + offsets,
                    mask=mask,
                    other=0.0,
                )

                col = (
                    offsets %
                    hidden_dim
                )

                b_value = tl.load(
                    b_ptr + col,
                    mask=mask,
                    other=0.0,
                )

                x_value += b_value

                inner = (
                    0.7978845608028654 *
                    (
                        x_value +
                        0.044715 *
                        x_value *
                        x_value *
                        x_value
                    )
                )

                y_value = (
                    0.5 *
                    x_value *
                    (
                        1.0 +
                        tl.math.tanh(inner)
                    )
                )

                tl.store(
                    y_ptr + offsets,
                    y_value,
                    mask=mask,
                )


            def triton_fn():
                output = torch.empty_like(x)

                n = x.numel()

                kernel[
                    lambda meta: (
                        triton.cdiv(
                            n,
                            meta["BLOCK"],
                        ),
                    )
                ](
                    x,
                    bias,
                    output,
                    n,
                    hidden_dim,
                    BLOCK=256,
                )

                return output


            triton_output = triton_fn()

            triton_time = benchmark_cuda(
                triton_fn,
                warmup,
                iterations,
            )

            print(
                f"Triton:             "
                f"{triton_time:.3f} us"
            )

            print(
                "Triton error:",
                error_metrics(
                    reference,
                    triton_output,
                ),
            )

        except Exception as exc:
            print(
                "Triton:             unavailable"
            )
            print(
                "                    ",
                exc,
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
        "LMLM / PyTorch speedup:",
        f"{eager_time / lmlm_time:.2f}x",
    )

    if compile_time:
        print(
            "LMLM / compile speedup:",
            f"{compile_time / lmlm_time:.2f}x",
        )

    if triton_time:
        print(
            "LMLM / Triton speedup:",
            f"{triton_time / lmlm_time:.2f}x",
        )

    print("-" * 72)


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
        default=25,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]

    run(
        rows=args.rows,
        hidden_dim=args.hidden_dim,
        dtype=dtype,
        approximate=not args.exact,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
