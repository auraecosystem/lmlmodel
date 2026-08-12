from __future__ import annotations

import torch

from .loader import get_cuda_ops


def fused_bias_gelu(
    input: torch.Tensor,
    bias: torch.Tensor,
    approximate: bool = True,
    block_size: int = 0,
) -> torch.Tensor:

    if input.device != bias.device:
        raise ValueError(
            "input and bias must be on the same device"
        )

    if input.dim() < 1:
        raise ValueError(
            "input must have at least one dimension"
        )

    if bias.dim() != 1:
        raise ValueError(
            "bias must be one-dimensional"
        )

    if input.shape[-1] != bias.shape[0]:
        raise ValueError(
            "input last dimension must equal "
            "bias dimension"
        )

    if input.is_cuda:
        get_cuda_ops()

    return torch.ops.lmlm.fused_bias_gelu(
        input,
        bias,
        approximate,
        block_size,
    )
