import torch

from lmlm import fused_bias_gelu

x = torch.randn(
    4096,
    4096,
    device="cuda",
    dtype=torch.bfloat16,
)

bias = torch.randn(
    4096,
    device="cuda",
    dtype=torch.bfloat16,
)

y = fused_bias_gelu(
    x,
    bias,
)
