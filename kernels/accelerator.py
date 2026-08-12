import torch
import torch.nn as nn
from kernels.compiler import get_cuda_ops


class FusedBiasGELU(nn.Module):
    """Fused Bias-GELU layer powered by custom .cu kernel with CPU fallback."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.cuda_ops = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            if self.cuda_ops is None:
                self.cuda_ops = get_cuda_ops()
            return self.cuda_ops.fused_bias_gelu(x, self.bias)
        return torch.nn.functional.gelu(x + self.bias)
