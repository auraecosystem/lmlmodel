from typing import Optional
import torch
import torch.nn as nn


class CUDAGraphDecoderEngine:
    """CUDA Graph executor for ultra-low latency autoregressive decoding."""

    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int,
        hidden_dim: int,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size

        # Pre-allocate fixed static memory addresses for graph inputs and outputs
        self.static_input = torch.zeros(
            (max_batch_size, 1), dtype=torch.long, device=device
        )
        self.static_output = torch.zeros(
            (max_batch_size, 1, hidden_dim),
            dtype=torch.float16,
            device=device,
        )

        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self._warmup_and_capture()

    def _warmup_and_capture(self) -> None:
        """Warming up CUDA allocator and capturing the execution graph."""
        # 1. Warmup runs on a dedicated stream to initialize CUDNN/CUBLAS handles and allocate memory
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(device=self.device))

        with torch.cuda.stream(s):
            for _ in range(3):
                out = self.model(self.static_input)
                self.static_output.copy_(out)

        torch.cuda.current_stream(device=self.device).wait_stream(s)

        # 2. Capture the graph recording ops targeting static_input -> static_output
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            out = self.model(self.static_input)
            self.static_output.copy_(out)

    def decode_step(self, active_token_ids: torch.Tensor) -> torch.Tensor:
        """Executes a decode step via CUDA graph replay without CPU launch overhead."""
        bsz = active_token_ids.size(0)

        # 1. Zero-out stale batch slots to prevent side-effects from unused tokens
        self.static_input.zero_()

        # 2. Copy dynamic input tensor into the captured static input buffer
        self.static_input[:bsz].copy_(active_token_ids)

        # 3. Replay CUDA Graph
        self.graph.replay()

        # 4. Return slice of the static output buffer matching current batch size
        return self.static_output[:bsz]
