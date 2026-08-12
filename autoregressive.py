import torch
import torch.nn as nn


class CUDAGraphDecoderEngine:
    """CUDA Graph executor for ultra-low latency autoregressive decoding."""
    def __init__(self, model: nn.Module, max_batch_size: int, hidden_dim: int, device: str = "cuda"):
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        
        # Static inputs/outputs required for graph capture
        self.static_input = torch.zeros((max_batch_size, 1), dtype=torch.long, device=device)
        self.static_output = torch.zeros((max_batch_size, 1, hidden_dim), dtype=torch.float16, device=device)
        
        self.graph: Optional[torch.cuda.GRAPH] = None
        self._warmup_and_capture()

    def _warmup_and_capture(self):
        """Warming up streams and capturing the execution graph."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):  # Warmup steps
                self.static_output = self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(s)

        # Graph Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)

    def decode_step(self, active_token_ids: torch.Tensor) -> torch.Tensor:
        """Executes a decode step via CUDA graph replay without CPU launch overhead."""
        bsz = active_token_ids.size(0)
        # Copy dynamic batch into static captured memory
        self.static_input[:bsz].copy_(active_token_ids)
        
        # Replay CUDA Graph
        self.graph.replay()
        
        return self.static_output[:bsz]
