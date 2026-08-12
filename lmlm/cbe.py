import asyncio
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class GenerationRequest:
    request_id: str
    prompt: str
    prompt_tokens: List[int]
    max_new_tokens: int
    generated_tokens: List[int] = field(default_factory=list)
    is_prefill: bool = True
    finished: bool = False


class ContinuousBatchingEngine:
    """Iteration-level scheduler with dynamic prefill and decode pipelining."""
    def __init__(self, kv_manager: PagedKVCacheManager, max_batch_size: int = 8):
        self.kv_manager = kv_manager
        self.max_batch_size = max_batch_size
        self.waiting_queue: List[GenerationRequest] = []
        self.running_batch: Dict[str, GenerationRequest] = {}

    def add_request(self, request: GenerationRequest):
        self.waiting_queue.append(request)

    async def step(self):
        """Executes a single iteration step across all active sequences."""
        # 1. Admit waiting requests if batch capacity & KV cache allows
        while self.waiting_queue and len(self.running_batch) < self.max_batch_size:
            req = self.waiting_queue[0]
            try:
                # Reserve KV space
                self.kv_manager.allocate(req.request_id, len(req.prompt_tokens))
                self.running_batch[req.request_id] = self.waiting_queue.pop(0)
            except RuntimeError:
                # GPU Memory Full - wait until running tasks complete
                break

        if not self.running_batch:
            return

        # 2. Separate into Prefill vs Decode phases
        prefill_reqs = [r for r in self.running_batch.values() if r.is_prefill]
        decode_reqs = [r for r in self.running_batch.values() if not r.is_prefill]

        # 3. Process Prefills (Prompt Chunk Processing)
        for req in prefill_reqs:
            # Execute prompt forward pass
            next_token = self._mock_forward(req.prompt_tokens)
            req.generated_tokens.append(next_token)
            req.is_prefill = False

        # 4. Process Decodes (Autoregressive Single-Token Step)
        if decode_reqs:
            # Batch decoding using CUDA Graph or PagedAttention kernel
            for req in decode_reqs:
                next_token = self._mock_forward([req.generated_tokens[-1]])
                req.generated_tokens.append(next_token)

                # Check completion conditions
                if len(req.generated_tokens) >= req.max_new_tokens or next_token == 0: # EOS
                    req.finished = True

        # 5. Eject Finished Requests
        finished_ids = [r.request_id for r in self.running_batch.values() if r.finished]
        for req_id in finished_ids:
            print(f"Request [{req_id}] finished. Cleaning up KV cache.")
            self.kv_manager.free(req_id)
            del self.running_batch[req_id]

    @staticmethod
    def _mock_forward(tokens: List[int]) -> int:
        """Simulates GPU forward pass returning token ID."""
        return (tokens[-1] + 1) % 1000
