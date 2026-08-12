from typing import Dict, List, Optional
import torch


class PhysicalBlock:
    """Represents a discrete contiguous block of GPU KV-cache memory."""
    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0

    def is_free(self) -> bool:
        return self.ref_count == 0


class PagedKVCacheManager:
    """Virtual memory page allocator for LLM/LMLM KV caches."""
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device: str = "cuda"
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        
        # Pre-allocate physical KV pool on GPU
        # Shape: [num_blocks, 2 (K/V), block_size, num_heads, head_dim]
        self.kv_pool = torch.zeros(
            (num_blocks, 2, block_size, num_heads, head_dim),
            dtype=torch.float16,
            device=device
        )
        
        self.blocks = [PhysicalBlock(i, block_size) for i in range(num_blocks)]
        self.free_blocks: List[int] = list(range(num_blocks))
        self.block_tables: Dict[str, List[int]] = {}

    def allocate(self, request_id: str, num_tokens: int) -> List[int]:
        """Allocates physical blocks for a new or expanding request."""
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < blocks_needed:
            raise RuntimeError("Out of GPU KV-Cache Memory (OOM)")

        allocated = []
        for _ in range(blocks_needed):
            block_id = self.free_blocks.pop(0)
            self.blocks[block_id].ref_count += 1
            allocated.append(block_id)

        if request_id not in self.block_tables:
            self.block_tables[request_id] = []
        self.block_tables[request_id].extend(allocated)
        return self.block_tables[request_id]

    def free(self, request_id: str):
        """Frees physical blocks associated with a completed request."""
        if request_id not in self.block_tables:
            return
        
        for block_id in self.block_tables[request_id]:
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].is_free():
                self.free_blocks.append(block_id)
        
        del self.block_tables[request_id]
