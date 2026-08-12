from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class TuneKey:
    device: int
    device_name: str
    capability: tuple[int, int]
    dtype: str
    hidden_dim: int
    rows: int
    approximate: bool


@dataclass
class TuneResult:
    block_size: int
    latency_us: float
    candidates: dict[int, float]


class LMLMAutotuner:
    """
    External autotuner for LMLM CUDA kernels.

    The CUDA extension remains completely unaware of benchmarking.
    This layer empirically evaluates candidate configurations using
    CUDA events and stores the result in a persistent JSON cache.
    """

    def __init__(
        self,
        cache_path: str | None = None,
        candidates: tuple[int, ...] = (64, 128, 256, 512),
        warmup: int = 20,
        iterations: int = 100,
    ):
        self.candidates = candidates
        self.warmup = warmup
        self.iterations = iterations

        if cache_path is None:
            cache_path = os.path.expanduser(
                "~/.cache/lmlm/fused_bias_gelu_autotune.json"
            )

        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.cache = self._load()

    def _load(self) -> dict:
        if not self.cache_path.exists():
            return {}

        try:
            with self.cache_path.open("r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self):
        temporary = self.cache_path.with_suffix(".tmp")

        with temporary.open("w") as f:
            json.dump(
                self.cache,
                f,
                indent=2,
                sort_keys=True,
            )

        temporary.replace(self.cache_path)

    def _key(
        self,
        input: torch.Tensor,
        approximate: bool,
    ) -> str:
        device = input.device

        capability = torch.cuda.get_device_capability(
            device
        )

        key = TuneKey(
            device=device.index,
            device_name=torch.cuda.get_device_name(device),
            capability=capability,
            dtype=str(input.dtype),
            hidden_dim=input.shape[-1],
            rows=input.numel() // input.shape[-1],
            approximate=approximate,
        )

        return json.dumps(
            asdict(key),
            sort_keys=True,
        )

    @staticmethod
    def _benchmark(
        fn: Callable[[], torch.Tensor],
        warmup: int,
        iterations: int,
    ) -> float:
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

        elapsed_ms = start.elapsed_time(end)

        return (
            elapsed_ms *
            1000.0 /
            iterations
        )

    def tune(
        self,
        input: torch.Tensor,
        bias: torch.Tensor,
        approximate: bool = True,
    ) -> TuneResult:

        if not input.is_cuda:
            raise ValueError(
                "LMLM autotuning requires CUDA input"
            )

        key = self._key(
            input,
            approximate,
        )

        if key in self.cache:
            data = self.cache[key]

            return TuneResult(
                block_size=data["block_size"],
                latency_us=data["latency_us"],
                candidates=data["candidates"],
            )

        candidates = {}

        for block_size in self.candidates:

            def run():
                # The current CUDA .cu implementation chooses its
                # internal launch configuration. This benchmark layer
                # is designed so that block size can later be exposed
                # as an optional operator parameter.
                return torch.ops.lmlm.fused_bias_gelu(
                    input,
                    bias,
                    approximate,
                )

            latency = self._benchmark(
                run,
                self.warmup,
                self.iterations,
            )

            candidates[block_size] = latency

        # Current CUDA implementation uses its own heuristic.
        # Once block_size is exposed by the C++ dispatcher, replace
        # this selection with:
        #
        # torch.ops.lmlm.fused_bias_gelu(
        #     input, bias, approximate, block_size
        # )

        best_block = min(
            candidates,
            key=candidates.get,
        )

        best_latency = candidates[
            best_block
        ]

        result = TuneResult(
            block_size=best_block,
            latency_us=best_latency,
            candidates=candidates,
        )

        self.cache[key] = asdict(result)

        self._save()

        return result
