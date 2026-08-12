from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import torch


@dataclass
class TuneResult:
    block_size: int
    latency_us: float
    candidates: dict[int, float]


class LMLMAutotuner:

    def __init__(
        self,
        cache_path: str | None = None,
        candidates=(64, 128, 256, 512),
        warmup=25,
        iterations=100,
    ):
        self.candidates = tuple(candidates)
        self.warmup = warmup
        self.iterations = iterations

        if cache_path is None:
            cache_path = os.path.expanduser(
                "~/.cache/lmlm/"
                "fused_bias_gelu.json"
            )

        self.cache_path = Path(cache_path)

        self.cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.cache = self._load()

    def _load(self):
        if not self.cache_path.exists():
            return {}

        try:
            with self.cache_path.open() as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self):
        tmp = self.cache_path.with_suffix(
            ".tmp"
        )

        with tmp.open("w") as f:
            json.dump(
                self.cache,
                f,
                indent=2,
                sort_keys=True,
            )

        tmp.replace(self.cache_path)

    def _key(
        self,
        input,
        approximate,
    ):
        device = input.device

        capability = (
            torch.cuda.get_device_capability(
                device
            )
        )

        return json.dumps(
            {
                "device":
                    torch.cuda.get_device_name(
                        device
                    ),
                "capability":
                    capability,
                "dtype":
                    str(input.dtype),
                "hidden_dim":
                    input.shape[-1],
                "rows":
                    input.numel() //
                    input.shape[-1],
                "approximate":
                    approximate,
            },
            sort_keys=True,
        )

    @staticmethod
    def _measure(
        fn,
        warmup,
        iterations,
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

    def tune(
        self,
        input,
        bias,
        approximate=True,
    ):
        if not input.is_cuda:
            raise RuntimeError(
                "Autotuning requires CUDA."
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

        results = {}

        for block in self.candidates:

            fn = lambda: (
                torch.ops.lmlm.fused_bias_gelu(
                    input,
                    bias,
                    approximate,
                    block,
                )
            )

            results[block] = self._measure(
                fn,
                self.warmup,
                self.iterations,
            )

        best = min(
            results,
            key=results.get,
        )

        result = TuneResult(
            block_size=best,
            latency_us=results[best],
            candidates=results,
        )

        self.cache[key] = asdict(result)

        self._save()

        return result
