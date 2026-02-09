#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""Benchmark EDT backends across sizes and devices.

Usage::

    # CPU only (scipy)
    python scripts/bench_edt.py

    # Include CUDA backends
    python scripts/bench_edt.py --cuda

    # Custom sizes / density
    python scripts/bench_edt.py --cuda --sizes 64 256 512 1024 --density 0.5
"""

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch

import sam3.model.edt as edt


@dataclass(frozen=True)
class BenchResult:
    backend: str
    size: int
    batch: int
    mean_ms: float
    std_ms: float


def _warmup_cuda():
    """Run a small allocation to get CUDA context initialised."""
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()


def _bench_fn(
    fn, data: torch.Tensor, *, warmup: int = 3, repeats: int = 10
) -> tuple[float, float]:
    """Time *fn(data)* and return (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn(data)
    if data.is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if data.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(data)
        if data.is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def run_benchmarks(
    sizes: list[int],
    batch: int,
    density: float,
    include_cuda: bool,
    repeats: int,
    workers: int | None = None,
) -> list[BenchResult]:
    results: list[BenchResult] = []

    # pre-define the callers
    cpu_methods: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    cuda_methods: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}

    def build_pair(
        fnt: Callable[[torch.Tensor, int | None], torch.Tensor],
        fns: Callable[[torch.Tensor], torch.Tensor],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        return partial(fnt, workers=workers) if workers != 1 else fns

    if edt._HAS_CV2:
        cpu_methods["edt-cv2"] = build_pair(edt._edt_cv2_threaded, edt._edt_cv2)  # ty:ignore[possibly-missing-attribute]
    if edt._HAS_TRITON:
        # has no multithreading
        cpu_methods["edt-triton"] = edt._edt_triton  # type:ignore[possibly-missing-attribute]
    if edt._HAS_CUPY:
        cpu_methods["edt-cupy"] = build_pair(edt._edt_cupy_threaded, edt._edt_cupy)  # type:ignore[possibly-missing-attribute]

    for size in sizes:
        torch.manual_seed(0)
        data_cpu = (torch.rand(batch, size, size) > density).float()

        for name, fn in cpu_methods.items():
            mean, std = _bench_fn(fn, data_cpu, repeats=repeats)
            results.append(BenchResult(name, size, batch, mean, std))

        if not include_cuda or not torch.cuda.is_available() or not cuda_methods:
            continue

        data_cuda = data_cpu.cuda()

        for name, fn in cuda_methods.items():
            mean, std = _bench_fn(fn, data_cuda, repeats=repeats)
            results.append(BenchResult(name, size, batch, mean, std))

    return results


def print_table(results: list[BenchResult]) -> None:
    header = (
        f"{'Backend':<16} {'Size':>6} {'Batch':>5} {'Mean (ms)':>10} {'Std (ms)':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.backend:<16} {r.size:>6} {r.batch:>5} "
            f"{r.mean_ms:>10.2f} {r.std_ms:>10.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark EDT backends")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
        help="Image sizes (H=W) to benchmark",
    )
    parser.add_argument("--batch", "-b", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="Fraction of foreground pixels (0-1)",
    )
    parser.add_argument("--cuda", action="store_true", help="Include CUDA backends")
    parser.add_argument("--repeats", "-r", type=int, default=10, help="Timing repeats")
    parser.add_argument(
        "--workers", "-w", type=int, default=1, help="Number of workers"
    )
    args = parser.parse_args(argv)

    if args.workers <= 0:
        args.workers = None

    print(
        f"Backends available: scipy=True, cupy={edt._HAS_CUPY}, triton={edt._HAS_TRITON}"
    )
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Multithreading: {args.workers is not None and args.workers > 1}")
    print()

    if args.cuda:
        _warmup_cuda()

    results = run_benchmarks(
        sizes=args.sizes,
        batch=args.batch,
        density=args.density,
        include_cuda=args.cuda,
        repeats=args.repeats,
        workers=args.workers,
    )

    print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
