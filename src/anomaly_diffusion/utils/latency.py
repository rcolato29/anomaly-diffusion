"""Latency measurement.

Times a callable with warmup and device synchronization, returning p50/p95 and throughput.
"""

from __future__ import annotations

import time

import numpy as np
import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def measure_latency(
    fn, device: torch.device, batch_size: int = 1, n_warmup: int = 3, n_runs: int = 20
) -> dict:
    """Time fn (one scored batch per call). Returns latency percentiles in ms.

    batch_size is used only to convert timings to throughput (images/sec).
    """
    for _ in range(n_warmup):
        fn()
    synchronize(device)

    times = []
    for _ in range(n_runs):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)

    t = np.array(times)
    per_image = t / batch_size
    return {
        "p50_ms": float(np.percentile(per_image, 50) * 1e3),
        "p95_ms": float(np.percentile(per_image, 95) * 1e3),
        "mean_ms": float(per_image.mean() * 1e3),
        "throughput_ips": float(batch_size / t.mean()),
        "batch_size": batch_size,
        "n_runs": n_runs,
    }
