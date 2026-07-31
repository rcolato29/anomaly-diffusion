import torch

from anomaly_diffusion.utils.latency import measure_latency


def test_measure_latency_keys_and_ordering():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        _ = torch.randn(64, 64) @ torch.randn(64, 64)

    stats = measure_latency(fn, torch.device("cpu"), batch_size=4, n_warmup=2, n_runs=5)
    for k in ("p50_ms", "p95_ms", "mean_ms", "throughput_ips"):
        assert k in stats and stats[k] > 0
    assert stats["p95_ms"] >= stats["p50_ms"]
    assert calls["n"] == 2 + 5  # warmup + timed runs
