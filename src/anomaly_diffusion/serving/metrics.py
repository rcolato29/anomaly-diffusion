"""Prometheus instrumentation.

Only ML-meaningful metrics (the latency histogram, images scored, the active NFE budget,
and the model-native drift signal).
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "Per-request /predict latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
PREDICT_IMAGES = Counter("predict_images_total", "Total images scored")
ACTIVE_NFE = Gauge("active_nfe", "DDIM reverse steps per scale (the NFE budget)")
INPUT_DRIFT_Z = Gauge("input_drift_z", "Live-window input-drift z-score (model-native)")


def render_latest() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
