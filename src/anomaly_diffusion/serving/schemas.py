"""Pydantic response models for the serving API."""

from __future__ import annotations

from pydantic import BaseModel


class Prediction(BaseModel):
    score: float
    is_anomaly: bool | None  # None when no decision threshold is configured
    heatmap_png_b64: str


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    device: str | None = None
