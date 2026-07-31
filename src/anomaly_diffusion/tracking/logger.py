"""Thin Weights & Biases wrapper with an offline-safe default.

If no API key is present (env WANDB_API_KEY / a prior wandb login) the logger
falls back to WANDB_MODE=offline so runs work with no account. Set mode="online"
explicitly, or mode="disabled" to no-op entirely (used in tests/CI).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def _resolve_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "online":
        return "online"
    return "offline"


class WandbLogger:
    def __init__(
        self,
        project: str = "anomaly-diffusion",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        mode: str = "auto",
    ):
        self.mode = _resolve_mode(mode)
        self._run = None
        if self.mode == "disabled":
            return
        import wandb

        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, config=config or {}, mode=self.mode)

    def log_scalars(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def log_images(self, tag: str, images: np.ndarray, step: int | None = None) -> None:
        """Log a batch of images. images is (N, H, W, C) in [0, 1]."""
        if self._run is None:
            return
        grid = [self._wandb.Image(img) for img in images]
        self._run.log({tag: grid}, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
