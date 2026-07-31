"""Exponential moving average of model parameters.

Sampling/scoring use the EMA weights, which are markedly more stable than the raw
training weights for diffusion models.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers(), strict=True):
            ema_b.copy_(b)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)
