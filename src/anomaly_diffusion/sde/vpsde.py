"""Variance-Preserving SDE (the continuous-time limit of DDPM).

    dx = -1/2 beta(t) x dt + sqrt(beta(t)) dW,    beta(t) = beta_min + t (beta_max - beta_min)

Let B(t) = integral_0^t beta(s) ds = beta_min t + 1/2 (beta_max - beta_min) t^2. The
perturbation kernel is closed form:

    p_{0t}(x_t | x0) = N( x0 * exp(-1/2 B(t)),  (1 - exp(-B(t))) I ).

"""

from __future__ import annotations

import torch

from anomaly_diffusion.sde.base import SDE, _broadcast


class VPSDE(SDE):
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        t_min: float = 1e-5,
        t_max: float = 1.0,
    ):
        super().__init__(t_min=t_min, t_max=t_max)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """-1/2 B(t), so that mean = x0 * exp(_log_mean_coeff)."""
        return -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * _broadcast(self.beta(t), x) * x

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.beta(t))

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_mean_coeff = self._log_mean_coeff(t)
        mean = _broadcast(torch.exp(log_mean_coeff), x0) * x0
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device)
