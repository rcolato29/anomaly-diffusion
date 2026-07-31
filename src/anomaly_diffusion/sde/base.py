"""Abstract forward/reverse SDE interface.

A concrete SDE specifies the forward drift and diffusion, the Gaussian perturbation-kernel
marginals, and how to sample the prior. The reverse-time SDE (Anderson, 1982) is derived
here from those coefficients and a score function, so subclasses do not reimplement it.
"""

from __future__ import annotations

import abc

import torch

# a score function maps (x, t) -> an estimate of grad_x log p_t(x)
ScoreFn = "callable"


class SDE(abc.ABC):
    """Base class for an Ito SDE dx = f(x, t) dt + g(t) dW on t in [0, 1]."""

    def __init__(self, t_min: float = 1e-5, t_max: float = 1.0):
        # t_min avoids the t -> 0 singularity in the perturbation-kernel std
        self.t_min = t_min
        self.t_max = t_max

    @abc.abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward drift f(x, t)."""

    @abc.abstractmethod
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion coefficient g(t) (scalar per batch element)."""

    @abc.abstractmethod
    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and std of the perturbation kernel p_{0t}(x_t | x0) = N(mean, std^2 I)."""

    @abc.abstractmethod
    def prior_sampling(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Draw a sample from the tractable prior p_1."""

    def perturb(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample x_t ~ p_{0t}(.|x0) and return (x_t, noise z, std).

        The DSM target is -z / std (the closed-form score of the Gaussian kernel).
        """
        mean, std = self.marginal_prob(x0, t)
        z = torch.randn_like(x0)
        xt = mean + _broadcast(std, x0) * z
        return xt, z, std

    def sample_t(self, batch: int, device: torch.device) -> torch.Tensor:
        """Sample t ~ Uniform(t_min, t_max), one per batch element."""
        return torch.rand(batch, device=device) * (self.t_max - self.t_min) + self.t_min

    def reverse_drift(
        self, score_fn, x: torch.Tensor, t: torch.Tensor, probability_flow: bool = False
    ) -> torch.Tensor:
        """Reverse-time drift.

        SDE form: f(x, t) - g(t)^2 * score
        PF-ODE form: f(x, t) - 0.5 * g(t)^2 * score (shares the marginals {p_t})
        """
        g = self.diffusion(t)
        g2 = _broadcast(g * g, x)
        coeff = 0.5 if probability_flow else 1.0
        return self.drift(x, t) - coeff * g2 * score_fn(x, t)

    def reverse_diffusion(self, t: torch.Tensor, probability_flow: bool = False) -> torch.Tensor:
        """Reverse diffusion coefficient (zero for the probability-flow ODE)."""
        if probability_flow:
            return torch.zeros_like(self.diffusion(t))
        return self.diffusion(t)


def _broadcast(coeff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a per-batch scalar to broadcast against x of shape (B, ...)."""
    return coeff.reshape(-1, *([1] * (x.ndim - 1)))
