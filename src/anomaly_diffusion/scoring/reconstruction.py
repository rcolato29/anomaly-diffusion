"""Partial-diffusion reconstruction scoring.

Noise a test image to a moderate level t*, integrate the reverse dynamics back to t=0, and
measure the residual. Normal content reconstructs faithfully, while defects are pulled
toward the normal manifold the model learned from defect-free data, leaving a large
per-pixel residual that serves as the localization heatmap. Residuals are aggregated over
several t* (small t* preserves fine detail, larger t* covers bigger defects).

The reverse pass defaults to the deterministic probability-flow ODE, which gives more
reproducible residuals than the stochastic reverse SDE.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torchvision.transforms.functional import gaussian_blur

from anomaly_diffusion.sampling.sampler import ddim_reconstruct, reverse_integrate
from anomaly_diffusion.sde.base import SDE, _broadcast


@torch.no_grad()
def reconstruct(
    score_fn,
    sde: SDE,
    x0: torch.Tensor,
    t_star: float,
    n_steps: int = 100,
    probability_flow: bool = True,
    solver: str = "pf_ode",
) -> torch.Tensor:
    """Noise x0 to t_star via the perturbation kernel, then denoise back to 0.

    solver="pf_ode" runs the Euler PF-ODE (accurate, many NFE). solver="ddim" is the
    fast deterministic path (few NFE).
    """
    t = torch.full((x0.shape[0],), float(t_star), device=x0.device)
    mean, std = sde.marginal_prob(x0, t)
    x_t = mean + _broadcast(std, x0) * torch.randn_like(x0)
    if solver == "ddim":
        return ddim_reconstruct(score_fn, sde, x_t, float(t_star), n_steps)
    return reverse_integrate(score_fn, sde, x_t, float(t_star), n_steps, probability_flow)


@torch.no_grad()
def anomaly_map(
    score_fn,
    sde: SDE,
    x0: torch.Tensor,
    t_stars: Sequence[float] = (0.1, 0.25, 0.4),
    n_steps: int = 100,
    probability_flow: bool = True,
    smooth_sigma: float = 4.0,
    solver: str = "pf_ode",
) -> torch.Tensor:
    """Per-pixel anomaly heatmap of shape (B, 1, H, W), aggregated over t_stars.

    The residual is the per-pixel L1 distance averaged over channels. A Gaussian blur
    suppresses high-frequency reconstruction noise so the map reflects defect regions.
    """
    maps = []
    for t_star in t_stars:
        x_hat = reconstruct(score_fn, sde, x0, t_star, n_steps, probability_flow, solver=solver)
        residual = (x0 - x_hat).abs().mean(dim=1, keepdim=True)
        maps.append(residual)
    amap = torch.stack(maps, dim=0).mean(dim=0)

    if smooth_sigma and smooth_sigma > 0:
        k = int(2 * round(3 * smooth_sigma) + 1)  # odd kernel covering ~3 sigma
        amap = gaussian_blur(amap, kernel_size=[k, k], sigma=[smooth_sigma, smooth_sigma])
    return amap


def image_score(amap: torch.Tensor, method: str = "max", topk_frac: float = 0.01) -> torch.Tensor:
    """Reduce a (B, 1, H, W) heatmap to a per-image scalar score of shape (B,)."""
    flat = amap.flatten(1)
    if method == "max":
        return flat.max(dim=1).values
    if method == "mean":
        return flat.mean(dim=1)
    if method == "topk":
        k = max(1, int(flat.shape[1] * topk_frac))
        return flat.topk(k, dim=1).values.mean(dim=1)
    raise ValueError(f"Unknown image_score method: {method!r}")
