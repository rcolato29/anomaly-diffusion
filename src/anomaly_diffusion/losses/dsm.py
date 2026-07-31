"""Denoising score matching (Vincent, 2011).

Regress s_theta(x_t, t) onto the closed-form score of the Gaussian perturbation kernel.
With x_t = mean + std * z and z ~ N(0, I), that score is -z / std, so the per-element
residual is s_theta(x_t, t) + z / std, averaged over batch and pixels.

Weightings:
  * "std2" (default): lambda(t) = std^2, equivalent to noise prediction (Song et al., 2021).
  * "likelihood": lambda(t) = g(t)^2 (= beta(t) for VP), the likelihood-weighted bound.
"""

from __future__ import annotations

import torch

from anomaly_diffusion.sde.base import SDE, _broadcast


def dsm_loss(
    score_fn,
    sde: SDE,
    x0: torch.Tensor,
    weighting: str = "std2",
) -> torch.Tensor:
    """Mean denoising-score-matching loss over a batch.

    Args:
        score_fn: callable (x, t) -> score estimate, shape like x.
        sde: the forward SDE providing the perturbation kernel.
        x0: clean data batch, shape (B, C, H, W).
        weighting: "std2" or "likelihood".
    """
    t = sde.sample_t(x0.shape[0], x0.device)
    xt, z, std = sde.perturb(x0, t)
    score = score_fn(xt, t)
    std_b = _broadcast(std, x0)

    if weighting == "std2":
        # || std * s_theta + z ||^2  (per element), averaged.
        residual = std_b * score + z
    elif weighting == "likelihood":
        # lambda(t) = g(t)^2: weight the raw score residual (-z/std) by g^2.
        g2 = _broadcast(sde.diffusion(t) ** 2, x0)
        residual = torch.sqrt(g2) * (score + z / std_b)
    else:
        raise ValueError(f"Unknown weighting: {weighting!r}")

    return residual.pow(2).mean()
