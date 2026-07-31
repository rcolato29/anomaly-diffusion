"""Basic reverse-time integrators.

Euler-Maruyama for the reverse SDE, and the deterministic probability-flow ODE variant
(probability_flow=True). reverse_integrate runs the backward dynamics from an
arbitrary start state/time, so it serves both unconditional sampling (start from the prior
at t_max) and partial-diffusion reconstruction (start from a noised input at t*). See
ddim_reconstruct for the fast deterministic path with an explicit NFE budget.
"""

from __future__ import annotations

import torch

from anomaly_diffusion.sde.base import SDE, _broadcast


@torch.no_grad()
def reverse_integrate(
    score_fn,
    sde: SDE,
    x_init: torch.Tensor,
    t_start: float,
    n_steps: int,
    probability_flow: bool = False,
) -> torch.Tensor:
    """Integrate the reverse SDE/ODE from t_start down to sde.t_min.

    Returns the final mean state (no noise added on the last step).
    """
    x = x_init
    x_mean = x_init
    ts = torch.linspace(t_start, sde.t_min, n_steps + 1, device=x.device)
    for i in range(n_steps):
        t = ts[i].expand(x.shape[0])
        dt = ts[i + 1] - ts[i]
        drift = sde.reverse_drift(score_fn, x, t, probability_flow=probability_flow)
        x_mean = x + drift * dt
        if probability_flow:
            x = x_mean
        else:
            g = sde.reverse_diffusion(t, probability_flow=False)
            noise = torch.randn_like(x)
            x = x_mean + _broadcast(g, x) * torch.sqrt(-dt) * noise
    return x_mean


@torch.no_grad()
def ddim_reconstruct(
    score_fn,
    sde: SDE,
    x_t: torch.Tensor,
    t_start: float,
    n_steps: int,
) -> torch.Tensor:
    """Deterministic DDIM reverse integration from t_start to t_min (Song et al., 2021).

    DDIM's x0-reparameterisation reaches good reconstruction quality in far fewer function
    evaluations than the raw Euler PF-ODE, so n_steps here is the NFE budget. The score '
    relates to the noise by score = -eps / std, and each step re-predicts x0 then
    re-noises to the next (lower) time.
    """
    x = x_t
    x0_pred = x_t
    ts = torch.linspace(t_start, sde.t_min, n_steps + 1, device=x.device)
    ones = torch.ones_like(x)
    for i in range(n_steps):
        t_cur = ts[i].expand(x.shape[0])
        m_cur, std_cur = sde.marginal_prob(ones, t_cur)  # m_cur = alpha, std_cur = sigma
        std_cur = _broadcast(std_cur, x)
        eps = -score_fn(x, t_cur) * std_cur  # score = -eps/std  ->  eps = -score*std
        x0_pred = (x - std_cur * eps) / m_cur
        m_next, std_next = sde.marginal_prob(ones, ts[i + 1].expand(x.shape[0]))
        x = m_next * x0_pred + _broadcast(std_next, x) * eps
    return x0_pred


@torch.no_grad()
def euler_maruyama_sample(
    score_fn,
    sde: SDE,
    shape: torch.Size,
    device: torch.device,
    n_steps: int = 500,
    probability_flow: bool = False,
) -> torch.Tensor:
    """Unconditional sampling: integrate from the prior at t_max back to t_min."""
    x = sde.prior_sampling(shape, device)
    return reverse_integrate(score_fn, sde, x, sde.t_max, n_steps, probability_flow)
