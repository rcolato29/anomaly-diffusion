"""Exact log-likelihood via the probability-flow ODE.

The PF-ODE drift f_tilde = f(x,t) - 1/2 g(t)^2 s_theta(x,t) is a deterministic, invertible
map sharing the marginals {p_t} with the SDE, so the model is a continuous normalizing flow.
Integrating the change-of-variables rule from data (t_min) to prior (t_max) gives

    log p_0(x_0) = log p_prior(x_T) + integral_{t_min}^{t_max} tr(d f_tilde / dx) dt.

The trace is estimated with the Hutchinson estimator tr(J) approx E_eps[eps^T J eps]
(one fixed eps per trajectory), using the vector-Jacobian product from autograd. An exact
mode (loop over dims) is available for testing. The ODE solver is torchdiffeq.
"""

from __future__ import annotations

import math

import torch
from torchdiffeq import odeint

from anomaly_diffusion.sde.base import SDE, _broadcast


def _pf_ode_drift(score_fn, sde: SDE, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    g2 = _broadcast(sde.diffusion(t) ** 2, x)
    return sde.drift(x, t) - 0.5 * g2 * score_fn(x, t)


def _hutchinson_div(drift: torch.Tensor, x: torch.Tensor, eps: list[torch.Tensor]) -> torch.Tensor:
    div = torch.zeros(x.shape[0], device=x.device)
    for e in eps:
        vjp = torch.autograd.grad(drift, x, grad_outputs=e, retain_graph=True)[0]
        div = div + (vjp * e).flatten(1).sum(dim=1)
    return div / len(eps)


def _exact_div(drift: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Exact divergence by summing e_i^T J e_i over all dims. Testing only (O(D) vjps)."""
    div = torch.zeros(x.shape[0], device=x.device)
    flat_dim = x[0].numel()
    for i in range(flat_dim):
        e = torch.zeros_like(x).flatten(1)
        e[:, i] = 1.0
        e = e.view_as(x)
        vjp = torch.autograd.grad(drift, x, grad_outputs=e, retain_graph=True)[0]
        div = div + (vjp * e).flatten(1).sum(dim=1)
    return div


def log_likelihood(
    score_fn,
    sde: SDE,
    x0: torch.Tensor,
    n_steps: int = 50,
    method: str = "rk4",
    hutchinson_samples: int = 1,
    eps_type: str = "rademacher",
    exact: bool = False,
    t0: float | None = None,
) -> torch.Tensor:
    """Return log p_0(x0) in nats, shape (B,).

    method is any torchdiffeq solver. For fixed-step solvers (rk4/euler), the
    step is (t_max - t0) / n_steps. exact=True uses the O(D) exact divergence
    (small inputs / tests only).
    """
    device = x0.device
    batch = x0.shape[0]
    t0 = sde.t_min if t0 is None else t0
    t1 = sde.t_max

    eps: list[torch.Tensor] = []
    if not exact:
        for _ in range(hutchinson_samples):
            if eps_type == "rademacher":
                e = torch.randint(0, 2, x0.shape, device=device, dtype=x0.dtype) * 2 - 1
            else:
                e = torch.randn_like(x0)
            eps.append(e)

    def ode_func(t, state):
        x = state[0].detach().requires_grad_(True)
        with torch.enable_grad():
            t_vec = torch.full((batch,), float(t), device=device)
            drift = _pf_ode_drift(score_fn, sde, x, t_vec)
            div = _exact_div(drift, x) if exact else _hutchinson_div(drift, x, eps)
        return (drift.detach(), div.detach())

    state0 = (x0, torch.zeros(batch, device=device))
    t_eval = torch.tensor([t0, t1], device=device)
    if method in ("rk4", "euler"):
        out = odeint(
            ode_func, state0, t_eval, method=method, options={"step_size": (t1 - t0) / n_steps}
        )
    else:
        out = odeint(ode_func, state0, t_eval, method=method, rtol=1e-4, atol=1e-4)

    x_T, delta = out[0][-1], out[1][-1]
    dim = x0[0].numel()
    prior_logp = -0.5 * (x_T.flatten(1) ** 2).sum(dim=1) - 0.5 * dim * math.log(2 * math.pi)
    return prior_logp + delta


def bits_per_dim(logp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Convert nats log-likelihood to bits/dim (lower likelihood -> higher bits/dim)."""
    dim = x[0].numel()
    return -logp / (dim * math.log(2))
