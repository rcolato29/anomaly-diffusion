import torch

from anomaly_diffusion.sde.vpsde import VPSDE


def test_beta_schedule_bounds():
    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    assert torch.isclose(sde.beta(torch.tensor(0.0)), torch.tensor(0.1))
    assert torch.isclose(sde.beta(torch.tensor(1.0)), torch.tensor(20.0))


def test_marginal_limits():
    sde = VPSDE()
    x0 = torch.randn(4, 3, 8, 8)
    # t -> 0: mean -> x0, std -> 0
    mean, std = sde.marginal_prob(x0, torch.full((4,), 1e-5))
    assert torch.allclose(mean, x0, atol=1e-2)
    assert torch.all(std < 1e-2)
    # t -> 1: variance preserved toward unit Gaussian
    _, std1 = sde.marginal_prob(x0, torch.ones(4))
    assert torch.all(std1 > 0.99)


def test_marginal_matches_monte_carlo():
    torch.manual_seed(0)
    sde = VPSDE()
    x0 = torch.ones(1, 1, 1, 1) * 0.7
    t = torch.tensor([0.5])
    mean, std = sde.marginal_prob(x0, t)
    samples = torch.stack([sde.perturb(x0, t)[0] for _ in range(20000)])
    assert torch.allclose(samples.mean(), mean.squeeze(), atol=2e-2)
    assert torch.allclose(samples.std(), std.squeeze(), atol=2e-2)


def test_kernel_score_identity():
    # the DSM target -z/std equals the closed-form kernel score -(xt - mean)/std^2
    sde = VPSDE()
    x0 = torch.randn(2, 3, 4, 4)
    t = torch.full((2,), 0.3)
    xt, z, std = sde.perturb(x0, t)
    mean, _ = sde.marginal_prob(x0, t)
    std_b = std.reshape(-1, 1, 1, 1)
    kernel_score = -(xt - mean) / std_b**2
    assert torch.allclose(kernel_score, -z / std_b, atol=1e-5)


def test_probability_flow_halves_score_term():
    sde = VPSDE()
    x = torch.randn(2, 3, 4, 4)
    t = torch.full((2,), 0.4)
    const = torch.ones_like(x)

    def score_fn(x_, t_):
        return const

    sde_drift = sde.reverse_drift(score_fn, x, t, probability_flow=False)
    ode_drift = sde.reverse_drift(score_fn, x, t, probability_flow=True)
    f = sde.drift(x, t)
    # SDE subtracts g^2 * score, ODE subtracts half that, so ode - sde = +0.5 g^2 score
    g2 = sde.diffusion(t).reshape(-1, 1, 1, 1) ** 2
    assert torch.allclose(ode_drift - sde_drift, 0.5 * g2 * const, atol=1e-5)
    assert torch.allclose(sde_drift, f - g2 * const, atol=1e-5)
