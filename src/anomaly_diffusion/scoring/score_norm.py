"""Score-norm anomaly scoring.

At low noise, the magnitude ||s_theta(x, t)|| measures how hard the model pushes a point back
toward the learned manifold, and is large for off-manifold (anomalous) inputs. It needs a
single forward pass per image, unlike the multi-step reconstruction and likelihood scores.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from anomaly_diffusion.sde.base import SDE


@torch.no_grad()
def score_norm(
    score_fn, sde: SDE, x0: torch.Tensor, t_levels: Sequence[float] = (0.01,)
) -> torch.Tensor:
    """Per-image score L2 norm, averaged over the given low-noise levels. Shape (B,)."""
    norms = []
    for t in t_levels:
        t_vec = torch.full((x0.shape[0],), float(t), device=x0.device)
        s = score_fn(x0, t_vec)
        norms.append(s.flatten(1).norm(dim=1))
    return torch.stack(norms, dim=0).mean(dim=0)
