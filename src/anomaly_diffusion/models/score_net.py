"""Time-conditioned score network.

The U-Net backbone is imported from diffusers. Only the SDE-convention wiring is
hand-written. UNet2DModel predicts the noise eps. The score relates to it by

    s_theta(x, t) = grad_x log p_t(x) = -eps_theta(x, t) / std(t),

so this wrapper converts the network's noise prediction into a score using the SDE's
perturbation-kernel std. Continuous t in [0, 1] is rescaled into the embedding's
natural range before being passed to the U-Net's timestep embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import UNet2DModel

from anomaly_diffusion.sde.base import SDE, _broadcast


class ScoreNet(nn.Module):
    def __init__(
        self,
        sde: SDE,
        image_size: int = 256,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 2, 4),
        layers_per_block: int = 2,
        time_embed_scale: float = 999.0,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.sde = sde
        self.time_embed_scale = time_embed_scale
        block_out_channels = tuple(base_channels * m for m in channel_mults)
        down = ("DownBlock2D",) * (len(channel_mults) - 1) + ("AttnDownBlock2D",)
        up = ("AttnUpBlock2D",) + ("UpBlock2D",) * (len(channel_mults) - 1)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=in_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down,
            up_block_types=up,
            norm_num_groups=norm_num_groups,
        )

    def predict_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Raw noise prediction eps_theta(x, t)."""
        return self.unet(x, t * self.time_embed_scale).sample

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score estimate s_theta(x, t) = -eps_theta / std(t)."""
        eps = self.predict_noise(x, t)
        _, std = self.sde.marginal_prob(x, t)
        return -eps / _broadcast(std, x)
