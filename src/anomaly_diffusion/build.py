"""Construct core objects from a (Hydra) config mapping.

Kept separate from the entrypoint scripts so it can be unit-tested without Hydra.
"""

from __future__ import annotations

from typing import Any

from anomaly_diffusion.models.score_net import ScoreNet
from anomaly_diffusion.sde.base import SDE
from anomaly_diffusion.sde.vpsde import VPSDE


def build_sde(cfg: Any) -> SDE:
    name = cfg["name"]
    if name == "vp":
        return VPSDE(
            beta_min=cfg["beta_min"],
            beta_max=cfg["beta_max"],
            t_min=cfg["t_min"],
            t_max=cfg["t_max"],
        )
    raise NotImplementedError(f"SDE {name!r} is not implemented. Only 'vp' is available.")


def build_model(sde: SDE, model_cfg: Any, data_cfg: Any) -> ScoreNet:
    return ScoreNet(
        sde=sde,
        image_size=data_cfg["image_size"],
        in_channels=3,
        base_channels=model_cfg["base_channels"],
        channel_mults=tuple(model_cfg["channel_mults"]),
        layers_per_block=model_cfg["layers_per_block"],
        time_embed_scale=model_cfg["time_embed_scale"],
        norm_num_groups=model_cfg.get("norm_num_groups", 32),
    )
