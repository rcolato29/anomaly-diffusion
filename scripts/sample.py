"""Sample from a trained checkpoint to confirm the model learned the manifold.

Example:
    uv run python scripts/sample.py checkpoint=outputs/checkpoints/last.pt \
        sampling.n_images=16 sampling.n_steps=500
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torchvision.utils import save_image

from anomaly_diffusion.build import build_model, build_sde
from anomaly_diffusion.sampling.sampler import euler_maruyama_sample
from anomaly_diffusion.utils.device import resolve_device
from anomaly_diffusion.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    sde = build_sde(cfg.sde)
    model = build_model(sde, cfg.model, cfg.data).to(device)

    ckpt_path = cfg.get("checkpoint", "outputs/checkpoints/last.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()

    n_images = int(cfg.get("sampling", {}).get("n_images", 16))
    n_steps = int(cfg.get("sampling", {}).get("n_steps", 500))
    shape = torch.Size([n_images, 3, cfg.data.image_size, cfg.data.image_size])
    samples = euler_maruyama_sample(model, sde, shape, device, n_steps=n_steps)

    out = Path("outputs/samples.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image((samples.clamp(-1, 1) + 1) / 2, out, nrow=4)
    print(f"Saved samples to {out}")


if __name__ == "__main__":
    main()
