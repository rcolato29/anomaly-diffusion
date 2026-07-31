"""Convolutional autoencoder baseline.

A plain CAE trained to reconstruct normal images. The per-pixel reconstruction error is the
anomaly map. It serves as the weak-baseline floor in the results table, below the
feature-based methods (PatchCore, PaDiM).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur

from anomaly_diffusion.data.mvtec import build_dataloader
from anomaly_diffusion.eval.metrics import au_pro, image_auroc, pixel_auroc


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()

        def down(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True))

        def up(i, o):
            return nn.Sequential(
                nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True)
            )

        self.encoder = nn.Sequential(
            down(in_channels, base),
            down(base, base * 2),
            down(base * 2, base * 4),
            down(base * 4, base * 8),
        )
        self.decoder = nn.Sequential(
            up(base * 8, base * 4),
            up(base * 4, base * 2),
            up(base * 2, base),
            nn.ConvTranspose2d(base, in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@torch.no_grad()
def cae_anomaly_map(
    model: ConvAutoencoder, x: torch.Tensor, smooth_sigma: float = 4.0
) -> torch.Tensor:
    residual = (x - model(x)).abs().mean(dim=1, keepdim=True)
    if smooth_sigma > 0:
        k = int(2 * round(3 * smooth_sigma) + 1)
        residual = gaussian_blur(residual, kernel_size=[k, k], sigma=[smooth_sigma, smooth_sigma])
    return residual


def run_cae_baseline(cfg, device, max_steps: int = 5000, lr: float = 1e-3) -> dict:
    """Train the CAE on normal images, then report detection/localization metrics."""
    model = ConvAutoencoder().to(device)
    train_loader = build_dataloader(
        root=cfg.data.root,
        category=cfg.data.category,
        split="train",
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    step = 0
    while step < max_steps:
        for batch in train_loader:
            x = batch["image"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), x)
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                print(f"[cae] step {step} loss {loss.item():.4f}")
            if step >= max_steps:
                break

    model.eval()
    test_loader = build_dataloader(
        root=cfg.data.root,
        category=cfg.data.category,
        split="test",
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )
    labels, scores, maps, masks = [], [], [], []
    for batch in test_loader:
        x = batch["image"].to(device)
        amap = cae_anomaly_map(model, x).cpu()
        labels.extend(batch["label"].tolist())
        scores.extend(amap.flatten(1).max(dim=1).values.tolist())
        maps.append(amap[:, 0].numpy())
        masks.append(batch["mask"][:, 0].numpy())

    labels, maps, masks = np.array(labels), np.concatenate(maps), np.concatenate(masks)
    return {
        "image_auroc": image_auroc(labels, scores),
        "pixel_auroc": pixel_auroc(masks, maps),
        "au_pro": au_pro(maps, masks),
        "n_test": int(labels.size),
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2))
