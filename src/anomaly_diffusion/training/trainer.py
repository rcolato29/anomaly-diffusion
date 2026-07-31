"""Training loop for the score network.

A denoising objective with AMP, gradient clipping, EMA, checkpointing and resume. Logging
goes through the offline-safe WandbLogger.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from anomaly_diffusion.losses.dsm import dsm_loss
from anomaly_diffusion.models.score_net import ScoreNet
from anomaly_diffusion.sde.base import SDE
from anomaly_diffusion.training.ema import EMA
from anomaly_diffusion.utils.device import autocast_dtype


class Trainer:
    def __init__(
        self,
        model: ScoreNet,
        sde: SDE,
        loader: DataLoader,
        device: torch.device,
        lr: float = 2e-4,
        ema_decay: float = 0.999,
        grad_clip: float = 1.0,
        weighting: str = "std2",
        amp: bool = True,
        ckpt_dir: str | Path = "outputs/checkpoints",
        logger=None,
    ):
        self.model = model.to(device)
        self.sde = sde
        self.loader = loader
        self.device = device
        self.grad_clip = grad_clip
        self.weighting = weighting
        self.logger = logger

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.ema = EMA(self.model, decay=ema_decay)
        self.amp = amp and device.type == "cuda"
        self.amp_dtype = autocast_dtype(device)
        self.scaler = torch.amp.GradScaler(enabled=self.amp and self.amp_dtype == torch.float16)
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0

    def _train_step(self, x0: torch.Tensor) -> float:
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp):
            loss = dsm_loss(self.model, self.sde, x0, weighting=self.weighting)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.ema.update(self.model)
        return loss.item()

    def train(
        self,
        max_steps: int,
        log_every: int = 50,
        ckpt_every: int = 1000,
        on_checkpoint=None,
    ) -> None:
        """Train until max_steps.

        on_checkpoint is an optional no-arg callback invoked right after each
        checkpoint is written (including the final last.pt). The Modal launcher uses
        it to commit the outputs Volume so checkpoints survive even if the run is later
        interrupted.
        """
        self.model.train()
        data_iter = _cycle(self.loader)
        while self.step < max_steps:
            batch = next(data_iter)
            x0 = batch["image"].to(self.device)
            loss = self._train_step(x0)
            self.step += 1
            if self.step % log_every == 0:
                lr = self.opt.param_groups[0]["lr"]
                if self.logger is not None:
                    self.logger.log_scalars({"loss": loss, "lr": lr}, step=self.step)
                print(f"step {self.step:>7} | loss {loss:.4f}")
            if self.step % ckpt_every == 0:
                self.save(self.ckpt_dir / f"step_{self.step}.pt")
                if on_checkpoint is not None:
                    on_checkpoint()
        self.save(self.ckpt_dir / "last.pt")
        if on_checkpoint is not None:
            on_checkpoint()

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.ema.load_state_dict(ckpt["ema"])
        self.opt.load_state_dict(ckpt["opt"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]


def _cycle(loader: DataLoader):
    while True:
        yield from loader
