"""Config-driven training entrypoint shared by the CLI and the Modal launcher.

scripts/train.py wraps this with @hydra.main for local CLI runs. modal_app.py
composes a config from overrides inside the GPU container and calls the same function.
Keeping the body here means there is exactly one training path.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from anomaly_diffusion.build import build_model, build_sde
from anomaly_diffusion.data.mvtec import build_dataloader
from anomaly_diffusion.tracking.logger import WandbLogger
from anomaly_diffusion.training.trainer import Trainer
from anomaly_diffusion.utils.device import resolve_device
from anomaly_diffusion.utils.seed import seed_everything


def train_from_cfg(cfg: DictConfig, on_checkpoint=None) -> None:
    """Run training from a composed config.

    on_checkpoint is forwarded to Trainer.train (the Modal launcher passes a
    Volume-commit callback). Set training.resume_from to a checkpoint path to continue
    a previous run (restores step, model, EMA, optimizer, and AMP scaler).
    """
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    sde = build_sde(cfg.sde)
    model = build_model(sde, cfg.model, cfg.data)
    loader = build_dataloader(
        root=cfg.data.root,
        category=cfg.data.category,
        split="train",
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        augment=cfg.data.augment,
    )

    logger = WandbLogger(
        project=cfg.tracking.project,
        run_name=cfg.tracking.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.tracking.mode,
    )

    trainer = Trainer(
        model=model,
        sde=sde,
        loader=loader,
        device=device,
        lr=cfg.training.lr,
        ema_decay=cfg.training.ema_decay,
        grad_clip=cfg.training.grad_clip,
        weighting=cfg.training.weighting,
        amp=cfg.training.amp,
        ckpt_dir=cfg.training.ckpt_dir,
        logger=logger,
    )

    resume_from = cfg.training.get("resume_from")
    if resume_from:
        if not Path(resume_from).exists():
            raise FileNotFoundError(f"training.resume_from not found: {resume_from}")
        trainer.load(resume_from)
        print(f"Resumed from {resume_from} at step {trainer.step}")

    try:
        trainer.train(
            max_steps=cfg.training.max_steps,
            log_every=cfg.training.log_every,
            ckpt_every=cfg.training.ckpt_every,
            on_checkpoint=on_checkpoint,
        )
    finally:
        logger.finish()
