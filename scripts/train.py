"""Train the VP-SDE score network on one MVTec category (local CLI entrypoint).

Example:
    uv run python scripts/train.py data.category=bottle training.max_steps=50

For GPU training on Modal, see modal_app.py and the README.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from anomaly_diffusion.runner import train_from_cfg


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train_from_cfg(cfg)


if __name__ == "__main__":
    main()
