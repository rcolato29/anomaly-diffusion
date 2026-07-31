"""Run the likelihood-reliability study on an MVTec test split (local CLI).

Example:
    uv run python scripts/reliability.py data.category=bottle \
        checkpoint=outputs/checkpoints/last.pt

For GPU on Modal, see modal run modal_app.py::reliability.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from anomaly_diffusion.eval.reliability import reliability_study


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint", "outputs/checkpoints/last.pt")
    reliability_study(cfg, checkpoint, out_dir="outputs/reliability")


if __name__ == "__main__":
    main()
