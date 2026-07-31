"""Evaluate reconstruction-based anomaly detection on an MVTec test split (local CLI).

Example:
    uv run python scripts/eval.py data.category=bottle \
        checkpoint=outputs/checkpoints/last.pt

For GPU evaluation on Modal, see modal_app.py (modal run modal_app.py::evaluate).
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from anomaly_diffusion.eval.evaluate import evaluate_from_cfg


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint", "outputs/checkpoints/last.pt")
    evaluate_from_cfg(cfg, checkpoint, out_dir="outputs/eval")


if __name__ == "__main__":
    main()
