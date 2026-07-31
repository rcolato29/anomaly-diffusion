"""NFE-vs-quality curve + latency benchmark (local CLI).

Example:
    uv run python scripts/latency.py data.category=bottle \
        checkpoint=outputs/checkpoints/last.pt

For GPU on Modal (named hardware), see modal run modal_app.py::latency.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from anomaly_diffusion.eval.latency import nfe_quality_curve


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint", "outputs/checkpoints/last.pt")
    nfe_list = list(cfg.get("nfe_list", [5, 10, 20, 50, 100]))
    hardware = cfg.get("hardware", "local")
    nfe_quality_curve(cfg, checkpoint, nfe_list, hardware, out_dir="outputs/latency")


if __name__ == "__main__":
    main()
