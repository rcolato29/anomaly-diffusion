"""Launch the interactive Gradio demo locally.

Example:
    uv run python scripts/demo.py data.category=bottle \
        checkpoint=outputs/checkpoints/last.pt

Opens a browser: upload an image, see the anomaly score + heatmap overlay. For a GPU-hosted
public URL, use modal serve modal_app.py::demo.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from anomaly_diffusion.serving.demo import build_demo
from anomaly_diffusion.serving.inference import AnomalyDetector


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint", "outputs/checkpoints/last.pt")
    detector = AnomalyDetector(cfg, checkpoint)
    build_demo(detector).launch()


if __name__ == "__main__":
    main()
