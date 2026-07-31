# anomaly-diffusion

This repo explores score-based diffusion for one-class visual anomaly detection. Here, a VP-SDE score model is
trained on defect-free images only, then detects and localizes manufacturing defects on
MVTec AD through partial-diffusion reconstruction. It is benchmarked against PatchCore and
PaDiM and served with FastAPI with an decision threshold set via Extreme Value Theory.

The full write-up, derivation walkthroughs, and results can be found here: https://rcolato29.github.io/anomaly-diffusion/

## Setup

```bash
uv sync --extra dev
```

Place the MVTec AD dataset at the `data.root` in `configs/data/mvtec.yaml` (default
`data/mvtec_anomaly_detection`). Configs are composed with [Hydra](https://hydra.cc). One can override
any field on the CLI (e.g. `data.category=cable data.image_size=256`).

## Scripts

Local CLI (runs on CPU, MPS, or a single CUDA GPU):

```bash
# train one category on its defect-free images
uv run python scripts/train.py data.category=bottle training.max_steps=50000

# unconditional samples to sanity-check the learned manifold
uv run python scripts/sample.py checkpoint=outputs/checkpoints/last.pt

# evaluate: image/pixel AUROC, AU-PRO, and heatmaps
uv run python scripts/eval.py data.category=bottle checkpoint=outputs/checkpoints/last.pt

# reliability study: reconstruction vs score-norm vs likelihood vs typicality
uv run python scripts/reliability.py data.category=bottle checkpoint=outputs/checkpoints/last.pt

# NFE-vs-quality and latency curve
uv run python scripts/latency.py data.category=bottle checkpoint=outputs/checkpoints/last.pt
```

GPU training and evaluation is run on [Modal](https://modal.com). The image, GPU, volumes, and
secrets are defined in `modal_app.py`.

```bash
uv run modal setup.     # one-time auth
modal volume put anomaly-mvtec <local_mvtec> /mvtec_anomaly_detection
uv run modal run modal_app.py --category bottle --max-steps 50000
uv run modal run modal_app.py::evaluate --category bottle
```

Other entrypoints: `::reliability`, `::baseline_anomalib`, `::latency`, `::sweep`. W&B logging
is offline-safe. Set `WANDB_API_KEY` to log online.

## Serve

```bash
docker build -t anomaly-diffusion .
docker run -p 8000:8000 -v /path/to/checkpoints:/models anomaly-diffusion

# or run locally without Docker
MODEL_CHECKPOINT=outputs/checkpoints/last.pt \
  uv run uvicorn anomaly_diffusion.serving.app:app --port 8000
```

`POST /predict` returns `{score, is_anomaly, heatmap_png_b64}` per uploaded image. Point
`ANOMALY_CALIBRATION` at a committed `configs/serving/<category>.json` (produced by
`modal run modal_app.py::freeze_threshold`) so the decision uses the calibrated threshold.
`docker compose up --build` also starts Prometheus and a Grafana dashboard, and
`uv run python scripts/demo.py checkpoint=outputs/checkpoints/last.pt` launches a Gradio demo.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest
```
