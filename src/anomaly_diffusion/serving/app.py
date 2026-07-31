"""FastAPI service.

Health + readiness probes, input validation, structured logging, and a batch inference
path. The model is loaded once at startup. An EVT-calibrated threshold (loaded from a
committed calibration file) turns /predict into a decision, and the same calibration
seeds a model-native drift monitor. A capped /metrics endpoint exposes ML-meaningful
metrics only. Build the app with create_app so tests can inject a detector.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError

from anomaly_diffusion.serving import metrics as m
from anomaly_diffusion.serving.drift import DriftMonitor
from anomaly_diffusion.serving.inference import AnomalyDetector
from anomaly_diffusion.serving.schemas import HealthResponse, PredictResponse, ReadyResponse

log = structlog.get_logger()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB per image
MAX_BATCH = 16


def _load_from_env() -> tuple[AnomalyDetector, DriftMonitor | None]:
    """Compose the config + load the detector, linking the EVT threshold + drift reference
    from an optional calibration file (env ANOMALY_CALIBRATION)."""
    import hydra

    config_dir = os.environ.get("ANOMALY_CONFIG_DIR", os.path.abspath("configs"))
    checkpoint = os.environ["MODEL_CHECKPOINT"]

    calib = {}
    calib_path = os.environ.get("ANOMALY_CALIBRATION")
    if calib_path and Path(calib_path).exists():
        calib = json.loads(Path(calib_path).read_text())

    overrides = []
    if "threshold" in calib:
        overrides.append(f"scoring.threshold={calib['threshold']}")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    detector = AnomalyDetector(cfg, checkpoint)
    drift = None
    if "ref_mean" in calib and "ref_std" in calib:
        drift = DriftMonitor(calib["ref_mean"], calib["ref_std"])
    return detector, drift


def create_app(
    detector: AnomalyDetector | None = None, drift: DriftMonitor | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector, app.state.drift = _load_from_env()
        m.ACTIVE_NFE.set(app.state.detector.n_steps)
        log.info("model_loaded", device=str(app.state.detector.device))
        yield

    app = FastAPI(title="anomaly-diffusion", lifespan=None if detector else lifespan)
    if detector is not None:
        app.state.detector = detector
        app.state.drift = drift
        m.ACTIVE_NFE.set(detector.n_steps)

    def get_detector() -> AnomalyDetector:
        det = getattr(app.state, "detector", None)
        if det is None:
            raise HTTPException(status_code=503, detail="model not ready")
        return det

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        det = getattr(app.state, "detector", None)
        return ReadyResponse(ready=det is not None, device=str(det.device) if det else None)

    @app.get("/drift")
    def drift_status() -> dict:
        monitor = getattr(app.state, "drift", None)
        return monitor.status() if monitor else {"drift_z": None, "drifting": None}

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        body, content_type = m.render_latest()
        return Response(content=body, media_type=content_type)

    @app.post("/predict", response_model=PredictResponse)
    async def predict(
        files: list[UploadFile] = File(...),
        detector: AnomalyDetector = Depends(get_detector),
    ) -> PredictResponse:
        if not files or len(files) > MAX_BATCH:
            raise HTTPException(status_code=400, detail=f"send 1..{MAX_BATCH} images")
        images = []
        for f in files:
            raw = await f.read()
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=400, detail=f"{f.filename}: file too large")
            try:
                img = Image.open(BytesIO(raw))
                img.load()
                images.append(img)
            except (UnidentifiedImageError, OSError):
                raise HTTPException(
                    status_code=400, detail=f"{f.filename}: not a valid image"
                ) from None

        t0 = time.perf_counter()
        preds = detector.score_batch(images)
        elapsed = time.perf_counter() - t0

        m.PREDICT_LATENCY.observe(elapsed)
        m.PREDICT_IMAGES.inc(len(images))
        monitor = getattr(app.state, "drift", None)
        if monitor is not None:
            monitor.update(p["score"] for p in preds)
            z = monitor.drift_z()
            if z is not None:
                m.INPUT_DRIFT_Z.set(z)

        log.info("predict", n=len(images), latency_ms=round(elapsed * 1e3, 1))
        return PredictResponse(predictions=preds, latency_ms=elapsed * 1e3)

    return app


app = create_app() if os.environ.get("MODEL_CHECKPOINT") else None
