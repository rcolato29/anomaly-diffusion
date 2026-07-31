"""Interactive Gradio demo: upload an image -> anomaly heatmap.

Wraps the same AnomalyDetector the API uses, so the demo and the service share one code
path. The prediction function is factored out (make_predict_fn) so it can be unit-tested
without importing Gradio.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def _overlay(image: Image.Image, heatmap_b64: str, alpha: float = 0.5) -> Image.Image:
    """Blend a turbo-colormapped anomaly heatmap over the input image."""
    from matplotlib import colormaps

    hm = Image.open(io.BytesIO(base64.b64decode(heatmap_b64))).convert("L")
    base = image.convert("RGB").resize(hm.size)
    h = np.asarray(hm, dtype=float) / 255.0
    colored = Image.fromarray((colormaps["turbo"](h)[:, :, :3] * 255).astype("uint8"))
    return Image.blend(base, colored, alpha)


def make_predict_fn(detector):
    """Return the (image) -> (overlay, verdict-text) function used by the demo."""

    def predict(image: Image.Image | None):
        if image is None:
            return None, "Upload an image."
        res = detector.score_image(image)
        overlay = _overlay(image, res["heatmap_png_b64"])
        verdict = f"anomaly score: {res['score']:.4f}"
        if res["is_anomaly"] is not None:
            verdict += f"  ->  {'ANOMALY' if res['is_anomaly'] else 'normal'}"
        return overlay, verdict

    return predict


def build_demo(detector):
    """Build the Gradio interface around a loaded AnomalyDetector."""
    import gradio as gr

    return gr.Interface(
        fn=make_predict_fn(detector),
        inputs=gr.Image(type="pil", label="Input image"),
        outputs=[
            gr.Image(type="pil", label="Anomaly heatmap (turbo overlay)"),
            gr.Textbox(label="Result"),
        ],
        title="anomaly-diffusion",
        description=(
            "Score-based diffusion anomaly detection. Upload a product image, then the model "
            "reconstructs it toward the learned 'normal' manifold and highlights the residual "
            "(defects). Higher score = more anomalous."
        ),
        allow_flagging="never",
    )
