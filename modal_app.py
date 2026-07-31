"""Modal launcher for GPU training.

Runs the same train_from_cfg path as the local CLI, but inside a Modal GPU container.
Data and checkpoints live on Modal Volumes. The W&B API key is injected from a Modal
Secret so the offline-safe logger flips online automatically. See the README.

One-time setup:
    uv run modal setup # auth this machine
    modal volume create anomaly-mvtec # dataset volume
    modal secret create wandb WANDB_API_KEY=<key> # or create it in the dashboard
    modal volume put anomaly-mvtec <local_mvtec_dir> /mvtec_anomaly_detection

Train / sample / evaluate:
    uv run modal run modal_app.py --category bottle --max-steps 15000
    uv run modal run modal_app.py::sample --checkpoint /outputs/checkpoints/last.pt
    uv run modal run modal_app.py::evaluate --category bottle

Pull results back:
    modal volume get anomaly-outputs /checkpoints ./outputs/checkpoints
    modal volume get anomaly-outputs /eval ./outputs/eval
"""

from __future__ import annotations

import os

import modal

APP_NAME = "anomaly-diffusion"
GPU = os.environ.get("MODAL_GPU", "A100")
DATA_ROOT = "/data/mvtec_anomaly_detection"
CKPT_DIR = "/outputs/checkpoints"
RESULTS_DIR = "/outputs/results"

app = modal.App(APP_NAME)

_runtime = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "torchvision>=0.19",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "diffusers>=0.30",
        "hydra-core>=1.3",
        "wandb>=0.17",
        "numpy>=1.26",
        "pillow>=10.0",
        "tqdm>=4.66",
        "scikit-learn>=1.4",
        "torchdiffeq>=0.2.4",
        "matplotlib>=3.8",
    )
)


def _with_local(img: modal.Image) -> modal.Image:
    """Attach the project source + Hydra configs as the final image layers (must come last)."""
    return img.add_local_dir("src/anomaly_diffusion", "/root/anomaly_diffusion").add_local_dir(
        "configs", "/root/configs"
    )


image = _with_local(_runtime)

anomalib_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "libgl1")
    .pip_install(
        "anomalib[full]==1.1.1",
        "numpy<2",
        "scikit-learn",
        "scikit-image",
        "rich<14",
        "matplotlib<3.8",
    )
)

# Gradio layered on the base (only the demo function needs it), with local files attached last.
demo_image = _with_local(_runtime.pip_install("gradio>=4.0"))

data_volume = modal.Volume.from_name("anomaly-mvtec", create_if_missing=True)
outputs_volume = modal.Volume.from_name("anomaly-outputs", create_if_missing=True)
VOLUMES = {"/data": data_volume, "/outputs": outputs_volume}
SECRETS = [modal.Secret.from_name("wandb")]


def _compose(overrides: list[str]):
    """Build a Hydra config from CLI-style overrides inside the container."""
    import hydra

    with hydra.initialize_config_dir(config_dir="/root/configs", version_base=None):
        return hydra.compose(config_name="config", overrides=overrides)


@app.function(
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=8 * 60 * 60,
)
def train(overrides: list[str]) -> None:
    from anomaly_diffusion.runner import train_from_cfg

    cfg = _compose(overrides)
    train_from_cfg(cfg, on_checkpoint=outputs_volume.commit)


@app.function(
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=60 * 60,
)
def sample(
    checkpoint: str = f"{CKPT_DIR}/last.pt",
    image_size: int = 256,
    n_images: int = 16,
    n_steps: int = 500,
) -> None:
    """Generate samples from a checkpoint to inspect the learned manifold.

    Directly runnable: modal run modal_app.py::sample --checkpoint <path>.
    """
    import torch
    from torchvision.utils import save_image

    from anomaly_diffusion.build import build_model, build_sde
    from anomaly_diffusion.sampling.sampler import euler_maruyama_sample
    from anomaly_diffusion.utils.device import resolve_device
    from anomaly_diffusion.utils.seed import seed_everything

    cfg = _compose([f"data.image_size={image_size}"])
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    sde = build_sde(cfg.sde)
    model = build_model(sde, cfg.model, cfg.data).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()

    shape = torch.Size([n_images, 3, image_size, image_size])
    samples = euler_maruyama_sample(model, sde, shape, device, n_steps=n_steps)

    out = "/outputs/samples.png"
    save_image((samples.clamp(-1, 1) + 1) / 2, out, nrow=4)
    outputs_volume.commit()
    print(f"Saved samples to {out} (in the anomaly-outputs volume)")


@app.function(
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=2 * 60 * 60,
)
def evaluate(
    checkpoint: str = "",
    category: str = "bottle",
    image_size: int = 256,
) -> dict:
    """Reconstruction AUROC + AU-PRO + EVT threshold + heatmaps on an MVTec test split.

    Directly runnable: modal run modal_app.py::evaluate --category bottle.
    Heatmaps and a results JSON land in the anomaly-outputs Volume.
    """
    from anomaly_diffusion.eval.evaluate import evaluate_from_cfg

    checkpoint = checkpoint or f"{CKPT_DIR}/{category}/last.pt"
    cfg = _compose(
        [
            f"data.category={category}",
            f"data.root={DATA_ROOT}",
            f"data.image_size={image_size}",
            "tracking.mode=online",
        ]
    )
    metrics = evaluate_from_cfg(cfg, checkpoint, out_dir="/outputs")
    outputs_volume.commit()
    return metrics


@app.function(
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=3 * 60 * 60,
)
def reliability(
    checkpoint: str = "",
    category: str = "bottle",
    image_size: int = 256,
) -> dict:
    """Reliability study: reconstruction vs score-norm vs likelihood vs typicality.

    Directly runnable: modal run modal_app.py::reliability --category bottle.
    Writes the bits/dim histogram to the anomaly-outputs Volume under /reliability.
    """
    from anomaly_diffusion.eval.reliability import reliability_study

    checkpoint = checkpoint or f"{CKPT_DIR}/{category}/last.pt"
    cfg = _compose(
        [
            f"data.category={category}",
            f"data.root={DATA_ROOT}",
            f"data.image_size={image_size}",
            "tracking.mode=online",
        ]
    )
    metrics = reliability_study(cfg, checkpoint, out_dir="/outputs/reliability")
    outputs_volume.commit()
    return metrics


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=3 * 60 * 60)
def baseline_cae(category: str = "bottle", image_size: int = 256, max_steps: int = 5000) -> dict:
    """CAE baseline floor. modal run modal_app.py::baseline_cae."""
    from anomaly_diffusion.baselines.cae import run_cae_baseline, save_metrics
    from anomaly_diffusion.utils.device import resolve_device

    cfg = _compose(
        [f"data.category={category}", f"data.root={DATA_ROOT}", f"data.image_size={image_size}"]
    )
    metrics = run_cae_baseline(cfg, resolve_device(cfg.device), max_steps=max_steps)
    metrics |= {"category": category, "method": "cae"}
    save_metrics(metrics, f"{RESULTS_DIR}/{category}_cae.json")
    outputs_volume.commit()
    print(metrics)
    return metrics


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=2 * 60 * 60)
def latency(
    category: str = "bottle",
    checkpoint: str = "",
    image_size: int = 256,
    nfe_list: str = "5,10,20,50,100",
) -> dict:
    """DDIM NFE-vs-quality curve + p50/p95 latency on the A100.

    modal run modal_app.py::latency --category bottle. Writes the curve JSON + plot to
    the anomaly-outputs Volume under /latency.
    """
    from anomaly_diffusion.eval.latency import nfe_quality_curve

    checkpoint = checkpoint or f"{CKPT_DIR}/{category}/last.pt"
    cfg = _compose(
        [f"data.category={category}", f"data.root={DATA_ROOT}", f"data.image_size={image_size}"]
    )
    nfes = [int(n) for n in nfe_list.split(",")]
    result = nfe_quality_curve(cfg, checkpoint, nfes, GPU, out_dir="/outputs/latency")
    outputs_volume.commit()
    return result


@app.function(image=anomalib_image, gpu=GPU, volumes=VOLUMES, timeout=2 * 60 * 60)
def baseline_anomalib(category: str = "bottle", model_name: str = "patchcore") -> dict:
    """Imported PatchCore/PaDiM baseline via anomalib, isolated image.

    modal run modal_app.py::baseline_anomalib --category bottle --model-name patchcore.
    """
    import json
    from pathlib import Path

    from anomalib.data import MVTec
    from anomalib.engine import Engine
    from anomalib.models import Padim, Patchcore

    datamodule = MVTec(root=DATA_ROOT, category=category, image_size=(256, 256))
    model = {"patchcore": Patchcore, "padim": Padim}[model_name]()
    engine = Engine(
        accelerator="gpu",
        devices=1,
        image_metrics=["AUROC"],
        pixel_metrics=["AUROC", "AUPRO"],  # add AU-PRO so baselines match our localization metric
    )
    engine.fit(model=model, datamodule=datamodule)
    test = engine.test(model=model, datamodule=datamodule)

    # Map anomalib's metric keys to this project's canonical names so the results table
    # lines the baselines up with the diffusion rows.
    key_map = {"image_AUROC": "image_auroc", "pixel_AUROC": "pixel_auroc", "pixel_AUPRO": "au_pro"}
    raw = dict(test[0]) if test else {}
    metrics = {"category": category, "method": model_name}
    for k, v in raw.items():
        metrics[key_map.get(k, k)] = float(v)
    out = Path(f"{RESULTS_DIR}/{category}_{model_name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    outputs_volume.commit()
    print(metrics)
    return metrics


@app.function(image=image, volumes=VOLUMES, timeout=10 * 60)
def results_table() -> str:
    """Aggregate every /outputs/results/*.json into a Markdown table and save it."""
    from pathlib import Path

    from anomaly_diffusion.eval.results_table import collect_json_results, to_markdown

    rows = collect_json_results(RESULTS_DIR)
    md = to_markdown(rows, ["category", "method", "image_auroc", "pixel_auroc", "au_pro"], "run")
    Path(f"{RESULTS_DIR}/table.md").write_text(md)
    outputs_volume.commit()
    print(md)
    return md


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=60 * 60)
def freeze_mini_eval(
    category: str = "bottle", checkpoint: str = "", k_per_class: int = 8, floor_margin: float = 0.05
) -> dict:
    """Freeze the golden mini-eval AUROC + floor. Pull the JSON and commit it to the repo.

    modal run modal_app.py::freeze_mini_eval --category bottle, then
    modal volume get anomaly-outputs /mini_eval/bottle.json configs/mini_eval/bottle.json.
    """
    import json
    from pathlib import Path

    from anomaly_diffusion.eval.evaluate import _load_model
    from anomaly_diffusion.eval.mini_eval import build_manifest, freeze_golden
    from anomaly_diffusion.utils.device import resolve_device
    from anomaly_diffusion.utils.seed import seed_everything

    checkpoint = checkpoint or f"{CKPT_DIR}/{category}/last.pt"
    cfg = _compose([f"data.category={category}", f"data.root={DATA_ROOT}", "data.image_size=256"])
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    sde, model = _load_model(cfg, checkpoint, device)
    manifest = build_manifest(cfg.data.root, category, k_per_class)
    golden = freeze_golden(model, sde, cfg, manifest, device, floor_margin)

    out = Path(f"/outputs/mini_eval/{category}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, indent=2))
    outputs_volume.commit()
    print(f"golden AUROC {golden['golden_auroc']} | floor {golden['floor']}")
    return golden


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=60 * 60)
def metrics_gate(category: str = "bottle", checkpoint: str = "") -> dict:
    """The regression gate: fail (nonzero exit) if the frozen mini-eval AUROC drops.

    Reads the committed golden file at configs/mini_eval/<category>.json.
    modal run modal_app.py::metrics_gate --category bottle.
    """
    import json
    import sys
    from pathlib import Path

    from anomaly_diffusion.eval.evaluate import _load_model
    from anomaly_diffusion.eval.mini_eval import run_gate
    from anomaly_diffusion.utils.device import resolve_device
    from anomaly_diffusion.utils.seed import seed_everything

    golden_path = Path(f"/root/configs/mini_eval/{category}.json")
    if not golden_path.exists():
        print(f"No frozen golden for {category}. Run freeze_mini_eval first. Skipping.")
        return {"skipped": True}

    golden = json.loads(golden_path.read_text())
    checkpoint = checkpoint or f"{CKPT_DIR}/{category}/last.pt"
    cfg = _compose([f"data.category={category}", f"data.root={DATA_ROOT}", "data.image_size=256"])
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    sde, model = _load_model(cfg, checkpoint, device)

    result = run_gate(model, sde, cfg, device, golden)
    status = "PASS" if result["passed"] else "FAIL"
    print(f"[{status}] image AUROC {result['image_auroc']} vs floor {result['floor']}")
    if not result["passed"]:
        sys.exit(1)  # nonzero exit -> modal run fails -> CI build fails
    return result


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=60 * 60)
def freeze_threshold(
    category: str = "bottle", checkpoint: str = "", far: float = 0.01, n_ref: int = 128
) -> dict:
    """Calibrate the serving EVT threshold + drift reference on normal images.

    modal run modal_app.py::freeze_threshold --category bottle, then
    modal volume get anomaly-outputs /serving/bottle.json configs/serving/bottle.json
    and commit it. Serve with ANOMALY_CALIBRATION pointed at that file.
    """
    import json
    from pathlib import Path

    from anomaly_diffusion.data.mvtec import build_dataloader
    from anomaly_diffusion.serving.calibration import calibrate_evt_threshold
    from anomaly_diffusion.serving.inference import AnomalyDetector

    checkpoint = checkpoint or f"{CKPT_DIR}/{category}/last.pt"
    cfg = _compose([f"data.category={category}", f"data.root={DATA_ROOT}", "data.image_size=256"])
    detector = AnomalyDetector(cfg, checkpoint)
    loader = build_dataloader(
        root=cfg.data.root,
        category=category,
        split="train",
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
    )
    calib = calibrate_evt_threshold(detector, loader, far=far, max_images=n_ref)
    calib["category"] = category

    out = Path(f"/outputs/serving/{category}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calib, indent=2))
    outputs_volume.commit()
    print(f"EVT threshold {calib['threshold']:.4f} @ FAR {far} | ref_mean {calib['ref_mean']:.4f}")
    return calib


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS, timeout=60 * 60)
def drift_demo(category: str = "bottle", checkpoint: str = "", brightness: float = 1.4) -> dict:
    """Demonstrate the model-native drift signal under simulated input drift.

    Reference + a clean live window of normal images -> drift_z near 0. The same window with a
    brightness shift (mimicking a lighting change) -> drift_z spikes, even though the images
    are not defects. modal run modal_app.py::drift_demo --category bottle.
    """
    import torch

    from anomaly_diffusion.data.mvtec import build_dataloader
    from anomaly_diffusion.scoring.reconstruction import anomaly_map, image_score
    from anomaly_diffusion.serving.drift import DriftMonitor
    from anomaly_diffusion.serving.inference import AnomalyDetector

    checkpoint = checkpoint or f"{CKPT_DIR}/{category}/last.pt"
    cfg = _compose([f"data.category={category}", f"data.root={DATA_ROOT}", "data.image_size=256"])
    detector = AnomalyDetector(cfg, checkpoint)

    def scores(x):
        amap = anomaly_map(
            detector.model,
            detector.sde,
            x,
            detector.t_stars,
            detector.n_steps,
            probability_flow=True,
            smooth_sigma=detector.smooth_sigma,
            solver=detector.solver,
        )
        return image_score(amap, detector.image_score_method, detector.topk_frac).cpu().tolist()

    loader = build_dataloader(
        root=cfg.data.root,
        category=category,
        split="train",
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
    )
    batches = [b["image"].to(detector.device) for b in loader]
    ref = scores(torch.cat(batches[:2]))
    live_clean = scores(torch.cat(batches[2:4]))
    live_drift = scores((torch.cat(batches[2:4]) * brightness).clamp(-1, 1))  # lighting shift

    win = min(len(live_clean), len(ref))
    clean_mon = DriftMonitor.from_reference(ref, window=win)
    clean_mon.update(live_clean[:win])
    drift_mon = DriftMonitor.from_reference(ref, window=win)
    drift_mon.update(live_drift[:win])

    result = {"clean": clean_mon.status(), "drifted": drift_mon.status()}
    print(
        f"clean drift_z {result['clean']['drift_z']:.2f} | "
        f"drifted drift_z {result['drifted']['drift_z']:.2f}"
    )
    return result


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def serve():
    """GPU-served FastAPI app with a public URL.

    Dev (ephemeral URL, live-reload): uv run modal serve modal_app.py
    Deploy (persistent URL): uv run modal deploy modal_app.py
    Then use <url>/docs to upload an image and see the prediction + heatmap.
    """
    import json
    from pathlib import Path

    from anomaly_diffusion.serving.app import create_app
    from anomaly_diffusion.serving.drift import DriftMonitor
    from anomaly_diffusion.serving.inference import AnomalyDetector

    category = "bottle"
    checkpoint = f"{CKPT_DIR}/last.pt"  # bottle's original (pre-namespacing) checkpoint
    calib_path = Path(f"/root/configs/serving/{category}.json")  # committed calibration, if any
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}

    overrides = [f"data.category={category}", f"data.root={DATA_ROOT}"]
    if "threshold" in calib:
        overrides.append(f"scoring.threshold={calib['threshold']}")
    cfg = _compose(overrides)

    detector = AnomalyDetector(cfg, checkpoint)
    drift = DriftMonitor(calib["ref_mean"], calib["ref_std"]) if "ref_mean" in calib else None
    return create_app(detector=detector, drift=drift)


@app.function(image=demo_image, gpu=GPU, volumes=VOLUMES, secrets=SECRETS)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def demo():
    """GPU-hosted interactive Gradio demo. Upload an image and see the heatmap overlay.

    Dev: uv run modal serve modal_app.py::demo
    Deploy: uv run modal deploy modal_app.py::demo
    """
    import gradio as gr
    from fastapi import FastAPI

    from anomaly_diffusion.serving.demo import build_demo
    from anomaly_diffusion.serving.inference import AnomalyDetector

    cfg = _compose(["data.category=bottle", f"data.root={DATA_ROOT}"])
    detector = AnomalyDetector(cfg, f"{CKPT_DIR}/last.pt")
    return gr.mount_gradio_app(FastAPI(), build_demo(detector), path="/")


@app.local_entrypoint()
def main(
    category: str = "bottle",
    max_steps: int = 15000,
    resume: bool = False,
    extra: str = "",
) -> None:
    """Launch a training run.

    --resume continues from the last checkpoint on the outputs Volume (extend a run by
    pairing it with a larger --max-steps). --extra passes extra Hydra overrides,
    space-separated.
    """
    ckpt_dir = f"{CKPT_DIR}/{category}"  # per-category so a sweep doesn't overwrite
    overrides = [
        f"data.category={category}",
        f"data.root={DATA_ROOT}",
        f"training.max_steps={max_steps}",
        f"training.ckpt_dir={ckpt_dir}",
        "tracking.mode=online",
    ]
    if resume:
        overrides.append(f"training.resume_from={ckpt_dir}/last.pt")
    if extra:
        overrides += extra.split()
    train.remote(overrides)


@app.local_entrypoint()
def sweep(categories: str = "bottle", max_steps: int = 15000, baselines: bool = True) -> None:
    """Train + evaluate (+ baselines) across categories, then build the results table.

    modal run modal_app.py::sweep --categories bottle,cable,hazelnut.
    """
    for category in [c.strip() for c in categories.split(",") if c.strip()]:
        ckpt_dir = f"{CKPT_DIR}/{category}"
        train.remote(
            [
                f"data.category={category}",
                f"data.root={DATA_ROOT}",
                f"training.max_steps={max_steps}",
                f"training.ckpt_dir={ckpt_dir}",
                "tracking.mode=online",
            ]
        )
        evaluate.remote(category=category)
        if baselines:
            baseline_cae.remote(category=category)
            baseline_anomalib.remote(category=category, model_name="patchcore")
            baseline_anomalib.remote(category=category, model_name="padim")
    results_table.remote()
