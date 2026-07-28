#!/usr/bin/env python3
"""Authorized cloud runner for smoke, full train, resume, validation and test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from src.training.segmentation_preflight import verify_cloud_training_payload

ROOT = Path(__file__).resolve().parents[1]


def project_path(variable: str, default: str) -> Path:
    value = Path(os.getenv(variable, default))
    return value if value.is_absolute() else ROOT / value


DATASET = project_path(
    "LEAF_SEGMENTATION_DATASET",
    "data/leaf_detection/detector_dataset",
)
OUTPUTS = project_path("LEAF_SEGMENTATION_OUTPUT", "outputs/leaf_detection")
CLOUD_DIR = project_path("CLOUD_TRAINING_DIR", "cloud_training")
MODEL_NAME = os.getenv("SEGMENTATION_MODEL", "yolo26n-seg.pt")
DEVICE = os.getenv("SEGMENTATION_DEVICE", "0")
DEVICE_INDEX = int(DEVICE.split(",", maxsplit=1)[0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config inválida: {path}")
    return payload


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def verified_weights() -> Path:
    manifest_path = OUTPUTS / "cloud_preflight" / "weights_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(str(manifest["path"]))
    if not path.is_file() or sha256(path) != manifest["sha256"]:
        raise RuntimeError("Los pesos no coinciden con weights_manifest.json")
    return path


def base_gate() -> None:
    verify_cloud_training_payload(DATASET)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA dejó de estar disponible")
    free = shutil.disk_usage(OUTPUTS if OUTPUTS.exists() else ROOT).free
    if free < 10 * 1024**3:
        raise RuntimeError("Se requieren al menos 10 GiB libres para resultados")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    probe = OUTPUTS / ".persistence_probe"
    probe.write_text("persistent-output-check\n", encoding="utf-8")
    probe.unlink()


def metrics_dict(result: Any) -> dict[str, Any]:
    values = getattr(result, "results_dict", {})
    return {str(key): float(value) for key, value in values.items()}


def to_serializable(value: Any) -> Any:
    """Convert tensors/arrays into JSON-safe values for the run summary."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return str(value)


def resolve_trainer(model: Any, result: Any) -> Any:
    """Prefer the model's trainer; metrics objects usually lack one."""
    return getattr(model, "trainer", None) or getattr(result, "trainer", None)


def train_mode(mode: str, config_path: Path) -> None:
    from ultralytics import YOLO

    base_gate()
    config = load_yaml(config_path)
    config.pop("task", None)
    config["model"] = str(verified_weights())
    config["data"] = str(DATASET / "dataset.yaml")
    config["device"] = DEVICE
    config["project"] = str(OUTPUTS / "segmenter")
    model_path = config.pop("model")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(DEVICE_INDEX)
    summary_path = (
        OUTPUTS / "segmenter" / "smoke_summary.json"
        if mode == "smoke"
        else OUTPUTS / "segmenter" / "training_summary.json"
    )
    summary: dict[str, Any] = {
        "status": "running",
        "mode": mode,
        "config": str(config_path),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprints": verify_cloud_training_payload(DATASET),
    }
    try:
        model = YOLO(model_path)
        result = model.train(**config)
        trainer = resolve_trainer(model, result)
        save_dir = Path(
            str(getattr(trainer, "save_dir", None) or getattr(result, "save_dir", ""))
        )
        checkpoint = save_dir / "weights" / ("last.pt" if mode == "smoke" else "best.pt")
        selected_batch = getattr(getattr(trainer, "args", None), "batch", config["batch"])
        summary.update(
            {
                "status": "passed",
                "duration_seconds": time.monotonic() - started,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(DEVICE_INDEX),
                "metrics": metrics_dict(result),
                "loss": to_serializable(getattr(trainer, "loss_items", None)),
                "selected_batch": selected_batch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
                "errors": [],
            }
        )
        if mode == "smoke":
            final = load_yaml(CLOUD_DIR / "configs" / "train_yolo26n_seg.yaml")
            final["model"] = MODEL_NAME
            final["data"] = str(DATASET / "dataset.yaml")
            final["device"] = DEVICE
            final["project"] = str(OUTPUTS / "segmenter")
            if isinstance(selected_batch, int) and selected_batch > 0:
                final["batch"] = selected_batch
            final_path = OUTPUTS / "segmenter" / "configs" / "train_yolo26n_seg.final.yaml"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text(yaml.safe_dump(final, sort_keys=False), encoding="utf-8")
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "duration_seconds": time.monotonic() - started,
                "errors": [repr(exc)],
            }
        )
        write(summary_path, summary)
        raise
    write(summary_path, summary)


def resume_mode(checkpoint: Path, reason: str) -> None:
    from ultralytics import YOLO

    base_gate()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = YOLO(str(checkpoint))
    epoch = (
        model.ckpt.get("epoch")
        if isinstance(getattr(model, "ckpt", None), dict)
        else None
    )
    manifest = {
        "status": "authorized",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "epoch": epoch,
        "original_configuration": (
            str(OUTPUTS / "segmenter" / "configs" / "train_yolo26n_seg.final.yaml")
        ),
        "utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    write(OUTPUTS / "segmenter" / "resume_manifest.json", manifest)
    model.train(resume=True)


def evaluate_mode(checkpoint: Path, config_path: Path, split: str) -> None:
    from ultralytics import YOLO

    base_gate()
    if split not in {"val", "test"}:
        raise ValueError("Sólo val o test interno")
    config = load_yaml(config_path)
    config.pop("task", None)
    config["data"] = str(DATASET / "dataset.yaml")
    config["device"] = DEVICE
    config["project"] = str(OUTPUTS / "segmenter_evaluation")
    config["split"] = split
    config["name"] = f"yolo26n_seg_{split}"
    model = YOLO(str(checkpoint))
    result = model.val(**config, plots=True, save_json=True)
    sample_dir = DATASET / "images" / split
    samples = [str(path) for path in sorted(sample_dir.iterdir())[:12]]
    model.predict(
        source=samples,
        imgsz=int(config["imgsz"]),
        device=config["device"],
        save=True,
        project=str(OUTPUTS / "segmenter_evaluation"),
        name=f"yolo26n_seg_{split}_predictions",
        verbose=False,
    )
    write(
        OUTPUTS / "segmenter_evaluation" / f"{split}_summary.json",
        {
            "status": "passed",
            "split": split,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "metrics": metrics_dict(result),
            "pilot_used": False,
            "prediction_samples": len(samples),
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("mode", choices=("smoke", "train", "resume", "evaluate"))
    result.add_argument("--config", type=Path)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--split", choices=("val", "test"))
    result.add_argument("--reason", default="manual_authorized_resume")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.mode in {"smoke", "train"}:
        if args.config is None:
            raise SystemExit("--config es obligatorio")
        train_mode(args.mode, args.config)
    elif args.mode == "resume":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint es obligatorio")
        resume_mode(args.checkpoint, args.reason)
    else:
        if args.checkpoint is None or args.config is None or args.split is None:
            raise SystemExit("--checkpoint, --config y --split son obligatorios")
        evaluate_mode(args.checkpoint, args.config, args.split)


if __name__ == "__main__":
    main()
