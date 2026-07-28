#!/usr/bin/env python3
"""Authorized cloud runner for smoke, full train, resume, validation and test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import shutil
import tempfile
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
    serialized = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_yaml_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create a frozen YAML without ever replacing an earlier decision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"No se sobrescribe configuración congelada: {path}") from exc


def verified_weights() -> Path:
    manifest_path = OUTPUTS / "cloud_preflight" / "weights_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(str(manifest["path"]))
    if not path.is_file() or sha256(path) != manifest["sha256"]:
        raise RuntimeError("Los pesos no coinciden con weights_manifest.json")
    return path


def base_gate() -> dict[str, Any]:
    payload = verify_cloud_training_payload(DATASET)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA dejó de estar disponible")
    free = shutil.disk_usage(OUTPUTS if OUTPUTS.exists() else ROOT).free
    if free < 10 * 1024**3:
        raise RuntimeError("Se requieren al menos 10 GiB libres para resultados")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    probe = OUTPUTS / ".persistence_probe"
    probe.write_text("persistent-output-check\n", encoding="utf-8")
    probe.unlink()
    return payload


def timestamped_manifest_path(directory: Path, stamp: datetime | None = None) -> Path:
    """Manifest path that never collides with a previous resume record."""
    moment = stamp if stamp is not None else datetime.now(timezone.utc)
    return directory / f"resume_manifest_{moment.strftime('%Y%m%dT%H%M%S%fZ')}.json"


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
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return str(value)


def resolve_trainer(model: Any, result: Any) -> Any:
    """Prefer the model's trainer; metrics objects usually lack one."""
    return getattr(model, "trainer", None) or getattr(result, "trainer", None)


def numeric_values(value: Any) -> list[float]:
    """Flatten numeric tensor/list/mapping values while rejecting booleans."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, numbers.Real):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in numeric_values(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in numeric_values(item)]
    return []


def require_finite_numeric(field: str, value: Any) -> list[float]:
    values = numeric_values(value)
    if not values:
        raise RuntimeError(f"{field} no contiene valores numéricos")
    if not all(math.isfinite(item) for item in values):
        raise RuntimeError(f"{field} contiene NaN o infinito: {values}")
    return values


def selected_positive_batch(trainer: Any) -> int:
    candidates = (
        getattr(trainer, "batch_size", None),
        getattr(getattr(trainer, "args", None), "batch", None),
    )
    for value in candidates:
        if (
            isinstance(value, numbers.Integral)
            and not isinstance(value, bool)
            and int(value) > 0
        ):
            return int(value)
    raise RuntimeError(f"Batch efectivo inválido: {candidates!r}")


def expected_run_directory(config: dict[str, Any]) -> Path:
    name = config.get("name")
    if not isinstance(name, str) or not name.strip() or Path(name).name != name:
        raise RuntimeError(f"name de run inválido: {name!r}")
    return (OUTPUTS / "segmenter" / name).resolve()


def checkpoint_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Checkpoint obligatorio ausente: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def package_trace() -> dict[str, Any]:
    manifest_path = CLOUD_DIR / "package_manifest.json"
    if not manifest_path.is_file():
        return {"manifest": None}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "manifest": str(manifest_path),
        "package_version": manifest.get("package_version"),
        "git": manifest.get("git"),
        "parent_fingerprint": manifest.get("parent_fingerprint"),
        "split_fingerprints": manifest.get("split_fingerprints"),
    }


def require_confirmation(mode: str) -> None:
    variable = (
        "CONFIRM_SEGMENTATION_SMOKE_TRAINING"
        if mode == "smoke"
        else "CONFIRM_SEGMENTATION_TRAINING"
    )
    if os.getenv(variable) != "1":
        raise RuntimeError(f"{mode} no autorizado: use {variable}=1")


def train_mode(mode: str, config_path: Path) -> None:
    from ultralytics import YOLO

    payload = base_gate()
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    if config.pop("task", None) != "segment":
        raise RuntimeError("La configuración debe declarar task=segment")
    if config.get("seed") != 42 or config.get("deterministic") is not True:
        raise RuntimeError("La configuración debe conservar seed=42 y deterministic=true")
    requested_batch = config.get("batch")
    if mode == "smoke":
        if requested_batch != -1:
            raise RuntimeError("El smoke debe usar batch=-1 para medir AutoBatch")
        if config.get("epochs") != 1:
            raise RuntimeError("El smoke debe ejecutar exactamente una época")
    elif (
        not isinstance(requested_batch, numbers.Integral)
        or isinstance(requested_batch, bool)
        or int(requested_batch) <= 0
    ):
        raise RuntimeError(
            "El entrenamiento completo requiere un batch entero positivo congelado"
        )
    elif (
        not isinstance(config.get("epochs"), numbers.Integral)
        or isinstance(config.get("epochs"), bool)
        or int(config["epochs"]) <= 1
    ):
        raise RuntimeError("El entrenamiento completo requiere más de una época")
    config["model"] = str(verified_weights())
    config["data"] = str(DATASET / "dataset.yaml")
    config["device"] = DEVICE
    config["project"] = str(OUTPUTS / "segmenter")
    config["exist_ok"] = False
    expected_save_dir = expected_run_directory(config)
    if expected_save_dir.exists():
        raise RuntimeError(
            f"El run solicitado ya existe; no se permite baseline2 implícito: "
            f"{expected_save_dir}"
        )
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
        "config_sha256": sha256(config_path),
        "expected_save_dir": str(expected_save_dir),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprints": payload,
        "initial_weights": checkpoint_record(Path(model_path)),
        "source": package_trace(),
    }
    final_path = OUTPUTS / "segmenter" / "configs" / "train_yolo26n_seg.final.yaml"
    active_manifest_path = OUTPUTS / "segmenter" / "active_run_manifest.json"
    if mode == "smoke" and final_path.exists():
        raise RuntimeError(
            f"Ya existe una configuración final; no se sobrescribe: {final_path}"
        )
    if mode == "train":
        if active_manifest_path.exists():
            raise RuntimeError(
                f"Ya existe identidad de run activa: {active_manifest_path}"
            )
        write(
            active_manifest_path,
            {
                "schema_version": 1,
                "status": "running",
                "run_id": config["name"],
                "save_dir": str(expected_save_dir),
                "expected_last_checkpoint": str(
                    expected_save_dir / "weights" / "last.pt"
                ),
                "expected_best_checkpoint": str(
                    expected_save_dir / "weights" / "best.pt"
                ),
                "config": str(config_path),
                "config_sha256": summary["config_sha256"],
                "dataset_fingerprints": payload,
                "initial_weights": summary["initial_weights"],
                "started_utc": summary["started_utc"],
                "source": summary["source"],
            },
        )
    try:
        model = YOLO(model_path)
        result = model.train(**config)
        trainer = resolve_trainer(model, result)
        if trainer is None:
            raise RuntimeError("Ultralytics no expuso el trainer efectivo")
        save_dir = Path(
            str(getattr(trainer, "save_dir", None) or getattr(result, "save_dir", ""))
        ).resolve()
        if save_dir != expected_save_dir:
            raise RuntimeError(
                f"Directorio de run inesperado: {save_dir} != {expected_save_dir}"
            )
        selected_batch = selected_positive_batch(trainer)
        loss = to_serializable(getattr(trainer, "loss_items", None))
        require_finite_numeric("loss", loss)
        metrics = metrics_dict(result)
        require_finite_numeric("metrics", metrics)
        trainer_device = str(getattr(trainer, "device", ""))
        peak_vram = torch.cuda.max_memory_allocated(DEVICE_INDEX)
        if not trainer_device.startswith("cuda") or peak_vram <= 0:
            raise RuntimeError(
                "No se verificó uso efectivo de GPU: "
                f"device={trainer_device!r}, peak_vram={peak_vram}"
            )
        last = checkpoint_record(save_dir / "weights" / "last.pt")
        checkpoints = {"last": last}
        if mode == "train":
            checkpoints["best"] = checkpoint_record(save_dir / "weights" / "best.pt")
        effective_config = {
            "task": "segment",
            "model": model_path,
            **config,
            "batch": selected_batch,
        }
        effective_config_path = save_dir / "doctor_maiz_effective_config.yaml"
        write_yaml_exclusive(effective_config_path, effective_config)
        summary.update(
            {
                "status": "passed",
                "duration_seconds": time.monotonic() - started,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "save_dir": str(save_dir),
                "trainer_device": trainer_device,
                "peak_vram_bytes": peak_vram,
                "metrics": metrics,
                "loss": loss,
                "selected_batch": selected_batch,
                "checkpoints": checkpoints,
                "checkpoint": (
                    checkpoints["last"]["path"]
                    if mode == "smoke"
                    else checkpoints["best"]["path"]
                ),
                "checkpoint_sha256": (
                    checkpoints["last"]["sha256"]
                    if mode == "smoke"
                    else checkpoints["best"]["sha256"]
                ),
                "effective_config": str(effective_config_path),
                "effective_config_sha256": sha256(effective_config_path),
                "errors": [],
            }
        )
        if mode == "smoke":
            final = load_yaml(CLOUD_DIR / "configs" / "train_yolo26n_seg.yaml")
            final["model"] = MODEL_NAME
            final["data"] = str(DATASET / "dataset.yaml")
            final["device"] = DEVICE
            final["project"] = str(OUTPUTS / "segmenter")
            final["batch"] = selected_batch
            write_yaml_exclusive(final_path, final)
            summary["final_config"] = str(final_path)
            summary["final_config_sha256"] = sha256(final_path)
        else:
            active_manifest = json.loads(
                active_manifest_path.read_text(encoding="utf-8")
            )
            active_manifest.update(
                {
                    "status": "completed",
                    "completed_utc": summary["completed_utc"],
                    "selected_batch": selected_batch,
                    "trainer_device": trainer_device,
                    "checkpoints": checkpoints,
                    "effective_config": str(effective_config_path),
                    "effective_config_sha256": summary["effective_config_sha256"],
                }
            )
            write(active_manifest_path, active_manifest)
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "duration_seconds": time.monotonic() - started,
                "errors": [repr(exc)],
            }
        )
        if mode == "train" and active_manifest_path.is_file():
            active_manifest = json.loads(
                active_manifest_path.read_text(encoding="utf-8")
            )
            active_manifest.update(
                {
                    "status": "failed",
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(exc),
                }
            )
            write(active_manifest_path, active_manifest)
        write(summary_path, summary)
        raise
    write(summary_path, summary)


def resume_mode(checkpoint: Path, reason: str) -> None:
    from ultralytics import YOLO

    payload = base_gate()
    active_path = OUTPUTS / "segmenter" / "active_run_manifest.json"
    if not active_path.is_file():
        raise RuntimeError(f"Falta identidad activa: {active_path}")
    active = json.loads(active_path.read_text(encoding="utf-8"))
    if active.get("status") not in {"running", "failed"}:
        raise RuntimeError(f"Run no reanudable: {active.get('status')!r}")
    checkpoint = checkpoint.resolve()
    expected_checkpoint = Path(
        str(active.get("expected_last_checkpoint", ""))
    ).resolve()
    if checkpoint != expected_checkpoint:
        raise RuntimeError(
            f"Checkpoint fuera del run activo: {checkpoint} != {expected_checkpoint}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path = Path(str(active.get("config", "")))
    if (
        not config_path.is_file()
        or sha256(config_path) != active.get("config_sha256")
    ):
        raise RuntimeError("La configuración original del run cambió")
    if active.get("dataset_fingerprints") != payload:
        raise RuntimeError("El dataset ya no coincide con el run interrumpido")
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
        "original_configuration": str(config_path),
        "original_configuration_sha256": active["config_sha256"],
        "run_id": active.get("run_id"),
        "save_dir": active.get("save_dir"),
        "dataset_fingerprints": payload,
        "utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    # Un manifiesto por reanudación (histórico completo) y una copia con el
    # nombre estable que consultan los consumidores existentes.
    history_path = timestamped_manifest_path(OUTPUTS / "segmenter")
    write(history_path, manifest)
    write(OUTPUTS / "segmenter" / "resume_manifest.json", manifest)
    try:
        torch.cuda.reset_peak_memory_stats(DEVICE_INDEX)
        result = model.train(resume=True)
        trainer = resolve_trainer(model, result)
        if trainer is None:
            raise RuntimeError("Ultralytics no expuso el trainer al reanudar")
        save_dir = Path(str(getattr(trainer, "save_dir", ""))).resolve()
        if save_dir != Path(str(active["save_dir"])).resolve():
            raise RuntimeError("La reanudación cambió el directorio del run")
        selected_batch = selected_positive_batch(trainer)
        loss = to_serializable(getattr(trainer, "loss_items", None))
        require_finite_numeric("loss", loss)
        metrics = metrics_dict(result)
        require_finite_numeric("metrics", metrics)
        trainer_device = str(getattr(trainer, "device", ""))
        peak_vram = torch.cuda.max_memory_allocated(DEVICE_INDEX)
        if not trainer_device.startswith("cuda") or peak_vram <= 0:
            raise RuntimeError(
                "No se verificó uso efectivo de GPU durante la reanudación"
            )
        checkpoints = {
            "last": checkpoint_record(save_dir / "weights" / "last.pt"),
            "best": checkpoint_record(save_dir / "weights" / "best.pt"),
        }
        manifest.update(
            {
                "status": "completed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "selected_batch": selected_batch,
                "trainer_device": trainer_device,
                "peak_vram_bytes": peak_vram,
                "metrics": metrics,
                "loss": loss,
                "checkpoints": checkpoints,
            }
        )
        active.update(
            {
                "status": "completed",
                "completed_utc": manifest["completed_utc"],
                "selected_batch": selected_batch,
                "trainer_device": trainer_device,
                "peak_vram_bytes": peak_vram,
                "checkpoints": checkpoints,
                "resumed": True,
            }
        )
        training_summary_path = OUTPUTS / "segmenter" / "training_summary.json"
        training_summary = (
            json.loads(training_summary_path.read_text(encoding="utf-8"))
            if training_summary_path.is_file()
            else {}
        )
        training_summary.update(
            {
                "status": "passed",
                "mode": "train",
                "resumed": True,
                "completed_utc": manifest["completed_utc"],
                "save_dir": str(save_dir),
                "selected_batch": selected_batch,
                "trainer_device": trainer_device,
                "peak_vram_bytes": peak_vram,
                "metrics": metrics,
                "loss": loss,
                "checkpoints": checkpoints,
                "errors": [],
            }
        )
        write(active_path, active)
        write(training_summary_path, training_summary)
        write(history_path, manifest)
        write(OUTPUTS / "segmenter" / "resume_manifest.json", manifest)
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
            }
        )
        write(history_path, manifest)
        write(OUTPUTS / "segmenter" / "resume_manifest.json", manifest)
        raise


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
    if args.mode in {"smoke", "train", "resume"}:
        require_confirmation(args.mode)
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
