"""Read-only readiness checks for remote baseline training."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import stat
import sys
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

from src.data.class_audit import SUPPORTED_IMAGE_EXTENSIONS
from src.data.split_audit import SPLIT_COLUMNS, SPLIT_NAMES, VALID_ENVIRONMENTS
from src.models import MODEL_REGISTRY

IMAGE_SUFFIXES = set(SUPPORTED_IMAGE_EXTENSIONS)
REQUIRED_TRAINING_SCRIPTS = (
    Path("scripts/pipeline/create_splits.py"),
    Path("scripts/pipeline/train_baselines.py"),
)


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _writable_directory(path: Path) -> bool:
    """Check declared mode bits plus OS access without writing a probe file."""
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir():
        return False
    mode = candidate.stat().st_mode
    has_write_bit = bool(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return has_write_bit and os.access(candidate, os.W_OK)


def _load_config(path: Path) -> tuple[dict[str, Any], list[str], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuración inexistente: {path}")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Configuración inválida: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError("Configuración inválida: falta la sección dataset")
    classes = config["dataset"].get("classes")
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(item, str) for item in classes)
    ):
        raise ValueError("Configuración inválida: dataset.classes")
    baseline = config.get("baseline", {})
    seed = (
        baseline.get("seed", config["dataset"].get("seed")) if isinstance(baseline, dict) else None
    )
    if not isinstance(seed, int):
        raise ValueError("Configuración inválida: seed")
    return config, classes, seed


def _gpu_information() -> dict[str, Any]:
    available = torch.cuda.is_available()
    information: dict[str, Any] = {
        "available": available,
        "device_count": torch.cuda.device_count() if available else 0,
        "name": None,
        "vram_free_bytes": None,
        "vram_total_bytes": None,
    }
    if available:
        information["name"] = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        information["vram_free_bytes"] = free_bytes
        information["vram_total_bytes"] = total_bytes
    return information


def _check_scripts(project_root: Path, blockers: list[str]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for relative in REQUIRED_TRAINING_SCRIPTS:
        path = project_root / relative
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            blockers.append(f"Script no importable/sin sintaxis válida: {relative}: {exc}")
            statuses[relative.as_posix()] = "invalid"
        else:
            statuses[relative.as_posix()] = "valid"
    return statuses


def _check_splits(
    splits_dir: Path,
    dataset_root: Path,
    classes: list[str],
    blockers: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"directory": str(splits_dir.resolve()), "rows": {}}
    class_set = set(classes)
    for split in SPLIT_NAMES:
        path = splits_dir / f"{split}.csv"
        if not path.is_file():
            blockers.append(f"Split inexistente: {path}")
            result["rows"][split] = None
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            rows = list(reader)
        result["rows"][split] = len(rows)
        if columns != SPLIT_COLUMNS:
            blockers.append(f"Columnas inválidas en {path}: {columns}")
        for row_number, row in enumerate(rows, start=2):
            relative = Path(row.get("image_path", ""))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not (dataset_root / relative).is_file()
            ):
                blockers.append(f"Ruta inválida/inexistente en {path}:{row_number}")
            if row.get("label") not in class_set:
                blockers.append(f"Etiqueta inválida en {path}:{row_number}")
            if row.get("environment") not in VALID_ENVIRONMENTS:
                blockers.append(f"Entorno inválido en {path}:{row_number}")
    return result


def _check_dataset(dataset_root: Path, blockers: list[str], warnings: list[str]) -> dict[str, Any]:
    clean = dataset_root / "clean"
    result: dict[str, Any] = {
        "root": str(dataset_root.resolve()),
        "clean_directory": str(clean.resolve()),
        "total_images": 0,
        "counts_by_class": {},
        "counts_by_environment": {},
        "samples_verified": 0,
        "ignored_files": [],
    }
    if not clean.is_dir():
        blockers.append(f"Carpeta clean inexistente: {clean}")
        return result

    by_class: Counter[str] = Counter()
    by_environment: Counter[str] = Counter()
    samples: dict[tuple[str, str], Path] = {}
    ignored_files: list[str] = []
    for path in sorted(clean.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(clean)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            ignored_files.append(relative.as_posix())
            continue
        if len(relative.parts) < 3:
            warnings.append(f"Imagen fuera de clean/<clase>/<entorno>: {relative}")
            continue
        class_name, environment = relative.parts[:2]
        by_class[class_name] += 1
        by_environment[environment] += 1
        samples.setdefault((class_name, environment), path)
    result["total_images"] = sum(by_class.values())
    result["counts_by_class"] = dict(sorted(by_class.items()))
    result["counts_by_environment"] = dict(sorted(by_environment.items()))
    result["ignored_files"] = ignored_files
    if ignored_files:
        warnings.append(f"Archivos no soportados ignorados: {len(ignored_files)}")
    if not by_class:
        blockers.append(f"No se encontraron imágenes soportadas en {clean}")
    for path in samples.values():
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, SyntaxError, ValueError) as exc:
            blockers.append(f"Muestra de imagen ilegible: {path}: {exc}")
        else:
            result["samples_verified"] += 1
    return result


def run_training_preflight(
    *,
    project_root: Path,
    splits_dir: Path,
    dataset_root: Path,
    config_path: Path,
    models: list[str],
    device: str,
    check_dataset: bool,
    check_gpu: bool,
    output_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Run readiness checks and persist reports; no model or trainer is executed."""
    blockers: list[str] = []
    warnings: list[str] = []
    try:
        _, classes, seed = _load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        classes, seed = [], 0
        blockers.append(str(exc))

    unknown_models = sorted(set(models) - set(MODEL_REGISTRY.list_names()))
    if unknown_models:
        blockers.append(f"Modelos no registrados: {unknown_models}")

    gpu = _gpu_information() if check_gpu or device == "cuda" else {"checked": False}
    if device == "cuda" and not gpu.get("available", False):
        blockers.append("CUDA fue solicitado, pero no hay una GPU CUDA disponible")
    elif check_gpu and not gpu.get("available", False):
        warnings.append("No hay GPU CUDA disponible; el modo CPU aún puede validarse")

    if not _writable_directory(results_dir):
        blockers.append(f"Directorio de resultados no escribible: {results_dir}")
    if not _writable_directory(output_dir):
        blockers.append(f"Directorio de reporte no escribible: {output_dir}")

    scripts = _check_scripts(project_root, blockers)
    split_status = (
        _check_splits(splits_dir, dataset_root, classes, blockers)
        if classes
        else {
            "directory": str(splits_dir.resolve()),
            "rows": {},
            "skipped": "configuración inválida",
        }
    )
    dataset_status = (
        _check_dataset(dataset_root, blockers, warnings)
        if check_dataset
        else {"checked": False, "root": str(dataset_root.resolve())}
    )
    disk_reference = results_dir if results_dir.exists() else results_dir.parent
    while not disk_reference.exists() and disk_reference != disk_reference.parent:
        disk_reference = disk_reference.parent
    disk = shutil.disk_usage(disk_reference)

    report: dict[str, Any] = {
        "schema_version": 1,
        "ready": not blockers,
        "requested_device": device,
        "requested_models": models,
        "registered_models": MODEL_REGISTRY.list_names(),
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "cuda_runtime": torch.version.cuda,
            "gpu": gpu,
            "disk_free_bytes": disk.free,
        },
        "project": {
            "root": str(project_root.resolve()),
            "config": str(config_path.resolve()),
            "scripts": scripts,
            "results_directory": str(results_dir.resolve()),
            "results_writable": _writable_directory(results_dir),
            "report_directory": str(output_dir.resolve()),
        },
        "splits": split_status,
        "dataset": dataset_status,
        "configured_classes": classes,
        "expected_seed": seed,
        "blockers": blockers,
        "warnings": warnings,
        "safety": {
            "models_built": 0,
            "forward_calls": 0,
            "backward_calls": 0,
            "epochs_run": 0,
            "checkpoints_written": 0,
            "pretrained_weights_downloaded": 0,
        },
    }

    if _writable_directory(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "preflight_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        text_lines = [
            "DoctorMaiz — preflight de entrenamiento remoto",
            f"Estado: {'LISTO' if report['ready'] else 'BLOQUEADO'}",
            f"Dispositivo solicitado: {device}",
            f"Python: {report['environment']['python_executable']}",
            f"PyTorch: {report['environment']['torch']}",
            f"CUDA disponible: {gpu.get('available', 'no consultado')}",
            f"Blockers: {len(blockers)}",
            *[f"  - {item}" for item in blockers],
            f"Advertencias: {len(warnings)}",
            *[f"  - {item}" for item in warnings],
            "Seguridad: 0 modelos, 0 forward, 0 backward, 0 épocas, 0 checkpoints.",
        ]
        (output_dir / "preflight_report.txt").write_text(
            "\n".join(text_lines) + "\n",
            encoding="utf-8",
        )
    return report
