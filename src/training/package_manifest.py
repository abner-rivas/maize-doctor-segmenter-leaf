"""Deterministic manifest of files required for remote baseline training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_INPUTS: tuple[tuple[Path, str], ...] = (
    (Path("config"), "Configuración reproducible del dataset y pipeline"),
    (Path("src"), "Código fuente del paquete DoctorMaiz"),
    (Path("scripts"), "Entradas de validación, splits y entrenamiento"),
    (Path("outputs/splits/seed_42_baseline"), "Manifiestos baseline oficiales"),
    (Path("pyproject.toml"), "Dependencias y metadatos de instalación"),
    (Path("Makefile"), "Atajos operativos con protección de entrenamiento"),
    (Path(".env.example"), "Plantilla de variables de entorno"),
)
EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__", "aborted_runs"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pth", ".pt", ".ckpt"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(relative: Path) -> bool:
    return (
        bool(set(relative.parts) & EXCLUDED_PARTS) or relative.suffix.lower() in EXCLUDED_SUFFIXES
    )


def build_training_package_manifest(project_root: Path) -> dict[str, Any]:
    """Build a stable manifest without packaging the dataset or run artifacts."""
    missing = [
        relative.as_posix()
        for relative, _ in REQUIRED_INPUTS
        if not (project_root / relative).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Faltan entradas obligatorias: {missing}")

    entries: list[dict[str, Any]] = []
    for required_path, purpose in REQUIRED_INPUTS:
        absolute = project_root / required_path
        candidates = [absolute]
        if absolute.is_dir():
            candidates.extend(sorted(absolute.rglob("*")))
        for path in candidates:
            relative = path.relative_to(project_root)
            if _excluded(relative) or path.is_symlink():
                continue
            if not (path.is_dir() or path.is_file()):
                continue
            entries.append(
                {
                    "path": relative.as_posix(),
                    "type": "directory" if path.is_dir() else "file",
                    "size_bytes": path.stat().st_size if path.is_file() else 0,
                    "sha256": _sha256(path) if path.is_file() else None,
                    "required": True,
                    "purpose": purpose,
                }
            )
    entries.sort(key=lambda item: (str(item["path"]), str(item["type"])))
    return {
        "schema_version": 1,
        "project_root_name": project_root.name,
        "dataset_included": False,
        "archive_created": False,
        "required_inputs": [path.as_posix() for path, _ in REQUIRED_INPUTS],
        "excluded_categories": sorted(EXCLUDED_PARTS | {"checkpoints", "temporary_files"}),
        "entry_count": len(entries),
        "total_file_bytes": sum(int(item["size_bytes"]) for item in entries),
        "entries": entries,
    }


def write_training_package_manifest(project_root: Path, output_path: Path) -> dict[str, Any]:
    manifest = build_training_package_manifest(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
