"""Materialize trustworthy segmentation outputs as a classifier-ready dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASET_DIRECTORY = "classifier_dataset"
DEFAULT_ENVIRONMENT = "lab"
DEFAULT_RELIABILITY_STATUSES = ("reliable",)
SUPPORTED_STORAGE_MODES = ("hardlink", "copy")
EXPORT_COLUMNS: tuple[str, ...] = (
    "image_id",
    "class",
    "environment",
    "filename",
    "dataset_path",
    "source_artifact",
    "source_relative_path",
    "original_filename",
    "original_environment",
    "original_split",
    "reliability_status",
    "original_sha256",
    "duplicate_of",
    "storage_mode",
)
REQUIRED_MANIFEST_COLUMNS = {
    "image_id",
    "class",
    "filename",
    "relative_path",
    "reliability_status",
    "segmented_path",
    "status",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str, *, field: str) -> str:
    component = value.strip()
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
    ):
        raise ValueError(f"{field} no es un nombre de directorio seguro: {value!r}")
    return component


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el manifest de inferencia: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_MANIFEST_COLUMNS - fields)
        if missing:
            raise ValueError(f"Faltan columnas en {path.name}: {', '.join(missing)}")
        return [dict(row) for row in reader]


def _resolve_source_artifact(run_dir: Path, relative: str) -> Path:
    if not relative.strip():
        raise ValueError("el registro no contiene segmented_path")
    candidate = run_dir / relative
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"El artefacto sale del directorio de inferencia: {relative}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"El artefacto segmentado no es un archivo: {candidate}")
    return resolved


def _materialize(source: Path, target: Path, *, storage_mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if storage_mode == "hardlink":
        os.link(source, target)
    elif storage_mode == "copy":
        shutil.copy2(source, target)
    else:  # guarded by the public entry point; retained for direct internal use
        raise ValueError(f"storage_mode no soportado: {storage_mode}")


def _export_row(
    row: Mapping[str, str],
    *,
    run_dir: Path,
    staging_dir: Path,
    environment: str,
    storage_mode: str,
) -> dict[str, str]:
    class_name = _safe_component(str(row.get("class", "")), field="class")
    image_id = _safe_component(str(row.get("image_id", "")), field="image_id")
    source_relative = str(row.get("segmented_path", ""))
    source = _resolve_source_artifact(run_dir, source_relative)
    suffix = source.suffix.lower()
    if not suffix:
        raise ValueError(f"El artefacto segmentado no tiene extensión: {source_relative}")
    filename = f"{image_id}{suffix}"
    dataset_relative = Path(class_name) / environment / filename
    _materialize(source, staging_dir / dataset_relative, storage_mode=storage_mode)
    return {
        "image_id": image_id,
        "class": class_name,
        "environment": environment,
        "filename": filename,
        "dataset_path": dataset_relative.as_posix(),
        "source_artifact": source_relative,
        "source_relative_path": str(row.get("relative_path", "")),
        "original_filename": str(row.get("filename", "")),
        "original_environment": str(row.get("environment", "")),
        "original_split": str(row.get("split", "")),
        "reliability_status": str(row.get("reliability_status", "")),
        "original_sha256": str(row.get("sha256", "")),
        "duplicate_of": str(row.get("duplicate_of", "")),
        "storage_mode": storage_mode,
    }


def _write_manifest(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path, summary: Mapping[str, object]) -> None:
    included_statuses = summary["included_statuses"]
    if not isinstance(included_statuses, list):
        raise TypeError("included_statuses debe ser una lista")
    statuses = ", ".join(f"`{value}`" for value in included_statuses)
    lines = [
        "# Dataset para el clasificador",
        "",
        "Imágenes producidas por la inferencia del segmentador y agrupadas por clase.",
        "La estructura es `<clase>/lab/<image_id>.jpg`, compatible con loaders que",
        "descubren clases desde el primer nivel de directorios.",
        "",
        "## Selección",
        "",
        f"- Estados incluidos: {statuses}",
        f"- Imágenes incluidas: {summary['images_included']}",
        f"- Imágenes excluidas: {summary['images_excluded']}",
        f"- Almacenamiento: `{summary['storage_mode']}`",
        "",
        "`manifest.csv` conserva la procedencia de cada imagen y `summary.json`",
        "documenta los conteos. Las imágenes no confiables permanecen en el run de",
        "inferencia, pero no forman parte de este dataset de entrenamiento.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def export_classifier_dataset(
    inference_run: Path,
    *,
    output_dir: Path | None = None,
    reliability_statuses: Iterable[str] = DEFAULT_RELIABILITY_STATUSES,
    environment: str = DEFAULT_ENVIRONMENT,
    storage_mode: str = "hardlink",
) -> dict[str, object]:
    """Export selected segmented artifacts under ``<class>/lab`` atomically.

    The default accepts only ``reliable`` segmentations. Hard links make the export
    self-contained without duplicating the image bytes on the same filesystem.
    """

    run_dir = inference_run.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No existe el run de inferencia: {run_dir}")
    if storage_mode not in SUPPORTED_STORAGE_MODES:
        raise ValueError(
            f"storage_mode debe ser uno de: {', '.join(SUPPORTED_STORAGE_MODES)}"
        )
    normalized_statuses = tuple(
        sorted({status.strip().lower() for status in reliability_statuses if status.strip()})
    )
    if not normalized_statuses:
        raise ValueError("Debe incluirse al menos un reliability_status")
    environment_name = _safe_component(environment, field="environment")
    output = (
        output_dir.resolve()
        if output_dir is not None
        else run_dir / DEFAULT_DATASET_DIRECTORY
    )
    if output.exists():
        raise FileExistsError(f"No se sobrescribe el dataset existente: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_manifest = run_dir / "manifest.csv"
    rows = _read_manifest(source_manifest)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    exported: list[dict[str, str]] = []
    excluded_by_status: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    try:
        for row in rows:
            status = str(row.get("reliability_status", "")).strip().lower() or "missing"
            if str(row.get("status", "")).strip().lower() != "processed":
                excluded_by_status["error"] += 1
                continue
            if status not in normalized_statuses:
                excluded_by_status[status] += 1
                continue
            exported_row = _export_row(
                row,
                run_dir=run_dir,
                staging_dir=staging,
                environment=environment_name,
                storage_mode=storage_mode,
            )
            exported.append(exported_row)
            class_counts[exported_row["class"]] += 1

        _write_manifest(staging / "manifest.csv", exported)
        summary: dict[str, object] = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_inference_run": str(run_dir),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _sha256_file(source_manifest),
            "layout": "<class>/<environment>/<image_id>.<extension>",
            "environment": environment_name,
            "storage_mode": storage_mode,
            "included_statuses": list(normalized_statuses),
            "images_found": len(rows),
            "images_included": len(exported),
            "images_excluded": len(rows) - len(exported),
            "by_class": dict(sorted(class_counts.items())),
            "excluded_by_status": dict(sorted(excluded_by_status.items())),
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_readme(staging / "README.md", summary)
        staging.rename(output)
        return summary
    except Exception:
        shutil.rmtree(staging)
        raise
