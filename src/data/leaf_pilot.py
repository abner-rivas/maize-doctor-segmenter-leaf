"""Reproducible selection and materialization of a manual leaf-ROI pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

SPLIT_REQUIRED_COLUMNS = ("image_path", "label", "environment")
PILOT_COLUMNS = (
    "pilot_id",
    "pilot_image_path",
    "original_image_path",
    "original_filename",
    "image_sha256",
    "label",
    "split",
    "environment",
    "source_dataset",
    "selected_by",
    "annotation_status",
    "copy_mode",
)
SUPPORTED_COPY_MODES = ("copy", "hardlink", "symlink")
SUPPORTED_SELECTION_STRATEGIES = ("balanced", "random")
VALID_SPLITS = ("train", "val", "test")
VALID_ENVIRONMENTS = ("lab", "real")

ANNOTATION_GUIDE = """# Guía de anotación manual de la hoja principal

## Etiqueta

`0 = maize_leaf`

La enfermedad no se anota en esta fase: ya está conservada como etiqueta del clasificador.

## Regla principal

> Marcar solamente la hoja principal que debería analizar el clasificador.

## Cómo elegir la hoja principal

1. Priorizar la hoja que ocupa mayor área.
2. Priorizar la hoja más cercana al centro.
3. Priorizar la hoja con síntomas visibles.
4. Evitar marcar otras hojas del fondo.
5. No marcar lesiones individuales.
6. No marcar suelo, manos, tallos o cielo.

## Casos ambiguos

Usar `ambiguous` cuando existen varias hojas con importancia similar, la hoja principal está
demasiado cortada, no se distingue una hoja completa o no es posible decidir qué hoja analizar.
No elegir automáticamente una de varias hojas equivalentes.

## Casos no aptos

Usar `rejected` cuando no hay una hoja de maíz visible, la imagen está corrupta, la hoja es
extremadamente pequeña, la calidad es insuficiente o el objeto principal no es una hoja.

## Formatos aceptados

- YOLO bbox: `0 center_x center_y width height`, una sola línea por imagen.
- CSV: `pilot_id,x1,y1,x2,y2,status,notes`.

Una etiqueta YOLO vacía o con varias cajas requiere revisión y no se convierte automáticamente
en una anotación válida.
"""

PILOT_README = """# Piloto de regiones de interés

Este directorio es un artefacto derivado y reproducible. Las imágenes originales no se
modifican. `manifests/pilot_manifest.csv` conserva la ruta, clase, split, entorno y SHA-256 de
cada imagen. Coloque las etiquetas YOLO manuales en `labels/` usando `<pilot_id>.txt`, o prepare
un CSV con `pilot_id,x1,y1,x2,y2,status,notes`.

Revise `annotation_guide.md` antes de anotar. Esta fase no entrena ni integra un detector o el
clasificador.
"""


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    """Read a UTF-8 CSV and return rows plus its exact header."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV sin encabezado: {path}")
        return [dict(row) for row in reader], tuple(reader.fieldnames)


def write_csv_rows(path: Path, rows: Iterable[dict[str, object]], columns: Sequence[str]) -> None:
    """Write dictionaries using a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require_columns(columns: Sequence[str], required: Sequence[str], context: str) -> None:
    """Raise a clear error listing required CSV columns that are absent."""
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{context}: faltan columnas obligatorias: {', '.join(missing)}")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_split_name(split_csv: Path, explicit_split: str | None = None) -> str:
    """Infer official split name from train/val/test.csv or use an explicit value."""
    split = explicit_split or split_csv.stem.lower()
    if split not in VALID_SPLITS:
        raise ValueError(
            f"No se puede inferir el split desde {split_csv.name!r}; use --split-name "
            f"con uno de: {', '.join(VALID_SPLITS)}"
        )
    return split


def resolve_source_image(dataset_root: Path, image_path: str) -> Path:
    """Resolve official split paths relative to DATASET_ROOT, preserving absolute inputs."""
    candidate = Path(image_path)
    return candidate if candidate.is_absolute() else dataset_root / candidate


def _priority_scores(priority_manifest: Path | None) -> dict[str, tuple[int, float]]:
    if priority_manifest is None:
        return {}
    rows, columns = read_csv_rows(priority_manifest)
    require_columns(columns, ("image_path",), "priority manifest")
    scores: dict[str, tuple[int, float]] = {}
    compatible_signals = {"correct", "pred_label", "pred_prob"}.intersection(columns)
    if not compatible_signals:
        raise ValueError(
            "priority manifest incompatible: requiere image_path y al menos uno de "
            "correct, pred_label o pred_prob"
        )
    for row in rows:
        path = row.get("image_path", "").strip()
        if not path:
            continue
        is_error = False
        correct = row.get("correct", "").strip().lower()
        if correct in {"false", "0", "no"}:
            is_error = True
        if row.get("label") and row.get("pred_label"):
            is_error = row["label"] != row["pred_label"]
        try:
            probability = float(row.get("pred_prob", "1") or 1)
        except ValueError:
            probability = 1.0
        scores[path] = (0 if is_error else 1, probability)
    return scores


def _deduplicate_rows(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        key = Path(row["image_path"]).as_posix().casefold()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, duplicates


def _ordered_group(
    rows: Sequence[dict[str, str]],
    rng: random.Random,
    priority_scores: dict[str, tuple[int, float]],
) -> list[dict[str, str]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    shuffled.sort(
        key=lambda row: (
            priority_scores.get(row["image_path"], (2, 1.0)),
            0 if row.get("environment") == "real" else 1,
        )
    )
    return shuffled


def select_pilot_rows(
    rows: Sequence[dict[str, str]],
    samples: int,
    seed: int,
    strategy: str,
    *,
    priority_scores: dict[str, tuple[int, float]] | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """Select unique rows reproducibly, balancing classes with round-robin quotas."""
    if samples <= 0:
        raise ValueError("samples debe ser mayor que cero")
    if strategy not in SUPPORTED_SELECTION_STRATEGIES:
        raise ValueError(f"selection_strategy desconocida: {strategy}")
    unique_rows, duplicate_count = _deduplicate_rows(rows)
    warnings: list[str] = []
    if duplicate_count:
        warnings.append(f"Se omitieron {duplicate_count} rutas duplicadas del split")
    requested = min(samples, len(unique_rows))
    if samples > len(unique_rows):
        warnings.append(
            f"Se solicitaron {samples} imágenes, pero sólo hay {len(unique_rows)} disponibles"
        )
    rng = random.Random(seed)
    scores = priority_scores or {}
    if strategy == "random":
        candidates = list(unique_rows)
        rng.shuffle(candidates)
        return candidates[:requested], warnings

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in unique_rows:
        groups[row["label"]].append(row)
    class_names = sorted(groups)
    if not class_names:
        return [], warnings
    ideal = (requested + len(class_names) - 1) // len(class_names)
    for class_name in class_names:
        if len(groups[class_name]) < ideal:
            warnings.append(
                f"Clase {class_name}: {len(groups[class_name])} disponibles, "
                f"menos que el cupo ideal {ideal}; el resto será redistribuido"
            )
        groups[class_name] = _ordered_group(groups[class_name], rng, scores)

    selected: list[dict[str, str]] = []
    position = 0
    while len(selected) < requested:
        added = False
        for class_name in class_names:
            group = groups[class_name]
            if position < len(group) and len(selected) < requested:
                selected.append(group[position])
                added = True
        if not added:
            break
        position += 1
    return selected, warnings


def materialize_file(source: Path, destination: Path, mode: str) -> str:
    """Copy or link one file, failing clearly without silent mode fallback."""
    if mode not in SUPPORTED_COPY_MODES:
        raise ValueError(f"copy_mode desconocido: {mode}")
    if not source.is_file():
        raise FileNotFoundError(f"Imagen de origen inexistente: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"El destino ya existe y no será sobrescrito: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "copy":
            shutil.copy2(source, destination)
        elif mode == "hardlink":
            os.link(source, destination)
        else:
            destination.symlink_to(source.resolve())
    except OSError as exc:
        raise RuntimeError(
            f"No fue posible usar copy_mode={mode!r} para {source} -> {destination}: {exc}. "
            "Seleccione explícitamente otro modo; no se aplicó fallback."
        ) from exc
    return mode


def _selected_by(row: dict[str, str], scores: dict[str, tuple[int, float]], strategy: str) -> str:
    if strategy == "balanced" and row["image_path"] in scores:
        return "priority_balanced"
    return strategy


def build_pilot(
    split_csv: Path,
    dataset_root: Path,
    output_root: Path,
    *,
    samples: int,
    seed: int,
    environments: Sequence[str],
    classes: Sequence[str] | None,
    copy_mode: str,
    selection_strategy: str,
    priority_manifest: Path | None = None,
    split_name: str | None = None,
) -> dict[str, object]:
    """Build the pilot directory and return its reproducibility summary."""
    rows, columns = read_csv_rows(split_csv)
    require_columns(columns, SPLIT_REQUIRED_COLUMNS, "split CSV")
    split = infer_split_name(split_csv, split_name)
    invalid_environments = sorted(set(environments) - set(VALID_ENVIRONMENTS))
    if invalid_environments:
        raise ValueError(f"Entornos desconocidos: {', '.join(invalid_environments)}")
    if copy_mode not in SUPPORTED_COPY_MODES:
        raise ValueError(f"copy_mode desconocido: {copy_mode}")
    if selection_strategy not in SUPPORTED_SELECTION_STRATEGIES:
        raise ValueError(f"selection_strategy desconocida: {selection_strategy}")

    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    if output_root == dataset_root or output_root.is_relative_to(dataset_root):
        raise ValueError("La salida del piloto debe estar fuera de DATASET_ROOT")
    manifest_path = output_root / "manifests" / "pilot_manifest.csv"
    summary_path = output_root / "manifests" / "pilot_summary.json"
    if manifest_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"El piloto ya contiene manifiestos en {output_root}; use otro --output"
        )

    allowed_environments = set(environments)
    allowed_classes = set(classes) if classes else None
    filtered = [
        row
        for row in rows
        if row.get("environment") in allowed_environments
        and (allowed_classes is None or row.get("label") in allowed_classes)
    ]
    if not filtered:
        raise ValueError("Ninguna fila cumple los filtros de clase y entorno")
    scores = _priority_scores(priority_manifest)
    selected, warnings = select_pilot_rows(
        filtered,
        samples,
        seed,
        selection_strategy,
        priority_scores=scores,
    )

    labels_dir = output_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    pilot_rows: list[dict[str, object]] = []
    source_hashes: dict[Path, str] = {}
    for index, row in enumerate(selected, start=1):
        source = resolve_source_image(dataset_root, row["image_path"]).resolve()
        source_hash = sha256_file(source)
        source_hashes[source] = source_hash
        pilot_id = f"image_{index:04d}"
        suffix = source.suffix.lower() or ".jpg"
        pilot_relative = Path("images") / f"{pilot_id}{suffix}"
        destination = output_root / pilot_relative
        actual_mode = materialize_file(source, destination, copy_mode)
        if sha256_file(destination) != source_hash:
            raise RuntimeError(f"Hash distinto después de materializar {destination}")
        pilot_rows.append(
            {
                "pilot_id": pilot_id,
                "pilot_image_path": pilot_relative.as_posix(),
                "original_image_path": row["image_path"],
                "original_filename": source.name,
                "image_sha256": source_hash,
                "label": row["label"],
                "split": split,
                "environment": row["environment"],
                "source_dataset": row.get("source_dataset", "").strip() or "unknown",
                "selected_by": _selected_by(row, scores, selection_strategy),
                "annotation_status": "pending",
                "copy_mode": actual_mode,
            }
        )
    for source, expected_hash in source_hashes.items():
        if sha256_file(source) != expected_hash:
            raise RuntimeError(
                f"La imagen original cambió durante la creación del piloto: {source}"
            )

    write_csv_rows(manifest_path, pilot_rows, PILOT_COLUMNS)
    (output_root / "annotation_guide.md").write_text(ANNOTATION_GUIDE, encoding="utf-8")
    (output_root / "README.md").write_text(PILOT_README, encoding="utf-8")
    summary = {
        "schema_version": 1,
        "split_csv": str(split_csv.resolve()),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "requested_samples": samples,
        "selected_samples": len(pilot_rows),
        "seed": seed,
        "split": split,
        "environments": list(environments),
        "classes": sorted(set(row["label"] for row in pilot_rows)),
        "selection_strategy": selection_strategy,
        "copy_mode": copy_mode,
        "priority_manifest": str(priority_manifest.resolve()) if priority_manifest else None,
        "counts_by_class": dict(sorted(Counter(row["label"] for row in pilot_rows).items())),
        "counts_by_environment": dict(
            sorted(Counter(row["environment"] for row in pilot_rows).items())
        ),
        "warnings": warnings,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
