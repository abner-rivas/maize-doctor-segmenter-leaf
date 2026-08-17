"""Audit YOLO annotations without modifying the source dataset.

The audit scans every annotation file for aggregate metrics and draws a seeded
sample of images. Bounding boxes are supported; polygon-like and malformed
records are reported but never stop the complete run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_SPLITS = ("train", "valid", "val", "test")
PROBLEM_EXAMPLES_PER_TYPE = 10


@dataclass(frozen=True)
class YoloBox:
    """A validated normalized YOLO bounding box."""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class AnnotationResult:
    """Result of parsing one non-empty annotation line."""

    line_number: int
    kind: str
    raw: str
    class_id: int | None = None
    box: YoloBox | None = None
    message: str | None = None


@dataclass(frozen=True)
class LabelFileAudit:
    """Parsed contents and state of one YOLO label file."""

    path: Path
    results: tuple[AnnotationResult, ...]
    empty: bool = False
    read_error: str | None = None


@dataclass(frozen=True)
class ImageRecord:
    """An image discovered in one source split and its paired label, if any."""

    split: str
    path: Path
    relative_path: Path
    label_path: Path | None


@dataclass(frozen=True)
class SplitInventory:
    """Images, labels, and pair information found in one split."""

    split: str
    records: tuple[ImageRecord, ...]
    labels: tuple[Path, ...]
    orphan_labels: tuple[Path, ...]
    duplicate_image_keys: tuple[str, ...]
    duplicate_label_keys: tuple[str, ...]


class ProblemCollector:
    """Retain a bounded, representative set of problems by category."""

    def __init__(self, limit_per_type: int = PROBLEM_EXAMPLES_PER_TYPE) -> None:
        self.limit_per_type = limit_per_type
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[dict[str, object]]] = defaultdict(list)

    def add(
        self,
        problem_type: str,
        *,
        split: str,
        path: str,
        message: str,
        line: int | None = None,
    ) -> None:
        self.counts[problem_type] += 1
        examples = self.examples[problem_type]
        if len(examples) >= self.limit_per_type:
            return
        item: dict[str, object] = {
            "split": split,
            "path": path,
            "message": message,
        }
        if line is not None:
            item["line"] = line
        examples.append(item)


def validate_normalized_bbox(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> list[str]:
    """Return validation errors for normalized YOLO bbox values."""
    values = (center_x, center_y, width, height)
    if not all(math.isfinite(value) for value in values):
        return ["las coordenadas deben ser números finitos"]

    errors: list[str] = []
    if not 0.0 <= center_x <= 1.0:
        errors.append("center_x debe estar entre 0 y 1")
    if not 0.0 <= center_y <= 1.0:
        errors.append("center_y debe estar entre 0 y 1")
    if not 0.0 < width <= 1.0:
        errors.append("width debe ser mayor que 0 y menor o igual que 1")
    if not 0.0 < height <= 1.0:
        errors.append("height debe ser mayor que 0 y menor o igual que 1")
    return errors


def clamp_pixel_box(
    box: tuple[int, int, int, int], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Clamp integer bbox coordinates to valid image pixel indices."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Las dimensiones de imagen deben ser mayores que cero")
    left, top, right, bottom = box
    return (
        min(max(left, 0), image_width - 1),
        min(max(top, 0), image_height - 1),
        min(max(right, 0), image_width - 1),
        min(max(bottom, 0), image_height - 1),
    )


def _raw_pixel_box(
    box: YoloBox, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    return (
        round((box.center_x - box.width / 2.0) * image_width),
        round((box.center_y - box.height / 2.0) * image_height),
        round((box.center_x + box.width / 2.0) * image_width),
        round((box.center_y + box.height / 2.0) * image_height),
    )


def yolo_to_pixel_box(
    box: YoloBox, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Convert a normalized YOLO bbox to clamped pixel coordinates."""
    errors = validate_normalized_bbox(box.center_x, box.center_y, box.width, box.height)
    if errors:
        raise ValueError("; ".join(errors))
    raw_box = _raw_pixel_box(box, image_width, image_height)
    return clamp_pixel_box(raw_box, image_width, image_height)


def looks_like_polygon(parts: Sequence[str]) -> bool:
    """Return whether tokens resemble YOLO segmentation: class plus point pairs."""
    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
        return False
    try:
        class_id = int(parts[0])
        coordinates = [float(value) for value in parts[1:]]
    except ValueError:
        return False
    return class_id >= 0 and all(math.isfinite(value) for value in coordinates)


def parse_annotation_line(line: str, line_number: int = 1) -> AnnotationResult:
    """Classify one YOLO line as bbox, possible polygon, or invalid data."""
    stripped = line.strip()
    parts = stripped.split()

    if looks_like_polygon(parts):
        class_id = int(parts[0])
        coordinates = [float(value) for value in parts[1:]]
        out_of_range = any(not 0.0 <= value <= 1.0 for value in coordinates)
        message = "posible polígono YOLO; formato no soportado en esta fase"
        if out_of_range:
            message += "; contiene coordenadas fuera de [0, 1]"
        return AnnotationResult(
            line_number=line_number,
            kind="polygon",
            raw=stripped,
            class_id=class_id,
            message=message,
        )

    if len(parts) != 5:
        return AnnotationResult(
            line_number=line_number,
            kind="invalid",
            raw=stripped,
            message=f"se esperaban 5 valores para bbox y se encontraron {len(parts)}",
        )

    try:
        class_id = int(parts[0])
    except ValueError:
        return AnnotationResult(
            line_number=line_number,
            kind="invalid",
            raw=stripped,
            message="class_id debe ser un entero",
        )
    if class_id < 0:
        return AnnotationResult(
            line_number=line_number,
            kind="invalid",
            raw=stripped,
            class_id=class_id,
            message="class_id no puede ser negativo",
        )

    try:
        center_x, center_y, width, height = (float(value) for value in parts[1:])
    except ValueError:
        return AnnotationResult(
            line_number=line_number,
            kind="invalid",
            raw=stripped,
            class_id=class_id,
            message="las coordenadas bbox deben ser numéricas",
        )

    errors = validate_normalized_bbox(center_x, center_y, width, height)
    if errors:
        return AnnotationResult(
            line_number=line_number,
            kind="invalid",
            raw=stripped,
            class_id=class_id,
            message="; ".join(errors),
        )

    box = YoloBox(class_id, center_x, center_y, width, height)
    crosses_boundary = (
        center_x - width / 2.0 < 0.0
        or center_y - height / 2.0 < 0.0
        or center_x + width / 2.0 > 1.0
        or center_y + height / 2.0 > 1.0
    )
    return AnnotationResult(
        line_number=line_number,
        kind="bbox",
        raw=stripped,
        class_id=class_id,
        box=box,
        message=("bbox cruza el borde y será limitado al dibujarse" if crosses_boundary else None),
    )


def read_label_file(path: Path) -> LabelFileAudit:
    """Read all non-empty lines from a label file without propagating file errors."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        return LabelFileAudit(path=path, results=(), read_error=str(exc))

    non_empty_lines = [
        (line_number, line)
        for line_number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]
    results = tuple(
        parse_annotation_line(line, line_number) for line_number, line in non_empty_lines
    )
    return LabelFileAudit(path=path, results=results, empty=not non_empty_lines)


def label_path_for_image(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    """Build the conventional paired label path while preserving subdirectories."""
    return labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")


def _relative_key(path: Path, base_dir: Path) -> str:
    relative = path.relative_to(base_dir)
    return relative.with_suffix("").as_posix().casefold()


def _find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def _find_labels(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".txt"),
        key=lambda path: path.as_posix().casefold(),
    )


def discover_split(dataset_root: Path, split: str) -> SplitInventory:
    """Discover images, labels, pairs, and ambiguous names for one YOLO split."""
    split_dir = dataset_root / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images = _find_images(images_dir)
    labels = _find_labels(labels_dir)

    images_by_key: dict[str, list[Path]] = defaultdict(list)
    labels_by_key: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        images_by_key[_relative_key(image_path, images_dir)].append(image_path)
    for label_path in labels:
        labels_by_key[_relative_key(label_path, labels_dir)].append(label_path)

    records: list[ImageRecord] = []
    for image_path in images:
        key = _relative_key(image_path, images_dir)
        label_matches = labels_by_key.get(key, [])
        records.append(
            ImageRecord(
                split=split,
                path=image_path,
                relative_path=image_path.relative_to(images_dir),
                label_path=(
                    label_matches[0]
                    if len(label_matches) == 1 and len(images_by_key[key]) == 1
                    else None
                ),
            )
        )

    orphan_labels = tuple(
        label_path
        for label_path in labels
        if _relative_key(label_path, labels_dir) not in images_by_key
    )
    return SplitInventory(
        split=split,
        records=tuple(records),
        labels=tuple(labels),
        orphan_labels=orphan_labels,
        duplicate_image_keys=tuple(
            sorted(key for key, paths in images_by_key.items() if len(paths) > 1)
        ),
        duplicate_label_keys=tuple(
            sorted(key for key, paths in labels_by_key.items() if len(paths) > 1)
        ),
    )


def _dataset_relative(path: Path, dataset_root: Path) -> str:
    try:
        return path.relative_to(dataset_root).as_posix()
    except ValueError:
        return str(path)


def _resolve_dataset_path(explicit_root: Path | None) -> Path:
    if explicit_root is None:
        from src.config import get_dataset_root

        return get_dataset_root() / "raw" / "corn-leaf-roboflow"

    explicit_root = explicit_root.expanduser()
    if any((explicit_root / split).is_dir() for split in DEFAULT_SPLITS):
        return explicit_root
    nested = explicit_root / "raw" / "corn-leaf-roboflow"
    if nested.is_dir() or (explicit_root / "clean").is_dir() or (explicit_root / "raw").is_dir():
        return nested
    return explicit_root


def _resolve_output_path(explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        return explicit_output.expanduser()
    from src.config import get_output_root

    return get_output_root() / "leaf_detection" / "annotation_audit"


def _normalize_splits(values: Sequence[str]) -> list[str]:
    splits: list[str] = []
    for value in values:
        for split in value.split(","):
            split = split.strip()
            if not split:
                continue
            if Path(split).name != split or split in {".", ".."}:
                raise ValueError(f"Nombre de split inválido: {split!r}")
            if split not in splits:
                splits.append(split)
    if not splits:
        raise ValueError("Debe indicar al menos un split")
    return splits


def _record_problem_examples(
    inventories: Sequence[SplitInventory],
    label_audits: dict[Path, LabelFileAudit],
    dataset_root: Path,
    problems: ProblemCollector,
) -> None:
    for inventory in inventories:
        for record in inventory.records:
            if record.label_path is None:
                problems.add(
                    "image_without_label",
                    split=inventory.split,
                    path=_dataset_relative(record.path, dataset_root),
                    message="No existe una etiqueta .txt única con la misma ruta relativa y stem",
                )
        for label_path in inventory.orphan_labels:
            problems.add(
                "label_without_image",
                split=inventory.split,
                path=_dataset_relative(label_path, dataset_root),
                message="No existe una imagen compatible con la misma ruta relativa y stem",
            )
        for key in inventory.duplicate_image_keys:
            problems.add(
                "duplicate_image_stem",
                split=inventory.split,
                path=key,
                message="Varias extensiones de imagen comparten la misma clave de anotación",
            )
        for key in inventory.duplicate_label_keys:
            problems.add(
                "duplicate_label_stem",
                split=inventory.split,
                path=key,
                message="Varias etiquetas comparten la misma clave relativa",
            )

        for label_path in inventory.labels:
            audit = label_audits[label_path]
            relative = _dataset_relative(label_path, dataset_root)
            if audit.empty:
                problems.add(
                    "empty_label_file",
                    split=inventory.split,
                    path=relative,
                    message="El archivo no contiene líneas de anotación",
                )
            if audit.read_error:
                problems.add(
                    "label_read_error",
                    split=inventory.split,
                    path=relative,
                    message=audit.read_error,
                )
            for result in audit.results:
                if result.kind == "bbox":
                    if result.message:
                        problems.add(
                            "bbox_crosses_boundary",
                            split=inventory.split,
                            path=relative,
                            line=result.line_number,
                            message=result.message,
                        )
                    continue
                problems.add(
                    "possible_polygon" if result.kind == "polygon" else "invalid_annotation",
                    split=inventory.split,
                    path=relative,
                    line=result.line_number,
                    message=result.message or "Anotación no soportada",
                )


def _annotation_metrics(
    inventories: Sequence[SplitInventory], label_audits: dict[Path, LabelFileAudit]
) -> tuple[Counter[str], dict[int, Counter[str]]]:
    metrics: Counter[str] = Counter()
    by_class: dict[int, Counter[str]] = defaultdict(Counter)
    for inventory in inventories:
        for label_path in inventory.labels:
            audit = label_audits[label_path]
            metrics["empty_label_files"] += int(audit.empty)
            metrics["label_read_errors"] += int(audit.read_error is not None)
            for result in audit.results:
                if result.kind == "bbox":
                    metrics["valid_annotations"] += 1
                    if result.class_id is not None:
                        by_class[result.class_id]["valid_bboxes"] += 1
                elif result.kind == "polygon":
                    metrics["possible_polygons"] += 1
                    if result.class_id is not None:
                        by_class[result.class_id]["possible_polygons"] += 1
                else:
                    metrics["invalid_annotations"] += 1
                    if result.class_id is not None:
                        by_class[result.class_id]["invalid_annotations"] += 1
    return metrics, by_class


def _visual_output_path(output_root: Path, record: ImageRecord) -> Path:
    return (
        output_root
        / "images"
        / record.split
        / record.relative_path.parent
        / f"{record.relative_path.name}.audit.jpg"
    )


def _draw_audit_image(
    record: ImageRecord,
    label_audit: LabelFileAudit | None,
    output_root: Path,
    dataset_root: Path,
    problems: ProblemCollector,
) -> Path:
    warnings: list[str] = []
    image_error: str | None = None
    try:
        with Image.open(record.path) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        image_error = str(exc)
        image = Image.new("RGB", (900, 400), "#2b2b2b")
        warnings.append(f"IMAGEN ILEGIBLE: {image_error}")
        problems.add(
            "image_read_error",
            split=record.split,
            path=_dataset_relative(record.path, dataset_root),
            message=image_error,
        )

    width, height = image.size
    annotations = label_audit.results if label_audit else ()
    valid_boxes = [result for result in annotations if result.kind == "bbox" and result.box]

    if record.label_path is None:
        warnings.append("SIN ETIQUETA: no se encontró un .txt único con el mismo stem")
    elif label_audit is None:
        warnings.append("ETIQUETA NO PROCESADA")
    else:
        if label_audit.empty:
            warnings.append("ETIQUETA VACÍA")
        if label_audit.read_error:
            warnings.append(f"ERROR AL LEER ETIQUETA: {label_audit.read_error}")
        for result in annotations:
            if result.kind != "bbox" or result.message:
                warnings.append(
                    f"Línea {result.line_number}: {result.message or 'anotación no soportada'}"
                )

    canvas_draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    if image_error is None:
        for result in valid_boxes:
            assert result.box is not None
            pixel_box = yolo_to_pixel_box(result.box, width, height)
            color = "#00ff66"
            canvas_draw.rectangle(pixel_box, outline=color, width=max(2, min(width, height) // 250))
            label = f"class {result.class_id}"
            text_box = canvas_draw.textbbox((pixel_box[0], pixel_box[1]), label, font=font)
            canvas_draw.rectangle(text_box, fill="#003d1f")
            canvas_draw.text((pixel_box[0], pixel_box[1]), label, fill="white", font=font)

    metadata = [
        f"archivo: {_dataset_relative(record.path, dataset_root)}",
        f"split: {record.split} | dimensiones: {width}x{height}",
        f"anotaciones: {len(annotations)} | bbox válidos: {len(valid_boxes)}",
    ]
    wrapped_warnings: list[str] = []
    wrap_width = max(40, width // 7)
    for warning in warnings:
        wrapped_warnings.extend(textwrap.wrap(f"ADVERTENCIA: {warning}", width=wrap_width))
    header_lines = metadata + wrapped_warnings
    line_height = 16
    header_height = max(58, 8 + line_height * len(header_lines))
    canvas = Image.new("RGB", (width, height + header_height), "#202020")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(header_lines):
        color = "#ff7070" if index >= len(metadata) else "white"
        draw.text((8, 5 + index * line_height), line, fill=color, font=font)

    output_path = _visual_output_path(output_root, record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=92)
    return output_path


def _write_summary_csv(summary: dict[str, object], path: Path) -> None:
    rows: list[dict[str, object]] = []
    overall_keys = (
        "total_images_found",
        "total_labels_found",
        "audited_images",
        "images_without_label",
        "labels_without_image",
        "empty_label_files",
        "label_read_errors",
        "valid_annotations",
        "invalid_annotations",
        "possible_polygons",
        "image_read_errors",
    )
    for key in overall_keys:
        rows.append({"section": "overall", "name": "all", "metric": key, "count": summary[key]})

    for split, metrics in dict(summary["by_split"]).items():
        for metric, count in dict(metrics).items():
            rows.append({"section": "split", "name": split, "metric": metric, "count": count})

    for class_id, metrics in dict(summary["annotations_by_class"]).items():
        for metric, count in dict(metrics).items():
            rows.append({"section": "class", "name": class_id, "metric": metric, "count": count})

    for problem_type, count in dict(summary["problem_counts"]).items():
        rows.append(
            {
                "section": "problem",
                "name": problem_type,
                "metric": "occurrences",
                "count": count,
            }
        )

    for problem_type, examples in dict(summary["problem_examples"]).items():
        for example in list(examples):
            rows.append(
                {
                    "section": "problem_example",
                    "name": problem_type,
                    "metric": example.get("path", ""),
                    "count": "",
                    "details": example.get("message", ""),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("section", "name", "metric", "count", "details")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(
    dataset_root: Path,
    output_root: Path,
    samples: int,
    seed: int,
    splits: Sequence[str],
) -> dict[str, object]:
    """Run discovery, full label validation, visual sampling, and summaries."""
    if samples <= 0:
        raise ValueError("--samples debe ser mayor que cero")
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"No existe el directorio del dataset: {dataset_root}")
    if output_root == dataset_root or output_root.is_relative_to(dataset_root):
        raise ValueError(
            "--output debe estar fuera del dataset fuente para mantenerlo inmutable: "
            f"{output_root}"
        )

    inventories = [discover_split(dataset_root, split) for split in splits]
    if not any((dataset_root / split).is_dir() for split in splits):
        searched = ", ".join(str(dataset_root / split) for split in splits)
        raise FileNotFoundError(
            f"No se encontró ningún split solicitado. Rutas revisadas: {searched}"
        )

    all_records = [record for inventory in inventories for record in inventory.records]
    all_labels = [label for inventory in inventories for label in inventory.labels]
    label_audits = {label_path: read_label_file(label_path) for label_path in all_labels}
    problems = ProblemCollector()
    _record_problem_examples(inventories, label_audits, dataset_root, problems)
    annotation_metrics, annotations_by_class = _annotation_metrics(inventories, label_audits)

    rng = random.Random(seed)
    selected = rng.sample(all_records, k=min(samples, len(all_records)))
    selected.sort(key=lambda record: (record.split, record.relative_path.as_posix().casefold()))
    audited_by_split: Counter[str] = Counter()
    generated_images: list[str] = []
    for record in selected:
        label_audit = label_audits.get(record.label_path) if record.label_path else None
        output_path = _draw_audit_image(
            record, label_audit, output_root, dataset_root, problems
        )
        audited_by_split[record.split] += 1
        generated_images.append(output_path.relative_to(output_root).as_posix())

    by_split: dict[str, dict[str, int | bool]] = {}
    for inventory in inventories:
        by_split[inventory.split] = {
            "directory_found": (dataset_root / inventory.split).is_dir(),
            "images_found": len(inventory.records),
            "labels_found": len(inventory.labels),
            "images_audited": audited_by_split[inventory.split],
            "images_without_label": sum(record.label_path is None for record in inventory.records),
            "labels_without_image": len(inventory.orphan_labels),
        }

    summary: dict[str, object] = {
        "schema_version": 1,
        "annotation_scope": "all discovered label files in requested splits",
        "visual_sample_scope": "seeded sample across all discovered images",
        "total_images_found": len(all_records),
        "total_labels_found": len(all_labels),
        "audited_images": len(selected),
        "images_without_label": sum(record.label_path is None for record in all_records),
        "labels_without_image": sum(len(inventory.orphan_labels) for inventory in inventories),
        "empty_label_files": annotation_metrics["empty_label_files"],
        "label_read_errors": annotation_metrics["label_read_errors"],
        "valid_annotations": annotation_metrics["valid_annotations"],
        "invalid_annotations": annotation_metrics["invalid_annotations"],
        "possible_polygons": annotation_metrics["possible_polygons"],
        "image_read_errors": problems.counts["image_read_error"],
        "annotations_by_class": {
            str(class_id): {
                "valid_bboxes": counts["valid_bboxes"],
                "invalid_annotations": counts["invalid_annotations"],
                "possible_polygons": counts["possible_polygons"],
            }
            for class_id, counts in sorted(annotations_by_class.items())
        },
        "by_split": by_split,
        "problem_counts": dict(sorted(problems.counts.items())),
        "problem_examples": dict(sorted(problems.examples.items())),
        "seed": seed,
        "requested_samples": samples,
        "requested_splits": list(splits),
        "paths": {
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "visual_audits": str(output_root / "images"),
            "summary_json": str(output_root / "summary.json"),
            "summary_csv": str(output_root / "summary.csv"),
        },
        "generated_images": generated_images,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_summary_csv(summary, output_root / "summary.csv")
    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("debe ser un entero mayor que cero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the annotation audit."""
    parser = argparse.ArgumentParser(
        description=(
            "Audita anotaciones YOLO bbox y reporta polígonos o líneas incompatibles "
            "sin modificar el dataset fuente."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Ruta a corn-leaf-roboflow. También acepta la raíz que contiene "
            "raw/corn-leaf-roboflow. Sin este flag usa get_dataset_root()."
        ),
    )
    parser.add_argument("--samples", type=_positive_int, default=100, help="Imágenes a dibujar")
    parser.add_argument("--seed", type=int, default=42, help="Semilla del muestreo reproducible")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Directorio de salida; por defecto usa "
            "get_output_root()/leaf_detection/annotation_audit"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits separados por espacios o comas (default: train valid val test)",
    )
    return parser


def main() -> None:
    """Run the CLI and print the locations of generated artifacts."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        splits = _normalize_splits(args.splits)
        dataset_root = _resolve_dataset_path(args.dataset_root)
        output_root = _resolve_output_path(args.output)
        summary = run_audit(dataset_root, output_root, args.samples, args.seed, splits)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))

    paths = dict(summary["paths"])
    print(f"Auditoría completada: {summary['audited_images']} imágenes visualizadas")
    print(f"Resumen JSON: {paths['summary_json']}")
    print(f"Resumen CSV:  {paths['summary_csv']}")


if __name__ == "__main__":
    main()
