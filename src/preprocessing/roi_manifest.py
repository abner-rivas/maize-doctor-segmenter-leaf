"""Manual ROI import, final manifest construction, and row-level validation."""

from __future__ import annotations

import json
import math
import random
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from src.data.leaf_pilot import (
    PILOT_COLUMNS,
    VALID_ENVIRONMENTS,
    VALID_SPLITS,
    read_csv_rows,
    require_columns,
    sha256_file,
    write_csv_rows,
)
from src.data.loader import load_and_normalize_image
from src.preprocessing.leaf_processor import LeafImageProcessor, LeafProcessorConfig
from src.preprocessing.leaf_roi import (
    BoundingBox,
    bbox_area_ratio,
    bbox_height,
    bbox_requires_clipping,
    bbox_width,
    crop_leaf_region,
    image_to_rgb,
    validate_bbox,
)

MANUAL_STATUSES = ("annotated", "ambiguous", "rejected")
KNOWN_STATUSES = ("pending", *MANUAL_STATUSES)
ANNOTATION_EXTRA_COLUMNS = (
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "roi_width",
    "roi_height",
    "roi_area_ratio",
    "original_rotation_degrees",
    "roi_conversion_method",
    "roi_clipped",
    "notes",
    "annotation_warnings",
    "annotation_format",
)
IMPORTED_ANNOTATION_COLUMNS = (*PILOT_COLUMNS, *ANNOTATION_EXTRA_COLUMNS)
ROI_MANIFEST_COLUMNS = (
    "pilot_id",
    "image_path",
    "original_image_path",
    "image_sha256",
    "label",
    "split",
    "environment",
    "source_dataset",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "roi_width",
    "roi_height",
    "roi_area_ratio",
    "roi_confidence",
    "roi_source",
    "annotation_status",
    "notes",
)
ROI_VALIDATION_COLUMNS = ("pilot_id", "valid", "errors", "warnings")


@dataclass(frozen=True)
class ManualAnnotation:
    """Normalized result of importing one manual annotation."""

    status: str
    bbox: BoundingBox | None
    area_ratio: float | None
    notes: str
    warnings: tuple[str, ...] = ()
    original_rotation_degrees: float = 0.0
    conversion_method: str = "direct_bbox"
    clipped: bool = False
    geometry_converted: bool = False


@dataclass(frozen=True)
class CvatImageAnnotation:
    """One CVAT ``image`` element normalized to a pilot annotation."""

    xml_id: str
    name: str
    pilot_id: str
    width: int | None
    height: int | None
    annotation: ManualAnnotation


def resolve_pilot_image(
    manifest_path: Path,
    row: dict[str, str],
    image_root: Path | None = None,
    *,
    field: str = "pilot_image_path",
) -> Path:
    """Resolve a pilot-relative image path without requiring DATASET_ROOT."""
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"Fila {row.get('pilot_id', '?')} sin {field}")
    path = Path(value)
    if path.is_absolute():
        return path
    if image_root is not None:
        return image_root / path
    direct = manifest_path.parent / path
    if direct.exists():
        return direct
    return manifest_path.parent.parent / path


def _empty_annotation(
    status: str,
    notes: str,
    *warnings: str,
    rotation_degrees: float = 0.0,
    conversion_method: str = "direct_bbox",
    clipped: bool = False,
    geometry_converted: bool = False,
) -> ManualAnnotation:
    return ManualAnnotation(
        status,
        None,
        None,
        notes,
        tuple(warnings),
        rotation_degrees,
        conversion_method,
        clipped,
        geometry_converted,
    )


def _annotation_from_pixel_bbox(
    bbox_values: Sequence[int | float | str],
    image_width: int,
    image_height: int,
    min_area_ratio: float,
    *,
    notes: str = "",
    rotation_degrees: float = 0.0,
    conversion_method: str = "direct_bbox",
) -> ManualAnnotation:
    detection = validate_bbox(
        image_width,
        image_height,
        bbox_values,
        min_area_ratio,
        source="manual",
    )
    if not detection.detected or detection.bbox is None:
        return _empty_annotation(
            "ambiguous",
            notes or detection.reason or "bbox manual inválido",
            detection.reason or "bbox manual inválido",
            rotation_degrees=rotation_degrees,
            conversion_method=conversion_method,
            geometry_converted=detection.bbox is not None,
        )
    warnings: list[str] = []
    clipped = bbox_requires_clipping(bbox_values, image_width, image_height)
    if clipped:
        warnings.append("bbox manual limitado a los bordes")
    return ManualAnnotation(
        "annotated",
        detection.bbox,
        detection.area_ratio,
        notes,
        tuple(warnings),
        rotation_degrees,
        conversion_method,
        clipped,
        True,
    )


def parse_yolo_leaf_annotation(
    label_path: Path,
    image_width: int,
    image_height: int,
    min_area_ratio: float,
) -> ManualAnnotation:
    """Import one single-class YOLO bbox or return an explicit review state."""
    if not label_path.is_file():
        return _empty_annotation("pending", "", "imagen sin archivo de etiqueta")
    try:
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8-sig").splitlines()]
    except (OSError, UnicodeError) as exc:
        return _empty_annotation("ambiguous", str(exc), "no se pudo leer la etiqueta")
    lines = [line for line in lines if line]
    if not lines:
        return _empty_annotation("ambiguous", "etiqueta YOLO vacía", "etiqueta vacía")
    if len(lines) > 1:
        return _empty_annotation(
            "ambiguous",
            f"se encontraron {len(lines)} bbox; requiere revisión manual",
            "múltiples bbox no se seleccionan automáticamente",
        )
    parts = lines[0].split()
    if len(parts) > 5:
        return _empty_annotation(
            "ambiguous",
            "posible polígono YOLO no soportado",
            "formato de polígono no soportado",
        )
    if len(parts) != 5:
        return _empty_annotation(
            "ambiguous",
            f"bbox YOLO requiere 5 valores; recibió {len(parts)}",
            "formato YOLO inválido",
        )
    try:
        class_id = int(parts[0])
    except ValueError:
        return _empty_annotation("ambiguous", "class_id no entero", "class_id inválido")
    if class_id != 0:
        return _empty_annotation(
            "ambiguous",
            f"class_id {class_id} no corresponde a maize_leaf (0)",
            "clase YOLO distinta de 0",
        )
    try:
        center_x, center_y, width, height = (float(value) for value in parts[1:])
    except ValueError:
        return _empty_annotation("ambiguous", "coordenadas no numéricas", "YOLO inválido")
    values = (center_x, center_y, width, height)
    if not all(math.isfinite(value) for value in values):
        return _empty_annotation("ambiguous", "coordenadas no finitas", "YOLO inválido")
    if (
        not 0.0 <= center_x <= 1.0
        or not 0.0 <= center_y <= 1.0
        or not 0.0 < width <= 1.0
        or not 0.0 < height <= 1.0
    ):
        return _empty_annotation(
            "ambiguous",
            "coordenadas YOLO fuera del rango normalizado",
            "YOLO fuera de [0, 1]",
        )
    pixel_bbox = (
        (center_x - width / 2.0) * image_width,
        (center_y - height / 2.0) * image_height,
        (center_x + width / 2.0) * image_width,
        (center_y + height / 2.0) * image_height,
    )
    return _annotation_from_pixel_bbox(
        pixel_bbox,
        image_width,
        image_height,
        min_area_ratio,
    )


def _load_csv_annotations(path: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    rows, columns = read_csv_rows(path)
    require_columns(
        columns,
        ("pilot_id", "x1", "y1", "x2", "y2", "status", "notes"),
        "CSV de anotaciones",
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("pilot_id", "").strip()].append(row)
    return grouped, []


def _parse_cvat_dimension(value: str | None, field: str) -> int:
    """Parse one positive integer image dimension from CVAT XML."""
    if value is None:
        raise ValueError(f"{field} ausente")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} debe ser entero") from exc
    if parsed <= 0:
        raise ValueError(f"{field} debe ser mayor que cero")
    return parsed


def _parse_cvat_float(value: str | None, field: str, *, default: float | None = None) -> float:
    """Parse one finite CVAT numeric attribute."""
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{field} ausente")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} debe ser numérico") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} debe ser finito")
    return parsed


def rotated_bbox_to_axis_aligned(
    bbox: Sequence[int | float | str],
    rotation_degrees: int | float | str,
) -> tuple[float, float, float, float]:
    """Return the smallest axis-aligned bbox enclosing a CVAT rotated box.

    CVAT rotates rectangles clockwise around their center in image coordinates
    (where the y axis points downward). All four corners are transformed before
    their coordinate-wise extrema are taken.
    """
    if len(bbox) != 4:
        raise ValueError("bbox CVAT debe contener cuatro coordenadas")
    values = tuple(
        _parse_cvat_float(str(value), field)
        for value, field in zip(bbox, ("xtl", "ytl", "xbr", "ybr"), strict=True)
    )
    xtl, ytl, xbr, ybr = values
    if xbr <= xtl:
        raise ValueError("xbr debe ser mayor que xtl")
    if ybr <= ytl:
        raise ValueError("ybr debe ser mayor que ytl")
    rotation = _parse_cvat_float(str(rotation_degrees), "rotation")
    radians = math.radians(rotation)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    center_x = (xtl + xbr) / 2.0
    center_y = (ytl + ybr) / 2.0
    rotated_corners: list[tuple[float, float]] = []
    for x, y in ((xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr)):
        delta_x = x - center_x
        delta_y = y - center_y
        rotated_corners.append(
            (
                center_x + delta_x * cosine - delta_y * sine,
                center_y + delta_x * sine + delta_y * cosine,
            )
        )
    return (
        min(x for x, _ in rotated_corners),
        min(y for _, y in rotated_corners),
        max(x for x, _ in rotated_corners),
        max(y for _, y in rotated_corners),
    )


def _parse_cvat_image(
    element: ET.Element,
    min_area_ratio: float,
) -> CvatImageAnnotation:
    """Parse and validate one CVAT native ``image`` element."""
    xml_id = (element.get("id") or "").strip()
    name = (element.get("name") or "").strip()
    pilot_id = Path(name).stem if name else ""
    structural_errors: list[str] = []
    if not xml_id:
        structural_errors.append("id de imagen ausente")
    if not name:
        structural_errors.append("name de imagen ausente")
    try:
        width = _parse_cvat_dimension(element.get("width"), "width")
        height = _parse_cvat_dimension(element.get("height"), "height")
    except ValueError as exc:
        width = None
        height = None
        structural_errors.append(str(exc))

    boxes = element.findall("box")
    if len(boxes) != 1:
        detail = "imagen sin caja" if not boxes else f"imagen con {len(boxes)} cajas"
        annotation = _empty_annotation(
            "ambiguous",
            detail,
            "se requiere exactamente una caja maize_leaf",
        )
        return CvatImageAnnotation(xml_id, name, pilot_id, width, height, annotation)

    box = boxes[0]
    if box.get("label") != "maize_leaf":
        label = box.get("label", "")
        annotation = _empty_annotation(
            "ambiguous",
            f"etiqueta CVAT distinta de maize_leaf: {label!r}",
            "etiqueta CVAT incorrecta",
        )
        return CvatImageAnnotation(xml_id, name, pilot_id, width, height, annotation)

    try:
        rotation = _parse_cvat_float(box.get("rotation"), "rotation", default=0.0)
    except ValueError as exc:
        annotation = _empty_annotation("ambiguous", str(exc), "rotation CVAT inválida")
        return CvatImageAnnotation(xml_id, name, pilot_id, width, height, annotation)
    conversion_method = (
        "direct_bbox" if math.isclose(rotation % 360.0, 0.0, abs_tol=1e-12)
        else "rotated_to_axis_aligned"
    )
    if structural_errors:
        annotation = _empty_annotation(
            "ambiguous",
            " | ".join(structural_errors),
            "elemento image CVAT inválido",
            rotation_degrees=rotation,
            conversion_method=conversion_method,
        )
        return CvatImageAnnotation(xml_id, name, pilot_id, width, height, annotation)

    try:
        raw_bbox = tuple(
            _parse_cvat_float(box.get(field), field)
            for field in ("xtl", "ytl", "xbr", "ybr")
        )
        if raw_bbox[2] <= raw_bbox[0]:
            raise ValueError("xbr debe ser mayor que xtl")
        if raw_bbox[3] <= raw_bbox[1]:
            raise ValueError("ybr debe ser mayor que ytl")
        converted_bbox = (
            raw_bbox
            if conversion_method == "direct_bbox"
            else rotated_bbox_to_axis_aligned(raw_bbox, rotation)
        )
    except ValueError as exc:
        annotation = _empty_annotation(
            "ambiguous",
            str(exc),
            "coordenadas CVAT inválidas",
            rotation_degrees=rotation,
            conversion_method=conversion_method,
        )
    else:
        assert width is not None and height is not None
        annotation = _annotation_from_pixel_bbox(
            converted_bbox,
            width,
            height,
            min_area_ratio,
            rotation_degrees=rotation,
            conversion_method=conversion_method,
        )
    return CvatImageAnnotation(xml_id, name, pilot_id, width, height, annotation)


def load_cvat_xml_annotations(
    path: Path,
    min_area_ratio: float,
) -> tuple[dict[str, list[CvatImageAnnotation]], dict[str, int], list[str]]:
    """Load CVAT native XML, retaining every image and every validation issue."""
    if not path.is_file():
        raise FileNotFoundError(f"XML CVAT inexistente: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"XML CVAT inválido: {exc}") from exc
    if root.tag != "annotations":
        raise ValueError(f"XML CVAT inválido: raíz esperada 'annotations', recibió {root.tag!r}")

    grouped: dict[str, list[CvatImageAnnotation]] = defaultdict(list)
    warnings: list[str] = []
    records = [
        _parse_cvat_image(element, min_area_ratio)
        for element in root.findall("image")
    ]
    for record in records:
        grouped[record.pilot_id].append(record)
        if not record.pilot_id:
            warnings.append(
                f"imagen XML id={record.xml_id or '?'} sin name asociable a pilot_id"
            )
    stats = {
        "xml_images": len(records),
        "xml_boxes": sum(len(element.findall("box")) for element in root.findall("image")),
        "multiple_box_images": sum(
            len(element.findall("box")) > 1 for element in root.findall("image")
        ),
        "images_without_box": sum(
            not element.findall("box") for element in root.findall("image")
        ),
    }
    return grouped, stats, warnings


def _parse_csv_annotation(
    rows: Sequence[dict[str, str]],
    image_width: int,
    image_height: int,
    min_area_ratio: float,
) -> ManualAnnotation:
    if not rows:
        return _empty_annotation("pending", "", "imagen sin anotación CSV")
    if len(rows) > 1:
        return _empty_annotation(
            "ambiguous",
            f"pilot_id duplicado {len(rows)} veces en anotaciones",
            "anotaciones CSV duplicadas",
        )
    row = rows[0]
    status = row.get("status", "").strip().lower()
    notes = row.get("notes", "").strip()
    if status not in MANUAL_STATUSES:
        return _empty_annotation(
            "ambiguous",
            notes or f"estado desconocido: {status!r}",
            "estado de anotación inválido",
        )
    if status in {"ambiguous", "rejected"}:
        return _empty_annotation(status, notes)
    bbox_values = (row.get("x1", ""), row.get("y1", ""), row.get("x2", ""), row.get("y2", ""))
    return _annotation_from_pixel_bbox(
        bbox_values,
        image_width,
        image_height,
        min_area_ratio,
        notes=notes,
    )


def _annotation_fields(annotation: ManualAnnotation, annotation_format: str) -> dict[str, object]:
    bbox = annotation.bbox
    return {
        "annotation_status": annotation.status,
        "roi_x1": bbox[0] if bbox else "",
        "roi_y1": bbox[1] if bbox else "",
        "roi_x2": bbox[2] if bbox else "",
        "roi_y2": bbox[3] if bbox else "",
        "roi_width": bbox_width(bbox) if bbox else "",
        "roi_height": bbox_height(bbox) if bbox else "",
        "roi_area_ratio": annotation.area_ratio if annotation.area_ratio is not None else "",
        "original_rotation_degrees": annotation.original_rotation_degrees,
        "roi_conversion_method": annotation.conversion_method,
        "roi_clipped": annotation.clipped,
        "notes": annotation.notes,
        "annotation_warnings": " | ".join(annotation.warnings),
        "annotation_format": annotation_format,
    }


def import_manual_annotations(
    pilot_manifest: Path,
    annotations: Path,
    annotation_format: str,
    output: Path,
    min_area_ratio: float,
    *,
    overwrite: bool = False,
    image_root: Path | None = None,
) -> dict[str, object]:
    """Import YOLO, CSV, or CVAT XML into a traceable intermediate manifest."""
    if annotation_format not in {"yolo", "csv", "cvat_xml"}:
        raise ValueError("format debe ser yolo, csv o cvat_xml")
    if output.exists() and not overwrite:
        raise FileExistsError(f"La salida ya existe; use --overwrite para reemplazarla: {output}")
    if output.resolve() == pilot_manifest.resolve() and not overwrite:
        raise ValueError("No se sobrescribe el manifiesto original sin --overwrite")
    pilot_rows, columns = read_csv_rows(pilot_manifest)
    require_columns(columns, PILOT_COLUMNS, "pilot manifest")

    grouped_csv: dict[str, list[dict[str, str]]] = {}
    grouped_cvat: dict[str, list[CvatImageAnnotation]] = {}
    cvat_stats: dict[str, int] = {}
    global_warnings: list[str] = []
    if annotation_format == "csv":
        grouped_csv, _ = _load_csv_annotations(annotations)
        known_ids = {row["pilot_id"] for row in pilot_rows}
        unknown_ids = sorted(set(grouped_csv) - known_ids - {""})
        if unknown_ids:
            global_warnings.append(
                f"pilot_id desconocidos en CSV: {', '.join(unknown_ids[:10])}"
            )
    elif annotation_format == "yolo":
        if not annotations.is_dir():
            raise FileNotFoundError(f"Directorio de etiquetas YOLO inexistente: {annotations}")
        known_ids = {row["pilot_id"] for row in pilot_rows}
        orphan_labels = sorted(
            path.stem
            for path in annotations.glob("*.txt")
            if path.stem not in known_ids
        )
        if orphan_labels:
            global_warnings.append(
                f"etiquetas sin imagen/pilot_id: {', '.join(orphan_labels[:10])}"
            )
    else:
        grouped_cvat, cvat_stats, cvat_warnings = load_cvat_xml_annotations(
            annotations,
            min_area_ratio,
        )
        global_warnings.extend(cvat_warnings)
        known_ids = {row["pilot_id"] for row in pilot_rows}
        unknown_ids = sorted(set(grouped_cvat) - known_ids - {""})
        if unknown_ids:
            global_warnings.append(
                f"pilot_id desconocidos en XML CVAT: {', '.join(unknown_ids[:10])}"
            )

    output_rows: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    conversion_counts: Counter[str] = Counter()
    clipped_count = 0
    for pilot_row in pilot_rows:
        pilot_id = pilot_row["pilot_id"]
        try:
            image_path = resolve_pilot_image(pilot_manifest, pilot_row, image_root)
            image = load_and_normalize_image(image_path)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            annotation = _empty_annotation("rejected", str(exc), "imagen inexistente o ilegible")
        else:
            if annotation_format == "yolo":
                annotation = parse_yolo_leaf_annotation(
                    annotations / f"{pilot_id}.txt",
                    image.width,
                    image.height,
                    min_area_ratio,
                )
            elif annotation_format == "csv":
                annotation = _parse_csv_annotation(
                    grouped_csv.get(pilot_id, []),
                    image.width,
                    image.height,
                    min_area_ratio,
                )
            else:
                cvat_records = grouped_cvat.get(pilot_id, [])
                if not cvat_records:
                    annotation = _empty_annotation(
                        "pending",
                        "",
                        "imagen sin anotación XML CVAT",
                    )
                elif len(cvat_records) > 1:
                    annotation = _empty_annotation(
                        "ambiguous",
                        f"pilot_id duplicado {len(cvat_records)} veces en XML CVAT",
                        "múltiples elementos image no se seleccionan automáticamente",
                    )
                else:
                    record = cvat_records[0]
                    if record.width != image.width or record.height != image.height:
                        annotation = _empty_annotation(
                            "ambiguous",
                            (
                                f"dimensiones XML {record.width}x{record.height} no coinciden "
                                f"con imagen {image.width}x{image.height}"
                            ),
                            "dimensiones CVAT inconsistentes",
                            rotation_degrees=record.annotation.original_rotation_degrees,
                            conversion_method=record.annotation.conversion_method,
                            clipped=record.annotation.clipped,
                        )
                    else:
                        annotation = record.annotation
        merged: dict[str, object] = dict(pilot_row)
        merged.update(_annotation_fields(annotation, annotation_format))
        output_rows.append(merged)
        status_counts[annotation.status] += 1
        if annotation.geometry_converted:
            conversion_counts[annotation.conversion_method] += 1
        if annotation.status == "annotated":
            clipped_count += annotation.clipped

    write_csv_rows(output, output_rows, IMPORTED_ANNOTATION_COLUMNS)
    summary = {
        "schema_version": 1,
        "pilot_manifest": str(pilot_manifest.resolve()),
        "annotations": str(annotations.resolve()),
        "format": annotation_format,
        "output": str(output.resolve()),
        "min_area_ratio": min_area_ratio,
        "rows": len(output_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "conversion_counts": dict(sorted(conversion_counts.items())),
        "clipped_rows": clipped_count,
        "valid_rows": status_counts["annotated"],
        "invalid_rows": len(output_rows) - status_counts["annotated"],
        "warnings": global_warnings,
    }
    summary.update(cvat_stats)
    summary_path = output.with_name(f"{output.stem}_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_roi_manifest(imported_manifest: Path, output: Path, *, overwrite: bool = False) -> int:
    """Create the final ROI manifest while preserving ambiguous and rejected states."""
    if output.exists() and not overwrite:
        raise FileExistsError(f"El manifiesto ROI ya existe: {output}")
    rows, columns = read_csv_rows(imported_manifest)
    require_columns(columns, IMPORTED_ANNOTATION_COLUMNS, "imported annotation manifest")
    roi_rows: list[dict[str, object]] = []
    for row in rows:
        status = row["annotation_status"]
        annotated = status == "annotated"
        roi_rows.append(
            {
                "pilot_id": row["pilot_id"],
                "image_path": row["pilot_image_path"],
                "original_image_path": row["original_image_path"],
                "image_sha256": row["image_sha256"],
                "label": row["label"],
                "split": row["split"],
                "environment": row["environment"],
                "source_dataset": row["source_dataset"] or "unknown",
                "roi_x1": row["roi_x1"] if annotated else "",
                "roi_y1": row["roi_y1"] if annotated else "",
                "roi_x2": row["roi_x2"] if annotated else "",
                "roi_y2": row["roi_y2"] if annotated else "",
                "roi_width": row["roi_width"] if annotated else "",
                "roi_height": row["roi_height"] if annotated else "",
                "roi_area_ratio": row["roi_area_ratio"] if annotated else "",
                "roi_confidence": 1.0 if annotated else "",
                "roi_source": "manual" if status in MANUAL_STATUSES else "",
                "annotation_status": status,
                "notes": row["notes"] or row["annotation_warnings"],
            }
        )
    write_csv_rows(output, roi_rows, ROI_MANIFEST_COLUMNS)
    return len(roi_rows)


def _parse_finite_float(value: str, field: str) -> tuple[float | None, str | None]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None, f"{field} debe ser numérico"
    if not math.isfinite(parsed):
        return None, f"{field} debe ser finito"
    return parsed, None


def _resolve_roi_image(manifest_path: Path, row: dict[str, str], image_root: Path | None) -> Path:
    adapted = dict(row)
    adapted["pilot_image_path"] = row.get("image_path", "")
    return resolve_pilot_image(manifest_path, adapted, image_root)


def _validate_annotation_bbox(
    row: dict[str, str],
    image_width: int,
    image_height: int,
    min_area_ratio: float,
) -> tuple[list[str], list[str], BoundingBox | None, float | None]:
    errors: list[str] = []
    warnings: list[str] = []
    status = row.get("annotation_status", "")
    coordinate_fields = ("roi_x1", "roi_y1", "roi_x2", "roi_y2")
    if status != "annotated":
        roi_fields = (
            *coordinate_fields,
            "roi_width",
            "roi_height",
            "roi_area_ratio",
            "roi_confidence",
        )
        if any(row.get(field, "").strip() for field in roi_fields):
            errors.append("estados no anotados deben tener datos ROI vacíos")
        if status == "pending":
            errors.append("anotación pendiente")
        return errors, warnings, None, None

    values: list[float] = []
    for field in coordinate_fields:
        parsed, error = _parse_finite_float(row.get(field, ""), field)
        if error:
            errors.append(error)
        elif parsed is not None:
            values.append(parsed)
    if errors:
        return errors, warnings, None, None
    confidence, confidence_error = _parse_finite_float(
        row.get("roi_confidence", "1") or "1",
        "roi_confidence",
    )
    if confidence_error:
        errors.append(confidence_error)
        return errors, warnings, None, None
    detection = validate_bbox(
        image_width,
        image_height,
        values,
        min_area_ratio,
        confidence=confidence,
        source=row.get("roi_source", "manual") or "manual",
    )
    if not detection.detected or detection.bbox is None:
        errors.append(detection.reason or "bbox inválido")
        return errors, warnings, detection.bbox, detection.area_ratio
    if bbox_requires_clipping(values, image_width, image_height):
        errors.append("bbox fuera de los límites de la imagen")
    bbox = detection.bbox
    expected_width = bbox_width(bbox)
    expected_height = bbox_height(bbox)
    expected_ratio = bbox_area_ratio(bbox, image_width, image_height)
    parsed_width, width_error = _parse_finite_float(row.get("roi_width", ""), "roi_width")
    parsed_height, height_error = _parse_finite_float(row.get("roi_height", ""), "roi_height")
    parsed_ratio, ratio_error = _parse_finite_float(
        row.get("roi_area_ratio", ""), "roi_area_ratio"
    )
    errors.extend(error for error in (width_error, height_error, ratio_error) if error)
    if parsed_width is not None and parsed_width != expected_width:
        errors.append(f"roi_width inconsistente: {parsed_width:g} != {expected_width}")
    if parsed_height is not None and parsed_height != expected_height:
        errors.append(f"roi_height inconsistente: {parsed_height:g} != {expected_height}")
    if parsed_ratio is not None and not math.isclose(parsed_ratio, expected_ratio, rel_tol=1e-6):
        errors.append(f"roi_area_ratio inconsistente: {parsed_ratio:.8f} != {expected_ratio:.8f}")
    return errors, warnings, bbox, expected_ratio


def _coverage(rows: Sequence[dict[str, str]], areas: Sequence[float]) -> dict[str, object]:
    def counts(field: str) -> dict[str, int]:
        return dict(sorted(Counter(row.get(field, "") or "unknown" for row in rows).items()))

    area_summary: dict[str, float | int | None] = {
        "count": len(areas),
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
    }
    if areas:
        area_summary.update(
            {
                "min": min(areas),
                "max": max(areas),
                "mean": statistics.fmean(areas),
                "median": statistics.median(areas),
            }
        )
    return {
        "by_class": counts("label"),
        "by_split": counts("split"),
        "by_environment": counts("environment"),
        "by_status": counts("annotation_status"),
        "by_source_dataset": counts("source_dataset"),
        "area_ratio": area_summary,
    }


def _draw_box(image: Image.Image, bbox: BoundingBox, color: tuple[int, int, int]) -> Image.Image:
    canvas = image_to_rgb(image)
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = bbox
    draw.rectangle(
        (x1, y1, x2 - 1, y2 - 1),
        outline=color,
        width=max(2, min(image.size) // 200),
    )
    return canvas


def _fit_panel(image: Image.Image, size: tuple[int, int] = (240, 180)) -> Image.Image:
    image = image_to_rgb(image)
    image.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (25, 25, 25))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _write_preview(
    image: Image.Image,
    bbox: BoundingBox,
    pilot_id: str,
    output: Path,
    processor_config: LeafProcessorConfig,
) -> None:
    processor = LeafImageProcessor(processor_config)
    result = processor.process(image, bbox, source="manual")
    if result.processed_image is None or result.expanded_bbox is None:
        return
    panels = [
        _fit_panel(image),
        _fit_panel(_draw_box(image, bbox, (255, 210, 0))),
        _fit_panel(_draw_box(image, result.expanded_bbox, (0, 255, 100))),
        _fit_panel(crop_leaf_region(image, result.expanded_bbox)),
        _fit_panel(result.processed_image),
    ]
    titles = ("original", "bbox", "margen", "recorte", "letterbox")
    title_height = 24
    contact = Image.new("RGB", (sum(panel.width for panel in panels), 180 + title_height), "white")
    draw = ImageDraw.Draw(contact)
    font = ImageFont.load_default()
    x_offset = 0
    for panel, title in zip(panels, titles, strict=True):
        contact.paste(panel, (x_offset, title_height))
        draw.text((x_offset + 5, 5), title, fill="black", font=font)
        x_offset += panel.width
    output.mkdir(parents=True, exist_ok=True)
    contact.save(output / f"{pilot_id}_preview.jpg", quality=92)


def validate_roi_manifest_rows(
    manifest_path: Path,
    rows: Sequence[dict[str, str]],
    *,
    valid_classes: set[str],
    min_area_ratio: float,
    image_root: Path | None = None,
    preview_samples: int = 0,
    preview_output: Path | None = None,
    preview_seed: int = 42,
    processor_config: LeafProcessorConfig | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Validate every ROI row, detect leakage, and optionally render previews."""
    results: list[dict[str, object]] = []
    resolved_paths: list[str] = []
    hashes: list[str] = []
    areas: list[float] = []
    preview_candidates: list[tuple[Path, dict[str, str], BoundingBox]] = []
    for row in rows:
        errors: list[str] = []
        warnings: list[str] = []
        resolved_path = ""
        actual_hash = ""
        pilot_id = row.get("pilot_id", "")
        if row.get("label") not in valid_classes:
            errors.append(f"clase desconocida: {row.get('label', '')}")
        if row.get("split") not in VALID_SPLITS:
            errors.append(f"split inválido: {row.get('split', '')}")
        if row.get("environment") not in VALID_ENVIRONMENTS:
            errors.append(f"entorno inválido: {row.get('environment', '')}")
        if row.get("annotation_status") not in KNOWN_STATUSES:
            errors.append(f"annotation_status inválido: {row.get('annotation_status', '')}")
        try:
            image_path = _resolve_roi_image(manifest_path, row, image_root).resolve()
            resolved_path = str(image_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"imagen inexistente: {image_path}")
            actual_hash = sha256_file(image_path)
            if actual_hash != row.get("image_sha256", ""):
                errors.append("SHA-256 no coincide")
            image = load_and_normalize_image(image_path)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))
            bbox = None
            area = None
        else:
            bbox_errors, bbox_warnings, bbox, area = _validate_annotation_bbox(
                row,
                image.width,
                image.height,
                min_area_ratio,
            )
            errors.extend(bbox_errors)
            warnings.extend(bbox_warnings)
            if area is not None and row.get("annotation_status") == "annotated":
                areas.append(area)
            if bbox is not None and not bbox_errors:
                preview_candidates.append((image_path, row, bbox))
        resolved_paths.append(resolved_path)
        hashes.append(actual_hash)
        results.append(
            {
                "pilot_id": pilot_id,
                "valid": not errors,
                "errors": " | ".join(errors),
                "warnings": " | ".join(warnings),
            }
        )

    ids_to_indices: dict[str, list[int]] = defaultdict(list)
    paths_to_indices: dict[str, list[int]] = defaultdict(list)
    hashes_to_indices: dict[str, list[int]] = defaultdict(list)
    names_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        ids_to_indices[row.get("pilot_id", "")].append(index)
        if index < len(resolved_paths) and resolved_paths[index]:
            paths_to_indices[resolved_paths[index]].append(index)
        original_name = Path(
            row.get("original_image_path", "") or row.get("image_path", "")
        ).name.casefold()
        if original_name:
            names_to_indices[original_name].append(index)
        if index < len(hashes) and hashes[index]:
            hashes_to_indices[hashes[index]].append(index)

    def add_issue(indices: Sequence[int], message: str, *, warning: bool = False) -> None:
        field = "warnings" if warning else "errors"
        for index in indices:
            previous = str(results[index][field])
            results[index][field] = " | ".join(part for part in (previous, message) if part)
            if not warning:
                results[index]["valid"] = False

    for pilot_id, indices in ids_to_indices.items():
        if len(indices) > 1:
            add_issue(indices, f"pilot_id duplicado: {pilot_id}")
    for path, indices in paths_to_indices.items():
        if len(indices) > 1:
            add_issue(indices, f"ruta duplicada: {path}")
    cross_split_hashes = 0
    for digest, indices in hashes_to_indices.items():
        if len(indices) <= 1:
            continue
        splits = {rows[index].get("split", "") for index in indices}
        if len(splits) > 1:
            add_issue(indices, f"fuga por hash entre splits: {digest}")
            cross_split_hashes += 1
        else:
            add_issue(indices, f"hash duplicado dentro del split: {digest}", warning=True)
    duplicate_name_groups = 0
    for name, indices in names_to_indices.items():
        if len(indices) > 1:
            add_issue(
                indices,
                f"nombre repetido (advertencia, no fuga confirmada): {name}",
                warning=True,
            )
            duplicate_name_groups += 1

    if preview_samples > 0 and preview_output is not None:
        candidates = list(preview_candidates)
        random.Random(preview_seed).shuffle(candidates)
        config = processor_config or LeafProcessorConfig(min_area_ratio=min_area_ratio)
        for image_path, row, bbox in candidates[:preview_samples]:
            image = load_and_normalize_image(image_path)
            _write_preview(image, bbox, row["pilot_id"], preview_output, config)

    summary = {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "total_rows": len(rows),
        "valid_rows": sum(bool(result["valid"]) for result in results),
        "invalid_rows": sum(not bool(result["valid"]) for result in results),
        "coverage": _coverage(rows, areas),
        "leakage": {
            "duplicate_pilot_id_groups": sum(
                len(indices) > 1 for indices in ids_to_indices.values()
            ),
            "duplicate_path_groups": sum(len(indices) > 1 for indices in paths_to_indices.values()),
            "duplicate_hash_groups": sum(
                len(indices) > 1 for indices in hashes_to_indices.values()
            ),
            "cross_split_hash_groups": cross_split_hashes,
            "duplicate_filename_groups": duplicate_name_groups,
        },
        "preview_requested": preview_samples,
        "preview_generated": min(preview_samples, len(preview_candidates))
        if preview_output is not None
        else 0,
    }
    return results, summary


def validate_roi_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    valid_classes: set[str],
    min_area_ratio: float,
    image_root: Path | None = None,
    preview_samples: int = 0,
    preview_output: Path | None = None,
    preview_seed: int = 42,
    processor_config: LeafProcessorConfig | None = None,
) -> dict[str, object]:
    """Validate a final ROI CSV and persist row details and aggregate summary."""
    rows, columns = read_csv_rows(manifest_path)
    require_columns(columns, ROI_MANIFEST_COLUMNS, "ROI manifest")
    validation_rows, summary = validate_roi_manifest_rows(
        manifest_path,
        rows,
        valid_classes=valid_classes,
        min_area_ratio=min_area_ratio,
        image_root=image_root,
        preview_samples=preview_samples,
        preview_output=preview_output,
        preview_seed=preview_seed,
        processor_config=processor_config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "roi_validation_rows.csv",
        validation_rows,
        ROI_VALIDATION_COLUMNS,
    )
    (output_dir / "roi_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
