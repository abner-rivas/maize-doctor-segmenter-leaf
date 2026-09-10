# ruff: noqa: E501
"""End-to-end, auditable inference over recursively discovered image datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import TextIO, cast

import yaml
from PIL import Image, ImageDraw, ImageOps

from src.config import PROJECT_ROOT, get_output_root
from src.data.loader import load_and_normalize_image
from src.preprocessing.leaf_mask import binary_mask_image
from src.preprocessing.segmented_leaf_processor import (
    SegmentedLeafProcessingResult,
    SegmentedLeafProcessor,
    mask_processor_config_from_mapping,
)
from src.segmentation.leaf_segmenter import LeafInstance, UltralyticsLeafSegmenter
from src.segmentation.quality import (
    SegmentationQualityGateConfig,
    SegmentationStatus,
    assess_segmentation,
)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}
MANIFEST_COLUMNS = (
    "image_id",
    "source_path",
    "relative_path",
    "split",
    "class",
    "environment",
    "filename",
    "sha256",
    "duplicate_of",
    "image_width",
    "image_height",
    "status",
    "reliability_status",
    "rejection_reason",
    "quality_gate_reasons",
    "selected_proposal",
    "num_proposals",
    "eligible_proposals",
    "confidence",
    "selection_margin",
    "selected_score",
    "mask_area_px",
    "mask_area_ratio",
    "bbox_area_ratio",
    "mask_bbox_ratio",
    "normalized_perimeter",
    "output_mode",
    "original_path",
    "prediction_path",
    "mask_path",
    "overlay_path",
    "segmented_path",
    "full_image_path",
    "final_path",
    "metadata_path",
    "original_thumbnail_path",
    "mask_thumbnail_path",
    "overlay_thumbnail_path",
    "final_thumbnail_path",
    "inference_time_ms",
    "selection_traces",
    "warnings",
    "error",
)
INVENTORY_COLUMNS = (
    "image_id",
    "source_path",
    "relative_path",
    "split",
    "class",
    "environment",
    "filename",
    "extension",
    "sha256",
    "duplicate_of",
    "image_width",
    "image_height",
    "valid",
    "error",
)


@dataclass(frozen=True)
class DatasetImage:
    """One discovered image plus immutable inventory evidence."""

    image_id: str
    source_path: str
    relative_path: str
    split: str
    class_name: str
    environment: str
    filename: str
    extension: str
    sha256: str
    duplicate_of: str
    image_width: int | None
    image_height: int | None
    valid: bool
    error: str

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["class"] = row.pop("class_name")
        return row


class CapturingSegmenter:
    """Transparent delegate that retains the latest raw model proposals for audit."""

    def __init__(self, delegate: UltralyticsLeafSegmenter) -> None:
        self.delegate = delegate
        self.last_instances: tuple[LeafInstance, ...] = ()

    def segment(self, image: Image.Image) -> tuple[LeafInstance, ...]:
        self.last_instances = tuple(self.delegate.segment(image))
        return self.last_instances

    def to_metadata(self) -> dict[str, object]:
        return self.delegate.to_metadata()


@dataclass(frozen=True)
class InferenceRuntime:
    processor: SegmentedLeafProcessor
    segmenter: CapturingSegmenter
    quality_gate: SegmentationQualityGateConfig
    reject_multiple_eligible: bool
    config: dict[str, object]
    config_path: Path
    checkpoint: Path
    device: str | int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_id(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20]
    return f"image_{digest}"


def discover_image_paths(dataset_root: Path, *, limit: int | None = None) -> list[Path]:
    root = dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"No existe la raíz del dataset: {root}")
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit debe ser positivo")
        paths = paths[:limit]
    return paths


def _load_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"No existe metadata: {resolved}")
    with resolved.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        path_field = next(
            (
                candidate
                for candidate in ("file_name", "relative_path", "path", "filename")
                if candidate in fields
            ),
            None,
        )
        if path_field is None:
            raise ValueError(f"Metadata sin columna de ruta reconocible: {resolved}")
        metadata: dict[str, dict[str, str]] = {}
        for raw in reader:
            key = str(raw.get(path_field, "")).strip().replace("\\", "/").lstrip("./")
            if key:
                metadata[key] = {str(name): str(value or "") for name, value in raw.items()}
    info_path = resolved.with_name("dataset_infos.json")
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        configurations = [value for value in info.values() if isinstance(value, dict)]
        if len(configurations) == 1:
            splits = configurations[0].get("splits")
            if isinstance(splits, dict) and len(splits) == 1:
                default_split = str(next(iter(splits)))
                for row in metadata.values():
                    row.setdefault("split", default_split)
    return metadata


def _metadata_for(relative_path: str, metadata: Mapping[str, dict[str, str]]) -> dict[str, str]:
    direct = metadata.get(relative_path)
    if direct is not None:
        return direct
    clean_prefixed = metadata.get(f"clean/{relative_path}")
    if clean_prefixed is not None:
        return clean_prefixed
    return {}


def _labels_from_path(
    relative_path: str,
    metadata: Mapping[str, dict[str, str]],
) -> tuple[str, str, str]:
    parts = Path(relative_path).parts
    split = ""
    split_index: int | None = None
    for index, part in enumerate(parts[:-1]):
        normalized = SPLIT_ALIASES.get(part.lower())
        if normalized is not None:
            split = normalized
            split_index = index
            break
    meta = _metadata_for(relative_path, metadata)
    class_name = str(meta.get("label") or meta.get("class") or meta.get("class_name") or "")
    environment = str(meta.get("environment") or meta.get("domain") or "")
    if not class_name:
        if split_index is not None and split_index + 1 < len(parts) - 1:
            class_name = parts[split_index + 1]
        elif len(parts) >= 2:
            class_name = parts[0]
    if not environment and len(parts) >= 3:
        environment = parts[-2]
    metadata_split = str(meta.get("split") or meta.get("subset") or "").lower()
    if not split and metadata_split:
        split = SPLIT_ALIASES.get(metadata_split, metadata_split)
    return split, class_name, environment


def inventory_dataset(
    dataset_root: Path,
    *,
    metadata_path: Path | None = None,
    limit: int | None = None,
    progress_every: int = 250,
) -> tuple[list[DatasetImage], dict[str, object]]:
    root = dataset_root.resolve()
    metadata = _load_metadata(metadata_path)
    paths = discover_image_paths(root, limit=limit)
    all_files = sum(1 for path in root.rglob("*") if path.is_file())
    first_by_hash: dict[str, str] = {}
    records: list[DatasetImage] = []
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(root).as_posix()
        split, class_name, environment = _labels_from_path(relative, metadata)
        image_hash = ""
        width: int | None = None
        height: int | None = None
        error = ""
        valid = False
        try:
            image_hash = sha256_file(path)
            with Image.open(path) as opened:
                opened.verify()
            with Image.open(path) as opened:
                oriented = ImageOps.exif_transpose(opened)
                width, height = oriented.size
                if width <= 0 or height <= 0:
                    raise ValueError(f"dimensiones inválidas: {width}x{height}")
            valid = True
        except Exception as exc:  # per-file isolation is intentional
            error = f"{type(exc).__name__}: {exc}"
        duplicate_of = first_by_hash.get(image_hash, "") if image_hash else ""
        if image_hash and not duplicate_of:
            first_by_hash[image_hash] = relative
        records.append(
            DatasetImage(
                image_id=_image_id(relative),
                source_path=str(path.resolve()),
                relative_path=relative,
                split=split,
                class_name=class_name,
                environment=environment,
                filename=path.name,
                extension=path.suffix.lower(),
                sha256=image_hash,
                duplicate_of=duplicate_of,
                image_width=width,
                image_height=height,
                valid=valid,
                error=error,
            )
        )
        if progress_every > 0 and (index % progress_every == 0 or index == len(paths)):
            print(f"Inventory: {index}/{len(paths)}", flush=True)
    split_counts = Counter(record.split for record in records if record.split)
    class_counts = Counter(record.class_name for record in records if record.class_name)
    extension_counts = Counter(record.extension for record in records)
    duplicate_records = sum(bool(record.duplicate_of) for record in records)
    duplicate_hashes = Counter(record.sha256 for record in records if record.sha256)
    summary: dict[str, object] = {
        "dataset_root": str(root),
        "metadata_path": str(metadata_path.resolve()) if metadata_path else None,
        "total_files": all_files,
        "image_files": len(records),
        "valid_images": sum(record.valid for record in records),
        "invalid_images": sum(not record.valid for record in records),
        "duplicates": duplicate_records,
        "duplicate_groups": sum(count > 1 for count in duplicate_hashes.values()),
        "splits": dict(sorted(split_counts.items())),
        "classes": dict(sorted(class_counts.items())),
        "extensions": dict(sorted(extension_counts.items())),
    }
    return records, summary


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_inventory(output_dir: Path, records: Sequence[DatasetImage], summary: object) -> None:
    csv_path = output_dir / "inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        inventory_columns: list[str] = list(INVENTORY_COLUMNS)
        writer = csv.DictWriter(handle, fieldnames=inventory_columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    jsonl_path = output_dir / "inventory.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_row(), ensure_ascii=False, allow_nan=False) + "\n")
    _write_json(output_dir / "inventory_summary.json", summary)


def load_inventory(output_dir: Path) -> tuple[list[DatasetImage], dict[str, object]]:
    records: list[DatasetImage] = []
    with (output_dir / "inventory.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row["class_name"] = row.pop("class")
            records.append(DatasetImage(**row))
    summary = json.loads((output_dir / "inventory_summary.json").read_text(encoding="utf-8"))
    return records, summary


def load_resume_manifest(
    output_dir: Path,
    records: Sequence[DatasetImage],
) -> list[dict[str, object]]:
    """Load a crash-safe manifest after proving it is an inventory prefix."""

    csv_path = output_dir / "manifest.csv"
    jsonl_path = output_dir / "manifest.jsonl"
    if not csv_path.is_file() or not jsonl_path.is_file():
        raise FileNotFoundError("Resume requiere manifest.csv y manifest.jsonl")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError("Las columnas de manifest.csv no coinciden con el esquema vigente")
        csv_rows = list(reader)
    jsonl_rows: list[dict[str, object]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Línea vacía en manifest.jsonl: {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Registro JSONL inválido en línea {line_number}")
            jsonl_rows.append(value)
    if len(csv_rows) != len(jsonl_rows):
        raise ValueError(
            "Los manifests no están alineados: "
            f"CSV={len(csv_rows)}, JSONL={len(jsonl_rows)}"
        )
    if len(jsonl_rows) > len(records):
        raise ValueError("El manifest contiene más registros que el inventario")
    expected = [(record.image_id, record.relative_path) for record in records[: len(jsonl_rows)]]
    csv_prefix = [(row.get("image_id", ""), row.get("relative_path", "")) for row in csv_rows]
    jsonl_prefix = [
        (str(row.get("image_id", "")), str(row.get("relative_path", "")))
        for row in jsonl_rows
    ]
    if csv_prefix != jsonl_prefix:
        raise ValueError("manifest.csv y manifest.jsonl no describen los mismos registros")
    if jsonl_prefix != expected:
        raise ValueError("El manifest no es un prefijo exacto y ordenado del inventario")
    return jsonl_rows


def load_segmentation_config(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("La configuración de segmentación debe ser un mapping")
    return loaded


def resolve_device(device: str | None) -> str | int:
    if device is not None:
        normalized = device.strip()
        return int(normalized) if normalized.isdigit() else normalized
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_runtime(config_path: Path, *, device: str | None = None) -> InferenceRuntime:
    resolved_config = config_path.resolve()
    config = load_segmentation_config(resolved_config)
    segmentation = config.get("segmentation")
    if not isinstance(segmentation, Mapping):
        raise ValueError("segmentation debe ser un mapping")
    quality = segmentation.get("quality_gate")
    if not isinstance(quality, Mapping):
        raise ValueError("segmentation.quality_gate debe ser un mapping")
    reject_multiple = quality.get("reject_multiple_eligible")
    if not isinstance(reject_multiple, bool):
        raise ValueError("reject_multiple_eligible debe ser booleano")
    raw_proposal_confidence = segmentation.get(
        "proposal_confidence_threshold",
        segmentation.get("confidence_threshold"),
    )
    if raw_proposal_confidence is None:
        raise ValueError("falta proposal_confidence_threshold")
    checkpoint = (get_output_root() / str(segmentation["checkpoint"])).resolve()
    selected_device = resolve_device(device)
    delegate = UltralyticsLeafSegmenter(
        checkpoint,
        image_size=int(segmentation["image_size"]),
        proposal_confidence_threshold=float(raw_proposal_confidence),
        iou_threshold=float(segmentation["iou_threshold"]),
        max_detections=int(segmentation["max_detections"]),
        device=selected_device,
        expected_version=str(segmentation["ultralytics_version"]),
    )
    segmenter = CapturingSegmenter(delegate)
    processor = SegmentedLeafProcessor(
        segmenter,
        mask_processor_config_from_mapping(segmentation),
    )
    return InferenceRuntime(
        processor=processor,
        segmenter=segmenter,
        quality_gate=SegmentationQualityGateConfig.from_mapping(quality),
        reject_multiple_eligible=reject_multiple,
        config=config,
        config_path=resolved_config,
        checkpoint=checkpoint,
        device=selected_device,
    )


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source.resolve(), start=destination.parent.resolve()))


def _save_jpeg(image: Image.Image, path: Path, *, quality: int = 88) -> None:
    image.convert("RGB").save(path, format="JPEG", quality=quality)


def _overlay(original: Image.Image, mask: Image.Image | None) -> Image.Image:
    if mask is None:
        return original.copy()
    tint = Image.new("RGB", original.size, (0, 190, 90))
    return Image.composite(Image.blend(original, tint, 0.42), original, mask)


def _proposal_visualization(
    original: Image.Image,
    instances: Sequence[LeafInstance],
    selected_index: int | None,
) -> Image.Image:
    canvas = original.copy().convert("RGB")
    palette = ((0, 190, 90), (255, 175, 0), (20, 130, 255), (220, 60, 180))
    draw = ImageDraw.Draw(canvas)
    for position, instance in enumerate(instances):
        source_index = int(getattr(instance, "source_index", position))
        try:
            mask = binary_mask_image(instance.mask, expected_size=original.size)
        except Exception:
            draw.text(
                (8, 8 + 18 * position),
                f"#{source_index} invalid proposal mask",
                fill=(255, 40, 40),
                stroke_width=2,
                stroke_fill="black",
            )
            continue
        color = (255, 40, 40) if source_index == selected_index else palette[position % len(palette)]
        tint = Image.new("RGB", original.size, color)
        canvas = Image.composite(Image.blend(canvas, tint, 0.32), canvas, mask)
        draw = ImageDraw.Draw(canvas)
        left, top, right, bottom = instance.bbox
        width = 5 if source_index == selected_index else 2
        draw.rectangle((left, top, right, bottom), outline=color, width=width)
        draw.text(
            (left + 4, top + 4),
            f"#{source_index} {instance.confidence:.3f}",
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )
    return canvas


def _thumbnail(image: Image.Image, path: Path, *, size: tuple[int, int] = (320, 240)) -> None:
    contained = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (24, 28, 34))
    offset = ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2)
    canvas.paste(contained, offset)
    _save_jpeg(canvas, path, quality=78)


def _selected_trace(result: SegmentedLeafProcessingResult) -> dict[str, object]:
    for trace in result.selection_traces:
        if trace.source_index == result.selected_instance:
            return trace.to_metadata()
    return {}


def _proposal_metadata(instances: Sequence[LeafInstance]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position, instance in enumerate(instances):
        try:
            area_ratio: float | None = instance.area_ratio
            mask_error = None
        except Exception as exc:
            area_ratio = None
            mask_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "source_index": int(getattr(instance, "source_index", position)),
                "confidence": instance.confidence,
                "class_id": instance.class_id,
                "bbox": list(instance.bbox),
                "mask_area_ratio": area_ratio,
                "mask_error": mask_error,
            }
        )
    return rows


def _base_manifest_row(record: DatasetImage) -> dict[str, object]:
    return {
        "image_id": record.image_id,
        "source_path": record.source_path,
        "relative_path": record.relative_path,
        "split": record.split,
        "class": record.class_name,
        "environment": record.environment,
        "filename": record.filename,
        "sha256": record.sha256,
        "duplicate_of": record.duplicate_of,
        "image_width": record.image_width,
        "image_height": record.image_height,
        **{name: "" for name in MANIFEST_COLUMNS[11:]},
    }


def _artifact_path(output_dir: Path, relative: str) -> Path:
    return output_dir / relative


def _process_one(
    record: DatasetImage,
    runtime: InferenceRuntime,
    output_dir: Path,
) -> dict[str, object]:
    row = _base_manifest_row(record)
    if not record.valid:
        row.update(
            {
                "status": "error",
                "reliability_status": "error",
                "error": record.error or "imagen inválida",
            }
        )
        error_path = output_dir / "errors" / f"{record.image_id}.txt"
        error_path.write_text(str(row["error"]) + "\n", encoding="utf-8")
        return row

    source = Path(record.source_path)
    try:
        original = load_and_normalize_image(str(source))
        started = time.perf_counter()
        processing = runtime.processor.process(original, source_image=source)
        inference_time_ms = (time.perf_counter() - started) * 1000.0
        instances = runtime.segmenter.last_instances
        assessment = assess_segmentation(
            processing,
            reject_multiple_eligible=runtime.reject_multiple_eligible,
            quality_gate=runtime.quality_gate,
        )
        metrics = assessment.quality_gate_metrics
        selected = _selected_trace(processing)
        output_mode = (
            "segmented" if assessment.status is SegmentationStatus.RELIABLE else "full_image"
        )

        original_relative = f"original/{record.image_id}{record.extension}"
        _relative_symlink(source, _artifact_path(output_dir, original_relative))

        mask = processing.mask or Image.new("L", original.size, 0)
        mask_relative = f"masks/{record.image_id}.png"
        mask.save(_artifact_path(output_dir, mask_relative), format="PNG", compress_level=1)

        overlay = _overlay(original, processing.mask)
        overlay_relative = f"overlays/{record.image_id}.jpg"
        _save_jpeg(overlay, _artifact_path(output_dir, overlay_relative))

        prediction = _proposal_visualization(original, instances, processing.selected_instance)
        prediction_relative = f"predictions/{record.image_id}.jpg"
        _save_jpeg(prediction, _artifact_path(output_dir, prediction_relative))

        segmented_relative = ""
        if processing.masked_image is not None:
            segmented_relative = f"segmented/{record.image_id}.jpg"
            _save_jpeg(
                processing.masked_image,
                _artifact_path(output_dir, segmented_relative),
            )

        full_image_relative = ""
        if output_mode == "full_image":
            full_image_relative = f"full_image/{record.image_id}{record.extension}"
            _relative_symlink(source, _artifact_path(output_dir, full_image_relative))
        final_relative = segmented_relative if output_mode == "segmented" else full_image_relative
        if not final_relative:
            raise RuntimeError("el pipeline no produjo un resultado final")

        category_relative = f"{assessment.status.value}/{record.image_id}{Path(final_relative).suffix}"
        _relative_symlink(
            _artifact_path(output_dir, final_relative),
            _artifact_path(output_dir, category_relative),
        )

        thumb_root = "gallery/thumbnails"
        original_thumb = f"{thumb_root}/{record.image_id}_original.jpg"
        mask_thumb = f"{thumb_root}/{record.image_id}_mask.jpg"
        overlay_thumb = f"{thumb_root}/{record.image_id}_overlay.jpg"
        final_thumb = f"{thumb_root}/{record.image_id}_final.jpg"
        _thumbnail(original, _artifact_path(output_dir, original_thumb))
        _thumbnail(mask, _artifact_path(output_dir, mask_thumb))
        _thumbnail(overlay, _artifact_path(output_dir, overlay_thumb))
        final_image = processing.masked_image if output_mode == "segmented" else original
        if final_image is None:
            final_image = original
        _thumbnail(final_image, _artifact_path(output_dir, final_thumb))

        metadata_relative = f"metadata/{record.image_id}.json"
        structured = {
            "inventory": record.to_row(),
            "model_prediction": _proposal_metadata(instances),
            "processing": processing.to_metadata(),
            "reliability_assessment": assessment.to_metadata(),
            "output_mode": output_mode,
            "artifacts": {
                "original": original_relative,
                "prediction": prediction_relative,
                "mask": mask_relative,
                "overlay": overlay_relative,
                "segmented": segmented_relative or None,
                "full_image": full_image_relative or None,
                "final": final_relative,
            },
            "inference_time_ms": inference_time_ms,
        }
        _write_json(_artifact_path(output_dir, metadata_relative), structured)
        row.update(
            {
                "status": "processed",
                "reliability_status": assessment.status.value,
                "rejection_reason": assessment.reason or "",
                "quality_gate_reasons": json.dumps(
                    list(assessment.quality_gate_reasons), ensure_ascii=False
                ),
                "selected_proposal": processing.selected_instance,
                "num_proposals": processing.number_of_instances,
                "eligible_proposals": assessment.eligible_instances,
                "confidence": processing.confidence,
                "selection_margin": metrics.get("instance_score_margin"),
                "selected_score": selected.get("score"),
                "mask_area_px": metrics.get("area_pixels"),
                "mask_area_ratio": processing.mask_area_ratio,
                "bbox_area_ratio": metrics.get("bbox_area_ratio"),
                "mask_bbox_ratio": metrics.get("mask_bbox_ratio"),
                "normalized_perimeter": metrics.get("normalized_perimeter"),
                "output_mode": output_mode,
                "original_path": original_relative,
                "prediction_path": prediction_relative,
                "mask_path": mask_relative,
                "overlay_path": overlay_relative,
                "segmented_path": segmented_relative,
                "full_image_path": full_image_relative,
                "final_path": final_relative,
                "metadata_path": metadata_relative,
                "original_thumbnail_path": original_thumb,
                "mask_thumbnail_path": mask_thumb,
                "overlay_thumbnail_path": overlay_thumb,
                "final_thumbnail_path": final_thumb,
                "inference_time_ms": inference_time_ms,
                "selection_traces": json.dumps(
                    [trace.to_metadata() for trace in processing.selection_traces],
                    ensure_ascii=False,
                ),
                "warnings": json.dumps(list(processing.warnings), ensure_ascii=False),
                "error": "",
            }
        )
        return row
    except Exception as exc:  # one bad image must not stop the dataset
        runtime.segmenter.last_instances = ()
        message = f"{type(exc).__name__}: {exc}"
        row.update(
            {
                "status": "error",
                "reliability_status": "error",
                "error": message,
            }
        )
        error_path = output_dir / "errors" / f"{record.image_id}.txt"
        error_path.write_text(
            message + "\n\n" + traceback.format_exc(),
            encoding="utf-8",
        )
        return row


def _jsonl_row(row: Mapping[str, object]) -> dict[str, object]:
    result = dict(row)
    for name in ("quality_gate_reasons", "selection_traces", "warnings"):
        value = result.get(name)
        if isinstance(value, str) and value:
            result[name] = json.loads(value)
        elif value == "":
            result[name] = []
    return result


def _group_summary(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    result: dict[str, object] = {}
    for name, group in sorted(grouped.items()):
        result[name] = {
            "total": len(group),
            "processed": sum(row.get("status") == "processed" for row in group),
            "errors": sum(row.get("status") == "error" for row in group),
            "reliable": sum(row.get("reliability_status") == "reliable" for row in group),
            "uncertain": sum(row.get("reliability_status") == "uncertain" for row in group),
            "failed": sum(row.get("reliability_status") == "failed" for row in group),
            "segmented": sum(row.get("output_mode") == "segmented" for row in group),
            "full_image": sum(row.get("output_mode") == "full_image" for row in group),
        }
    return result


def build_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    inventory_summary: Mapping[str, object],
    total_wall_seconds: float,
) -> dict[str, object]:
    processed = [row for row in rows if row.get("status") == "processed"]
    errors = [row for row in rows if row.get("status") == "error"]
    reliability = Counter(str(row.get("reliability_status")) for row in processed)
    output_modes = Counter(str(row.get("output_mode")) for row in processed)
    rejection_reasons = Counter(
        str(row["rejection_reason"])
        for row in processed
        if row.get("rejection_reason") not in (None, "")
    )
    def numeric_values(name: str) -> list[float]:
        values: list[float] = []
        for row in processed:
            value = row.get(name)
            if isinstance(value, (int, float, str)) and value != "":
                values.append(float(value))
        return values

    ratios = numeric_values("mask_area_ratio")
    areas = numeric_values("mask_area_px")
    timings = numeric_values("inference_time_ms")
    denominator = len(processed)
    return {
        "schema_version": 1,
        "inventory": dict(inventory_summary),
        "images_found": len(rows),
        "images_processed": len(processed),
        "errors": len(errors),
        "reliable": reliability["reliable"],
        "uncertain": reliability["uncertain"],
        "failed": reliability["failed"],
        "reliability_rate_percent": (
            reliability["reliable"] * 100.0 / denominator if denominator else 0.0
        ),
        "uncertain_rate_percent": (
            reliability["uncertain"] * 100.0 / denominator if denominator else 0.0
        ),
        "segmented": output_modes["segmented"],
        "full_image": output_modes["full_image"],
        "no_detection": rejection_reasons["no_detection"],
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "masks": {
            "count": len(ratios),
            "area_px_mean": fmean(areas) if areas else None,
            "mask_ratio_mean": fmean(ratios) if ratios else None,
            "mask_ratio_min": min(ratios) if ratios else None,
            "mask_ratio_max": max(ratios) if ratios else None,
        },
        "time": {
            "total_seconds": total_wall_seconds,
            "inference_mean_ms": fmean(timings) if timings else None,
            "inference_sum_seconds": sum(timings) / 1000.0,
        },
        "by_class": _group_summary(rows, "class"),
        "by_split": _group_summary(rows, "split"),
        "error_details": [
            {
                "image_id": row.get("image_id"),
                "relative_path": row.get("relative_path"),
                "error": row.get("error"),
            }
            for row in errors
        ],
    }


def _gallery_record(row: Mapping[str, object]) -> dict[str, object]:
    def gallery_relative(value: object) -> str:
        return f"../{value}" if value else ""

    def thumbnail_relative(value: object) -> str:
        text = str(value or "")
        prefix = "gallery/"
        return text[len(prefix) :] if text.startswith(prefix) else text

    return {
        "image_id": row.get("image_id"),
        "filename": row.get("filename"),
        "relative_path": row.get("relative_path"),
        "class": row.get("class") or "unknown",
        "split": row.get("split") or "unknown",
        "status": row.get("status"),
        "reliability": row.get("reliability_status"),
        "reason": row.get("rejection_reason") or "",
        "confidence": row.get("confidence"),
        "mask_ratio": row.get("mask_area_ratio"),
        "output_mode": row.get("output_mode"),
        "error": row.get("error") or "",
        "original_thumb": thumbnail_relative(row.get("original_thumbnail_path")),
        "mask_thumb": thumbnail_relative(row.get("mask_thumbnail_path")),
        "overlay_thumb": thumbnail_relative(row.get("overlay_thumbnail_path")),
        "final_thumb": thumbnail_relative(row.get("final_thumbnail_path")),
        "original": gallery_relative(row.get("original_path")),
        "mask": gallery_relative(row.get("mask_path")),
        "overlay": gallery_relative(row.get("overlay_path")),
        "final": gallery_relative(row.get("final_path")),
    }


GALLERY_HTML = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Doctor Maíz — segmentación del dataset completo</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #101419; color: #eef2f5; }
    header { position: sticky; top: 0; z-index: 2; background: #151b22f5; padding: 16px; border-bottom: 1px solid #34404c; }
    h1 { margin: 0 0 12px; font-size: 1.25rem; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; }
    button, select, input { color: inherit; background: #222b35; border: 1px solid #465462; border-radius: 6px; padding: 8px 10px; }
    button.active { background: #176b45; border-color: #36b979; }
    #count { margin: 10px 0 0; color: #aeb9c4; }
    main { padding: 16px; }
    #cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; }
    .card { background: #171e26; border: 1px solid #303b46; border-radius: 10px; overflow: hidden; }
    .tiles { display: grid; grid-template-columns: 1fr 1fr; }
    figure { margin: 0; border: 1px solid #222b35; background: #0b0e12; }
    figure img { width: 100%; aspect-ratio: 4/3; object-fit: contain; display: block; }
    figcaption { padding: 5px 8px; font-size: .75rem; color: #bcc7d1; }
    .meta { padding: 10px 12px; font-size: .86rem; line-height: 1.5; overflow-wrap: anywhere; }
    .meta strong { color: #fff; }
    .reliable { border-color: #25885a; }
    .uncertain { border-color: #a87920; }
    .failed, .error { border-color: #a13b3b; }
    .pager { display: flex; justify-content: center; align-items: center; gap: 10px; margin: 18px; }
    .missing { display: grid; place-items: center; min-height: 160px; color: #7f8b96; }
  </style>
</head>
<body>
<header>
  <h1>Doctor Maíz — segmentación del dataset completo</h1>
  <div class="controls" id="statusButtons">
    <button data-status="all" class="active">Todos</button>
    <button data-status="reliable">Reliable</button>
    <button data-status="uncertain">Uncertain</button>
    <button data-status="failed">Failed</button>
    <button data-status="error">Errores</button>
    <select id="classFilter"><option value="all">Todas las clases</option></select>
    <select id="splitFilter"><option value="all">Todos los splits</option></select>
    <select id="reasonFilter"><option value="all">Todos los motivos</option></select>
    <input id="search" type="search" placeholder="Buscar archivo o ruta">
  </div>
  <div id="count"></div>
</header>
<main>
  <div id="cards"></div>
  <div class="pager"><button id="prev">Anterior</button><span id="page"></span><button id="next">Siguiente</button></div>
</main>
<script src="data.js"></script>
<script>
const pageSize = 100;
let page = 1, status = "all";
const cards = document.querySelector("#cards");
const esc = value => String(value ?? "").replace(/[&<>\"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'\"':"&quot;","'":"&#39;"}[c]));
const fmt = value => value === "" || value == null ? "—" : (typeof value === "number" ? value.toFixed(4) : value);
function options(id, key) {
  const select = document.querySelector(id);
  [...new Set(window.GALLERY_DATA.map(row => row[key]).filter(Boolean))].sort().forEach(value => {
    const option = document.createElement("option"); option.value = value; option.textContent = value; select.append(option);
  });
}
options("#classFilter", "class"); options("#splitFilter", "split"); options("#reasonFilter", "reason");
function imageTile(label, thumb, full) {
  if (!thumb) return `<figure><div class="missing">Sin artefacto</div><figcaption>${label}</figcaption></figure>`;
  return `<figure><a href="${esc(full || thumb)}"><img loading="lazy" src="${esc(thumb)}" alt="${label}"></a><figcaption>${label}</figcaption></figure>`;
}
function filtered() {
  const cls = document.querySelector("#classFilter").value;
  const split = document.querySelector("#splitFilter").value;
  const reason = document.querySelector("#reasonFilter").value;
  const query = document.querySelector("#search").value.trim().toLowerCase();
  return window.GALLERY_DATA.filter(row => {
    const statusMatch = status === "all" || (status === "error" ? row.status === "error" : row.reliability === status);
    return statusMatch && (cls === "all" || row.class === cls) && (split === "all" || row.split === split)
      && (reason === "all" || row.reason === reason)
      && (!query || `${row.filename} ${row.relative_path}`.toLowerCase().includes(query));
  });
}
function render() {
  const rows = filtered();
  const pages = Math.max(1, Math.ceil(rows.length / pageSize)); page = Math.min(page, pages);
  const visible = rows.slice((page - 1) * pageSize, page * pageSize);
  cards.innerHTML = visible.map(row => `<article class="card ${esc(row.status === "error" ? "error" : row.reliability)}">
    <div class="tiles">${imageTile("Original", row.original_thumb, row.original)}${imageTile("Máscara", row.mask_thumb, row.mask)}${imageTile("Overlay", row.overlay_thumb, row.overlay)}${imageTile("Resultado final", row.final_thumb, row.final)}</div>
    <div class="meta"><strong>${esc(row.filename)}</strong><br>${esc(row.relative_path)}<br>
    clase=${esc(row.class)} · split=${esc(row.split)} · status=${esc(row.status)}<br>
    reliability=${esc(row.reliability)} · output=${esc(row.output_mode || "—")}<br>
    confidence=${esc(fmt(row.confidence))} · mask_ratio=${esc(fmt(row.mask_ratio))}<br>
    motivo=${esc(row.reason || "—")}${row.error ? `<br>error=${esc(row.error)}` : ""}</div></article>`).join("");
  document.querySelector("#count").textContent = `${rows.length.toLocaleString()} resultados`; document.querySelector("#page").textContent = `${page} / ${pages}`;
  document.querySelector("#prev").disabled = page <= 1; document.querySelector("#next").disabled = page >= pages;
}
document.querySelectorAll("#statusButtons button[data-status]").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll("#statusButtons button[data-status]").forEach(item => item.classList.remove("active")); button.classList.add("active"); status = button.dataset.status; page = 1; render();
}));
["#classFilter", "#splitFilter", "#reasonFilter", "#search"].forEach(id => document.querySelector(id).addEventListener("input", () => { page = 1; render(); }));
document.querySelector("#prev").addEventListener("click", () => { page--; render(); }); document.querySelector("#next").addEventListener("click", () => { page++; render(); });
render();
</script>
</body>
</html>
"""


def write_gallery(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    gallery = output_dir / "gallery"
    data = [_gallery_record(row) for row in rows]
    (gallery / "data.js").write_text(
        "window.GALLERY_DATA = "
        + json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    (gallery / "index.html").write_text(GALLERY_HTML, encoding="utf-8")


def _git_value(args: Sequence[str], *, cwd: Path) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def build_run_metadata(
    runtime: InferenceRuntime,
    *,
    dataset_root: Path,
    dataset_source: str,
    dataset_commit: str,
    source_repository: Path | None,
    command: str,
    inventory_summary: Mapping[str, object],
) -> dict[str, object]:
    segmenter_metadata = runtime.segmenter.to_metadata()
    project_status = _git_value(("status", "--porcelain"), cwd=PROJECT_ROOT)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "project": {
            "root": str(PROJECT_ROOT),
            "commit": _git_value(("rev-parse", "HEAD"), cwd=PROJECT_ROOT),
            "branch": _git_value(("branch", "--show-current"), cwd=PROJECT_ROOT),
            "dirty": bool(project_status),
        },
        "dataset": {
            "root": str(dataset_root.resolve()),
            "source": dataset_source,
            "commit": dataset_commit,
            "source_repository": str(source_repository.resolve()) if source_repository else None,
            "inventory": dict(inventory_summary),
        },
        "checkpoint": {
            "path": str(runtime.checkpoint),
            "sha256": segmenter_metadata["segmenter_checkpoint_sha256"],
        },
        "segmenter": segmenter_metadata,
        "quality_gate": {
            "thresholds": runtime.quality_gate.to_metadata(),
            "reject_multiple_eligible": runtime.reject_multiple_eligible,
        },
        "config": {
            "path": str(runtime.config_path),
            "sha256": sha256_file(runtime.config_path),
            "values": runtime.config,
        },
        "runtime": {
            "python": sys.version,
            "device": runtime.device,
        },
    }


def _create_output_directories(output_dir: Path) -> None:
    for name in (
        "original",
        "predictions",
        "masks",
        "overlays",
        "segmented",
        "full_image",
        "reliable",
        "uncertain",
        "failed",
        "errors",
        "metadata",
        "gallery/thumbnails",
        "runtime_cache/matplotlib",
        "runtime_cache/ultralytics",
    ):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def write_readme(
    output_dir: Path,
    *,
    summary: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    checkpoint = cast(Mapping[str, object], metadata["checkpoint"])
    dataset = cast(Mapping[str, object], metadata["dataset"])
    content = f"""# Doctor Maíz — full dataset segmentation

Ejecución auditable del pipeline vigente sobre todas las imágenes válidas del dataset.
No se entrenó ningún modelo ni se modificaron thresholds, selección o quality gate.

## Provenance

- Dataset: `{dataset['source']}`
- Dataset commit: `{dataset['commit']}`
- Dataset root: `{dataset['root']}`
- Checkpoint: `{checkpoint['path']}`
- Checkpoint SHA256: `{checkpoint['sha256']}`
- Comando: `{metadata['command']}`

## Resultado

- Encontradas: {summary['images_found']}
- Procesadas: {summary['images_processed']}
- Errores: {summary['errors']}
- Reliable: {summary['reliable']}
- Uncertain: {summary['uncertain']}
- Failed: {summary['failed']}
- Segmented: {summary['segmented']}
- Full image: {summary['full_image']}

Abra `gallery/index.html` directamente en un navegador. Los manifests CSV/JSONL,
metadata por imagen y artefactos de predicción, máscara, overlay y resultado final
utilizan rutas relativas a este directorio.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def print_summary(summary: Mapping[str, object], output_dir: Path) -> None:
    reasons = summary["rejection_reasons"]
    reason_lines = "\n".join(
        f"  {name}: {count}" for name, count in reasons.items()  # type: ignore[union-attr]
    ) or "  (ninguno)"
    print(
        f"""
====================================================
DOCTOR MAÍZ — FULL DATASET SEGMENTATION
====================================================

Images discovered:       {summary['images_found']}
Images processed:        {summary['images_processed']}
Errors:                  {summary['errors']}

Reliable:                {summary['reliable']}
Uncertain:               {summary['uncertain']}
Failed:                  {summary['failed']}

Segmented:               {summary['segmented']}
Full image fallback:     {summary['full_image']}

Reliability rate:        {summary['reliability_rate_percent']:.2f}%
Uncertain rate:          {summary['uncertain_rate_percent']:.2f}%

Rejection reasons:
{reason_lines}

Output:  {output_dir}
Gallery: {output_dir / 'gallery' / 'index.html'}
====================================================
""",
        flush=True,
    )


def _manifest_handles(
    output_dir: Path,
    *,
    append: bool = False,
) -> tuple[TextIO, csv.DictWriter, TextIO]:
    mode = "a" if append else "x"
    csv_handle = (output_dir / "manifest.csv").open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
    if not append:
        writer.writeheader()
    jsonl_handle = (output_dir / "manifest.jsonl").open(mode, encoding="utf-8")
    return csv_handle, writer, jsonl_handle


def _elapsed_before_resume(
    output_dir: Path,
    run_metadata: Mapping[str, object],
) -> float:
    runtime = run_metadata.get("runtime")
    runtime_mapping = runtime if isinstance(runtime, Mapping) else {}
    accumulated = float(runtime_mapping.get("accumulated_wall_seconds_before_current", 0.0))
    started_at = runtime_mapping.get("current_started_at") or run_metadata.get("created_at")
    if not isinstance(started_at, str):
        raise ValueError("run_metadata.json no contiene una fecha de inicio válida")
    started = datetime.fromisoformat(started_at)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    last_flush = datetime.fromtimestamp(
        (output_dir / "manifest.jsonl").stat().st_mtime,
        tz=timezone.utc,
    )
    return accumulated + max(0.0, (last_flush - started).total_seconds())


def _validate_resume_provenance(
    previous: Mapping[str, object],
    current: Mapping[str, object],
) -> None:
    for section, key in (
        ("dataset", "root"),
        ("dataset", "source"),
        ("dataset", "commit"),
        ("checkpoint", "sha256"),
        ("config", "sha256"),
    ):
        old_section = previous.get(section)
        new_section = current.get(section)
        old_value = old_section.get(key) if isinstance(old_section, Mapping) else None
        new_value = new_section.get(key) if isinstance(new_section, Mapping) else None
        if old_value != new_value:
            raise ValueError(
                f"Resume rechazado: cambió la procedencia {section}.{key} "
                f"({old_value!r} != {new_value!r})"
            )


def run_full_dataset(
    *,
    dataset_root: Path,
    output_dir: Path,
    config_path: Path,
    metadata_path: Path | None,
    dataset_source: str,
    dataset_commit: str,
    source_repository: Path | None,
    device: str | None,
    limit: int | None,
    progress_every: int,
    inventory_only: bool,
    reuse_inventory: bool,
    resume: bool,
    command: str,
) -> dict[str, object]:
    root = dataset_root.resolve()
    output = output_dir.resolve()
    if reuse_inventory and resume:
        raise ValueError("reuse_inventory y resume son mutuamente excluyentes")
    if resume and inventory_only:
        raise ValueError("resume no es compatible con inventory_only")
    if resume and limit is not None:
        raise ValueError("resume usa el inventario persistido y no acepta limit")
    previous_metadata: dict[str, object] | None = None
    previous_rows: list[dict[str, object]] = []
    prior_wall_seconds = 0.0
    if resume:
        if not output.is_dir():
            raise FileNotFoundError(f"No existe el run que se desea reanudar: {output}")
        if (output / "summary.json").exists():
            raise FileExistsError(f"El run ya está finalizado y no se sobrescribe: {output}")
        records, inventory_summary = load_inventory(output)
        if Path(str(inventory_summary["dataset_root"])).resolve() != root:
            raise ValueError("El inventory existente corresponde a otro dataset_root")
        previous_rows = load_resume_manifest(output, records)
        metadata_value = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata_value, dict):
            raise ValueError("run_metadata.json debe contener un objeto")
        previous_metadata = metadata_value
        prior_wall_seconds = _elapsed_before_resume(output, previous_metadata)
    elif reuse_inventory:
        if not output.is_dir():
            raise FileNotFoundError(f"No existe el inventario reutilizable: {output}")
        if (output / "manifest.csv").exists() or (output / "manifest.jsonl").exists():
            raise FileExistsError(f"El run ya contiene inferencia y no se sobrescribe: {output}")
        records, inventory_summary = load_inventory(output)
        if Path(str(inventory_summary["dataset_root"])).resolve() != root:
            raise ValueError("El inventory existente corresponde a otro dataset_root")
    else:
        if output.exists():
            raise FileExistsError(f"No se sobrescribe el run: {output}")
        output.mkdir(parents=True)
        records, inventory_summary = inventory_dataset(
            root,
            metadata_path=metadata_path,
            limit=limit,
            progress_every=progress_every,
        )
        write_inventory(output, records, inventory_summary)
    if inventory_only:
        print(json.dumps(inventory_summary, indent=2, sort_keys=True, ensure_ascii=False))
        return dict(inventory_summary)

    _create_output_directories(output)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "runtime_cache" / "matplotlib"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output / "runtime_cache" / "ultralytics"))
    runtime = build_runtime(config_path, device=device)
    run_metadata = build_run_metadata(
        runtime,
        dataset_root=root,
        dataset_source=dataset_source,
        dataset_commit=dataset_commit,
        source_repository=source_repository,
        command=command,
        inventory_summary=inventory_summary,
    )
    current_started_at = str(run_metadata["created_at"])
    if previous_metadata is not None:
        _validate_resume_provenance(previous_metadata, run_metadata)
        original_command = previous_metadata.get("command")
        prior_commands = previous_metadata.get("commands")
        commands = (
            list(prior_commands)
            if isinstance(prior_commands, list)
            else [original_command]
            if isinstance(original_command, str)
            else []
        )
        commands.append(command)
        run_metadata["created_at"] = previous_metadata.get("created_at", current_started_at)
        run_metadata["commands"] = commands
        run_metadata["resumed_at"] = current_started_at
    runtime_metadata = cast(dict[str, object], run_metadata["runtime"])
    runtime_metadata["accumulated_wall_seconds_before_current"] = prior_wall_seconds
    runtime_metadata["current_started_at"] = current_started_at
    _write_json(output / "run_metadata.json", run_metadata)

    rows = list(previous_rows)
    started = time.perf_counter()
    csv_handle, csv_writer, jsonl_handle = _manifest_handles(output, append=resume)
    try:
        completed = len(rows)
        if resume:
            percent = completed * 100.0 / len(records) if records else 100.0
            print(
                f"Resuming dataset: {completed}/{len(records)} [{percent:.1f}%]",
                flush=True,
            )
        for index, record in enumerate(records[completed:], start=completed + 1):
            row = _process_one(record, runtime, output)
            rows.append(row)
            csv_writer.writerow(row)
            jsonl_handle.write(
                json.dumps(_jsonl_row(row), ensure_ascii=False, allow_nan=False) + "\n"
            )
            csv_handle.flush()
            jsonl_handle.flush()
            if progress_every > 0 and (index % progress_every == 0 or index == len(records)):
                percent = index * 100.0 / len(records) if records else 100.0
                print(
                    f"Segmenting dataset: {index}/{len(records)} [{percent:.1f}%]",
                    flush=True,
                )
    finally:
        csv_handle.close()
        jsonl_handle.close()
    total_wall_seconds = prior_wall_seconds + time.perf_counter() - started
    summary = build_summary(
        rows,
        inventory_summary=inventory_summary,
        total_wall_seconds=total_wall_seconds,
    )
    run_metadata["segmenter"] = runtime.segmenter.to_metadata()
    runtime_metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    runtime_metadata["total_wall_seconds"] = total_wall_seconds
    _write_json(output / "run_metadata.json", run_metadata)
    _write_json(output / "summary.json", summary)
    write_gallery(output, rows)
    write_readme(output, summary=summary, metadata=run_metadata)
    print_summary(summary, output)
    return summary


def exact_command(
    argv: Sequence[str] | None = None,
    *,
    module: str | None = None,
) -> str:
    arguments = list(sys.argv if argv is None else argv)
    if module is not None:
        arguments = [sys.executable, "-m", module, *arguments[1:]]
    return shlex.join(arguments)
