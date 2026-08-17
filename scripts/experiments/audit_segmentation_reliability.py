"""Run the reproducible, human-labeled leaf-segmentation reliability audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageOps

from src.config import PROJECT_ROOT, get_dataset_root, get_output_root
from src.data.leaf_pilot import sha256_file
from src.data.loader import load_and_normalize_image
from src.evaluation.segmentation_reliability import (
    MaskQualityLabel,
    build_reliability_audit_summary,
)
from src.preprocessing.leaf_roi import image_to_rgb
from src.preprocessing.segmented_leaf_processor import (
    SegmentedLeafProcessingResult,
    SegmentedLeafProcessor,
    mask_processor_config_from_mapping,
)
from src.segmentation.leaf_segmenter import UltralyticsLeafSegmenter
from src.segmentation.quality import (
    SegmentationAssessment,
    SegmentationQualityGateConfig,
    SegmentationStatus,
    assess_segmentation,
    assess_segmentation_legacy,
)

DEFAULT_MANIFEST = (
    PROJECT_ROOT / "scripts" / "experiments" / "manifests" / "segmentation_reliability_audit_v1.csv"
)
DEFAULT_OUTPUT = (
    get_output_root() / "leaf_detection" / "validation_real_pipeline" / "reliability_gate_audit_v1"
)
BOOLEAN_COLUMNS = (
    "multi_leaf",
    "severe_fall_armyworm",
    "blur",
    "occlusion",
    "partial_leaf",
    "complex_background",
    "small_leaf",
    "large_leaf",
)
REQUIRED_COLUMNS = (
    "image_id",
    "relative_path",
    "source_label",
    "environment",
    "quality_label",
    "issue_type",
    *BOOLEAN_COLUMNS,
    "rationale",
)


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("la configuración debe ser un mapping")
    return loaded


def _strict_bool(value: str, *, line_number: int, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "sí", "si"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"{name} inválido en línea {line_number}: {value!r}")


def read_audit_manifest(
    path: Path,
    *,
    dataset_root: Path,
    raw_dir: str,
) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"faltan columnas en el manifest: {missing}")
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        for line_number, raw in enumerate(reader, start=2):
            image_id = raw["image_id"].strip()
            if not image_id or image_id in seen:
                raise ValueError(f"image_id vacío o duplicado en línea {line_number}")
            seen.add(image_id)
            try:
                quality = MaskQualityLabel(raw["quality_label"].strip())
            except ValueError as exc:
                raise ValueError(f"quality_label inválida en línea {line_number}") from exc
            relative_path = raw["relative_path"].strip()
            image_path = dataset_root / raw_dir / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(f"imagen inexistente en línea {line_number}: {image_path}")
            row: dict[str, object] = {
                "image_id": image_id,
                "relative_path": relative_path,
                "image_path": str(image_path.resolve()),
                "source_label": raw["source_label"].strip(),
                "environment": raw["environment"].strip(),
                "quality_label": quality.value,
                "issue_type": raw["issue_type"].strip(),
                "rationale": raw["rationale"].strip(),
            }
            for name in BOOLEAN_COLUMNS:
                row[name] = _strict_bool(raw[name], line_number=line_number, name=name)
            rows.append(row)
    if not rows:
        raise ValueError("el manifest de auditoría está vacío")
    return rows


def _eligible_trace(processing: SegmentedLeafProcessingResult) -> Mapping[str, object]:
    for trace in processing.selection_traces:
        if trace.source_index == processing.selected_instance:
            return trace.to_metadata()
    return {}


def _render_panel(
    processing: SegmentedLeafProcessingResult,
    *,
    quality_label: str,
    legacy_status: str,
    proposed_status: str,
) -> Image.Image:
    original = image_to_rgb(processing.original_image)
    if processing.mask is None:
        mask = Image.new("L", original.size, 0)
        overlay = original.copy()
        masked = Image.new("RGB", original.size, 0)
    else:
        mask = processing.mask
        tint = Image.new("RGB", original.size, (0, 180, 80))
        overlay = Image.composite(Image.blend(original, tint, 0.42), original, mask)
        masked = processing.masked_image or Image.new("RGB", original.size, 0)
    tiles = (original, overlay, mask.convert("RGB"), masked)
    labels = ("original", "overlay", "mask", "masked")
    tile_size = (300, 225)
    header_height = 42
    canvas = Image.new("RGB", (tile_size[0] * len(tiles), tile_size[1] + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 4),
        f"quality={quality_label} legacy={legacy_status} proposed={proposed_status}",
        fill="black",
    )
    for index, (tile, label) in enumerate(zip(tiles, labels, strict=True)):
        fitted = ImageOps.contain(tile, tile_size)
        x = index * tile_size[0] + (tile_size[0] - fitted.width) // 2
        y = header_height + (tile_size[1] - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        draw.text((index * tile_size[0] + 8, 23), label, fill="black")
    return canvas


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no hay filas para {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assessment_fields(assessment: SegmentationAssessment, prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_status": assessment.status.value,
        f"{prefix}_reason": assessment.reason,
    }


def _audit_row(
    case: Mapping[str, object],
    processing: SegmentedLeafProcessingResult,
    legacy: SegmentationAssessment,
    proposed: SegmentationAssessment,
    panel_path: Path | None,
) -> dict[str, object]:
    selected = _eligible_trace(processing)
    geometry = proposed.quality_gate_metrics
    segmenter = processing.segmenter_metadata
    row = dict(case)
    row.update(_assessment_fields(legacy, "legacy"))
    row.update(_assessment_fields(proposed, "proposed"))
    row.update(
        {
            "automatic_status": proposed.status.value,
            "segmentation_available": proposed.status is SegmentationStatus.RELIABLE,
            "quality_gate_reason": proposed.reason,
            "quality_gate_reasons": ";".join(proposed.quality_gate_reasons),
            "quality_gate_version": proposed.quality_gate_version,
            "proposal_confidence_threshold": segmenter.get("proposal_confidence_threshold"),
            "selected_proposal_confidence": selected.get("confidence"),
            "selection_confidence": processing.confidence,
            "mask_area_ratio": processing.mask_area_ratio,
            "bbox_area_ratio": geometry.get("bbox_area_ratio"),
            "mask_bbox_ratio": geometry.get("mask_bbox_ratio"),
            "number_of_instances": processing.number_of_instances,
            "eligible_instances": proposed.eligible_instances,
            "selected_instance": processing.selected_instance,
            "fallback_used": processing.fallback_used,
            "fallback_reason": processing.fallback_reason,
            "selected_relative_area": selected.get("relative_area"),
            "selected_center_proximity": selected.get("center_proximity"),
            "selected_score": selected.get("score"),
            "instance_score_margin": geometry.get("instance_score_margin"),
            "bbox_width_ratio": geometry.get("bbox_width_ratio"),
            "bbox_height_ratio": geometry.get("bbox_height_ratio"),
            "bbox_aspect_ratio": geometry.get("bbox_aspect_ratio"),
            "bbox_center_x_ratio": geometry.get("bbox_center_x_ratio"),
            "bbox_center_y_ratio": geometry.get("bbox_center_y_ratio"),
            "border_contact_count": geometry.get("border_contact_count"),
            "connected_components": geometry.get("connected_components"),
            "largest_component_ratio": geometry.get("largest_component_ratio"),
            "area_pixels": geometry.get("area_pixels"),
            "perimeter_edges": geometry.get("perimeter_edges"),
            "perimeter_area_ratio": geometry.get("perimeter_area_ratio"),
            "normalized_perimeter": geometry.get("normalized_perimeter"),
            "visual_panel": str(panel_path) if panel_path is not None else None,
        }
    )
    return row


def _build_processor(
    config: Mapping[str, object],
    *,
    checkpoint_override: Path | None,
    device: str | None,
    proposal_confidence_threshold: float | None = None,
    selection_confidence_threshold: float | None = None,
) -> tuple[SegmentedLeafProcessor, SegmentationQualityGateConfig, bool]:
    segmentation = config.get("segmentation")
    if not isinstance(segmentation, Mapping):
        raise ValueError("segmentation debe ser un mapping")
    quality = segmentation.get("quality_gate")
    if not isinstance(quality, Mapping):
        raise ValueError("segmentation.quality_gate debe ser un mapping")
    reject_multiple = quality.get("reject_multiple_eligible")
    if not isinstance(reject_multiple, bool):
        raise ValueError("reject_multiple_eligible debe ser booleano")
    checkpoint = checkpoint_override or get_output_root() / str(segmentation["checkpoint"])
    configured_proposal_confidence = proposal_confidence_threshold
    if configured_proposal_confidence is None:
        raw_proposal_confidence = segmentation.get(
            "proposal_confidence_threshold",
            segmentation.get("confidence_threshold"),
        )
        if raw_proposal_confidence is None:
            raise ValueError("falta proposal_confidence_threshold")
        configured_proposal_confidence = float(raw_proposal_confidence)
    segmenter = UltralyticsLeafSegmenter(
        checkpoint,
        image_size=int(segmentation["image_size"]),
        proposal_confidence_threshold=configured_proposal_confidence,
        iou_threshold=float(segmentation["iou_threshold"]),
        max_detections=int(segmentation["max_detections"]),
        device=device,
        expected_version=str(segmentation["ultralytics_version"]),
    )
    processor = SegmentedLeafProcessor(
        segmenter,
        mask_processor_config_from_mapping(
            segmentation,
            selection_confidence_threshold=selection_confidence_threshold,
        ),
    )
    return processor, SegmentationQualityGateConfig.from_mapping(quality), reject_multiple


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"No se sobrescribe la auditoría: {args.output_dir}")
    config = _load_config(args.config)
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("paths debe ser un mapping")
    cases = read_audit_manifest(
        args.manifest,
        dataset_root=get_dataset_root(),
        raw_dir=str(paths["raw_dir"]),
    )
    processor, quality_gate, reject_multiple = _build_processor(
        config,
        checkpoint_override=args.segmenter_checkpoint,
        device=args.segmenter_device,
        proposal_confidence_threshold=args.proposal_confidence_threshold,
        selection_confidence_threshold=args.selection_confidence_threshold,
    )

    args.output_dir.mkdir(parents=True)
    visuals_dir = args.output_dir / "visuals"
    if not args.no_visuals:
        visuals_dir.mkdir()
    rows: list[dict[str, object]] = []
    structured: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        processing = processor.process(
            load_and_normalize_image(str(case["image_path"])),
            source_image=str(case["image_path"]),
        )
        legacy = assess_segmentation_legacy(processing)
        proposed = assess_segmentation(
            processing,
            reject_multiple_eligible=reject_multiple,
            quality_gate=quality_gate,
        )
        panel_path = None
        if not args.no_visuals:
            panel_path = visuals_dir / f"{index:03d}_{case['image_id']}.jpg"
            _render_panel(
                processing,
                quality_label=str(case["quality_label"]),
                legacy_status=legacy.status.value,
                proposed_status=proposed.status.value,
            ).save(panel_path, quality=92)
        rows.append(_audit_row(case, processing, legacy, proposed, panel_path))
        structured.append(
            {
                "case": case,
                "processing": processing.to_metadata(),
                "legacy_assessment": legacy.to_metadata(),
                "proposed_assessment": proposed.to_metadata(),
            }
        )

    _write_csv(args.output_dir / "audit_metrics.csv", rows)
    false_reliable = [
        row
        for row in rows
        if row["proposed_status"] == "reliable" and row["quality_label"] != "GOOD"
    ]
    if false_reliable:
        _write_csv(args.output_dir / "false_reliable.csv", false_reliable)
    summary = build_reliability_audit_summary(rows)
    summary["provenance"] = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "segmenter": processor.segmenter.to_metadata(),
        "quality_gate_thresholds": quality_gate.to_metadata(),
        "reject_multiple_eligible": reject_multiple,
        "visual_panels": not args.no_visuals,
    }
    (args.output_dir / "structured_results.json").write_text(
        json.dumps(structured, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--segmenter-device", default=None)
    parser.add_argument("--segmenter-checkpoint", type=Path, default=None)
    parser.add_argument("--proposal-confidence-threshold", type=float, default=None)
    parser.add_argument("--selection-confidence-threshold", type=float, default=None)
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "segmentation.yaml",
    )
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    args.config = args.config.resolve()
    return args


def main() -> None:
    summary = run_audit(_parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
