"""Compare historical full-image inference against manual-ROI inference.

This is a read-only diagnostic: it never trains, updates, or writes a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import PROJECT_ROOT, set_global_seed
from src.data.leaf_pilot import read_csv_rows, require_columns, sha256_file, write_csv_rows
from src.data.loader import load_and_normalize_image
from src.data.transforms import CornTransformFactory
from src.models import MODEL_REGISTRY, resolve_input_size
from src.preprocessing.leaf_processor import (
    BASELINE_FULL,
    BASELINE_ROI,
    PROCESSOR_VERSION,
    LeafImageProcessor,
    LeafProcessingProfile,
    LeafProcessorConfig,
    ProfileProcessingResult,
)
from src.preprocessing.leaf_roi import BoundingBox, crop_leaf_region
from src.preprocessing.roi_manifest import ROI_MANIFEST_COLUMNS

EXPERIMENT_NAME = "diagnostic_full_vs_manual_roi"
PREDICTION_METADATA_COLUMNS = (
    "processing_profile",
    "roi_source",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "roi_area_ratio",
    "roi_confidence",
    "margin_ratio",
    "target_width",
    "target_height",
    "padding_value",
    "preserve_aspect_ratio",
    "fallback_used",
    "fallback_reason",
    "roi_manifest_path",
    "roi_manifest_sha256",
    "processor_version",
)
PREDICTION_COLUMNS = (
    "pilot_id",
    "image_path",
    "true_label",
    "annotation_status",
    "included_in_metrics",
    "exclusion_reason",
    "predicted_label",
    "confidence",
    "correct",
    "loss",
    *PREDICTION_METADATA_COLUMNS,
)
COMPARISON_COLUMNS = (
    "pilot_id",
    "image_path",
    "true_label",
    "annotation_status",
    "included_in_metrics",
    "exclusion_reason",
    "full_prediction",
    "full_confidence",
    "full_correct",
    "roi_prediction",
    "roi_confidence",
    "roi_correct",
    "confidence_delta",
    "prediction_changed",
    "roi_area_ratio",
    "roi_source",
    "fallback_used",
)
_AREA_IN_NOTES = re.compile(r"area_ratio\s+([0-9]+(?:\.[0-9]+)?)")


class CheckpointCompatibilityError(ValueError):
    """Raised when an existing checkpoint cannot represent the requested model."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Validated model and immutable checkpoint provenance."""

    model: torch.nn.Module
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    target_size: tuple[int, int]
    sha256: str
    summary_path: str | None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe la configuración: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuración YAML inválida: {path}")
    return loaded


def _validate_class_mapping(mapping: dict[str, Any], configured: Sequence[str]) -> dict[str, int]:
    normalized = {str(label): int(index) for label, index in mapping.items()}
    expected_indices = list(range(len(normalized)))
    if sorted(normalized.values()) != expected_indices:
        raise CheckpointCompatibilityError(
            "class_to_idx del checkpoint no usa índices contiguos desde cero"
        )
    if set(normalized) != set(configured):
        missing = sorted(set(configured) - set(normalized))
        extra = sorted(set(normalized) - set(configured))
        raise CheckpointCompatibilityError(
            f"clases incompatibles; faltantes={missing}, adicionales={extra}"
        )
    return normalized


def _checkpoint_context(
    checkpoint_path: Path,
    model_name: str,
    config: dict[str, Any],
) -> tuple[dict[str, int], tuple[int, int], str | None]:
    configured_classes = [str(label) for label in config["dataset"]["classes"]]
    class_to_idx = {label: index for index, label in enumerate(configured_classes)}
    base_size = tuple(int(value) for value in config["dataset"]["target_size"])
    target_size = resolve_input_size(model_name, base_size)
    summary_path = checkpoint_path.parent / "summary.json"
    if not summary_path.is_file():
        return class_to_idx, target_size, None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_model = summary.get("model")
    if summary_model is not None and summary_model != model_name:
        raise CheckpointCompatibilityError(
            f"checkpoint documentado para {summary_model!r}, no para {model_name!r}"
        )
    if "class_to_idx" in summary:
        class_to_idx = _validate_class_mapping(summary["class_to_idx"], configured_classes)
    declared_classes = summary.get("num_classes")
    if declared_classes is not None and int(declared_classes) != len(class_to_idx):
        raise CheckpointCompatibilityError(
            f"summary declara {declared_classes} clases; se esperaban {len(class_to_idx)}"
        )
    image_size = summary.get("image_size")
    if isinstance(image_size, list) and len(image_size) == 2:
        target_size = (int(image_size[0]), int(image_size[1]))
    return class_to_idx, target_size, str(summary_path.resolve())


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise CheckpointCompatibilityError("checkpoint vacío o sin state_dict")
    if not all(isinstance(key, str) for key in checkpoint):
        raise CheckpointCompatibilityError("state_dict contiene claves no textuales")
    if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise CheckpointCompatibilityError("state_dict contiene valores que no son tensores")
    return checkpoint


def load_compatible_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    config: dict[str, Any],
    device: torch.device,
) -> LoadedCheckpoint:
    """Validate path, architecture, class count, and state shapes without downloading."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")
    if model_name not in MODEL_REGISTRY:
        raise CheckpointCompatibilityError(
            f"Modelo {model_name!r} no registrado; disponibles: {MODEL_REGISTRY.list_names()}"
        )
    digest = sha256_file(checkpoint_path)
    class_to_idx, target_size, summary_path = _checkpoint_context(
        checkpoint_path, model_name, config
    )
    try:
        raw_checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(f"no se pudo leer el checkpoint: {exc}") from exc
    state_dict = _extract_state_dict(raw_checkpoint)
    model = MODEL_REGISTRY.build(
        model_name,
        num_classes=len(class_to_idx),
        pretrained=False,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint incompatible con {model_name} y {len(class_to_idx)} clases: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return LoadedCheckpoint(
        model=model,
        class_to_idx=class_to_idx,
        idx_to_class={index: label for label, index in class_to_idx.items()},
        target_size=target_size,
        sha256=digest,
        summary_path=summary_path,
    )


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` with explicit availability checks."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda solicitado, pero CUDA no está disponible")
    return torch.device(requested)


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"valor no finito: {value}")
    return converted


def _reported_area_ratio(row: dict[str, str]) -> float | None:
    area = _optional_float(row.get("roi_area_ratio"))
    if area is not None:
        return area
    match = _AREA_IN_NOTES.search(row.get("notes", ""))
    return float(match.group(1)) if match else None


def _optional_bbox(row: dict[str, str]) -> BoundingBox | None:
    names = ("roi_x1", "roi_y1", "roi_x2", "roi_y2")
    if any(not row.get(name, "").strip() for name in names):
        return None
    return tuple(int(float(row[name])) for name in names)  # type: ignore[return-value]


def metric_inclusion(row: dict[str, str]) -> tuple[bool, str]:
    """Keep every row in outputs while excluding non-annotated cases from metrics."""
    status = row.get("annotation_status", "").strip()
    if status == "annotated":
        return True, ""
    area = _reported_area_ratio(row)
    area_reason = f"; roi_area_ratio={area:.6f}" if area is not None else ""
    return False, f"annotation_status={status or 'missing'}{area_reason}"


def _resolve_image_path(manifest_path: Path, row: dict[str, str]) -> Path:
    candidate = Path(row["image_path"])
    if candidate.is_absolute():
        return candidate
    direct = manifest_path.parent / candidate
    if direct.is_file():
        return direct
    return manifest_path.parent.parent / candidate


def _predict(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    true_index: int,
    idx_to_class: dict[int, str],
    device: torch.device,
) -> tuple[str, float, float]:
    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0).to(device))
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(idx_to_class):
            raise CheckpointCompatibilityError(
                f"salida del modelo {tuple(logits.shape)} incompatible con "
                f"{len(idx_to_class)} clases"
            )
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = probabilities.max(dim=1)
        loss = F.cross_entropy(
            logits,
            torch.tensor([true_index], dtype=torch.long, device=device),
        )
    return (
        idx_to_class[int(predicted_index.item())],
        float(confidence.item()),
        float(loss.item()),
    )


def _prediction_row(
    manifest_row: dict[str, str],
    prepared: ProfileProcessingResult,
    prediction: str,
    confidence: float,
    loss: float,
    included: bool,
    exclusion_reason: str,
) -> dict[str, object]:
    true_label = manifest_row["label"]
    return {
        "pilot_id": manifest_row["pilot_id"],
        "image_path": manifest_row["image_path"],
        "true_label": true_label,
        "annotation_status": manifest_row["annotation_status"],
        "included_in_metrics": included,
        "exclusion_reason": exclusion_reason,
        "predicted_label": prediction,
        "confidence": confidence,
        "correct": prediction == true_label,
        "loss": loss,
        **prepared.metadata,
    }


def compare_predictions(
    full_rows: Sequence[dict[str, object]],
    roi_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Build deterministic paired outcomes, including corrected and worsened cases."""
    if len(full_rows) != len(roi_rows):
        raise ValueError("las variantes no contienen la misma cantidad de predicciones")
    comparisons: list[dict[str, object]] = []
    for full, roi in zip(full_rows, roi_rows, strict=True):
        if full["pilot_id"] != roi["pilot_id"]:
            raise ValueError("las predicciones no están alineadas por pilot_id")
        comparisons.append(
            {
                "pilot_id": full["pilot_id"],
                "image_path": full["image_path"],
                "true_label": full["true_label"],
                "annotation_status": full["annotation_status"],
                "included_in_metrics": full["included_in_metrics"],
                "exclusion_reason": full["exclusion_reason"],
                "full_prediction": full["predicted_label"],
                "full_confidence": full["confidence"],
                "full_correct": full["correct"],
                "roi_prediction": roi["predicted_label"],
                "roi_confidence": roi["confidence"],
                "roi_correct": roi["correct"],
                "confidence_delta": float(roi["confidence"]) - float(full["confidence"]),
                "prediction_changed": roi["predicted_label"] != full["predicted_label"],
                "roi_area_ratio": roi["roi_area_ratio"],
                "roi_source": roi["roi_source"],
                "fallback_used": roi["fallback_used"],
            }
        )
    return comparisons


def _metrics(
    rows: Sequence[dict[str, object]],
    class_names: Sequence[str],
) -> dict[str, object]:
    included = [row for row in rows if bool(row["included_in_metrics"])]
    y_true = [str(row["true_label"]) for row in included]
    y_pred = [str(row["predicted_label"]) for row in included]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(class_names),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(class_names),
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(class_names))
    return {
        "images": len(included),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "loss": float(sum(float(row["loss"]) for row in included) / len(included)),
        "confusion_matrix": {
            "labels": list(class_names),
            "values": matrix.tolist(),
        },
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(class_names)
        },
    }


def build_summary(
    full_rows: Sequence[dict[str, object]],
    roi_rows: Sequence[dict[str, object]],
    comparisons: Sequence[dict[str, object]],
    class_names: Sequence[str],
) -> dict[str, object]:
    """Aggregate only rows explicitly marked as included in primary metrics."""
    included = [row for row in comparisons if bool(row["included_in_metrics"])]
    corrected = [row for row in included if not row["full_correct"] and row["roi_correct"]]
    worsened = [row for row in included if row["full_correct"] and not row["roi_correct"]]
    changed = [row for row in included if row["prediction_changed"]]
    return {
        "experiment": EXPERIMENT_NAME,
        "official_baseline": False,
        "training_performed": False,
        "total_images": len(comparisons),
        "included_images": len(included),
        "excluded_images": len(comparisons) - len(included),
        "excluded_cases": [
            {
                "pilot_id": row["pilot_id"],
                "reason": row["exclusion_reason"],
            }
            for row in comparisons
            if not row["included_in_metrics"]
        ],
        "baseline_full": _metrics(full_rows, class_names),
        "baseline_roi": _metrics(roi_rows, class_names),
        "comparison": {
            "changed_predictions": len(changed),
            "corrected_errors": len(corrected),
            "lost_correct_predictions": len(worsened),
            "mean_confidence_delta": float(
                sum(float(row["confidence_delta"]) for row in included) / len(included)
            ),
            "fallbacks": sum(bool(row["fallback_used"]) for row in included),
            "corrected_pilot_ids": [row["pilot_id"] for row in corrected],
            "worsened_pilot_ids": [row["pilot_id"] for row in worsened],
            "changed_pilot_ids": [row["pilot_id"] for row in changed],
        },
    }


def _fit_panel(image: Image.Image, size: tuple[int, int] = (260, 220)) -> Image.Image:
    panel = image.copy()
    panel.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(panel, ((size[0] - panel.width) // 2, (size[1] - panel.height) // 2))
    return canvas


def _draw_bbox(image: Image.Image, bbox: BoundingBox | None) -> Image.Image:
    rendered = image.copy()
    if bbox is not None:
        draw = ImageDraw.Draw(rendered)
        draw.rectangle(bbox, outline=(255, 210, 0), width=max(2, min(image.size) // 150))
    return rendered


def _render_preview(
    image: Image.Image,
    roi_prepared: ProfileProcessingResult,
    comparison: dict[str, object],
) -> Image.Image:
    roi_result = roi_prepared.roi_result
    bbox = roi_result.clipped_bbox if roi_result else None
    expanded = roi_result.expanded_bbox if roi_result else None
    crop = crop_leaf_region(image, expanded) if expanded is not None else image
    panels = (
        _fit_panel(image),
        _fit_panel(_draw_bbox(image, bbox)),
        _fit_panel(crop),
        _fit_panel(roi_prepared.image),
    )
    titles = (
        "imagen completa",
        "bounding box",
        "recorte con margen",
        "letterbox",
    )
    header_height = 62
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), panels[0].height + header_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    details = (
        f"{comparison['pilot_id']} | real={comparison['true_label']} | "
        f"full={comparison['full_prediction']} ({float(comparison['full_confidence']):.4f}) | "
        f"roi={comparison['roi_prediction']} ({float(comparison['roi_confidence']):.4f})"
    )
    draw.text((6, 5), details, fill="black", font=font)
    x_offset = 0
    for panel, title in zip(panels, titles, strict=True):
        draw.text((x_offset + 6, 33), title, fill="black", font=font)
        canvas.paste(panel, (x_offset, header_height))
        x_offset += panel.width
    return canvas


def _write_previews(
    output_dir: Path,
    manifest_path: Path,
    manifest_rows: Sequence[dict[str, str]],
    roi_profile: LeafProcessingProfile,
    comparisons: Sequence[dict[str, object]],
) -> None:
    comparison_by_id = {str(row["pilot_id"]): row for row in comparisons}
    for directory in (
        output_dir / "previews",
        output_dir / "improved_cases",
        output_dir / "worsened_cases",
        output_dir / "changed_predictions",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    for row in manifest_rows:
        image = load_and_normalize_image(_resolve_image_path(manifest_path, row))
        prepared = roi_profile.prepare(
            image,
            _optional_bbox(row),
            confidence=_optional_float(row.get("roi_confidence")) or 1.0,
            source=row.get("roi_source", "") or "manual",
            fallback_reason=row.get("notes", "") or None,
            reported_area_ratio=_reported_area_ratio(row),
        )
        comparison = comparison_by_id[row["pilot_id"]]
        preview = _render_preview(image, prepared, comparison)
        filename = f"{row['pilot_id']}_comparison.jpg"
        preview.save(output_dir / "previews" / filename, quality=92)
        if (
            comparison["included_in_metrics"]
            and not comparison["full_correct"]
            and comparison["roi_correct"]
        ):
            preview.save(output_dir / "improved_cases" / filename, quality=92)
        if (
            comparison["included_in_metrics"]
            and comparison["full_correct"]
            and not comparison["roi_correct"]
        ):
            preview.save(output_dir / "worsened_cases" / filename, quality=92)
        if comparison["included_in_metrics"] and comparison["prediction_changed"]:
            preview.save(output_dir / "changed_predictions" / filename, quality=92)


def run_experiment(
    *,
    checkpoint_path: Path,
    model_name: str,
    roi_manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    device_name: str = "auto",
) -> dict[str, object]:
    """Run paired read-only inference and persist deterministic tabular results."""
    config = _load_yaml(config_path)
    if config.get("processing_profile", BASELINE_FULL) != BASELINE_FULL:
        raise ValueError("processing_profile global debe continuar en baseline_full")
    if bool(config.get("leaf_detection", {}).get("enabled", False)):
        raise ValueError("leaf_detection.enabled debe continuar en false para este diagnóstico")
    device = resolve_device(device_name)
    loaded = load_compatible_checkpoint(checkpoint_path, model_name, config, device)
    rows, columns = read_csv_rows(roi_manifest_path)
    require_columns(columns, ROI_MANIFEST_COLUMNS, "ROI manifest")
    rows = sorted(rows, key=lambda row: row["pilot_id"])
    if not rows:
        raise ValueError("ROI manifest sin filas")

    leaf_config = config["leaf_detection"]
    roi_processor = LeafImageProcessor(
        LeafProcessorConfig(
            margin_ratio=float(leaf_config["margin_ratio"]),
            min_area_ratio=float(leaf_config["min_area_ratio"]),
            target_size=loaded.target_size,
            padding_value=int(leaf_config["padding_value"]),
            fallback=str(leaf_config["fallback"]),
            preserve_aspect_ratio=bool(leaf_config["preserve_aspect_ratio"]),
        )
    )
    full_profile = LeafProcessingProfile(
        BASELINE_FULL,
        processor=roi_processor,
        roi_manifest_path=roi_manifest_path,
    )
    roi_profile = LeafProcessingProfile(
        BASELINE_ROI,
        processor=roi_processor,
        roi_manifest_path=roi_manifest_path,
    )
    transform = CornTransformFactory(
        config_path=str(config_path),
        target_size=loaded.target_size,
    ).get_pipeline("inference")
    set_global_seed(int(config["dataset"]["seed"]))

    full_predictions: list[dict[str, object]] = []
    roi_predictions: list[dict[str, object]] = []
    for row in rows:
        true_label = row["label"]
        if true_label not in loaded.class_to_idx:
            raise CheckpointCompatibilityError(f"clase del piloto no soportada: {true_label}")
        image_path = _resolve_image_path(roi_manifest_path, row)
        image = load_and_normalize_image(image_path)
        included, reason = metric_inclusion(row)
        confidence = _optional_float(row.get("roi_confidence")) or 1.0
        area_ratio = _reported_area_ratio(row)
        full_applied = full_profile.apply(
            image,
            stage="inference",
            normalization=transform,
            reported_area_ratio=area_ratio,
        )
        roi_applied = roi_profile.apply(
            image,
            _optional_bbox(row),
            stage="inference",
            normalization=transform,
            confidence=confidence,
            source=row.get("roi_source", "") or "manual",
            fallback_reason=row.get("notes", "") or None,
            reported_area_ratio=area_ratio,
        )
        true_index = loaded.class_to_idx[true_label]
        full_prediction, full_confidence, full_loss = _predict(
            loaded.model,
            full_applied.output,  # type: ignore[arg-type]
            true_index,
            loaded.idx_to_class,
            device,
        )
        roi_prediction, roi_confidence, roi_loss = _predict(
            loaded.model,
            roi_applied.output,  # type: ignore[arg-type]
            true_index,
            loaded.idx_to_class,
            device,
        )
        full_predictions.append(
            _prediction_row(
                row,
                full_applied.prepared,
                full_prediction,
                full_confidence,
                full_loss,
                included,
                reason,
            )
        )
        roi_predictions.append(
            _prediction_row(
                row,
                roi_applied.prepared,
                roi_prediction,
                roi_confidence,
                roi_loss,
                included,
                reason,
            )
        )

    comparisons = compare_predictions(full_predictions, roi_predictions)
    class_names = [
        loaded.idx_to_class[index] for index in range(len(loaded.idx_to_class))
    ]
    summary = build_summary(full_predictions, roi_predictions, comparisons, class_names)
    if sha256_file(checkpoint_path) != loaded.sha256:
        raise RuntimeError("el checkpoint cambió durante el diagnóstico")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "predictions_full.csv", full_predictions, PREDICTION_COLUMNS)
    write_csv_rows(output_dir / "predictions_roi.csv", roi_predictions, PREDICTION_COLUMNS)
    write_csv_rows(output_dir / "comparison.csv", comparisons, COMPARISON_COLUMNS)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run_metadata = {
        "experiment": EXPERIMENT_NAME,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": model_name,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": loaded.sha256,
        "checkpoint_summary_path": loaded.summary_path,
        "checkpoint_modified": False,
        "num_classes": len(loaded.class_to_idx),
        "class_to_idx": loaded.class_to_idx,
        "target_size": list(loaded.target_size),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "roi_manifest_path": str(roi_manifest_path.resolve()),
        "roi_manifest_sha256": sha256_file(roi_manifest_path),
        "processor_version": PROCESSOR_VERSION,
        "processing_profiles": [BASELINE_FULL, BASELINE_ROI],
        "global_processing_profile": config.get("processing_profile"),
        "leaf_detection_enabled": leaf_config["enabled"],
        "training_performed": False,
        "optimizer_created": False,
        "backward_called": False,
        "checkpoints_written": 0,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_previews(output_dir, roi_manifest_path, rows, roi_profile, comparisons)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.list_names())
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "dataset.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run_experiment(
            checkpoint_path=args.checkpoint,
            model_name=args.model,
            roi_manifest_path=args.roi_manifest,
            config_path=args.config,
            output_dir=args.output,
            device_name=args.device,
        )
    except (
        CheckpointCompatibilityError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(f"Experimento: {EXPERIMENT_NAME}")
    print(f"Imágenes incluidas: {summary['included_images']}")
    print(f"Imágenes excluidas: {summary['excluded_images']}")
    print(f"Salida: {args.output.resolve()}")
    print("Entrenamiento: no")


if __name__ == "__main__":
    main()
