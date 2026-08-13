"""Metrics for a human-labeled segmentation Reliability Gate audit."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np

from src.segmentation.quality import SegmentationStatus


class MaskQualityLabel(str, Enum):
    """Visual quality of a leaf-segmentation result."""

    GOOD = "GOOD"
    AMBIGUOUS = "AMBIGUOUS"
    BAD = "BAD"


AUDIT_NUMERIC_FIELDS = (
    "selected_proposal_confidence",
    "mask_area_ratio",
    "bbox_area_ratio",
    "mask_bbox_ratio",
    "bbox_width_ratio",
    "bbox_height_ratio",
    "bbox_aspect_ratio",
    "bbox_center_x_ratio",
    "bbox_center_y_ratio",
    "border_contact_count",
    "connected_components",
    "largest_component_ratio",
    "perimeter_edges",
    "perimeter_area_ratio",
    "normalized_perimeter",
    "number_of_instances",
    "eligible_instances",
    "selected_relative_area",
    "selected_center_proximity",
    "selected_score",
    "instance_score_margin",
)


def _quality(row: Mapping[str, object]) -> MaskQualityLabel:
    try:
        return MaskQualityLabel(str(row["quality_label"]))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"quality_label inválida: {row.get('quality_label')!r}") from exc


def _gate_metrics(
    rows: Sequence[Mapping[str, object]],
    status_field: str,
) -> dict[str, object]:
    status_counts = Counter(str(row[status_field]) for row in rows)
    reliable = [row for row in rows if str(row[status_field]) == SegmentationStatus.RELIABLE.value]
    reliable_good = [row for row in reliable if _quality(row) is MaskQualityLabel.GOOD]
    false_reliable = [row for row in reliable if _quality(row) is not MaskQualityLabel.GOOD]
    good = [row for row in rows if _quality(row) is MaskQualityLabel.GOOD]
    good_rejected = [
        row for row in good if str(row[status_field]) != SegmentationStatus.RELIABLE.value
    ]
    return {
        "total_audited": len(rows),
        "status_counts": {
            status.value: status_counts.get(status.value, 0) for status in SegmentationStatus
        },
        "reliable_good": len(reliable_good),
        "false_reliable": len(false_reliable),
        "false_reliable_ids": [str(row["image_id"]) for row in false_reliable],
        "good_masks_rejected": len(good_rejected),
        "good_masks_rejected_ids": [str(row["image_id"]) for row in good_rejected],
        "reliability_precision": (len(reliable_good) / len(reliable) if reliable else None),
        "segmentation_coverage": len(reliable) / len(rows) if rows else None,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def build_feature_distributions(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, dict[str, float | int] | None]]:
    """Summarize observed values without imputing missing measurements."""
    distributions: dict[str, dict[str, dict[str, float | int] | None]] = {}
    for field in AUDIT_NUMERIC_FIELDS:
        by_quality: dict[str, dict[str, float | int] | None] = {}
        for quality in MaskQualityLabel:
            values: list[float] = []
            for row in rows:
                value = row.get(field)
                if _quality(row) is quality and isinstance(value, (int, float)):
                    values.append(float(value))
            by_quality[quality.value] = _distribution(values)
        distributions[field] = by_quality
    return distributions


def _subgroups(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    groups: dict[str, list[Mapping[str, object]]] = {
        "environment:lab": [row for row in rows if row.get("environment") == "lab"],
        "environment:real": [row for row in rows if row.get("environment") == "real"],
    }
    for field in (
        "multi_leaf",
        "severe_fall_armyworm",
        "blur",
        "occlusion",
        "partial_leaf",
        "complex_background",
        "small_leaf",
        "large_leaf",
    ):
        groups[field] = [row for row in rows if row.get(field) is True]
    return {name: subset for name, subset in groups.items() if subset}


def build_reliability_audit_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare the frozen legacy gate with the proposed transparent gate."""
    if not rows:
        raise ValueError("la auditoría no puede estar vacía")
    for row in rows:
        _quality(row)
        for field in ("legacy_status", "proposed_status", "image_id"):
            if field not in row:
                raise ValueError(f"falta la columna de auditoría {field!r}")

    legacy = _gate_metrics(rows, "legacy_status")
    proposed = _gate_metrics(rows, "proposed_status")
    quality_counts = Counter(_quality(row).value for row in rows)
    by_subgroup: dict[str, object] = {}
    for name, subset in _subgroups(rows).items():
        by_subgroup[name] = {
            "quality_counts": dict(Counter(_quality(row).value for row in subset)),
            "legacy": _gate_metrics(subset, "legacy_status"),
            "proposed": _gate_metrics(subset, "proposed_status"),
        }
    legacy_false = _metric_int(legacy, "false_reliable")
    proposed_false = _metric_int(proposed, "false_reliable")
    legacy_good_rejected = _metric_int(legacy, "good_masks_rejected")
    proposed_good_rejected = _metric_int(proposed, "good_masks_rejected")
    return {
        "audit": "segmentation_reliability_gate_v1",
        "total_audited": len(rows),
        "quality_counts": {
            quality.value: quality_counts.get(quality.value, 0) for quality in MaskQualityLabel
        },
        "legacy_gate": legacy,
        "proposed_gate": proposed,
        "delta": {
            "reliability_precision": _subtract_optional(
                proposed["reliability_precision"], legacy["reliability_precision"]
            ),
            "segmentation_coverage": _subtract_optional(
                proposed["segmentation_coverage"], legacy["segmentation_coverage"]
            ),
            "false_reliable": proposed_false - legacy_false,
            "good_masks_rejected": proposed_good_rejected - legacy_good_rejected,
        },
        "feature_distributions": build_feature_distributions(rows),
        "by_subgroup": by_subgroup,
    }


def _metric_int(metrics: Mapping[str, object], key: str) -> int:
    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"la métrica {key!r} debe ser entera")
    return value


def _subtract_optional(left: Any, right: Any) -> float | None:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) - float(right)
    return None
