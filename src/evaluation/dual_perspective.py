"""Metrics for controlled full-image versus segmented-leaf experiments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.inference.dual_perspective import DualPerspectiveResult, SegmentationStatus


@dataclass(frozen=True)
class DualPerspectiveExperimentRecord:
    """One labeled case plus tags required by the controlled analysis."""

    filename: str
    ground_truth: str
    environment: str
    result: DualPerspectiveResult
    multi_leaf: bool = False
    severe_fall_armyworm: bool = False

    def to_metadata(self) -> dict[str, object]:
        segmented = self.result.segmented_leaf.prediction
        return {
            "file": self.filename,
            "ground_truth": self.ground_truth,
            "environment": self.environment,
            "multi_leaf": self.multi_leaf,
            "severe_fall_armyworm": self.severe_fall_armyworm,
            "segmentation_available": self.result.segmented_leaf.available,
            "segmentation_status": self.result.segmentation.status.value,
            "segmentation_reason": self.result.segmentation.reason,
            "full_image_prediction": self.result.full_image.class_name,
            "full_image_confidence": self.result.full_image.confidence,
            "segmented_leaf_prediction": (segmented.class_name if segmented is not None else None),
            "segmented_leaf_confidence": (segmented.confidence if segmented is not None else None),
            "agreement": self.result.agreement,
            "full_image_correct": (self.result.full_image.class_name == self.ground_truth),
            "segmented_leaf_correct": (
                segmented.class_name == self.ground_truth if segmented is not None else None
            ),
        }


def _classification_metrics(
    records: Sequence[DualPerspectiveExperimentRecord],
    class_names: Sequence[str],
    *,
    segmented: bool,
) -> dict[str, object]:
    selected: list[tuple[str, str, str]] = []
    for record in records:
        prediction = (
            record.result.segmented_leaf.prediction if segmented else record.result.full_image
        )
        if prediction is not None:
            selected.append((record.ground_truth, prediction.class_name, record.environment))
    labels = list(class_names)
    if not selected:
        return {
            "images": 0,
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "confusion_matrix": {
                "labels": labels,
                "values": [[0 for _ in labels] for _ in labels],
            },
            "per_class": {},
            "by_environment": {},
        }

    y_true = [truth for truth, _, _ in selected]
    y_pred = [prediction for _, prediction, _ in selected]
    precision, recall, f1, support = cast(
        tuple[Any, Any, Any, Any],
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=cast(Any, 0),
        ),
    )
    macro_precision, macro_recall, macro_f1, _ = cast(
        tuple[Any, Any, Any, Any],
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=cast(Any, 0),
        ),
    )
    by_environment: dict[str, dict[str, object]] = {}
    for environment in sorted({environment for _, _, environment in selected}):
        pairs = [
            (truth, prediction) for truth, prediction, current in selected if current == environment
        ]
        by_environment[environment] = {
            "images": len(pairs),
            "accuracy": float(
                accuracy_score(
                    [truth for truth, _ in pairs],
                    [prediction for _, prediction in pairs],
                )
            ),
        }
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "images": len(selected),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": {"labels": labels, "values": matrix.tolist()},
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "by_environment": by_environment,
    }


def _subset_accuracy(
    records: Sequence[DualPerspectiveExperimentRecord],
) -> dict[str, object]:
    segmented = [
        record for record in records if record.result.segmented_leaf.prediction is not None
    ]
    return {
        "images": len(records),
        "full_image_accuracy": (
            sum(record.result.full_image.class_name == record.ground_truth for record in records)
            / len(records)
            if records
            else None
        ),
        "segmented_leaf_images": len(segmented),
        "segmented_leaf_accuracy": (
            sum(
                record.result.segmented_leaf.prediction.class_name == record.ground_truth
                for record in segmented
                if record.result.segmented_leaf.prediction is not None
            )
            / len(segmented)
            if segmented
            else None
        ),
    }


def build_dual_perspective_summary(
    records: Sequence[DualPerspectiveExperimentRecord],
    class_names: Sequence[str],
) -> dict[str, object]:
    """Aggregate paired results without fusion or confidence comparison."""
    if not records:
        raise ValueError("records no puede estar vacío")
    unknown = sorted({record.ground_truth for record in records} - set(class_names))
    if unknown:
        raise ValueError(f"ground truth fuera del mapeo de clases: {unknown}")

    paired = [record for record in records if record.result.segmented_leaf.available]
    agreements = [record for record in paired if record.result.agreement is True]
    disagreements = [record for record in paired if record.result.agreement is False]
    failed = [
        record
        for record in records
        if record.result.segmentation.status is SegmentationStatus.FAILED
    ]
    multi_leaf = [record for record in records if record.multi_leaf]
    severe_fall = [record for record in records if record.severe_fall_armyworm]
    status_counts = Counter(record.result.segmentation.status.value for record in records)

    return {
        "experiment": "dual_perspective_full_vs_segmented_leaf",
        "experimental": True,
        "training_performed": False,
        "fusion_applied": False,
        "total_images": len(records),
        "full_image": _classification_metrics(
            records,
            class_names,
            segmented=False,
        ),
        "segmented_leaf": _classification_metrics(
            records,
            class_names,
            segmented=True,
        ),
        "segmentation": {
            "coverage": len(paired) / len(records),
            "available_images": len(paired),
            "status_counts": {
                status.value: status_counts.get(status.value, 0) for status in SegmentationStatus
            },
        },
        "views": {
            "paired_images": len(paired),
            "agreement_rate": len(agreements) / len(paired) if paired else None,
            "disagreement_rate": len(disagreements) / len(paired) if paired else None,
        },
        "conditional_accuracy": {
            "agreement_true": _subset_accuracy(agreements),
            "agreement_false": _subset_accuracy(disagreements),
            "segmentation_failed": _subset_accuracy(failed),
            "multi_leaf": _subset_accuracy(multi_leaf),
            "severe_fall_armyworm": _subset_accuracy(severe_fall),
        },
    }
