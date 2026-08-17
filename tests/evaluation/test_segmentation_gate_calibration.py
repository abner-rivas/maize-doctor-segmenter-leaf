"""Tests for frozen-audit quality-gate calibration."""

from __future__ import annotations

from src.evaluation.segmentation_gate_calibration import (
    calibrate_gate,
    calibrated_status,
    evaluate_gate,
)
from src.segmentation.quality import SegmentationQualityGateConfig


def _row(
    image_id: str,
    quality: str,
    *,
    area: float = 0.30,
    bbox_ratio: float = 0.90,
    perimeter: float = 4.0,
) -> dict[str, str]:
    return {
        "image_id": image_id,
        "quality_label": quality,
        "segmentation_available": "True",
        "fallback_used": "False",
        "eligible_instances": "1",
        "instance_score_margin": "",
        "mask_area_ratio": str(area),
        "mask_bbox_ratio": str(bbox_ratio),
        "normalized_perimeter": str(perimeter),
    }


def test_calibrated_status_replays_combined_geometry_rule() -> None:
    gate = SegmentationQualityGateConfig(
        large_mask_area_ratio=0.25,
        min_large_mask_bbox_ratio=0.80,
        max_large_mask_normalized_perimeter=8.0,
    )
    suspicious = _row("ambiguous", "AMBIGUOUS", area=0.29, bbox_ratio=0.75, perimeter=8.31)
    assert calibrated_status(suspicious, gate) == "uncertain"
    assert calibrated_status(_row("good", "GOOD"), gate) == "reliable"


def test_evaluate_gate_reports_false_reliable_and_good_coverage() -> None:
    rows = [_row("good", "GOOD"), _row("ambiguous", "AMBIGUOUS")]
    metrics = evaluate_gate(rows, SegmentationQualityGateConfig())
    assert metrics["reliable_good"] == 1
    assert metrics["false_reliable"] == 1
    assert metrics["reliability_precision"] == 0.5
    assert metrics["good_mask_coverage"] == 1.0


def test_calibration_meets_precision_without_rejecting_good_mask() -> None:
    rows = [
        _row("good", "GOOD"),
        _row(
            "ambiguous",
            "AMBIGUOUS",
            area=0.29,
            bbox_ratio=0.75,
            perimeter=8.31,
        ),
    ]
    recommended, sweep = calibrate_gate(
        rows,
        baseline=SegmentationQualityGateConfig(),
        minimum_precision=1.0,
    )
    assert len(sweep) == 640
    assert recommended["reliability_precision"] == 1.0
    assert recommended["reliable_good"] == 1
    assert recommended["false_reliable_ids"] == []
