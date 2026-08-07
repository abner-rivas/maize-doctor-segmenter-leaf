from __future__ import annotations

from unittest import TestCase

from src.evaluation.dual_perspective import (
    DualPerspectiveExperimentRecord,
    build_dual_perspective_summary,
)
from src.inference.classifier import ClassificationPrediction, RankedClassPrediction
from src.inference.dual_perspective import (
    DualPerspectiveResult,
    SegmentationAssessment,
    SegmentationStatus,
    SegmentedLeafView,
)


def _prediction(name: str, confidence: float = 0.8) -> ClassificationPrediction:
    index = 0 if name == "healthy" else 1
    top = RankedClassPrediction(name, index, confidence)
    return ClassificationPrediction(name, index, confidence, (top,))


def _result(
    full: str,
    segmented: str | None,
    status: SegmentationStatus,
) -> DualPerspectiveResult:
    full_prediction = _prediction(full)
    segmented_prediction = _prediction(segmented) if segmented else None
    return DualPerspectiveResult(
        full_image=full_prediction,
        segmented_leaf=SegmentedLeafView(
            available=segmented_prediction is not None,
            prediction=segmented_prediction,
            reason=None if segmented_prediction else status.value,
        ),
        segmentation=SegmentationAssessment(
            status=status,
            reason=None if status is SegmentationStatus.RELIABLE else status.value,
            number_of_instances=1 if segmented_prediction else 0,
            eligible_instances=1 if segmented_prediction else 0,
            selected_instance=0 if segmented_prediction else None,
            confidence=0.9 if segmented_prediction else None,
            mask_area_ratio=0.5 if segmented_prediction else None,
            metadata={},
        ),
        agreement=(full == segmented) if segmented_prediction else None,
    )


class DualPerspectiveMetricsTests(TestCase):
    def setUp(self) -> None:
        self.records = [
            DualPerspectiveExperimentRecord(
                "one.jpg",
                "healthy",
                "lab",
                _result("healthy", "healthy", SegmentationStatus.RELIABLE),
            ),
            DualPerspectiveExperimentRecord(
                "two.jpg",
                "healthy",
                "real",
                _result("healthy", "common_rust", SegmentationStatus.RELIABLE),
                multi_leaf=True,
            ),
            DualPerspectiveExperimentRecord(
                "three.jpg",
                "common_rust",
                "real",
                _result("healthy", None, SegmentationStatus.FAILED),
                severe_fall_armyworm=True,
            ),
            DualPerspectiveExperimentRecord(
                "four.jpg",
                "healthy",
                "real",
                _result("healthy", None, SegmentationStatus.UNCERTAIN),
                multi_leaf=True,
            ),
        ]
        self.classes = ["healthy", "common_rust"]

    def test_coverage_and_view_rates_use_the_correct_denominators(self) -> None:
        summary = build_dual_perspective_summary(self.records, self.classes)

        self.assertEqual(summary["segmentation"]["coverage"], 0.5)  # type: ignore[index]
        self.assertEqual(summary["views"]["paired_images"], 2)  # type: ignore[index]
        self.assertEqual(summary["views"]["agreement_rate"], 0.5)  # type: ignore[index]
        self.assertEqual(summary["views"]["disagreement_rate"], 0.5)  # type: ignore[index]

    def test_statuses_and_conditional_accuracies_are_reported(self) -> None:
        summary = build_dual_perspective_summary(self.records, self.classes)
        segmentation = summary["segmentation"]
        conditional = summary["conditional_accuracy"]

        self.assertEqual(
            segmentation["status_counts"],
            {  # type: ignore[index]
                "reliable": 2,
                "uncertain": 1,
                "failed": 1,
            },
        )
        self.assertEqual(
            conditional["agreement_false"]["full_image_accuracy"],  # type: ignore[index]
            1.0,
        )
        self.assertEqual(
            conditional["agreement_false"]["segmented_leaf_accuracy"],  # type: ignore[index]
            0.0,
        )
        self.assertEqual(
            conditional["segmentation_failed"]["full_image_accuracy"],  # type: ignore[index]
            0.0,
        )

    def test_current_classification_metrics_remain_available_by_view(self) -> None:
        summary = build_dual_perspective_summary(self.records, self.classes)

        self.assertEqual(summary["full_image"]["images"], 4)  # type: ignore[index]
        self.assertEqual(summary["segmented_leaf"]["images"], 2)  # type: ignore[index]
        self.assertIn("macro_f1", summary["full_image"])  # type: ignore[operator]
        self.assertIn("confusion_matrix", summary["segmented_leaf"])  # type: ignore[operator]
        self.assertEqual(summary["full_image"]["by_environment"]["lab"]["images"], 1)  # type: ignore[index]

    def test_empty_records_and_unknown_truth_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "vacío"):
            build_dual_perspective_summary([], self.classes)
        with self.assertRaisesRegex(ValueError, "fuera del mapeo"):
            build_dual_perspective_summary(self.records, ["healthy"])
