from __future__ import annotations

from unittest import TestCase

from src.evaluation.segmentation_reliability import (
    build_feature_distributions,
    build_reliability_audit_summary,
)


class SegmentationReliabilityMetricsTests(TestCase):
    def setUp(self) -> None:
        self.rows: list[dict[str, object]] = [
            {
                "image_id": "good_one",
                "quality_label": "GOOD",
                "environment": "lab",
                "legacy_status": "reliable",
                "proposed_status": "reliable",
                "mask_area_ratio": 0.6,
                "multi_leaf": False,
            },
            {
                "image_id": "bad_one",
                "quality_label": "BAD",
                "environment": "real",
                "legacy_status": "reliable",
                "proposed_status": "uncertain",
                "mask_area_ratio": 0.5,
                "multi_leaf": False,
            },
            {
                "image_id": "good_multi",
                "quality_label": "GOOD",
                "environment": "real",
                "legacy_status": "uncertain",
                "proposed_status": "reliable",
                "mask_area_ratio": 0.2,
                "multi_leaf": True,
            },
            {
                "image_id": "ambiguous",
                "quality_label": "AMBIGUOUS",
                "environment": "real",
                "legacy_status": "failed",
                "proposed_status": "failed",
                "mask_area_ratio": None,
                "multi_leaf": True,
            },
        ]

    def test_summary_measures_reliability_precision_and_coverage(self) -> None:
        summary = build_reliability_audit_summary(self.rows)
        legacy = summary["legacy_gate"]
        proposed = summary["proposed_gate"]

        self.assertEqual(legacy["reliability_precision"], 0.5)  # type: ignore[index]
        self.assertEqual(proposed["reliability_precision"], 1.0)  # type: ignore[index]
        self.assertEqual(legacy["segmentation_coverage"], 0.5)  # type: ignore[index]
        self.assertEqual(proposed["segmentation_coverage"], 0.5)  # type: ignore[index]
        self.assertEqual(legacy["good_masks_rejected"], 1)  # type: ignore[index]
        self.assertEqual(proposed["false_reliable"], 0)  # type: ignore[index]

    def test_subgroups_and_missing_feature_values_are_supported(self) -> None:
        summary = build_reliability_audit_summary(self.rows)
        distributions = build_feature_distributions(self.rows)

        self.assertIn("multi_leaf", summary["by_subgroup"])  # type: ignore[operator]
        self.assertEqual(distributions["mask_area_ratio"]["GOOD"]["count"], 2)  # type: ignore[index]
        self.assertIsNone(distributions["normalized_perimeter"]["BAD"])

    def test_invalid_or_empty_audit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "vacía"):
            build_reliability_audit_summary([])
        with self.assertRaisesRegex(ValueError, "quality_label"):
            build_reliability_audit_summary(
                [
                    {
                        "image_id": "broken",
                        "quality_label": "UNKNOWN",
                        "legacy_status": "failed",
                        "proposed_status": "failed",
                    }
                ]
            )
