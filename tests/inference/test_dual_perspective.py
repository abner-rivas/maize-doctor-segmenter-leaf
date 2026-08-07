from __future__ import annotations

import base64
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from unittest import TestCase

import numpy as np
from PIL import Image, ImageDraw

from src.inference.classifier import ClassificationPrediction, RankedClassPrediction
from src.inference.dual_perspective import (
    DualPerspectiveConfig,
    SegmentationQualityGateConfig,
    SegmentationStatus,
    classify_dual_perspective,
)
from src.preprocessing.segmented_leaf_processor import (
    MASK_BLACK,
    LeafMaskProcessorConfig,
    SegmentedLeafProcessor,
)
from src.segmentation.leaf_segmenter import LeafInstance


def _mask(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(
        (bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1),
        fill=255,
    )
    return mask


def _instance(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    confidence: float,
    source_index: int,
) -> LeafInstance:
    return LeafInstance(
        mask=_mask(size, bbox),
        confidence=confidence,
        bbox=bbox,
        source_index=source_index,
    )


def _instance_from_mask(
    mask: Image.Image,
    confidence: float = 0.95,
    source_index: int = 0,
) -> LeafInstance:
    bbox = mask.getbbox()
    if bbox is None:
        bbox = (0, 0, 1, 1)
    return LeafInstance(
        mask=mask,
        confidence=confidence,
        bbox=bbox,
        source_index=source_index,
    )


def _prediction(class_name: str, confidence: float = 0.8) -> ClassificationPrediction:
    index = 0 if class_name == "healthy" else 1
    ranked = RankedClassPrediction(class_name, index, confidence)
    return ClassificationPrediction(class_name, index, confidence, (ranked,))


class _StaticSegmenter:
    def __init__(self, instances: Sequence[LeafInstance]) -> None:
        self.instances = tuple(instances)

    def segment(self, image: Image.Image) -> tuple[LeafInstance, ...]:
        return self.instances

    def to_metadata(self) -> dict[str, object]:
        return {"segmenter_model": "static", "proposal_confidence_threshold": 0.5}


class _RaisingSegmenter:
    def segment(self, image: Image.Image) -> tuple[LeafInstance, ...]:
        raise RuntimeError("synthetic segmentation failure")

    def to_metadata(self) -> dict[str, object]:
        return {"segmenter_model": "raising"}


class _PredictionSequence:
    def __init__(self, *predictions: ClassificationPrediction) -> None:
        self.predictions = predictions
        self.calls: list[Image.Image] = []

    def __call__(self, image: Image.Image) -> ClassificationPrediction:
        self.calls.append(image)
        return self.predictions[len(self.calls) - 1]


class DualPerspectiveTests(TestCase):
    def setUp(self) -> None:
        self.size = (100, 80)
        self.image = Image.new("RGB", self.size, (20, 120, 40))
        self.policy = DualPerspectiveConfig()

    def _processor(
        self,
        instances: Sequence[LeafInstance],
    ) -> SegmentedLeafProcessor:
        return SegmentedLeafProcessor(
            _StaticSegmenter(instances),
            LeafMaskProcessorConfig(processing_profile=MASK_BLACK),
        )

    def _clear_leaf(self) -> LeafInstance:
        return _instance(self.size, (20, 10, 80, 70), 0.95, 0)

    def test_reliable_segmentation_and_equal_predictions(self) -> None:
        classifier = _PredictionSequence(
            _prediction("healthy", 0.81),
            _prediction("healthy", 0.92),
        )

        result = classify_dual_perspective(
            self.image,
            classifier=classifier,
            leaf_processor=self._processor((self._clear_leaf(),)),
            config=self.policy,
        )

        self.assertEqual(len(classifier.calls), 2)
        self.assertIs(classifier.calls[0], self.image)
        self.assertEqual(result.segmentation.status, SegmentationStatus.RELIABLE)
        self.assertTrue(result.segmented_leaf.available)
        self.assertTrue(result.agreement)
        self.assertEqual(result.full_image.confidence, 0.81)
        self.assertEqual(result.segmented_leaf.prediction.confidence, 0.92)  # type: ignore[union-attr]

    def test_reliable_segmentation_and_different_predictions(self) -> None:
        classifier = _PredictionSequence(
            _prediction("healthy"),
            _prediction("common_rust"),
        )

        result = classify_dual_perspective(
            self.image,
            classifier=classifier,
            leaf_processor=self._processor((self._clear_leaf(),)),
        )

        self.assertFalse(result.agreement)
        self.assertNotEqual(
            result.full_image.class_name,
            result.segmented_leaf.prediction.class_name,  # type: ignore[union-attr]
        )
        self.assertFalse(result.to_metadata()["fusion_applied"])

    def test_no_detection_keeps_full_image_and_agreement_is_null(self) -> None:
        classifier = _PredictionSequence(_prediction("healthy"))

        result = classify_dual_perspective(
            self.image,
            classifier=classifier,
            leaf_processor=self._processor(()),
        )

        self.assertEqual(len(classifier.calls), 1)
        self.assertEqual(result.full_image.class_name, "healthy")
        self.assertFalse(result.segmented_leaf.available)
        self.assertEqual(result.segmentation.status, SegmentationStatus.FAILED)
        self.assertEqual(result.segmentation.reason, "no_detection")
        self.assertIsNone(result.agreement)

    def test_low_segmentation_confidence_is_failed(self) -> None:
        low = _instance(self.size, (20, 10, 80, 70), 0.49, 0)

        result = classify_dual_perspective(
            self.image,
            classifier=_PredictionSequence(_prediction("healthy")),
            leaf_processor=self._processor((low,)),
        )

        self.assertEqual(result.segmentation.reason, "low_segmentation_confidence")
        self.assertFalse(result.segmented_leaf.available)

    def test_multiple_eligible_leaves_are_uncertain(self) -> None:
        instances = (
            _instance(self.size, (5, 5, 45, 60), 0.90, 0),
            _instance(self.size, (50, 10, 95, 70), 0.88, 1),
        )

        result = classify_dual_perspective(
            self.image,
            classifier=_PredictionSequence(_prediction("healthy")),
            leaf_processor=self._processor(instances),
        )

        self.assertEqual(result.segmentation.status, SegmentationStatus.UNCERTAIN)
        self.assertEqual(
            result.segmentation.reason,
            "ambiguous_multiple_eligible_leaves",
        )
        self.assertFalse(result.segmented_leaf.available)

    def test_clearly_selected_multi_leaf_can_be_reliable(self) -> None:
        instances = (
            _instance(self.size, (10, 5, 80, 75), 0.98, 0),
            _instance(self.size, (85, 2, 98, 15), 0.70, 1),
        )
        policy = DualPerspectiveConfig(reject_multiple_eligible=False)
        classifier = _PredictionSequence(
            _prediction("healthy"),
            _prediction("healthy"),
        )

        result = classify_dual_perspective(
            self.image,
            classifier=classifier,
            leaf_processor=self._processor(instances),
            config=policy,
        )

        self.assertEqual(result.segmentation.status, SegmentationStatus.RELIABLE)
        margin = result.segmentation.quality_gate_metrics["instance_score_margin"]
        self.assertIsInstance(margin, float)
        self.assertGreaterEqual(margin, 0.33)  # type: ignore[operator]
        self.assertEqual(len(classifier.calls), 2)

    def test_ambiguous_multi_leaf_uses_score_margin_when_enabled(self) -> None:
        instances = (
            _instance(self.size, (5, 5, 45, 60), 0.90, 0),
            _instance(self.size, (50, 10, 95, 70), 0.88, 1),
        )
        classifier = _PredictionSequence(_prediction("healthy"))

        result = classify_dual_perspective(
            self.image,
            classifier=classifier,
            leaf_processor=self._processor(instances),
            config=DualPerspectiveConfig(reject_multiple_eligible=False),
        )

        self.assertEqual(result.segmentation.status, SegmentationStatus.UNCERTAIN)
        self.assertEqual(result.segmentation.reason, "ambiguous_instance_score_margin")
        self.assertEqual(len(classifier.calls), 1)

    def test_evident_subsegmentation_geometry_is_uncertain(self) -> None:
        array = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        for start in range(0, self.size[0], 5):
            array[:, start : start + 3] = 255
        irregular = Image.fromarray(array, mode="L")
        classifier = _PredictionSequence(_prediction("fall_armyworm"))

        result = classify_dual_perspective(
            self.image,
            classifier=classifier,
            leaf_processor=self._processor((_instance_from_mask(irregular),)),
            config=self.policy,
        )

        self.assertEqual(result.segmentation.status, SegmentationStatus.UNCERTAIN)
        self.assertEqual(result.segmentation.reason, "suspicious_large_mask_geometry")
        self.assertFalse(result.segmented_leaf.available)
        self.assertEqual(len(classifier.calls), 1)

    def test_abnormally_large_mask_is_uncertain(self) -> None:
        array = np.ones((self.size[1], self.size[0]), dtype=np.uint8) * 255
        array[0, 0] = 0
        near_full = Image.fromarray(array, mode="L")

        result = classify_dual_perspective(
            self.image,
            classifier=_PredictionSequence(_prediction("healthy")),
            leaf_processor=self._processor((_instance_from_mask(near_full),)),
            config=self.policy,
        )

        self.assertEqual(result.segmentation.status, SegmentationStatus.UNCERTAIN)
        self.assertEqual(result.segmentation.reason, "excessive_mask_area_ratio")

    def test_real_67220087_mask_cannot_pass_silently_as_reliable(self) -> None:
        fixture = Path(__file__).parents[1] / "fixtures" / "segmentation" / "67220087_mask.png.b64"
        mask = Image.open(BytesIO(base64.b64decode(fixture.read_text().strip()))).convert("L")
        image = Image.new("RGB", mask.size, (20, 120, 40))
        classifier = _PredictionSequence(_prediction("fall_armyworm"))

        result = classify_dual_perspective(
            image,
            classifier=classifier,
            leaf_processor=self._processor(
                (_instance_from_mask(mask, confidence=0.6492254137992859),)
            ),
            config=self.policy,
        )

        metrics = result.segmentation.quality_gate_metrics
        self.assertAlmostEqual(metrics["mask_area_ratio"], 0.5418925383)  # type: ignore[arg-type]
        self.assertAlmostEqual(metrics["mask_bbox_ratio"], 0.6695724980)  # type: ignore[arg-type]
        self.assertAlmostEqual(metrics["normalized_perimeter"], 9.9700470867)  # type: ignore[arg-type]
        self.assertEqual(result.segmentation.status, SegmentationStatus.UNCERTAIN)
        self.assertEqual(result.segmentation.reason, "suspicious_large_mask_geometry")
        self.assertFalse(result.segmented_leaf.available)
        self.assertEqual(len(classifier.calls), 1)

    def test_invalid_mask_uses_full_image_fallback(self) -> None:
        invalid = LeafInstance(
            mask=Image.new("L", self.size, 0),
            confidence=0.99,
            bbox=(0, 0, 1, 1),
            source_index=0,
        )

        result = classify_dual_perspective(
            self.image,
            classifier=_PredictionSequence(_prediction("common_rust", 0.77)),
            leaf_processor=self._processor((invalid,)),
        )

        self.assertEqual(result.full_image.confidence, 0.77)
        self.assertEqual(result.segmentation.reason, "invalid_mask")
        self.assertFalse(result.segmented_leaf.available)

    def test_segmenter_exception_never_suppresses_full_image(self) -> None:
        processor = SegmentedLeafProcessor(
            _RaisingSegmenter(),
            LeafMaskProcessorConfig(processing_profile=MASK_BLACK),
        )

        result = classify_dual_perspective(
            self.image,
            classifier=_PredictionSequence(_prediction("healthy", 0.83)),
            leaf_processor=processor,
        )

        self.assertEqual(result.full_image.class_name, "healthy")
        self.assertEqual(result.segmentation.reason, "segmentation_error")
        self.assertEqual(
            result.segmentation.metadata["error_type"],
            "RuntimeError",
        )
        self.assertIsNone(result.agreement)

    def test_mapping_controls_multi_leaf_policy_and_profile(self) -> None:
        policy = DualPerspectiveConfig.from_mapping(
            {
                "dual_perspective": {
                    "segmented_profile": "mask_black",
                    "reject_multiple_eligible": False,
                    "quality_gate": {
                        "max_mask_area_ratio": 0.999,
                        "large_mask_area_ratio": 0.50,
                        "min_large_mask_bbox_ratio": 0.70,
                        "max_large_mask_normalized_perimeter": 8.0,
                        "min_multi_instance_score_margin": 0.33,
                    },
                }
            }
        )

        self.assertEqual(policy.segmented_profile, MASK_BLACK)
        self.assertFalse(policy.reject_multiple_eligible)
        self.assertEqual(
            policy.quality_gate,
            SegmentationQualityGateConfig(),
        )

    def test_quality_gate_metadata_contains_reasons_metrics_and_thresholds(self) -> None:
        instances = (
            _instance(self.size, (5, 5, 45, 60), 0.90, 0),
            _instance(self.size, (50, 10, 95, 70), 0.88, 1),
        )

        result = classify_dual_perspective(
            self.image,
            classifier=_PredictionSequence(_prediction("healthy")),
            leaf_processor=self._processor(instances),
            config=DualPerspectiveConfig(reject_multiple_eligible=False),
        )
        gate = result.to_metadata()["segmentation"]["quality_gate"]  # type: ignore[index]

        self.assertEqual(gate["reasons"], ["ambiguous_instance_score_margin"])
        self.assertIn("instance_score_margin", gate["metrics"])
        self.assertEqual(gate["thresholds"]["min_multi_instance_score_margin"], 0.33)
