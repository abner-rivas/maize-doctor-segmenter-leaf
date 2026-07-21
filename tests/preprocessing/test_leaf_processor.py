"""Tests for fallbacks and end-to-end in-memory leaf processing."""

import json
import math
from unittest import TestCase

from PIL import Image

from src.preprocessing.leaf_processor import (
    FALLBACK_CENTER_CROP,
    FALLBACK_ORIGINAL,
    FALLBACK_REJECT,
    LeafImageProcessor,
    LeafProcessorConfig,
    apply_fallback,
)


class FallbackTests(TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (100, 80), (10, 20, 30))

    def test_original_returns_independent_full_image(self) -> None:
        result = apply_fallback(self.image, FALLBACK_ORIGINAL)

        self.assertFalse(result.rejected)
        self.assertEqual(result.image.size, self.image.size)  # type: ignore[union-attr]
        self.assertEqual(result.bbox, (0, 0, 100, 80))
        self.assertIsNot(result.image, self.image)

    def test_center_crop_uses_configurable_axis_ratio(self) -> None:
        result = apply_fallback(self.image, FALLBACK_CENTER_CROP, center_crop_ratio=0.5)

        self.assertEqual(result.bbox, (25, 20, 75, 60))
        self.assertEqual(result.image.size, (50, 40))  # type: ignore[union-attr]

    def test_reject_is_controlled_without_image(self) -> None:
        result = apply_fallback(self.image, FALLBACK_REJECT, reason="bbox inválido")

        self.assertTrue(result.rejected)
        self.assertIsNone(result.image)
        self.assertEqual(result.reason, "bbox inválido")

    def test_unknown_fallback_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "fallback desconocido"):
            apply_fallback(self.image, "unknown")


class LeafImageProcessorIntegrationTests(TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (200, 100), (30, 120, 40))

    def test_valid_bbox_returns_consistent_image_and_metadata(self) -> None:
        processor = LeafImageProcessor(
            LeafProcessorConfig(
                margin_ratio=0.1,
                min_area_ratio=0.01,
                target_size=(224, 320),
                padding_value=(2, 3, 4),
            )
        )

        result = processor.process(self.image, (50, 20, 150, 80), confidence=0.9)
        metadata = result.to_metadata()

        self.assertFalse(result.fallback_used)
        self.assertEqual(result.clipped_bbox, (50, 20, 150, 80))
        self.assertEqual(result.expanded_bbox, (40, 14, 160, 86))
        self.assertEqual(result.crop_size, (120, 72))
        self.assertEqual(result.processed_image.size, (320, 224))  # type: ignore[union-attr]
        self.assertEqual(metadata["processed_size"], [320, 224])
        self.assertEqual(metadata["detection_result"]["source"], "manual")  # type: ignore[index]

    def test_invalid_bbox_uses_original_fallback(self) -> None:
        processor = LeafImageProcessor(
            LeafProcessorConfig(min_area_ratio=0.0, fallback=FALLBACK_ORIGINAL)
        )

        result = processor.process(self.image, (40, 20, 10, 80))

        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback, FALLBACK_ORIGINAL)
        self.assertEqual(result.crop_size, self.image.size)
        self.assertEqual(result.processed_image.size, (224, 224))  # type: ignore[union-attr]
        self.assertTrue(result.detection_result.fallback_used)

    def test_invalid_bbox_uses_center_crop_fallback(self) -> None:
        processor = LeafImageProcessor(
            LeafProcessorConfig(
                min_area_ratio=0.0,
                fallback=FALLBACK_CENTER_CROP,
                center_crop_ratio=0.5,
            )
        )

        result = processor.process(self.image, (0, 0, 0, 30))

        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_bbox, (50, 25, 150, 75))
        self.assertEqual(result.crop_size, (100, 50))
        self.assertEqual(result.processed_size, (224, 224))

    def test_invalid_bbox_uses_reject_fallback(self) -> None:
        processor = LeafImageProcessor(
            LeafProcessorConfig(min_area_ratio=0.0, fallback=FALLBACK_REJECT)
        )

        result = processor.process(self.image, (0, 0, 0, 30))

        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback, FALLBACK_REJECT)
        self.assertIsNone(result.processed_image)
        self.assertIsNone(result.processed_size)
        self.assertIn("x2", result.warnings[0])

    def test_small_bbox_uses_fallback_because_of_minimum_area(self) -> None:
        processor = LeafImageProcessor(
            LeafProcessorConfig(min_area_ratio=0.15, fallback=FALLBACK_ORIGINAL)
        )

        result = processor.process(self.image, (0, 0, 10, 10))

        self.assertTrue(result.fallback_used)
        self.assertAlmostEqual(result.detection_result.area_ratio, 0.005)
        self.assertEqual(result.clipped_bbox, (0, 0, 10, 10))

    def test_clipped_bbox_is_recorded_as_warning(self) -> None:
        processor = LeafImageProcessor(
            LeafProcessorConfig(margin_ratio=0.0, min_area_ratio=0.0)
        )

        result = processor.process(self.image, (-10, 0, 100, 100))

        self.assertEqual(result.clipped_bbox, (0, 0, 100, 100))
        self.assertIn("limitado", result.warnings[0])

    def test_stretch_mode_is_available_but_isolated(self) -> None:
        processor = LeafImageProcessor(
            LeafProcessorConfig(
                margin_ratio=0.0,
                min_area_ratio=0.0,
                target_size=(100, 300),
                preserve_aspect_ratio=False,
            )
        )

        result = processor.process(self.image, (0, 0, 100, 100))

        self.assertEqual(result.processed_size, (300, 100))
        self.assertEqual(result.padding, (0, 0, 0, 0))
        self.assertFalse(result.preserve_aspect_ratio)

    def test_nan_bbox_fallback_metadata_is_valid_json(self) -> None:
        processor = LeafImageProcessor(
            LeafProcessorConfig(min_area_ratio=0.0, fallback=FALLBACK_ORIGINAL)
        )

        result = processor.process(self.image, (0, 0, math.nan, 20))
        serialized = json.dumps(result.to_metadata(), allow_nan=False)

        self.assertTrue(result.fallback_used)
        self.assertIn("nan", serialized)

    def test_processor_does_not_modify_original(self) -> None:
        before = self.image.tobytes()

        LeafImageProcessor().process(self.image, (0, 0, 200, 100))

        self.assertEqual(self.image.size, (200, 100))
        self.assertEqual(self.image.tobytes(), before)
