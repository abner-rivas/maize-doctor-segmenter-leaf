"""Tests for bbox validation, margin expansion, and RGB crops."""

import math
from unittest import TestCase

from PIL import Image

from src.preprocessing.leaf_roi import (
    InvalidBoundingBoxError,
    bbox_area,
    bbox_area_ratio,
    bbox_height,
    bbox_width,
    crop_leaf_region,
    expand_bbox,
    validate_bbox,
)


class BoundingBoxValidationTests(TestCase):
    def test_completely_valid_bbox(self) -> None:
        result = validate_bbox(200, 100, (20, 10, 180, 90), 0.1)

        self.assertTrue(result.detected)
        self.assertEqual(result.bbox, (20, 10, 180, 90))
        self.assertAlmostEqual(result.area_ratio, 0.64)
        self.assertEqual(bbox_width(result.bbox), 160)  # type: ignore[arg-type]
        self.assertEqual(bbox_height(result.bbox), 80)  # type: ignore[arg-type]
        self.assertEqual(bbox_area(result.bbox), 12800)  # type: ignore[arg-type]
        area_ratio = bbox_area_ratio(result.bbox, 200, 100)  # type: ignore[arg-type]
        self.assertAlmostEqual(area_ratio, 0.64)

    def test_negative_coordinates_are_clipped(self) -> None:
        result = validate_bbox(100, 80, (-20, -10, 40, 30), 0.0)

        self.assertTrue(result.detected)
        self.assertEqual(result.bbox, (0, 0, 40, 30))

    def test_coordinates_above_image_size_are_clipped(self) -> None:
        result = validate_bbox(100, 80, (20, 10, 140, 120), 0.0)

        self.assertTrue(result.detected)
        self.assertEqual(result.bbox, (20, 10, 100, 80))

    def test_convertible_float_values_are_rounded_outward(self) -> None:
        result = validate_bbox(100, 80, ("10.8", "5.2", "30.1", "20.6"), 0.0)

        self.assertEqual(result.bbox, (10, 5, 31, 21))

    def test_x2_not_greater_than_x1_is_rejected(self) -> None:
        result = validate_bbox(100, 80, (20, 10, 20, 30), 0.0)

        self.assertFalse(result.detected)
        self.assertIn("x2", result.reason or "")

    def test_y2_not_greater_than_y1_is_rejected(self) -> None:
        result = validate_bbox(100, 80, (20, 30, 40, 10), 0.0)

        self.assertFalse(result.detected)
        self.assertIn("y2", result.reason or "")

    def test_nan_is_rejected(self) -> None:
        result = validate_bbox(100, 80, (0, 0, math.nan, 20), 0.0)

        self.assertFalse(result.detected)
        self.assertIn("finito", result.reason or "")

    def test_infinity_is_rejected(self) -> None:
        result = validate_bbox(100, 80, (0, 0, math.inf, 20), 0.0)

        self.assertFalse(result.detected)
        self.assertIn("finito", result.reason or "")

    def test_region_empty_after_clipping_is_rejected(self) -> None:
        result = validate_bbox(100, 80, (-20, 5, -1, 30), 0.0)

        self.assertFalse(result.detected)
        self.assertIn("vacío", result.reason or "")

    def test_minimum_area_ratio_is_enforced(self) -> None:
        result = validate_bbox(100, 100, (0, 0, 10, 10), 0.15)

        self.assertFalse(result.detected)
        self.assertEqual(result.bbox, (0, 0, 10, 10))
        self.assertAlmostEqual(result.area_ratio, 0.01)
        self.assertIn("min_area_ratio", result.reason or "")

    def test_zero_sized_image_is_controlled_rejection(self) -> None:
        result = validate_bbox(0, 100, (0, 0, 10, 10), 0.0)

        self.assertFalse(result.detected)
        self.assertIn("mayores que cero", result.reason or "")


class BoundingBoxMarginTests(TestCase):
    def test_zero_margin_preserves_bbox(self) -> None:
        bbox = (100, 50, 500, 350)

        self.assertEqual(expand_bbox(bbox, 1000, 500, 0), bbox)

    def test_eight_percent_margin_expands_each_side(self) -> None:
        self.assertEqual(
            expand_bbox((100, 50, 500, 350), 1000, 500, 0.08),
            (68, 26, 532, 374),
        )

    def test_margin_stops_at_left_edge(self) -> None:
        self.assertEqual(expand_bbox((5, 20, 55, 60), 100, 100, 0.2)[0], 0)

    def test_margin_stops_at_right_edge(self) -> None:
        self.assertEqual(expand_bbox((50, 20, 95, 60), 100, 100, 0.2)[2], 100)

    def test_margin_stops_at_top_edge(self) -> None:
        self.assertEqual(expand_bbox((20, 5, 60, 55), 100, 100, 0.2)[1], 0)

    def test_margin_stops_at_bottom_edge(self) -> None:
        self.assertEqual(expand_bbox((20, 50, 60, 95), 100, 100, 0.2)[3], 100)

    def test_negative_margin_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "margin_ratio"):
            expand_bbox((10, 10, 20, 20), 100, 100, -0.1)

    def test_margin_greater_than_one_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "margin_ratio"):
            expand_bbox((10, 10, 20, 20), 100, 100, 1.01)


class CropLeafRegionTests(TestCase):
    def test_rgb_crop(self) -> None:
        image = Image.new("RGB", (100, 80), (10, 20, 30))

        cropped = crop_leaf_region(image, (10, 5, 60, 45))

        self.assertEqual(cropped.mode, "RGB")
        self.assertEqual(cropped.size, (50, 40))
        self.assertEqual(cropped.getpixel((0, 0)), (10, 20, 30))

    def test_grayscale_crop_becomes_rgb(self) -> None:
        cropped = crop_leaf_region(Image.new("L", (40, 30), 90), (5, 5, 25, 20))

        self.assertEqual(cropped.mode, "RGB")
        self.assertEqual(cropped.getpixel((0, 0)), (90, 90, 90))

    def test_rgba_transparency_is_composited(self) -> None:
        image = Image.new("RGBA", (20, 20), (255, 0, 0, 0))

        cropped = crop_leaf_region(image, (0, 0, 20, 20), transparency_background=(1, 2, 3))

        self.assertEqual(cropped.mode, "RGB")
        self.assertEqual(cropped.getpixel((0, 0)), (1, 2, 3))

    def test_horizontal_and_vertical_regions_keep_geometry(self) -> None:
        horizontal = crop_leaf_region(Image.new("RGB", (100, 50)), (5, 10, 95, 30))
        vertical = crop_leaf_region(Image.new("RGB", (50, 100)), (10, 5, 30, 95))

        self.assertEqual(horizontal.size, (90, 20))
        self.assertEqual(vertical.size, (20, 90))

    def test_one_pixel_region_is_geometrically_valid(self) -> None:
        cropped = crop_leaf_region(Image.new("RGB", (10, 10)), (4, 4, 5, 5))

        self.assertEqual(cropped.size, (1, 1))

    def test_crop_does_not_modify_original_image(self) -> None:
        image = Image.new("RGB", (20, 20), (4, 5, 6))
        before = image.tobytes()

        crop_leaf_region(image, (5, 5, 15, 15))

        self.assertEqual(image.size, (20, 20))
        self.assertEqual(image.tobytes(), before)

    def test_empty_crop_is_rejected_with_specific_error(self) -> None:
        with self.assertRaises(InvalidBoundingBoxError):
            crop_leaf_region(Image.new("RGB", (20, 20)), (10, 10, 10, 15))
