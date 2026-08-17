"""Tests for exact binary masking without interpolation or source mutation."""

from unittest import TestCase

import numpy as np
from PIL import Image

from src.preprocessing.leaf_mask import (
    InvalidLeafMaskError,
    apply_leaf_mask,
    binary_mask_array,
    mask_area_ratio,
    mask_bbox,
    mask_geometry,
)


class LeafMaskTests(TestCase):
    def test_background_is_exactly_black_and_leaf_pixels_are_unchanged(self) -> None:
        pixels = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(6, 8, 3)
        image = Image.fromarray(pixels, mode="RGB")
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:5, 2:7] = 1
        before = image.tobytes()

        result = apply_leaf_mask(image, mask)
        output = np.asarray(result)

        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, image.size)
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.all(output[mask == 0] == (0, 0, 0)))
        np.testing.assert_array_equal(output[mask == 1], pixels[mask == 1])
        self.assertEqual(image.tobytes(), before)

    def test_custom_rgb_background_is_supported(self) -> None:
        image = Image.new("RGB", (4, 3), (20, 30, 40))
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[1, 1] = 255

        result = apply_leaf_mask(image, mask, (1, 2, 3))

        self.assertEqual(result.getpixel((0, 0)), (1, 2, 3))
        self.assertEqual(result.getpixel((1, 1)), (20, 30, 40))

    def test_different_spatial_resolution_is_rejected_not_resized(self) -> None:
        with self.assertRaisesRegex(InvalidLeafMaskError, "no coincide"):
            apply_leaf_mask(
                Image.new("RGB", (10, 8)),
                np.ones((10, 8), dtype=np.uint8),
            )

    def test_interpolated_non_binary_values_are_rejected(self) -> None:
        mask = np.asarray([[0, 12], [255, 0]], dtype=np.uint8)

        with self.assertRaisesRegex(InvalidLeafMaskError, "binaria"):
            binary_mask_array(mask)

    def test_empty_mask_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidLeafMaskError, "vacía"):
            apply_leaf_mask(Image.new("RGB", (4, 4)), np.zeros((4, 4), dtype=np.uint8))

    def test_mask_geometry_uses_half_open_bbox(self) -> None:
        mask = np.zeros((10, 20), dtype=np.uint8)
        mask[2:8, 3:17] = 1

        self.assertEqual(mask_bbox(mask), (3, 2, 17, 8))
        self.assertAlmostEqual(mask_area_ratio(mask), 84 / 200)

    def test_horizontal_and_vertical_masks_keep_their_shape(self) -> None:
        horizontal = np.ones((20, 100), dtype=np.bool_)
        vertical = np.ones((100, 20), dtype=np.bool_)

        self.assertEqual(binary_mask_array(horizontal).shape, (20, 100))
        self.assertEqual(binary_mask_array(vertical).shape, (100, 20))

    def test_mask_geometry_is_exact_and_reports_disconnected_regions(self) -> None:
        mask = np.zeros((10, 20), dtype=np.uint8)
        mask[1:5, 2:8] = 1
        mask[7:9, 15:18] = 1

        geometry = mask_geometry(mask)

        self.assertEqual(geometry.area_pixels, 30)
        self.assertEqual(geometry.bbox, (2, 1, 18, 9))
        self.assertEqual(geometry.connected_components, 2)
        self.assertAlmostEqual(geometry.largest_component_ratio, 24 / 30)
        self.assertEqual(geometry.border_contact_count, 0)
        self.assertEqual(geometry.perimeter_edges, 30)
        self.assertAlmostEqual(geometry.mask_bbox_ratio, 30 / (16 * 8))
