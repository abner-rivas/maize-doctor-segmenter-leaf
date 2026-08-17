"""Unit tests for the phase-one YOLO annotation audit."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.dataset.audit_leaf_annotations import (
    YoloBox,
    clamp_pixel_box,
    label_path_for_image,
    looks_like_polygon,
    parse_annotation_line,
    read_label_file,
    run_audit,
    validate_normalized_bbox,
    yolo_to_pixel_box,
)


class YoloCoordinateTests(TestCase):
    def test_yolo_bbox_converts_to_pixels(self) -> None:
        box = YoloBox(class_id=2, center_x=0.5, center_y=0.5, width=0.5, height=0.5)

        self.assertEqual(yolo_to_pixel_box(box, 100, 80), (25, 20, 75, 60))

    def test_full_image_bbox_is_clamped_to_pixel_indices(self) -> None:
        box = YoloBox(class_id=0, center_x=0.5, center_y=0.5, width=1.0, height=1.0)

        self.assertEqual(yolo_to_pixel_box(box, 100, 80), (0, 0, 99, 79))

    def test_pixel_coordinates_are_clamped_to_image_edges(self) -> None:
        self.assertEqual(clamp_pixel_box((-8, 4, 120, 90), 100, 80), (0, 4, 99, 79))

    def test_normalized_values_are_validated(self) -> None:
        self.assertEqual(validate_normalized_bbox(0.5, 0.5, 0.2, 0.4), [])
        self.assertIn("center_x", validate_normalized_bbox(1.1, 0.5, 0.2, 0.4)[0])
        self.assertTrue(validate_normalized_bbox(0.5, 0.5, 0.0, 0.4))
        self.assertTrue(validate_normalized_bbox(float("nan"), 0.5, 0.2, 0.4))


class AnnotationParsingTests(TestCase):
    def test_invalid_value_count_is_reported(self) -> None:
        result = parse_annotation_line("0 0.5 0.5 0.2", line_number=7)

        self.assertEqual(result.kind, "invalid")
        self.assertEqual(result.line_number, 7)
        self.assertIn("5 valores", result.message or "")

    def test_invalid_normalized_bbox_is_reported(self) -> None:
        result = parse_annotation_line("0 1.5 0.5 0.2 0.2")

        self.assertEqual(result.kind, "invalid")
        self.assertIn("center_x", result.message or "")

    def test_valid_bbox_crossing_image_edge_is_marked_for_clamping(self) -> None:
        result = parse_annotation_line("1 0.05 0.5 0.2 0.4")

        self.assertEqual(result.kind, "bbox")
        self.assertIn("borde", result.message or "")

    def test_polygon_format_is_detected(self) -> None:
        parts = "3 0.1 0.1 0.8 0.1 0.8 0.9 0.1 0.9".split()

        self.assertTrue(looks_like_polygon(parts))
        self.assertEqual(parse_annotation_line(" ".join(parts)).kind, "polygon")

    def test_six_value_record_is_unsupported_not_polygon(self) -> None:
        result = parse_annotation_line("0 0.1 0.2 0.3 0.4 0.5")

        self.assertFalse(looks_like_polygon(result.raw.split()))
        self.assertEqual(result.kind, "invalid")

    def test_empty_label_file_is_preserved_as_empty(self) -> None:
        with TemporaryDirectory() as directory:
            label_path = Path(directory) / "empty.txt"
            label_path.write_text("\n  \n", encoding="utf-8")

            audit = read_label_file(label_path)

        self.assertTrue(audit.empty)
        self.assertEqual(audit.results, ())
        self.assertIsNone(audit.read_error)


class ImageLabelPairingTests(TestCase):
    def test_label_path_uses_same_relative_stem(self) -> None:
        images_dir = Path("dataset/train/images")
        labels_dir = Path("dataset/train/labels")
        image_path = images_dir / "nested" / "leaf.sample.JPG"

        self.assertEqual(
            label_path_for_image(image_path, images_dir, labels_dir),
            labels_dir / "nested" / "leaf.sample.txt",
        )

    def test_output_cannot_be_written_inside_source_dataset(self) -> None:
        with TemporaryDirectory() as directory:
            dataset_root = Path(directory) / "dataset"
            (dataset_root / "train").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "fuera del dataset"):
                run_audit(dataset_root, dataset_root / "audit", 1, 42, ("train",))
