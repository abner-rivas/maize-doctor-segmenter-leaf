"""Tests for external segmentation dataset audit primitives."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.data.segmentation_audit import (
    DatasetInventory,
    audit_cache_is_current,
    build_audit_input_fingerprint,
    compare_yolo_coco,
    compute_image_statistics,
    deterministic_sample,
    evaluate_coco_recovery,
    find_exact_duplicates,
    load_coco_segmentation,
    load_yolo_dataset,
    parse_yolo_segmentation_line,
    polygon_area,
    polygon_touches_border,
)


class YoloSegmentationParserTests(TestCase):
    def test_valid_line(self) -> None:
        parsed = parse_yolo_segmentation_line("0 0.1 0.1 0.9 0.1 0.5 0.9")

        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.class_id, 0)
        self.assertEqual(len(parsed.points), 3)

    def test_five_field_row_is_explicit_bbox_format(self) -> None:
        parsed = parse_yolo_segmentation_line("1 0.1 0.1 0.9 0.9")

        self.assertFalse(parsed.valid)
        self.assertEqual(
            {issue["issue_type"] for issue in parsed.issues},
            {"bbox_format_in_segmentation_label"},
        )
        self.assertEqual(parsed.annotation_format, "yolo_bbox")
        self.assertEqual(parsed.points, [])

    def test_incomplete_coordinate_pair(self) -> None:
        parsed = parse_yolo_segmentation_line("0 0 0 1 0 1 1 0.5")

        self.assertFalse(parsed.valid)
        self.assertIn(
            "incomplete_coordinate_pair",
            {issue["issue_type"] for issue in parsed.issues},
        )

    def test_concatenated_numeric_token(self) -> None:
        parsed = parse_yolo_segmentation_line(
            "0 0.1 0.1 0.87665343915343920.5684523809523809 0.2 0.5 0.9"
        )

        self.assertFalse(parsed.valid)
        self.assertIn(
            "concatenated_numeric_token",
            {issue["issue_type"] for issue in parsed.issues},
        )

    def test_non_numeric_token(self) -> None:
        parsed = parse_yolo_segmentation_line("0 0.1 0.1 abc 0.2 0.5 0.9")

        self.assertIn(
            "non_numeric_token",
            {issue["issue_type"] for issue in parsed.issues},
        )

    def test_out_of_range_coordinate(self) -> None:
        parsed = parse_yolo_segmentation_line("0 0.1 0.1 1.2 0.2 0.5 0.9")

        self.assertIn(
            "coordinate_out_of_range",
            {issue["issue_type"] for issue in parsed.issues},
        )

    def test_nan_and_infinity(self) -> None:
        parsed = parse_yolo_segmentation_line("0 nan 0.1 inf 0.2 0.5 0.9")

        self.assertEqual(
            sum(
                issue["issue_type"] == "non_finite_coordinate"
                for issue in parsed.issues
            ),
            2,
        )

    def test_empty_line(self) -> None:
        parsed = parse_yolo_segmentation_line("  ")

        self.assertFalse(parsed.valid)
        self.assertEqual(parsed.issues[0]["issue_type"], "empty_line")

    def test_self_intersecting_polygon(self) -> None:
        parsed = parse_yolo_segmentation_line("0 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.9")

        self.assertFalse(parsed.valid)
        self.assertIn(
            "self_intersection",
            {issue["issue_type"] for issue in parsed.issues},
        )
        self.assertIn(
            "non_simple_polygon",
            {issue["issue_type"] for issue in parsed.issues},
        )

    def test_repeated_vertex(self) -> None:
        parsed = parse_yolo_segmentation_line(
            "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9 0.9 0.1"
        )

        self.assertFalse(parsed.valid)
        self.assertIn(
            "repeated_vertex",
            {issue["issue_type"] for issue in parsed.issues},
        )

    def test_zero_length_edge(self) -> None:
        parsed = parse_yolo_segmentation_line(
            "0 0.1 0.1 0.9 0.1 0.9 0.1 0.1 0.9"
        )

        self.assertFalse(parsed.valid)
        self.assertIn(
            "zero_length_edge",
            {issue["issue_type"] for issue in parsed.issues},
        )

    def test_insufficient_unique_vertices(self) -> None:
        parsed = parse_yolo_segmentation_line("0 0.1 0.1 0.9 0.1 0.1 0.1")

        self.assertFalse(parsed.valid)
        self.assertIn(
            "insufficient_unique_vertices",
            {issue["issue_type"] for issue in parsed.issues},
        )

    def test_zero_area_polygon(self) -> None:
        parsed = parse_yolo_segmentation_line("0 0.1 0.1 0.5 0.5 0.9 0.9")

        self.assertFalse(parsed.valid)
        self.assertIn(
            "zero_or_near_zero_area",
            {issue["issue_type"] for issue in parsed.issues},
        )

    def test_polygon_area_and_border(self) -> None:
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        self.assertAlmostEqual(polygon_area(points), 1.0)
        borders = polygon_touches_border(points)
        self.assertTrue(all(borders.values()))


class SegmentationDatasetAuditTests(TestCase):
    def _inventory(self, root: Path) -> DatasetInventory:
        yolo = root / "yolo"
        images = yolo / "train" / "images"
        labels = yolo / "train" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        first = images / "first.jpg"
        second = images / "second.jpg"
        Image.new("RGB", (100, 80), "green").save(first)
        Image.new("RGB", (100, 80), "green").save(second)
        (labels / "first.txt").write_text(
            "0 0.1 0.1 0.9 0.1 0.5 0.9\n"
            "1 0.2 0.2 0.3 0.2 0.25 0.3\n",
            encoding="utf-8",
        )
        (labels / "second.txt").write_text("", encoding="utf-8")
        coco_root = root / "coco"
        coco_train = coco_root / "train"
        coco_train.mkdir(parents=True)
        coco_json = coco_train / "_annotations.coco.json"
        coco_json.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 1, "file_name": "first.jpg", "width": 100, "height": 80},
                        {"id": 2, "file_name": "second.jpg", "width": 100, "height": 80},
                    ],
                    "categories": [
                        {"id": 1, "name": "leaf"},
                        {"id": 2, "name": "lesion"},
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "segmentation": [[10, 8, 90, 8, 50, 72]],
                            "area": 2560,
                        },
                        {
                            "id": 2,
                            "image_id": 1,
                            "category_id": 2,
                            "segmentation": [[20, 16, 30, 16, 25, 24]],
                            "area": 40,
                        },
                        {
                            "id": 3,
                            "image_id": 2,
                            "category_id": 1,
                            "segmentation": [[10, 8, 90, 8, 50, 72]],
                            "area": 2560,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return DatasetInventory(
            source="fixture",
            yolo_root=yolo,
            coco_root=coco_root,
            image_dir=images,
            label_dir=labels,
            coco_json=coco_json,
            images=[first, second],
            labels=[labels / "first.txt", labels / "second.txt"],
            class_names={0: "leaf", 1: "lesion"},
            license_name="CC BY 4.0",
        )

    def test_multiple_polygons_classes_empty_file_and_correspondence(self) -> None:
        with TemporaryDirectory() as directory:
            inventory = self._inventory(Path(directory))
            image_rows = compute_image_statistics(inventory)

            polygons, issues, mismatches, parsed = load_yolo_dataset(
                inventory, image_rows
            )

            self.assertEqual(len(polygons), 2)
            self.assertEqual({row["class_id"] for row in polygons}, {0, 1})
            self.assertEqual(len(parsed["first.jpg"]), 2)
            self.assertEqual(parsed["second.jpg"], [])
            self.assertTrue(
                any(issue["issue_type"] == "empty_label_file" for issue in issues)
            )
            self.assertEqual(mismatches, [])

    def test_yolo_coco_comparison_and_recovery(self) -> None:
        with TemporaryDirectory() as directory:
            inventory = self._inventory(Path(directory))
            image_rows = compute_image_statistics(inventory)
            _, issues, _, parsed = load_yolo_dataset(inventory, image_rows)
            coco = load_coco_segmentation(inventory.coco_json)

            rows = compare_yolo_coco(inventory, parsed, issues, coco)

            first = next(row for row in rows if row["filename"] == "first.jpg")
            second = next(row for row in rows if row["filename"] == "second.jpg")
            self.assertEqual(first["annotation_count_delta"], 0)
            self.assertEqual(second["annotation_count_delta"], -1)
            empty_issue = next(
                issue for issue in issues if issue["issue_type"] == "empty_label_file"
            )
            self.assertFalse(empty_issue["coco_recovery_possible"])
            self.assertEqual(empty_issue["recovery_decision"], "manual_review")

    def test_exact_duplicate_against_pilot(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = self._inventory(root)
            image_rows = compute_image_statistics(inventory)
            pilot = root / "pilot"
            pilot.mkdir()
            pilot_copy = pilot / "pilot.jpg"
            pilot_copy.write_bytes(inventory.images[0].read_bytes())

            rows, summary = find_exact_duplicates(image_rows, pilot)

            self.assertTrue(rows)
            self.assertTrue(summary["pilot_leakage_detected"])
            self.assertEqual(summary["pilot_cross_group_counts"]["fixture"], 1)

    def test_visual_sample_is_deterministic(self) -> None:
        rows = [{"filename": f"image_{index:03d}.jpg"} for index in range(30)]

        first = deterministic_sample(rows, 10, 42)
        second = deterministic_sample(list(reversed(rows)), 10, 42)

        self.assertEqual(first, second)


class AuditFingerprintTests(TestCase):
    def _fingerprint_fixture(
        self,
        root: Path,
    ) -> tuple[dict[str, DatasetInventory], Path]:
        audit = SegmentationDatasetAuditTests()
        inventory = audit._inventory(root)
        pilot_root = root / "pilot_root"
        pilot_images = pilot_root / "images"
        pilot_images.mkdir(parents=True)
        Image.new("RGB", (20, 20), "yellow").save(pilot_images / "pilot.jpg")
        return {"fixture": inventory}, pilot_root

    def test_fingerprint_is_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            inventories, pilot = self._fingerprint_fixture(Path(directory))

            first = build_audit_input_fingerprint(inventories, pilot)
            second = build_audit_input_fingerprint(dict(reversed(inventories.items())), pilot)

            self.assertEqual(first, second)

    def test_txt_change_invalidates_cache(self) -> None:
        with TemporaryDirectory() as directory:
            inventories, pilot = self._fingerprint_fixture(Path(directory))
            before = build_audit_input_fingerprint(inventories, pilot)
            cached = {
                "cache_schema_version": before["cache_schema_version"],
                "parser_schema_version": before["parser_schema_version"],
                "input_fingerprint": before,
            }
            inventories["fixture"].labels[0].write_text(
                "0 0.1 0.1 0.8 0.1 0.5 0.8\n",
                encoding="utf-8",
            )
            after = build_audit_input_fingerprint(inventories, pilot)

            self.assertNotEqual(before["global_sha256"], after["global_sha256"])
            self.assertFalse(audit_cache_is_current(cached, after))

    def test_image_change_invalidates_cache(self) -> None:
        with TemporaryDirectory() as directory:
            inventories, pilot = self._fingerprint_fixture(Path(directory))
            before = build_audit_input_fingerprint(inventories, pilot)
            cached = {
                "cache_schema_version": before["cache_schema_version"],
                "parser_schema_version": before["parser_schema_version"],
                "input_fingerprint": before,
            }
            Image.new("RGB", (100, 80), "blue").save(
                inventories["fixture"].images[0]
            )
            after = build_audit_input_fingerprint(inventories, pilot)

            self.assertNotEqual(before["global_sha256"], after["global_sha256"])
            self.assertFalse(audit_cache_is_current(cached, after))

    def test_pilot_change_invalidates_cache(self) -> None:
        with TemporaryDirectory() as directory:
            inventories, pilot = self._fingerprint_fixture(Path(directory))
            before = build_audit_input_fingerprint(inventories, pilot)
            cached = {
                "cache_schema_version": before["cache_schema_version"],
                "parser_schema_version": before["parser_schema_version"],
                "input_fingerprint": before,
            }
            Image.new("RGB", (20, 20), "red").save(pilot / "images" / "pilot.jpg")
            after = build_audit_input_fingerprint(inventories, pilot)

            self.assertNotEqual(before["global_sha256"], after["global_sha256"])
            self.assertFalse(audit_cache_is_current(cached, after))


class CocoRecoveryEvidenceTests(TestCase):
    @staticmethod
    def _annotation(
        *,
        category_id: int = 1,
        bbox: list[float] | None = None,
    ) -> dict[str, object]:
        return {
            "id": 7,
            "image_id": 1,
            "category_id": category_id,
            "bbox": bbox or [30, 40, 40, 20],
            "segmentation": [[30, 40, 70, 40, 70, 60, 30, 60]],
        }

    def _evaluate(
        self,
        annotations: list[dict[str, object]],
        *,
        categories: dict[int, str] | None = None,
    ) -> dict[str, object]:
        return evaluate_coco_recovery(
            source="corn",
            raw_line="0 0.5 0.5 0.4 0.2",
            points=[],
            original_class_id=0,
            original_class_name="leaf",
            semantic_role="full_leaf",
            coco_annotations=annotations,
            coco_categories=categories or {1: "leaf"},
            width=100,
            height=100,
        )

    def test_unique_valid_recovery(self) -> None:
        result = self._evaluate([self._annotation()])

        self.assertEqual(result["recovery_decision"], "recover_from_coco")
        self.assertEqual(result["recovery_candidate_count"], 1)
        self.assertEqual(result["bbox_max_abs_error"], 0.0)
        self.assertEqual(result["bbox_iou"], 1.0)
        self.assertTrue(result["class_match"])
        self.assertTrue(result["semantic_role_match"])
        self.assertTrue(result["topology_valid"])

    def test_ambiguous_candidates_are_not_recovered(self) -> None:
        result = self._evaluate([self._annotation(), self._annotation()])

        self.assertEqual(result["recovery_decision"], "manual_review")
        self.assertEqual(result["recovery_candidate_count"], 2)

    def test_incompatible_class_is_not_recovered(self) -> None:
        result = self._evaluate(
            [self._annotation(category_id=2)],
            categories={2: "lesion"},
        )

        self.assertEqual(result["recovery_decision"], "manual_review")
        self.assertFalse(result["class_match"])

    def test_incompatible_bbox_is_not_recovered(self) -> None:
        result = self._evaluate([self._annotation(bbox=[10, 10, 10, 10])])

        self.assertEqual(result["recovery_decision"], "manual_review")
        self.assertEqual(result["recovery_candidate_count"], 0)
