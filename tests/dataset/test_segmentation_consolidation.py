"""Tests for controlled external segmentation consolidation."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.data.segmentation_audit import sha256_file
from src.data.segmentation_consolidation import (
    decide_annotation,
    recover_yolo_annotation_from_coco,
    remap_yolo_polygon_line,
    roboflow_original_base,
    roboflow_variant_group,
    select_image_actions,
    source_files_fingerprint,
    validate_consolidated_dataset,
)


class SegmentationConsolidationTests(TestCase):
    def test_remap_class_to_maize_leaf(self) -> None:
        line = "7 0.1 0.1 0.9 0.1 0.5 0.9"

        remapped = remap_yolo_polygon_line(line)

        self.assertEqual(remapped, "0 0.1 0.1 0.9 0.1 0.5 0.9")

    def test_lesion_is_excluded_even_when_recoverable(self) -> None:
        decision = decide_annotation(
            "lesion",
            known_class=True,
            valid_polygon=False,
            recovery_available=True,
        )

        self.assertEqual(decision, "exclude_lesion")

    def test_bbox_row_is_recovered_from_matching_coco_polygon(self) -> None:
        raw = "1 0.5 0.5 0.4 0.2"
        annotation = {
            "id": 10,
            "category_id": 2,
            "bbox": [30, 40, 40, 20],
            "segmentation": [[30, 40, 70, 40, 70, 60, 30, 60]],
        }

        recovered = recover_yolo_annotation_from_coco(
            source="corn_leaf_diseases_classification",
            raw_line=raw,
            original_class_id=1,
            original_class_name="leaf",
            semantic_role="full_leaf",
            coco_entry={
                "image": {"width": 100, "height": 100},
                "annotations": [annotation],
            },
            coco_categories={2: "leaf"},
            image_match_unique=True,
        )

        self.assertIsNotNone(recovered)
        line, evidence = recovered or ("", {})
        self.assertTrue(line.startswith("0 "))
        self.assertEqual(evidence["recovery_candidate_count"], 1)
        self.assertLess(float(evidence["bbox_max_abs_error"]), 1e-12)
        self.assertEqual(evidence["recovery_match_method"], (
            "unique_image_class_semantic_bbox_topology"
        ))

    def test_bbox_recovery_rejects_ambiguous_geometry(self) -> None:
        annotation = {
            "id": 10,
            "category_id": 2,
            "bbox": [30, 40, 40, 20],
            "segmentation": [[30, 40, 70, 40, 70, 60, 30, 60]],
        }

        recovered = recover_yolo_annotation_from_coco(
            source="corn_leaf_diseases_classification",
            raw_line="1 0.5 0.5 0.4 0.2",
            original_class_id=1,
            original_class_name="leaf",
            semantic_role="full_leaf",
            coco_entry={
                "image": {"width": 100, "height": 100},
                "annotations": [annotation, {**annotation, "id": 11}],
            },
            coco_categories={2: "leaf"},
            image_match_unique=True,
        )

        self.assertIsNone(recovered)

    def test_malformed_polygon_cannot_be_remapped(self) -> None:
        with self.assertRaises(ValueError):
            remap_yolo_polygon_line("1 0.5 0.5 0.4 0.2")

    def test_self_intersecting_polygon_cannot_be_remapped(self) -> None:
        with self.assertRaises(ValueError):
            remap_yolo_polygon_line(
                "1 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.9"
            )

    def test_image_label_correspondence_and_single_class(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            all_root = root / "all"
            images = all_root / "images"
            labels = all_root / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            image = images / "sample.jpg"
            Image.new("RGB", (32, 24), "green").save(image)
            label = labels / "sample.txt"
            label.write_text("0 0.1 0.1 0.9 0.1 0.5 0.9\n", encoding="utf-8")
            manifest = [
                {
                    "decision": "include_after_remap",
                    "consolidated_image_path": "/future/all/images/sample.jpg",
                    "consolidated_label_path": "/future/all/labels/sample.txt",
                }
            ]

            issues, summary = validate_consolidated_dataset(
                all_root,
                manifest,
                set(),
            )

            self.assertEqual(issues, [])
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["annotations"], 1)

    def test_missing_label_is_reported(self) -> None:
        with TemporaryDirectory() as directory:
            all_root = Path(directory) / "all"
            images = all_root / "images"
            labels = all_root / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            Image.new("RGB", (16, 16), "green").save(images / "orphan.jpg")

            issues, summary = validate_consolidated_dataset(all_root, [], set())

            self.assertFalse(summary["passed"])
            self.assertIn(
                "image_without_label",
                {row["issue_type"] for row in issues},
            )

    def test_exact_duplicates_are_deterministic(self) -> None:
        rows = [
            {
                "source_dataset": "b",
                "filename": "second.jpg",
                "original_image_path": "/second.jpg",
                "image_sha256": "same",
                "perceptual_hash": "hash",
                "roboflow_variant_group": "group",
            },
            {
                "source_dataset": "a",
                "filename": "first.jpg",
                "original_image_path": "/first.jpg",
                "image_sha256": "same",
                "perceptual_hash": "hash",
                "roboflow_variant_group": "group",
            },
        ]

        first_actions, first_rows = select_image_actions(rows, set())
        second_actions, second_rows = select_image_actions(
            list(reversed(rows)),
            set(),
        )

        self.assertEqual(first_actions, second_actions)
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first_actions[("a", "first.jpg")], "keep")
        self.assertEqual(first_actions[("b", "second.jpg")], "exclude_duplicate")

    def test_pilot_hash_blocks_candidate(self) -> None:
        row = {
            "source_dataset": "source",
            "filename": "image.jpg",
            "original_image_path": "/image.jpg",
            "image_sha256": "pilot-hash",
            "perceptual_hash": "hash",
            "roboflow_variant_group": "group",
        }

        actions, _ = select_image_actions([row], {"pilot-hash"})

        self.assertEqual(
            actions[("source", "image.jpg")],
            "exclude_pilot_leakage",
        )

    def test_roboflow_variants_share_group(self) -> None:
        first = "leaf_001_jpg.rf.abc123.jpg"
        second = "leaf_001_jpg.rf.xyz789.jpg"

        self.assertEqual(roboflow_original_base(first), "leaf_001")
        self.assertEqual(
            roboflow_variant_group("corn", first),
            roboflow_variant_group("corn", second),
        )

    def test_source_fingerprint_confirms_preservation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("immutable\n", encoding="utf-8")
            before = source_files_fingerprint([source])
            copied = root / "copy.txt"

            shutil.copy2(source, copied)
            after = source_files_fingerprint([source])

            self.assertEqual(before, after)
            self.assertEqual(sha256_file(source), sha256_file(copied))
