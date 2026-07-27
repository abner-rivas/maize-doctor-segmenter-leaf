"""Regression tests for source-backed human-review previews."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.data.segmentation_review_preview import (
    BLOCKED_STATUS,
    ReviewGeometryResolver,
    _match_coco_entry,
    render_review_preview,
    validate_review_preview_rows,
)


def _case(
    image: Path,
    label: Path,
    *,
    filename: str | None = None,
    line: str = "",
    class_id: str = "0",
    decision: str = "manual_review",
) -> dict[str, str]:
    return {
        "source_dataset": "source",
        "filename": filename or image.name,
        "original_image_path": str(image),
        "original_label_path": str(label),
        "original_line_number": line,
        "original_class_id": class_id,
        "original_class_name": "leaf",
        "decision": decision,
        "review_reason": "eda_stratified_visual_review",
        "reviewer_decision": "",
        "review_status": "pending",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class ReviewPreviewResolutionTests(TestCase):
    def _roots(self, root: Path) -> tuple[Path, Path, Path, Path]:
        external = root / "external"
        train = external / "source_yolo26" / "train"
        image = train / "images" / "leaf_jpg.rf.sourcehash.jpg"
        label = train / "labels" / "leaf_jpg.rf.sourcehash.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir()
        Image.new("RGB", (100, 100), "green").save(image)
        manifests = root / "manifests"
        manifests.mkdir()
        return external, manifests, image, label

    def test_source_polygon_renders_when_absent_from_consolidated_pool(self) -> None:
        with TemporaryDirectory() as directory:
            external, manifests, image, label = self._roots(Path(directory))
            label.write_text(
                "0 0.1 0.1 0.9 0.1 0.5 0.9\n",
                encoding="utf-8",
            )
            case = _case(image, label, decision="exclude_invalid")
            resolved = ReviewGeometryResolver(external, manifests).resolve(case)
            preview = Path(directory) / "preview.jpg"

            rendered = render_review_preview(case, resolved, preview)

            self.assertEqual(resolved["geometry_source"], "yolo_original")
            self.assertGreaterEqual(rendered["instances_rendered"], 1)
            self.assertGreater(rendered["mask_pixels_rendered"], 0)
            self.assertTrue(preview.is_file())

    def test_roboflow_suffix_fallback_matches_unique_coco_filename(self) -> None:
        entry = {"image": {"file_name": "leaf_jpg.rf.first.jpg"}, "annotations": []}
        coco = {"by_filename_all": {"leaf_jpg.rf.first.jpg": [entry]}}

        matched = _match_coco_entry(coco, "leaf_jpg.rf.second.jpg")

        self.assertIs(matched, entry)

    def test_self_intersection_remains_visible(self) -> None:
        with TemporaryDirectory() as directory:
            external, manifests, image, label = self._roots(Path(directory))
            label.write_text(
                "0 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.9\n",
                encoding="utf-8",
            )
            case = _case(image, label, line="1")
            case["review_reason"] = "topologically_invalid_full_leaf"
            resolved = ReviewGeometryResolver(external, manifests).resolve(case)

            rendered = render_review_preview(
                case,
                resolved,
                Path(directory) / "topology.jpg",
            )

            self.assertEqual(resolved["render_status"], "rendered_invalid_original")
            self.assertIn("self_intersection", resolved["reason"])
            self.assertGreater(rendered["mask_pixels_rendered"], 0)

    def test_coco_recovery_uses_matched_annotation_id(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            external, manifests, image, label = self._roots(root)
            label.write_text("0 0.5 0.5 0.02 0.02\n", encoding="utf-8")
            case = _case(
                image,
                label,
                line="1",
                decision="recover_from_coco",
            )
            recovered = {
                **case,
                "recovery_evidence": json.dumps(
                    {
                        "matched_annotation_id": 7,
                        "recovery_match_method": "unique_bbox",
                        "bbox_max_abs_error": 0.0,
                        "bbox_iou": 1.0,
                    }
                ),
                "consolidated_label_path": "",
                "consolidated_line_number": "",
            }
            _write_csv(manifests / "recovered_annotations.csv", [recovered])
            coco_path = (
                external
                / "source_coco_segmentation"
                / "train"
                / "_annotations.coco.json"
            )
            coco_path.parent.mkdir(parents=True)
            coco_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": 1,
                                "file_name": image.name,
                                "width": 100,
                                "height": 100,
                            }
                        ],
                        "annotations": [
                            {
                                "id": 7,
                                "image_id": 1,
                                "category_id": 1,
                                "segmentation": [[49, 49, 51, 49, 51, 51, 49, 51]],
                            }
                        ],
                        "categories": [{"id": 1, "name": "leaf"}],
                    }
                ),
                encoding="utf-8",
            )

            resolved = ReviewGeometryResolver(external, manifests).resolve(case)
            rendered = render_review_preview(
                case,
                resolved,
                root / "coco_preview.jpg",
            )

            self.assertEqual(resolved["geometry_source"], "coco_original")
            self.assertEqual(resolved["render_status"], "rendered_from_coco")
            self.assertEqual(resolved["recovery_method"], "unique_bbox")
            self.assertGreater(rendered["mask_pixels_rendered"], 0)
            self.assertLess(rendered["polygon_area_ratio"], 0.001)

    def test_instance_index_targets_requested_source_line(self) -> None:
        with TemporaryDirectory() as directory:
            external, manifests, image, label = self._roots(Path(directory))
            label.write_text(
                "\n".join(
                    (
                        "0 0.1 0.1 0.4 0.1 0.2 0.4",
                        "0 0.6 0.6 0.9 0.6 0.8 0.9",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            resolved = ReviewGeometryResolver(external, manifests).resolve(
                _case(image, label, line="2")
            )

            targets = [
                geometry
                for geometry in resolved["geometries"]
                if geometry.target
            ]
            self.assertEqual(len(resolved["geometries"]), 2)
            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0].instance_index, 2)

    def test_empty_annotation_is_explicitly_unavailable(self) -> None:
        with TemporaryDirectory() as directory:
            external, manifests, image, label = self._roots(Path(directory))
            label.write_text("", encoding="utf-8")

            resolved = ReviewGeometryResolver(external, manifests).resolve(
                _case(image, label)
            )

            self.assertEqual(resolved["render_status"], "no_geometry_available")
            self.assertEqual(resolved["reason"], "empty_annotation")
            self.assertEqual(resolved["geometries"], [])

    def test_validation_fails_if_known_geometry_renders_zero_instances(self) -> None:
        validation = validate_review_preview_rows(
            [
                {
                    "review_origin": "general",
                    "expected_instances": 1,
                    "instances_rendered": 0,
                    "mask_pixels_rendered": 0,
                    "render_status": "rendered",
                    "_preview_exists": True,
                }
            ],
            expected_total=1,
            expected_mandatory=0,
        )

        self.assertEqual(validation["global_status"], BLOCKED_STATUS)
        self.assertEqual(validation["known_geometry_with_zero_instances"], 1)

