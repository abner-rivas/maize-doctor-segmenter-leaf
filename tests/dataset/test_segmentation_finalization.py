"""Tests for deterministic application of human segmentation decisions."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.data.segmentation_consolidation import MANIFEST_COLUMNS, write_csv
from src.data.segmentation_finalization import _apply_reviews, decisions_fingerprint


def _manifest_row(root: Path, line_number: int) -> dict[str, str]:
    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "source_dataset": "source",
            "original_image_path": "/source/leaf.jpg",
            "original_line_number": str(line_number),
            "original_class_id": "0",
            "decision": "include_after_remap",
            "image_sha256": "digest",
            "target_class_id": "0",
            "target_class_name": "maize_leaf",
            "consolidated_image_path": str(root / "all/images/leaf.jpg"),
            "consolidated_label_path": str(root / "all/labels/leaf.txt"),
            "consolidated_line_number": str(line_number),
        }
    )
    return row


def _review(decision: str, line_number: str = "1") -> dict[str, str]:
    return {
        "review_case_id": f"case-{line_number}",
        "review_key": f"source|leaf.jpg|{line_number}|0",
        "source_dataset": "source",
        "filename": "leaf.jpg",
        "original_line_number": line_number,
        "original_class_id": "0",
        "reviewer_decision": decision,
        "review_reason": "human reason",
        "review_status": "completed",
    }


class SegmentationFinalizationTests(TestCase):
    def test_reannotation_removes_only_matching_geometry_and_renumbers(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "all/images").mkdir(parents=True)
            (root / "all/labels").mkdir()
            (root / "manifests").mkdir()
            Image.new("RGB", (8, 8), "green").save(root / "all/images/leaf.jpg")
            (root / "all/labels/leaf.txt").write_text(
                "0 0.1 0.1 0.2 0.1 0.1 0.2\n"
                "0 0.3 0.3 0.4 0.3 0.3 0.4\n",
                encoding="utf-8",
            )
            write_csv(
                root / "manifests/consolidation_manifest.csv",
                [_manifest_row(root, 1), _manifest_row(root, 2)],
                MANIFEST_COLUMNS,
            )

            manifest, application = _apply_reviews(
                root,
                Path("/final"),
                [_review("needs_reannotation")],
            )

            self.assertEqual(application["removed_annotations"], 1)
            self.assertEqual(
                (root / "all/labels/leaf.txt").read_text(encoding="utf-8"),
                "0 0.3 0.3 0.4 0.3 0.3 0.4\n",
            )
            included = [row for row in manifest if row["decision"] == "include_after_remap"]
            self.assertEqual(included[0]["consolidated_line_number"], 1)

    def test_decision_fingerprint_is_order_independent(self) -> None:
        first = _review("approved", "1")
        second = _review("exclude", "2")

        self.assertEqual(
            decisions_fingerprint([first, second]),
            decisions_fingerprint([second, first]),
        )
