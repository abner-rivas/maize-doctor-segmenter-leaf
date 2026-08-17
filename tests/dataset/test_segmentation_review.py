"""Tests for the human-review gate of the segmentation candidate pool."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.data.segmentation_review import (
    ReviewManifestError,
    build_dataset_lock,
    dataset_fingerprint,
    validate_review_manifests,
    write_applied_review_report,
    write_reannotation_queue,
)


def _review(
    *,
    filename: str = "leaf.jpg",
    decision: str = "",
    reason: str = "",
    status: str = "pending",
) -> dict[str, str]:
    return {
        "source_dataset": "source",
        "filename": filename,
        "original_image_path": f"/source/{filename}",
        "original_label_path": f"/source/{Path(filename).stem}.txt",
        "original_line_number": "1",
        "original_class_id": "0",
        "original_class_name": "leaf",
        "reviewer_decision": decision,
        "review_reason": reason,
        "review_status": status,
    }


class ReviewValidationTests(TestCase):
    def test_pending_mandatory_review_blocks_without_inventing_decision(self) -> None:
        summary = validate_review_manifests([_review()], [])

        self.assertEqual(len(summary["mandatory_pending"]), 1)
        self.assertEqual(summary["approved"], [])
        self.assertEqual(summary["excluded"], [])
        self.assertEqual(summary["reannotation"], [])

    def test_completed_review_requires_decision_and_reason(self) -> None:
        summary = validate_review_manifests(
            [_review(status="completed")],
            [],
        )

        self.assertEqual(len(summary["invalid"]), 1)
        self.assertIn("completed requiere reviewer_decision", summary["invalid"][0]["problems"])
        self.assertIn("completed requiere review_reason", summary["invalid"][0]["problems"])

    def test_completed_decisions_are_grouped(self) -> None:
        rows = [
            _review(
                filename="approved.jpg",
                decision="approved",
                reason="contorno correcto",
                status="completed",
            ),
            _review(
                filename="excluded.jpg",
                decision="exclude",
                reason="máscara incorrecta",
                status="completed",
            ),
            _review(
                filename="redo.jpg",
                decision="needs_reannotation",
                reason="requiere nueva máscara",
                status="completed",
            ),
        ]

        summary = validate_review_manifests(rows, [])

        self.assertEqual(len(summary["approved"]), 1)
        self.assertEqual(len(summary["excluded"]), 1)
        self.assertEqual(len(summary["reannotation"]), 1)
        self.assertEqual(summary["pending"], [])

    def test_conflicting_completed_decisions_are_rejected(self) -> None:
        mandatory = [
            _review(
                decision="approved",
                reason="válida",
                status="completed",
            )
        ]
        general = [
            _review(
                decision="exclude",
                reason="inválida",
                status="completed",
            )
        ]

        with self.assertRaises(ReviewManifestError):
            validate_review_manifests(mandatory, general)

    def test_needs_reannotation_writes_queue(self) -> None:
        with TemporaryDirectory() as directory:
            row = _review(
                decision="needs_reannotation",
                reason="máscara incompleta",
                status="completed",
            )
            summary = validate_review_manifests([row], [])
            target = Path(directory) / "reannotation_queue.csv"

            write_reannotation_queue(target, summary["reannotation"])

            self.assertTrue(target.is_file())
            self.assertIn("needs_reannotation", target.read_text(encoding="utf-8"))
            self.assertIn("/source/leaf.jpg", target.read_text(encoding="utf-8"))

    def test_empty_derived_review_reports_keep_their_schema(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            applied = root / "review_decisions_applied.csv"
            queue = root / "reannotation_queue.csv"

            write_applied_review_report(applied, [])
            write_reannotation_queue(queue, [])

            self.assertEqual(len(applied.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(len(queue.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(
                applied.read_text(encoding="utf-8").splitlines()[0].count(
                    "review_case_id"
                ),
                1,
            )


class DatasetLockTests(TestCase):
    def _dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        images = dataset / "all" / "images"
        labels = dataset / "all" / "labels"
        manifests = dataset / "manifests"
        images.mkdir(parents=True)
        labels.mkdir()
        manifests.mkdir()
        Image.new("RGB", (16, 16), "green").save(images / "leaf.jpg")
        (labels / "leaf.txt").write_text(
            "0 0.1 0.1 0.9 0.1 0.5 0.9\n",
            encoding="utf-8",
        )
        (dataset / "dataset.yaml").write_text("names:\n  0: maize_leaf\n")
        return dataset

    def test_dataset_fingerprint_is_deterministic_and_content_sensitive(self) -> None:
        with TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            first = dataset_fingerprint(dataset)
            second = dataset_fingerprint(dataset)
            (dataset / "all" / "labels" / "leaf.txt").write_text(
                "0 0.1 0.1 0.8 0.1 0.5 0.8\n",
                encoding="utf-8",
            )
            changed = dataset_fingerprint(dataset)

            self.assertEqual(first, second)
            self.assertNotEqual(first["sha256"], changed["sha256"])

    def test_lock_is_blocked_while_manual_review_is_pending(self) -> None:
        with TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory))
            reviews = validate_review_manifests([_review()], [])
            consolidation = {
                "source_fingerprint_after": {
                    "tree_sha256": "source-tree",
                    "file_count": 1,
                    "total_bytes": 10,
                },
                "source_files_unchanged": True,
                "counts": {"pilot_leakage": 0},
            }
            eda = {
                "input_fingerprint": {
                    "global_sha256": "eda-tree",
                    "file_count": 2,
                },
                "input_files_unchanged": True,
            }

            lock = build_dataset_lock(
                dataset,
                consolidation,
                eda,
                reviews,
                decisions_applied_from_sources=False,
                lock_date=date(2026, 7, 27),
            )

            self.assertEqual(lock["status"], "blocked_by_manual_review")
            self.assertEqual(lock["total_images"], 1)
            self.assertEqual(lock["total_masks"], 1)
            self.assertEqual(len(lock["pending_reviews"]), 1)
            self.assertFalse(lock["splits_created"])
            self.assertFalse(lock["training_performed"])
