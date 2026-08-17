"""Tests for ROI manifest integrity, leakage, coverage, and previews."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.data.leaf_pilot import read_csv_rows, sha256_file, write_csv_rows
from src.preprocessing.roi_manifest import (
    ROI_MANIFEST_COLUMNS,
    validate_roi_manifest,
)


class RoiManifestValidationTests(TestCase):
    def _image(
        self,
        root: Path,
        name: str,
        color: tuple[int, int, int],
    ) -> Path:
        path = root / "pilot" / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), color).save(path)
        return path

    def _row(
        self,
        image: Path,
        pilot_id: str,
        *,
        label: str = "healthy",
        split: str = "test",
        digest: str | None = None,
        bbox: tuple[int, int, int, int] = (10, 10, 90, 90),
    ) -> dict[str, object]:
        x1, y1, x2, y2 = bbox
        return {
            "pilot_id": pilot_id,
            "image_path": f"images/{image.name}",
            "original_image_path": f"clean/{label}/real/{image.name}",
            "image_sha256": digest or sha256_file(image),
            "label": label,
            "split": split,
            "environment": "real",
            "source_dataset": "unknown",
            "roi_x1": x1,
            "roi_y1": y1,
            "roi_x2": x2,
            "roi_y2": y2,
            "roi_width": x2 - x1,
            "roi_height": y2 - y1,
            "roi_area_ratio": (x2 - x1) * (y2 - y1) / 10_000,
            "roi_confidence": 1.0,
            "roi_source": "manual",
            "annotation_status": "annotated",
            "notes": "",
        }

    def _validate(
        self,
        root: Path,
        rows: list[dict[str, object]],
        *,
        preview_samples: int = 0,
    ) -> tuple[dict[str, object], list[dict[str, str]]]:
        manifest = root / "pilot" / "manifests" / "roi_manifest.csv"
        write_csv_rows(manifest, rows, ROI_MANIFEST_COLUMNS)
        output = root / "validation"
        preview = root / "previews" if preview_samples else None
        summary = validate_roi_manifest(
            manifest,
            output,
            valid_classes={"healthy", "common_rust"},
            min_area_ratio=0.15,
            preview_samples=preview_samples,
            preview_output=preview,
        )
        result_rows, _ = read_csv_rows(output / "roi_validation_rows.csv")
        return summary, result_rows

    def test_correct_hash_bbox_coverage_and_preview(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            healthy = self._image(root, "healthy.jpg", (20, 100, 20))
            rust = self._image(root, "rust.jpg", (120, 40, 20))

            summary, rows = self._validate(
                root,
                [
                    self._row(healthy, "image_0001"),
                    self._row(rust, "image_0002", label="common_rust"),
                ],
                preview_samples=1,
            )

            self.assertEqual(summary["valid_rows"], 2)
            self.assertEqual(summary["coverage"]["by_class"], {"common_rust": 1, "healthy": 1})
            self.assertEqual(summary["preview_generated"], 1)
            self.assertEqual(rows[0]["valid"], "True")
            self.assertEqual(len(list((root / "previews").glob("*_preview.jpg"))), 1)

    def test_wrong_hash_is_invalid(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = self._image(root, "leaf.jpg", (20, 100, 20))

            summary, rows = self._validate(
                root,
                [self._row(image, "image_0001", digest="0" * 64)],
            )

            self.assertEqual(summary["invalid_rows"], 1)
            self.assertIn("SHA-256", rows[0]["errors"])

    def test_same_hash_between_splits_reports_leakage(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._image(root, "first.jpg", (20, 100, 20))
            second = root / "pilot" / "images" / "second.jpg"
            second.write_bytes(first.read_bytes())

            summary, rows = self._validate(
                root,
                [
                    self._row(first, "image_0001", split="train"),
                    self._row(second, "image_0002", split="test"),
                ],
            )

            self.assertEqual(summary["leakage"]["cross_split_hash_groups"], 1)
            self.assertTrue(all("fuga por hash" in row["errors"] for row in rows))

    def test_duplicate_route_is_invalid(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = self._image(root, "leaf.jpg", (20, 100, 20))

            summary, rows = self._validate(
                root,
                [
                    self._row(image, "image_0001"),
                    self._row(image, "image_0002"),
                ],
            )

            self.assertEqual(summary["leakage"]["duplicate_path_groups"], 1)
            self.assertTrue(all("ruta duplicada" in row["errors"] for row in rows))

    def test_repeated_original_filename_is_only_a_warning(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._image(root, "first.jpg", (20, 100, 20))
            second = self._image(root, "second.jpg", (40, 120, 40))
            first_row = self._row(first, "image_0001")
            second_row = self._row(second, "image_0002")
            first_row["original_image_path"] = "clean/a/real/leaf.jpg"
            second_row["original_image_path"] = "clean/b/real/leaf.jpg"

            summary, rows = self._validate(root, [first_row, second_row])

            self.assertEqual(summary["valid_rows"], 2)
            self.assertEqual(summary["leakage"]["duplicate_filename_groups"], 1)
            self.assertTrue(all("nombre repetido" in row["warnings"] for row in rows))

    def test_invalid_bbox_and_minimum_area_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._image(root, "first.jpg", (20, 100, 20))
            second = self._image(root, "second.jpg", (40, 120, 40))

            summary, rows = self._validate(
                root,
                [
                    self._row(first, "image_0001", bbox=(90, 10, 20, 80)),
                    self._row(second, "image_0002", bbox=(10, 10, 20, 20)),
                ],
            )

            self.assertEqual(summary["invalid_rows"], 2)
            self.assertTrue(rows[0]["errors"])
            self.assertIn("min_area_ratio", rows[1]["errors"])

    def test_corrupt_image_is_rejected_without_crashing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "pilot" / "images" / "corrupt.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"not-an-image")

            summary, rows = self._validate(
                root,
                [self._row(image, "image_0001")],
            )

            self.assertEqual(summary["invalid_rows"], 1)
            self.assertTrue(rows[0]["errors"])

    def test_non_numeric_confidence_is_reported(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = self._image(root, "leaf.jpg", (20, 100, 20))
            row = self._row(image, "image_0001")
            row["roi_confidence"] = "manual"

            summary, rows = self._validate(root, [row])

            self.assertEqual(summary["invalid_rows"], 1)
            self.assertIn("roi_confidence", rows[0]["errors"])
