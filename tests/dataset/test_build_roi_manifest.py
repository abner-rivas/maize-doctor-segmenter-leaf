"""Tests for final ROI manifest construction."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.data.leaf_pilot import PILOT_COLUMNS, read_csv_rows, write_csv_rows
from src.preprocessing.roi_manifest import (
    ANNOTATION_EXTRA_COLUMNS,
    IMPORTED_ANNOTATION_COLUMNS,
    ROI_MANIFEST_COLUMNS,
    build_roi_manifest,
)


class RoiManifestBuildTests(TestCase):
    def test_expected_columns_manual_confidence_and_preserved_fields(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            imported = root / "imported.csv"
            base = {
                "pilot_id": "image_0001",
                "pilot_image_path": "images/image_0001.jpg",
                "original_image_path": "clean/healthy/real/original.jpg",
                "original_filename": "original.jpg",
                "image_sha256": "a" * 64,
                "label": "healthy",
                "split": "test",
                "environment": "real",
                "source_dataset": "unknown",
                "selected_by": "balanced",
                "annotation_status": "annotated",
                "copy_mode": "copy",
                "roi_x1": 10,
                "roi_y1": 20,
                "roi_x2": 90,
                "roi_y2": 80,
                "roi_width": 80,
                "roi_height": 60,
                "roi_area_ratio": 0.24,
                "notes": "hoja principal",
                "annotation_warnings": "",
                "annotation_format": "csv",
            }
            write_csv_rows(imported, [base], IMPORTED_ANNOTATION_COLUMNS)
            output = root / "roi_manifest.csv"

            count = build_roi_manifest(imported, output)
            rows, columns = read_csv_rows(output)

            self.assertEqual(count, 1)
            self.assertEqual(tuple(columns), ROI_MANIFEST_COLUMNS)
            self.assertEqual(rows[0]["roi_confidence"], "1.0")
            self.assertEqual(rows[0]["roi_source"], "manual")
            self.assertEqual(rows[0]["split"], "test")
            self.assertEqual(rows[0]["label"], "healthy")

    def test_ambiguous_and_rejected_cases_have_empty_coordinates(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            imported = root / "imported.csv"
            rows = []
            for index, status in enumerate(("ambiguous", "rejected"), start=1):
                row = {column: "" for column in (*PILOT_COLUMNS, *ANNOTATION_EXTRA_COLUMNS)}
                row.update(
                    {
                        "pilot_id": f"image_{index:04d}",
                        "pilot_image_path": f"images/image_{index:04d}.jpg",
                        "original_image_path": f"clean/healthy/real/{index}.jpg",
                        "image_sha256": str(index) * 64,
                        "label": "healthy",
                        "split": "test",
                        "environment": "real",
                        "source_dataset": "unknown",
                        "annotation_status": status,
                        "notes": "requiere revisión" if status == "ambiguous" else "sin hoja",
                    }
                )
                rows.append(row)
            write_csv_rows(imported, rows, IMPORTED_ANNOTATION_COLUMNS)
            output = root / "roi_manifest.csv"

            build_roi_manifest(imported, output)
            final_rows, _ = read_csv_rows(output)

            for row in final_rows:
                self.assertEqual(row["roi_x1"], "")
                self.assertEqual(row["roi_confidence"], "")
                self.assertEqual(row["roi_source"], "manual")
