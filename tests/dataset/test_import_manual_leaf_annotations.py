"""Tests for manual YOLO and CSV ROI annotation import."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.data.leaf_pilot import PILOT_COLUMNS, read_csv_rows, sha256_file, write_csv_rows
from src.preprocessing.roi_manifest import (
    import_manual_annotations,
    parse_yolo_leaf_annotation,
)


class YoloAnnotationParsingTests(TestCase):
    def _parse(self, content: str | None):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "image_0001.txt"
        if content is not None:
            path.write_text(content, encoding="utf-8")
        return parse_yolo_leaf_annotation(path, 200, 100, 0.01)

    def test_valid_yolo_bbox(self) -> None:
        result = self._parse("0 0.5 0.5 0.5 0.6\n")

        self.assertEqual(result.status, "annotated")
        self.assertEqual(result.bbox, (50, 20, 150, 80))
        self.assertAlmostEqual(result.area_ratio or 0, 0.3)

    def test_out_of_range_yolo_values(self) -> None:
        result = self._parse("0 1.2 0.5 0.5 0.5\n")

        self.assertEqual(result.status, "ambiguous")
        self.assertIn("rango", result.notes)

    def test_polygon_is_not_supported(self) -> None:
        result = self._parse("0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n")

        self.assertEqual(result.status, "ambiguous")
        self.assertIn("polígono", result.notes)

    def test_empty_label_is_ambiguous(self) -> None:
        result = self._parse("\n")

        self.assertEqual(result.status, "ambiguous")
        self.assertIn("vacía", result.notes)

    def test_multiple_lines_require_review(self) -> None:
        result = self._parse("0 0.3 0.5 0.2 0.4\n0 0.7 0.5 0.2 0.4\n")

        self.assertEqual(result.status, "ambiguous")
        self.assertIn("2 bbox", result.notes)

    def test_class_other_than_zero_is_ambiguous(self) -> None:
        result = self._parse("1 0.5 0.5 0.4 0.4\n")

        self.assertEqual(result.status, "ambiguous")
        self.assertIn("maize_leaf", result.notes)

    def test_image_without_label_remains_pending(self) -> None:
        result = self._parse(None)

        self.assertEqual(result.status, "pending")
        self.assertIn("sin archivo", result.warnings[0])


class ManualAnnotationImportTests(TestCase):
    def _pilot_fixture(self, root: Path, count: int = 3) -> Path:
        pilot_root = root / "pilot"
        rows: list[dict[str, object]] = []
        for index in range(1, count + 1):
            pilot_id = f"image_{index:04d}"
            relative = Path("images") / f"{pilot_id}.jpg"
            image_path = pilot_root / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (200, 100), (index * 20, 100, 40)).save(image_path)
            rows.append(
                {
                    "pilot_id": pilot_id,
                    "pilot_image_path": relative.as_posix(),
                    "original_image_path": f"clean/healthy/real/{pilot_id}.jpg",
                    "original_filename": f"{pilot_id}.jpg",
                    "image_sha256": sha256_file(image_path),
                    "label": "healthy",
                    "split": "test",
                    "environment": "real",
                    "source_dataset": "unknown",
                    "selected_by": "balanced",
                    "annotation_status": "pending",
                    "copy_mode": "copy",
                }
            )
        manifest = pilot_root / "manifests" / "pilot_manifest.csv"
        write_csv_rows(manifest, rows, PILOT_COLUMNS)
        return manifest

    def test_yolo_import_reports_orphan_and_missing_labels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._pilot_fixture(root, 2)
            labels = root / "labels"
            labels.mkdir()
            (labels / "image_0001.txt").write_text("0 0.5 0.5 0.5 0.6\n", encoding="utf-8")
            (labels / "orphan.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            output = root / "pilot" / "manifests" / "imported.csv"

            summary = import_manual_annotations(manifest, labels, "yolo", output, 0.01)
            rows, _ = read_csv_rows(output)

            self.assertEqual(rows[0]["annotation_status"], "annotated")
            self.assertEqual(rows[1]["annotation_status"], "pending")
            self.assertTrue(any("orphan" in warning for warning in summary["warnings"]))

    def test_csv_valid_ambiguous_rejected_and_notes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._pilot_fixture(root)
            annotations = root / "annotations.csv"
            with annotations.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("pilot_id", "x1", "y1", "x2", "y2", "status", "notes"),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "pilot_id": "image_0001",
                            "x1": 40,
                            "y1": 10,
                            "x2": 160,
                            "y2": 90,
                            "status": "annotated",
                            "notes": "hoja central",
                        },
                        {
                            "pilot_id": "image_0002",
                            "x1": "",
                            "y1": "",
                            "x2": "",
                            "y2": "",
                            "status": "ambiguous",
                            "notes": "dos hojas similares",
                        },
                        {
                            "pilot_id": "image_0003",
                            "x1": "",
                            "y1": "",
                            "x2": "",
                            "y2": "",
                            "status": "rejected",
                            "notes": "sin hoja",
                        },
                    ]
                )
            output = root / "pilot" / "manifests" / "imported.csv"

            import_manual_annotations(manifest, annotations, "csv", output, 0.01)
            rows, _ = read_csv_rows(output)

            self.assertEqual(
                [row["annotation_status"] for row in rows],
                ["annotated", "ambiguous", "rejected"],
            )
            self.assertEqual(rows[0]["notes"], "hoja central")
            self.assertEqual(rows[1]["roi_x1"], "")

    def test_csv_unknown_id_is_reported(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._pilot_fixture(root, 1)
            annotations = root / "annotations.csv"
            annotations.write_text(
                "pilot_id,x1,y1,x2,y2,status,notes\nunknown,1,1,10,10,annotated,x\n",
                encoding="utf-8",
            )

            summary = import_manual_annotations(
                manifest,
                annotations,
                "csv",
                root / "pilot" / "manifests" / "imported.csv",
                0.01,
            )

            self.assertTrue(any("unknown" in warning for warning in summary["warnings"]))

    def test_csv_invalid_coordinates_and_duplicates_are_ambiguous(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._pilot_fixture(root, 2)
            annotations = root / "annotations.csv"
            annotations.write_text(
                "pilot_id,x1,y1,x2,y2,status,notes\n"
                "image_0001,50,10,20,80,annotated,invertido\n"
                "image_0002,10,10,100,80,annotated,primera\n"
                "image_0002,20,20,120,90,annotated,segunda\n",
                encoding="utf-8",
            )
            output = root / "pilot" / "manifests" / "imported.csv"

            import_manual_annotations(manifest, annotations, "csv", output, 0.01)
            rows, _ = read_csv_rows(output)

            self.assertEqual(rows[0]["annotation_status"], "ambiguous")
            self.assertEqual(rows[1]["annotation_status"], "ambiguous")
            self.assertIn("duplicadas", rows[1]["annotation_warnings"])
