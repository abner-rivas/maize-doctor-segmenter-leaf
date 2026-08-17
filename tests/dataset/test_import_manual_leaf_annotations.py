"""Tests for manual YOLO and CSV ROI annotation import."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.data.leaf_pilot import PILOT_COLUMNS, read_csv_rows, sha256_file, write_csv_rows
from src.preprocessing.roi_manifest import (
    import_manual_annotations,
    load_cvat_xml_annotations,
    parse_yolo_leaf_annotation,
    rotated_bbox_to_axis_aligned,
)


def _write_pilot_fixture(
    root: Path,
    count: int = 3,
    *,
    size: tuple[int, int] = (200, 100),
) -> Path:
    pilot_root = root / "pilot"
    rows: list[dict[str, object]] = []
    for index in range(1, count + 1):
        pilot_id = f"image_{index:04d}"
        relative = Path("images") / f"{pilot_id}.jpg"
        image_path = pilot_root / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (index % 255, 100, 40)).save(image_path)
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
        return _write_pilot_fixture(root, count)

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


class CvatXmlAnnotationTests(TestCase):
    def _load_one(
        self,
        box: str,
        *,
        width: int = 200,
        height: int = 100,
        min_area_ratio: float = 0.01,
    ):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        xml_path = Path(directory.name) / "annotations.xml"
        xml_path.write_text(
            "<annotations>"
            f'<image id="0" name="images/image_0001.jpg" '
            f'width="{width}" height="{height}">{box}</image>'
            "</annotations>",
            encoding="utf-8",
        )
        grouped, stats, warnings = load_cvat_xml_annotations(
            xml_path,
            min_area_ratio,
        )
        return grouped["image_0001"][0].annotation, stats, warnings

    def test_direct_box_without_rotation(self) -> None:
        annotation, stats, _ = self._load_one(
            '<box label="maize_leaf" xtl="20" ytl="10" xbr="180" ybr="90"/>'
        )

        self.assertEqual(annotation.status, "annotated")
        self.assertEqual(annotation.bbox, (20, 10, 180, 90))
        self.assertEqual(annotation.conversion_method, "direct_bbox")
        self.assertEqual(annotation.original_rotation_degrees, 0.0)
        self.assertEqual(stats["xml_images"], 1)
        self.assertEqual(stats["xml_boxes"], 1)

    def test_rotation_90_degrees_swaps_axis_aligned_extents(self) -> None:
        annotation, _, _ = self._load_one(
            '<box label="maize_leaf" xtl="80" ytl="30" xbr="120" ybr="70" '
            'rotation="90"/>'
        )

        self.assertEqual(annotation.status, "annotated")
        self.assertEqual(annotation.bbox, (80, 30, 120, 70))
        self.assertEqual(annotation.conversion_method, "rotated_to_axis_aligned")

        converted = rotated_bbox_to_axis_aligned((70, 40, 130, 60), 90)
        self.assertAlmostEqual(converted[0], 90.0)
        self.assertAlmostEqual(converted[1], 20.0)
        self.assertAlmostEqual(converted[2], 110.0)
        self.assertAlmostEqual(converted[3], 80.0)

    def test_arbitrary_rotation_338_10_is_converted(self) -> None:
        annotation, _, _ = self._load_one(
            '<box label="maize_leaf" xtl="40" ytl="20" xbr="160" ybr="80" '
            'rotation="338.10"/>'
        )

        self.assertEqual(annotation.status, "annotated")
        self.assertEqual(annotation.bbox, (33, 0, 167, 100))
        self.assertAlmostEqual(annotation.original_rotation_degrees, 338.10)
        self.assertEqual(annotation.conversion_method, "rotated_to_axis_aligned")

    def test_negative_coordinates_are_clipped_with_warning(self) -> None:
        annotation, _, _ = self._load_one(
            '<box label="maize_leaf" xtl="-10" ytl="-5" xbr="100" ybr="80"/>'
        )

        self.assertEqual(annotation.bbox, (0, 0, 100, 80))
        self.assertTrue(annotation.clipped)
        self.assertIn("limitado", annotation.warnings[0])

    def test_coordinates_above_image_size_are_clipped(self) -> None:
        annotation, _, _ = self._load_one(
            '<box label="maize_leaf" xtl="50" ytl="20" xbr="220" ybr="110"/>'
        )

        self.assertEqual(annotation.bbox, (50, 20, 200, 100))
        self.assertTrue(annotation.clipped)

    def test_rotated_box_is_clipped_after_conversion(self) -> None:
        annotation, _, _ = self._load_one(
            '<box label="maize_leaf" xtl="0" ytl="0" xbr="190" ybr="90" '
            'rotation="20"/>'
        )

        self.assertEqual(annotation.status, "annotated")
        self.assertTrue(annotation.clipped)
        self.assertEqual(annotation.bbox, (0, 0, 200, 100))

    def test_box_empty_after_clipping_is_ambiguous(self) -> None:
        annotation, _, _ = self._load_one(
            '<box label="maize_leaf" xtl="-40" ytl="10" xbr="-10" ybr="80"/>'
        )

        self.assertEqual(annotation.status, "ambiguous")
        self.assertIn("vacío", annotation.notes)

    def test_area_below_minimum_is_detected(self) -> None:
        annotation, _, _ = self._load_one(
            '<box label="maize_leaf" xtl="80" ytl="40" xbr="120" ybr="60"/>',
            min_area_ratio=0.15,
        )

        self.assertEqual(annotation.status, "ambiguous")
        self.assertIn("min_area_ratio", annotation.notes)
        self.assertTrue(annotation.geometry_converted)

    def test_missing_multiple_and_wrong_label_boxes_are_not_selected(self) -> None:
        cases = (
            ("", "imagen sin caja"),
            (
                '<box label="maize_leaf" xtl="10" ytl="10" xbr="80" ybr="80"/>'
                '<box label="maize_leaf" xtl="90" ytl="10" xbr="180" ybr="80"/>',
                "2 cajas",
            ),
            ('<box label="weed" xtl="10" ytl="10" xbr="180" ybr="80"/>', "distinta"),
        )
        for boxes, expected in cases:
            with self.subTest(expected=expected):
                annotation, _, _ = self._load_one(boxes)
                self.assertEqual(annotation.status, "ambiguous")
                self.assertIn(expected, annotation.notes)

    def test_invalid_dimensions_and_non_finite_coordinates_are_ambiguous(self) -> None:
        invalid_dimension, _, _ = self._load_one(
            '<box label="maize_leaf" xtl="10" ytl="10" xbr="180" ybr="80"/>',
            width=0,
        )
        non_finite, _, _ = self._load_one(
            '<box label="maize_leaf" xtl="nan" ytl="10" xbr="180" ybr="80"/>'
        )

        self.assertEqual(invalid_dimension.status, "ambiguous")
        self.assertIn("mayor que cero", invalid_dimension.notes)
        self.assertEqual(non_finite.status, "ambiguous")
        self.assertIn("finito", non_finite.notes)

    def test_invalid_xml_raises_clear_error(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.xml"
            path.write_text("<annotations><image>", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "XML CVAT inválido"):
                load_cvat_xml_annotations(path, 0.01)

    def test_manifest_association_and_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_pilot_fixture(root, 1)
            xml_path = root / "annotations.xml"
            xml_path.write_text(
                "<annotations>"
                '<image id="17" name="images/image_0001.jpg" width="200" height="100">'
                '<box label="maize_leaf" xtl="20" ytl="10" xbr="180" ybr="90" '
                'rotation="5"/>'
                "</image></annotations>",
                encoding="utf-8",
            )
            output = root / "pilot" / "manifests" / "imported.csv"

            summary = import_manual_annotations(
                manifest,
                xml_path,
                "cvat_xml",
                output,
                0.01,
            )
            rows, columns = read_csv_rows(output)

            self.assertEqual(rows[0]["pilot_id"], "image_0001")
            self.assertEqual(rows[0]["annotation_status"], "annotated")
            self.assertEqual(rows[0]["roi_conversion_method"], "rotated_to_axis_aligned")
            self.assertEqual(rows[0]["original_rotation_degrees"], "5.0")
            self.assertIn("roi_clipped", columns)
            self.assertEqual(summary["status_counts"], {"annotated": 1})

    def test_unknown_pilot_id_is_reported(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_pilot_fixture(root, 1)
            xml_path = root / "annotations.xml"
            xml_path.write_text(
                "<annotations>"
                '<image id="0" name="images/image_9999.jpg" width="200" height="100">'
                '<box label="maize_leaf" xtl="20" ytl="10" xbr="180" ybr="90"/>'
                "</image></annotations>",
                encoding="utf-8",
            )

            summary = import_manual_annotations(
                manifest,
                xml_path,
                "cvat_xml",
                root / "pilot" / "manifests" / "imported.csv",
                0.01,
            )

            self.assertTrue(any("image_9999" in warning for warning in summary["warnings"]))
            self.assertEqual(summary["status_counts"], {"pending": 1})

    def test_complete_conversion_of_100_synthetic_images(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_pilot_fixture(root, 100)
            image_elements = []
            for index in range(1, 101):
                rotation = "" if index <= 48 else ' rotation="12.5"'
                image_elements.append(
                    f'<image id="{index - 1}" name="images/image_{index:04d}.jpg" '
                    'width="200" height="100">'
                    '<box label="maize_leaf" xtl="30" ytl="20" xbr="170" ybr="80"'
                    f"{rotation}/></image>"
                )
            xml_path = root / "annotations.xml"
            xml_path.write_text(
                "<annotations>" + "".join(image_elements) + "</annotations>",
                encoding="utf-8",
            )

            summary = import_manual_annotations(
                manifest,
                xml_path,
                "cvat_xml",
                root / "pilot" / "manifests" / "imported.csv",
                0.01,
            )

            self.assertEqual(summary["xml_images"], 100)
            self.assertEqual(summary["valid_rows"], 100)
            self.assertEqual(summary["invalid_rows"], 0)
            self.assertEqual(
                summary["conversion_counts"],
                {"direct_bbox": 48, "rotated_to_axis_aligned": 52},
            )
