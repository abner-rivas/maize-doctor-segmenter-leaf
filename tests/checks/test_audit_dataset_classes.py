"""Tests for the read-only configured/documented/physical class audit."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.data.class_audit import (
    COUNT_COLUMNS,
    audit_dataset_classes,
    audit_exit_code,
)


class DatasetClassAuditTests(TestCase):
    def _fixture(
        self,
        root: Path,
        configured: list[str],
        documented: dict[str, tuple[int, int]],
    ) -> tuple[Path, Path, Path]:
        dataset_root = root / "dataset"
        (dataset_root / "clean").mkdir(parents=True)
        config = root / "dataset.yaml"
        class_lines = "\n".join(f"    - {name}" for name in configured)
        config.write_text(
            "paths:\n  raw_dir: clean\ndataset:\n  classes:\n" + class_lines + "\n",
            encoding="utf-8",
        )
        documentation = root / "classes.md"
        lines = [
            "| Clase | Lab | Real | Total |",
            "|---|---:|---:|---:|",
        ]
        for class_name, (lab, real) in documented.items():
            lines.append(f"| `{class_name}` | {lab} | {real} | {lab + real} |")
        documentation.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dataset_root, config, documentation

    def _image(
        self,
        dataset_root: Path,
        class_name: str,
        environment: str,
        name: str,
    ) -> Path:
        path = dataset_root / "clean" / class_name / environment / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), "green").save(path)
        return path

    def _audit(
        self,
        root: Path,
        dataset_root: Path,
        config: Path,
        documentation: Path,
    ) -> dict[str, object]:
        return audit_dataset_classes(
            dataset_root,
            config,
            root / "audit",
            documentation_paths=[documentation],
        )

    def test_matching_classes_counts_and_valid_extensions(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, docs = self._fixture(
                root,
                ["healthy"],
                {"healthy": (1, 1)},
            )
            self._image(dataset, "healthy", "lab", "lab.JPG")
            self._image(dataset, "healthy", "real", "real.png")

            report = self._audit(root, dataset, config, docs)

            self.assertEqual(report["conclusion"], "classes_coherent")
            self.assertEqual(report["total_images"], 2)
            self.assertTrue(report["ready_for_splits"])
            with (root / "audit" / "class_counts.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), COUNT_COLUMNS)
                self.assertEqual(sum(int(row["image_count"]) for row in reader), 2)

    def test_configured_class_absent(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, docs = self._fixture(
                root,
                ["healthy", "lethal_necrosis"],
                {"healthy": (1, 1), "lethal_necrosis": (0, 1)},
            )
            self._image(dataset, "healthy", "lab", "lab.jpg")
            self._image(dataset, "healthy", "real", "real.jpg")

            report = self._audit(root, dataset, config, docs)

            self.assertEqual(report["missing_classes"], ["lethal_necrosis"])
            self.assertIn("configured_missing_on_disk", self._types(root))

    def test_additional_disk_class(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, docs = self._fixture(
                root,
                ["healthy"],
                {"healthy": (1, 1)},
            )
            self._image(dataset, "healthy", "lab", "lab.jpg")
            self._image(dataset, "healthy", "real", "real.jpg")
            self._image(dataset, "aphids_pest", "real", "aphid.jpg")

            report = self._audit(root, dataset, config, docs)

            self.assertEqual(report["additional_classes"], ["aphids_pest"])
            self.assertIn("disk_class_not_configured", self._types(root))

    def test_empty_class_is_reported(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, docs = self._fixture(
                root,
                ["healthy"],
                {"healthy": (0, 0)},
            )
            (dataset / "clean" / "healthy" / "lab").mkdir(parents=True)
            (dataset / "clean" / "healthy" / "real").mkdir()

            report = self._audit(root, dataset, config, docs)

            self.assertFalse(report["ready_for_splits"])
            self.assertIn("empty_class", self._types(root))

    def test_unexpected_environment_is_an_error(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, docs = self._fixture(
                root,
                ["healthy"],
                {"healthy": (0, 0)},
            )
            self._image(dataset, "healthy", "field", "leaf.jpg")

            report = self._audit(root, dataset, config, docs)

            self.assertFalse(report["ready_for_splits"])
            self.assertIn("unexpected_environment", self._types(root))

    def test_non_image_file_is_ignored_and_reported(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, docs = self._fixture(
                root,
                ["healthy"],
                {"healthy": (1, 1)},
            )
            self._image(dataset, "healthy", "lab", "lab.jpeg")
            self._image(dataset, "healthy", "real", "real.jpg")
            note = dataset / "clean" / "healthy" / "real" / "notes.txt"
            note.write_text("not an image", encoding="utf-8")

            report = self._audit(root, dataset, config, docs)

            self.assertEqual(report["total_images"], 2)
            self.assertIn("healthy/real/notes.txt", report["ignored_files"])
            self.assertIn("invalid_file", self._types(root))

    def test_missing_dataset_root(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, config, docs = self._fixture(root, ["healthy"], {"healthy": (0, 0)})

            with self.assertRaisesRegex(FileNotFoundError, "DATASET_ROOT"):
                self._audit(root, root / "missing", config, docs)

    def test_invalid_configuration(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            (dataset / "clean").mkdir(parents=True)
            config = root / "dataset.yaml"
            config.write_text("dataset:\n  classes: healthy\n", encoding="utf-8")
            docs = root / "classes.md"
            docs.write_text("# no table\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dataset.classes"):
                self._audit(root, dataset, config, docs)

    def test_aphids_present_and_lethal_configured_is_explicit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, docs = self._fixture(
                root,
                ["lethal_necrosis"],
                {"lethal_necrosis": (0, 1)},
            )
            self._image(dataset, "aphids_pest", "real", "aphid.jpg")

            report = self._audit(root, dataset, config, docs)

            self.assertEqual(report["missing_classes"], ["lethal_necrosis"])
            self.assertEqual(report["additional_classes"], ["aphids_pest"])
            self.assertEqual(report["split_recommendation"], "do_not_generate")

    def test_documented_classes_can_differ_from_disk(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, docs = self._fixture(
                root,
                ["healthy"],
                {"common_rust": (1, 0)},
            )
            self._image(dataset, "healthy", "real", "healthy.jpg")

            self._audit(root, dataset, config, docs)

            mismatch_types = self._types(root)
            self.assertIn("documented_missing_on_disk", mismatch_types)
            self.assertIn("disk_class_not_documented", mismatch_types)

    def test_documented_total_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, docs = self._fixture(
                root,
                ["healthy"],
                {"healthy": (0, 2)},
            )
            self._image(dataset, "healthy", "real", "healthy.jpg")

            report = self._audit(root, dataset, config, docs)

            self.assertEqual(report["total_images"], 1)
            self.assertIn("count_mismatch", self._types(root))

    def test_fail_on_mismatch_is_optional(self) -> None:
        report = {"mismatch_count": 1, "critical_mismatch_count": 1}

        self.assertEqual(audit_exit_code(report, fail_on_mismatch=True), 2)
        self.assertEqual(audit_exit_code(report, fail_on_mismatch=False), 0)
        self.assertEqual(
            audit_exit_code({"mismatch_count": 1, "critical_mismatch_count": 0}, True),
            0,
        )
        self.assertEqual(audit_exit_code({"critical_mismatch_count": 0}, True), 0)

    def _types(self, root: Path) -> set[str]:
        with (root / "audit" / "class_mismatches.csv").open(
            encoding="utf-8",
            newline="",
        ) as handle:
            return {row["mismatch_type"] for row in csv.DictReader(handle)}
