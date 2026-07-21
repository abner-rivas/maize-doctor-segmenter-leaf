"""Tests for reproducible pilot selection and file materialization."""

import csv
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.data.leaf_pilot import (
    PILOT_COLUMNS,
    build_pilot,
    materialize_file,
    read_csv_rows,
    select_pilot_rows,
    sha256_file,
)


def _selection_rows(per_class: int = 6) -> list[dict[str, str]]:
    return [
        {
            "image_path": f"clean/{label}/real/{label}_{index}.jpg",
            "label": label,
            "environment": "real",
        }
        for label in ("a", "b", "c")
        for index in range(per_class)
    ]


class PilotSelectionTests(TestCase):
    def test_same_seed_is_reproducible(self) -> None:
        rows = _selection_rows()

        first, _ = select_pilot_rows(rows, 9, 42, "balanced")
        second, _ = select_pilot_rows(rows, 9, 42, "balanced")

        self.assertEqual(first, second)

    def test_different_seed_changes_selection(self) -> None:
        rows = _selection_rows()

        first, _ = select_pilot_rows(rows, 9, 42, "balanced")
        second, _ = select_pilot_rows(rows, 9, 43, "balanced")

        self.assertNotEqual(first, second)

    def test_balanced_selection_covers_classes_evenly(self) -> None:
        selected, _ = select_pilot_rows(_selection_rows(), 9, 42, "balanced")

        self.assertEqual(Counter(row["label"] for row in selected), {"a": 3, "b": 3, "c": 3})

    def test_balanced_selection_prioritizes_real_environment(self) -> None:
        rows = [
            {
                "image_path": f"clean/a/{environment}/{environment}_{index}.jpg",
                "label": "a",
                "environment": environment,
            }
            for environment in ("lab", "real")
            for index in range(3)
        ]

        selected, _ = select_pilot_rows(rows, 3, 42, "balanced")

        self.assertTrue(all(row["environment"] == "real" for row in selected))

    def test_scarce_class_uses_all_and_redistributes(self) -> None:
        rows = [row for row in _selection_rows() if row["label"] != "c"]
        rows.append({"image_path": "clean/c/real/c_0.jpg", "label": "c", "environment": "real"})

        selected, warnings = select_pilot_rows(rows, 9, 42, "balanced")

        counts = Counter(row["label"] for row in selected)
        self.assertEqual(counts["c"], 1)
        self.assertEqual(len(selected), 9)
        self.assertTrue(any("redistribuido" in warning for warning in warnings))

    def test_duplicate_paths_are_not_selected_twice(self) -> None:
        rows = _selection_rows(2)
        rows.append(dict(rows[0]))

        selected, warnings = select_pilot_rows(rows, 20, 42, "balanced")

        paths = [row["image_path"] for row in selected]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(any("duplicadas" in warning for warning in warnings))

    def test_request_larger_than_available_returns_all(self) -> None:
        rows = _selection_rows(1)

        selected, warnings = select_pilot_rows(rows, 100, 42, "balanced")

        self.assertEqual(len(selected), 3)
        self.assertTrue(any("sólo hay 3" in warning for warning in warnings))


class PilotMaterializationTests(TestCase):
    def test_copy_hash_and_original_immutability(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpg"
            Image.new("RGB", (20, 10), "green").save(source)
            before = sha256_file(source)
            destination = root / "copy" / "image.jpg"

            actual_mode = materialize_file(source, destination, "copy")

            self.assertEqual(actual_mode, "copy")
            self.assertEqual(sha256_file(destination), before)
            self.assertEqual(sha256_file(source), before)

    def test_hardlink_and_symlink_modes_are_explicit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"leaf")

            self.assertEqual(materialize_file(source, root / "hard.bin", "hardlink"), "hardlink")
            self.assertEqual(materialize_file(source, root / "soft.bin", "symlink"), "symlink")
            self.assertTrue((root / "soft.bin").is_symlink())

    def test_missing_source_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                materialize_file(root / "missing.jpg", root / "out.jpg", "copy")

    def test_existing_destination_is_not_overwritten(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"source")
            destination.write_bytes(b"keep")

            with self.assertRaises(FileExistsError):
                materialize_file(source, destination, "copy")

            self.assertEqual(destination.read_bytes(), b"keep")


class PilotBuildTests(TestCase):
    def _create_fixture(self, root: Path) -> tuple[Path, Path]:
        dataset_root = root / "dataset"
        split_path = root / "test.csv"
        rows: list[dict[str, str]] = []
        for label in ("healthy", "common_rust"):
            for environment in ("real", "lab"):
                for index in range(2):
                    relative = (
                        Path("clean")
                        / label
                        / environment
                        / f"{label}_{environment}_{index}.jpg"
                    )
                    image_path = dataset_root / relative
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (30, 20), (index * 20, 100, 30)).save(image_path)
                    rows.append(
                        {
                            "image_path": relative.as_posix(),
                            "label": label,
                            "environment": environment,
                        }
                    )
        with split_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image_path", "label", "environment"))
            writer.writeheader()
            writer.writerows(rows)
        return dataset_root, split_path

    def test_environment_filter_structure_and_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root, split_path = self._create_fixture(root)
            output = root / "pilot"

            summary = build_pilot(
                split_path,
                dataset_root,
                output,
                samples=4,
                seed=42,
                environments=("real",),
                classes=None,
                copy_mode="copy",
                selection_strategy="balanced",
            )
            rows, columns = read_csv_rows(output / "manifests" / "pilot_manifest.csv")

            self.assertEqual(summary["selected_samples"], 4)
            self.assertEqual(tuple(columns), PILOT_COLUMNS)
            self.assertTrue(all(row["environment"] == "real" for row in rows))
            self.assertTrue(all(row["source_dataset"] == "unknown" for row in rows))
            self.assertTrue(all(row["annotation_status"] == "pending" for row in rows))
            self.assertTrue((output / "annotation_guide.md").is_file())
            self.assertTrue((output / "README.md").is_file())
            self.assertEqual(len(list((output / "images").iterdir())), 4)

    def test_missing_required_split_column_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            split_path = root / "test.csv"
            split_path.write_text("image_path,label\na.jpg,healthy\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "environment"):
                build_pilot(
                    split_path,
                    root / "dataset",
                    root / "pilot",
                    samples=1,
                    seed=42,
                    environments=("real",),
                    classes=None,
                    copy_mode="copy",
                    selection_strategy="balanced",
                )

    def test_same_original_filename_does_not_collide(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            split_path = root / "test.csv"
            rows = []
            for label, color in (("healthy", "green"), ("common_rust", "brown")):
                relative = Path("clean") / label / "real" / "leaf.jpg"
                source = dataset_root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (30, 20), color).save(source)
                rows.append(
                    {
                        "image_path": relative.as_posix(),
                        "label": label,
                        "environment": "real",
                    }
                )
            with split_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("image_path", "label", "environment"),
                )
                writer.writeheader()
                writer.writerows(rows)

            build_pilot(
                split_path,
                dataset_root,
                root / "pilot",
                samples=2,
                seed=42,
                environments=("real",),
                classes=None,
                copy_mode="copy",
                selection_strategy="balanced",
            )
            manifest_rows, _ = read_csv_rows(
                root / "pilot" / "manifests" / "pilot_manifest.csv"
            )

            self.assertEqual({row["original_filename"] for row in manifest_rows}, {"leaf.jpg"})
            self.assertEqual(
                len({row["pilot_image_path"] for row in manifest_rows}),
                2,
            )
