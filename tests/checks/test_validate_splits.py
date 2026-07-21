"""Tests for split manifests, leakage, coverage, and reproducibility validation."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.data.split_audit import (
    SPLIT_COLUMNS,
    SPLIT_COUNT_COLUMNS,
    SPLIT_NAMES,
    compare_split_directories,
    validate_splits,
)


class SplitValidationTests(TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        dataset = root / "dataset"
        config = root / "dataset.yaml"
        config.write_text(
            "dataset:\n"
            "  seed: 42\n"
            "  classes: [healthy, common_rust]\n"
            "baseline:\n"
            "  seed: 42\n"
            "  classes: [healthy, common_rust]\n",
            encoding="utf-8",
        )
        splits = root / "splits"
        for split_index, split in enumerate(SPLIT_NAMES):
            rows = []
            for label_index, label in enumerate(("healthy", "common_rust")):
                relative = Path("clean") / label / "real" / f"{split}.jpg"
                image = dataset / relative
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new(
                    "RGB",
                    (8, 6),
                    (20 + split_index * 50, 30 + label_index * 70, 40),
                ).save(image)
                rows.append(
                    {
                        "image_path": relative.as_posix(),
                        "label": label,
                        "environment": "real",
                    }
                )
            self._write_split(splits / f"{split}.csv", rows)
        return dataset, config, splits

    def _write_split(
        self,
        path: Path,
        rows: list[dict[str, str]],
        columns: tuple[str, ...] = SPLIT_COLUMNS,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _validate(
        self,
        root: Path,
        dataset: Path,
        config: Path,
        splits: Path,
        *,
        compare: Path | None = None,
    ) -> dict[str, object]:
        return validate_splits(
            splits,
            dataset,
            config,
            root / "validation",
            compare_dir=compare,
        )

    def test_valid_columns_routes_labels_environments_and_counts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, splits = self._fixture(root)

            report = self._validate(root, dataset, config, splits)

            self.assertTrue(report["valid"])
            self.assertEqual(report["total_rows"], 6)
            self.assertEqual(report["images_verified"], 6)
            self.assertEqual(report["leakage"]["cross_split_hash_groups"], 0)
            with (root / "validation" / "split_counts.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), SPLIT_COUNT_COLUMNS)
                self.assertEqual(sum(int(row["image_count"]) for row in reader), 6)

    def test_missing_route_is_invalid(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, splits = self._fixture(root)
            rows = self._read_rows(splits / "test.csv")
            rows[0]["image_path"] = "clean/healthy/real/missing.jpg"
            self._write_split(splits / "test.csv", rows)

            report = self._validate(root, dataset, config, splits)

            self.assertFalse(report["valid"])
            self.assertIn("missing_image", self._issue_types(report))

    def test_invalid_columns_label_and_environment(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, splits = self._fixture(root)
            rows = self._read_rows(splits / "val.csv")
            rows[0]["label"] = "unknown"
            rows[0]["environment"] = "field"
            for row in rows:
                row["extra"] = "x"
            self._write_split(
                splits / "val.csv",
                rows,
                (*SPLIT_COLUMNS, "extra"),
            )

            report = self._validate(root, dataset, config, splits)

            issue_types = self._issue_types(report)
            self.assertIn("invalid_columns", issue_types)
            self.assertIn("invalid_label", issue_types)
            self.assertIn("invalid_environment", issue_types)

    def test_duplicate_route_between_splits_is_leakage(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, splits = self._fixture(root)
            train_rows = self._read_rows(splits / "train.csv")
            test_rows = self._read_rows(splits / "test.csv")
            test_rows[0] = dict(train_rows[0])
            self._write_split(splits / "test.csv", test_rows)

            report = self._validate(root, dataset, config, splits)

            self.assertEqual(report["leakage"]["cross_split_route_groups"], 1)
            self.assertIn("duplicate_route", self._issue_types(report))

    def test_same_hash_on_different_routes_is_detected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, splits = self._fixture(root)
            train_rows = self._read_rows(splits / "train.csv")
            test_rows = self._read_rows(splits / "test.csv")
            first = dataset / train_rows[0]["image_path"]
            second = dataset / test_rows[0]["image_path"]
            second.write_bytes(first.read_bytes())

            report = self._validate(root, dataset, config, splits)

            self.assertEqual(report["leakage"]["cross_split_hash_groups"], 1)
            self.assertIn("cross_split_hash_leakage", self._issue_types(report))

    def test_corrupt_image_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, splits = self._fixture(root)
            rows = self._read_rows(splits / "test.csv")
            (dataset / rows[0]["image_path"]).write_bytes(b"not-an-image")

            report = self._validate(root, dataset, config, splits)

            self.assertFalse(report["valid"])
            self.assertIn("unreadable_image", self._issue_types(report))

    def test_baseline_class_cap_is_enforced(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, splits = self._fixture(root)
            config.write_text(config.read_text(encoding="utf-8") + "  max_images_per_class: 2\n")

            report = self._validate(root, dataset, config, splits)

            self.assertFalse(report["valid"])
            self.assertIn("baseline_cap_exceeded", self._issue_types(report))

    def test_exact_comparison_is_reproducible_with_seed_42(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, splits = self._fixture(root)
            comparison = root / "comparison"
            for split in SPLIT_NAMES:
                target = comparison / f"{split}.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((splits / f"{split}.csv").read_bytes())

            report = self._validate(
                root,
                dataset,
                config,
                splits,
                compare=comparison,
            )

            self.assertTrue(report["reproducibility"]["exactly_reproducible"])
            self.assertEqual(report["reproducibility"]["expected_seed"], 42)

    def test_reordered_rows_are_not_exactly_reproducible(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, splits = self._fixture(root)
            comparison = root / "comparison"
            for split in SPLIT_NAMES:
                rows = self._read_rows(splits / f"{split}.csv")
                self._write_split(comparison / f"{split}.csv", list(reversed(rows)))

            result = compare_split_directories(splits, comparison)

            self.assertFalse(result["exactly_reproducible"])
            self.assertTrue(result["same_route_sets"])
            self.assertFalse(result["same_order"])

    def test_validation_does_not_modify_images(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, config, splits = self._fixture(root)
            image = next((dataset / "clean").rglob("*.jpg"))
            before = image.read_bytes()

            self._validate(root, dataset, config, splits)

            self.assertEqual(image.read_bytes(), before)

    def _issue_types(self, report: dict[str, object]) -> set[str]:
        return {str(issue["type"]) for issue in report["issues"]}
