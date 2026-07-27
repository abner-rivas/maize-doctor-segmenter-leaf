"""Tests for deterministic preparation of leaf-detector annotation batches."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.data.leaf_detector_dataset import (
    CVAT_GUIDE,
    DATASET_YAML_TEMPLATE,
    build_detector_annotation_set,
    scan_split_candidates,
    select_detector_candidates,
)
from src.data.leaf_pilot import sha256_file


class LeafDetectorDatasetTests(TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        dataset = root / "dataset"
        split_dir = root / "splits"
        split_dir.mkdir()
        split_paths: dict[str, Path] = {}
        for split, per_class in (("train", 8), ("val", 5)):
            rows: list[dict[str, str]] = []
            for label in ("healthy", "rust"):
                for environment in ("real", "lab"):
                    for index in range(per_class):
                        relative = (
                            Path("clean")
                            / label
                            / environment
                            / f"{split}_{label}_{environment}_{index}.jpg"
                        )
                        image = dataset / relative
                        image.parent.mkdir(parents=True, exist_ok=True)
                        width = 30 + index * 3
                        height = 50 if index % 2 else 24
                        split_offset = 0 if split == "train" else 80
                        label_offset = 0 if label == "healthy" else 30
                        environment_offset = 0 if environment == "real" else 15
                        color = (
                            split_offset + label_offset + environment_offset + index,
                            100 + index,
                            30 + label_offset,
                        )
                        Image.new("RGB", (width, height), color).save(image)
                        rows.append(
                            {
                                "image_path": relative.as_posix(),
                                "label": label,
                                "environment": environment,
                            }
                        )
            path = split_dir / f"{split}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("image_path", "label", "environment")
                )
                writer.writeheader()
                writer.writerows(rows)
            split_paths[split] = path

        pilot = root / "pilot"
        pilot_images = pilot / "images"
        pilot_images.mkdir(parents=True)
        imported = pilot / "manifests" / "imported_annotations.csv"
        imported.parent.mkdir()
        columns = (
            "pilot_id",
            "pilot_image_path",
            "original_image_path",
            "image_sha256",
            "label",
            "environment",
            "source_dataset",
            "annotation_status",
            "roi_x1",
            "roi_y1",
            "roi_x2",
            "roi_y2",
            "roi_area_ratio",
            "original_rotation_degrees",
            "roi_conversion_method",
            "roi_clipped",
            "notes",
        )
        pilot_rows = []
        for index, status in enumerate(("annotated", "annotated", "ambiguous"), start=1):
            pilot_id = f"image_{index:04d}"
            image = pilot_images / f"{pilot_id}.jpg"
            Image.new("RGB", (100, 80), (20 * index, 90, 30)).save(image)
            pilot_rows.append(
                {
                    "pilot_id": pilot_id,
                    "pilot_image_path": f"images/{pilot_id}.jpg",
                    "original_image_path": f"clean/healthy/real/test_{index}.jpg",
                    "image_sha256": sha256_file(image),
                    "label": "healthy",
                    "environment": "real",
                    "source_dataset": "fixture",
                    "annotation_status": status,
                    "roi_x1": "10",
                    "roi_y1": "10",
                    "roi_x2": "80",
                    "roi_y2": "70",
                    "roi_area_ratio": "0.525",
                    "original_rotation_degrees": "12.0",
                    "roi_conversion_method": "rotated_to_axis_aligned",
                    "roi_clipped": "False",
                    "notes": "",
                }
            )
        with imported.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(pilot_rows)
        cvat = pilot / "annotations" / "cvat" / "annotations.xml"
        cvat.parent.mkdir(parents=True)
        cvat.write_text("<annotations/>", encoding="utf-8")
        return {
            "dataset": dataset,
            "train": split_paths["train"],
            "val": split_paths["val"],
            "pilot": pilot,
            "imported": imported,
            "cvat": cvat,
        }

    def test_selection_is_deterministic_and_diverse(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            candidates, _ = scan_split_candidates(paths["train"], paths["dataset"])
            first = select_detector_candidates(candidates, 12, 42)
            second = select_detector_candidates(candidates, 12, 42)

            self.assertEqual(
                [item.image_path for item in first],
                [item.image_path for item in second],
            )
            self.assertEqual(len({item.image_sha256 for item in first}), 12)
            self.assertEqual({item.environment for item in first}, {"real", "lab"})
            self.assertGreaterEqual(len({item.orientation for item in first}), 2)

    def test_build_counts_manifests_hashes_zips_and_no_training(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            output = root / "detector_dataset"

            summary = build_detector_annotation_set(
                paths["train"],
                paths["val"],
                paths["dataset"],
                paths["pilot"],
                paths["imported"],
                paths["cvat"],
                output,
                train_count=12,
                val_count=6,
                seed=42,
            )

            self.assertEqual(summary["counts"]["train"], 12)
            self.assertEqual(summary["counts"]["val"], 6)
            self.assertEqual(summary["counts"]["test_annotated"], 2)
            self.assertEqual(summary["counts"]["test_ambiguous"], 1)
            self.assertTrue(summary["leakage_zero"])
            self.assertFalse(summary["training_performed"])
            self.assertFalse(summary["weights_downloaded"])
            self.assertFalse(summary["labels_invented"])
            self.assertFalse((output / "yolo").exists())
            self.assertFalse((output / "annotation_batches" / "train" / "labels").exists())
            self.assertFalse((output / "annotation_batches" / "val" / "labels").exists())
            self.assertEqual(
                len(list((output / "annotation_batches" / "train" / "images").iterdir())),
                12,
            )
            self.assertEqual(
                len(list((output / "annotation_batches" / "val" / "images").iterdir())),
                6,
            )
            self.assertEqual(len(list((output / "test" / "labels").iterdir())), 2)

            with (output / "manifests" / "train_selection.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                train_rows = list(csv.DictReader(handle))
            with (output / "manifests" / "val_selection.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                val_rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["annotation_status"] == "pending" for row in train_rows))
            self.assertTrue(all(row["annotation_status"] == "pending" for row in val_rows))
            self.assertFalse(
                {row["original_image_path"] for row in train_rows}
                & {row["original_image_path"] for row in val_rows}
            )
            for row in [*train_rows, *val_rows]:
                copied = output / row["copied_image_path"]
                self.assertIn(copied.suffix, (".jpg", ".jpeg", ".png"))
                self.assertEqual(sha256_file(copied), row["image_sha256"])

            leakage = json.loads(
                (output / "manifests" / "leakage_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(leakage["zero_leakage"])
            self.assertEqual(leakage["total_overlap_signals"], 0)
            self.assertEqual(
                (output / "dataset.yaml.template").read_text(encoding="utf-8"),
                DATASET_YAML_TEMPLATE,
            )
            for split in ("train", "val"):
                archive_path = output / "cvat" / f"{split}_annotation_batch.zip"
                with zipfile.ZipFile(archive_path) as archive:
                    names = archive.namelist()
                    self.assertIn("annotation_guide.md", names)
                    self.assertFalse(any(name.startswith("labels/") for name in names))
                    self.assertEqual(
                        archive.read("annotation_guide.md").decode("utf-8"),
                        CVAT_GUIDE,
                    )

    def test_pilot_path_and_hash_are_excluded(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            with paths["train"].open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            pilot_source = paths["dataset"] / rows[0]["image_path"]
            candidates, exclusions = scan_split_candidates(
                paths["train"],
                paths["dataset"],
                excluded_paths={rows[0]["image_path"].casefold()},
                excluded_hashes={sha256_file(pilot_source)},
            )

            self.assertNotIn(rows[0]["image_path"], {item.image_path for item in candidates})
            self.assertEqual(exclusions["held_out_path"], 1)

    def test_exif_orientation_is_applied_to_selection_dimensions(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            relative = Path("clean/healthy/real/rotated.jpg")
            image_path = dataset / relative
            image_path.parent.mkdir(parents=True)
            image = Image.new("RGB", (40, 20), "green")
            exif = image.getexif()
            exif[274] = 6
            image.save(image_path, exif=exif)
            split = root / "train.csv"
            split.write_text(
                "image_path,label,environment\n"
                f"{relative.as_posix()},healthy,real\n",
                encoding="utf-8",
            )

            candidates, _ = scan_split_candidates(split, dataset)

            self.assertEqual((candidates[0].width, candidates[0].height), (20, 40))
            self.assertEqual(candidates[0].orientation, "portrait")

    def test_existing_output_is_not_overwritten(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._fixture(root)
            output = root / "detector_dataset"
            output.mkdir()

            with self.assertRaises(FileExistsError):
                build_detector_annotation_set(
                    paths["train"],
                    paths["val"],
                    paths["dataset"],
                    paths["pilot"],
                    paths["imported"],
                    paths["cvat"],
                    output,
                    train_count=4,
                    val_count=2,
                )
