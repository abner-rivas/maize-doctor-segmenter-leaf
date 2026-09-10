from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from src.inference.classifier_dataset import EXPORT_COLUMNS, export_classifier_dataset


def _write_inference_run(run: Path) -> None:
    segmented = run / "segmented"
    segmented.mkdir(parents=True)
    (segmented / "reliable.jpg").write_bytes(b"reliable-image")
    (segmented / "uncertain.jpg").write_bytes(b"uncertain-image")
    rows = [
        {
            "image_id": "reliable",
            "class": "common_rust",
            "filename": "source-a.jpg",
            "relative_path": "common_rust/real/source-a.jpg",
            "split": "train",
            "environment": "real",
            "status": "processed",
            "reliability_status": "reliable",
            "segmented_path": "segmented/reliable.jpg",
            "sha256": "original-a",
            "duplicate_of": "",
        },
        {
            "image_id": "uncertain",
            "class": "healthy",
            "filename": "source-b.jpg",
            "relative_path": "healthy/real/source-b.jpg",
            "split": "train",
            "environment": "real",
            "status": "processed",
            "reliability_status": "uncertain",
            "segmented_path": "segmented/uncertain.jpg",
            "sha256": "original-b",
            "duplicate_of": "",
        },
        {
            "image_id": "failed",
            "class": "healthy",
            "filename": "source-c.jpg",
            "relative_path": "healthy/real/source-c.jpg",
            "split": "train",
            "environment": "real",
            "status": "processed",
            "reliability_status": "failed",
            "segmented_path": "",
            "sha256": "original-c",
            "duplicate_of": "",
        },
    ]
    with (run / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_export_groups_reliable_images_under_class_and_lab(tmp_path: Path) -> None:
    run = tmp_path / "inference"
    _write_inference_run(run)

    summary = export_classifier_dataset(run)

    dataset = run / "classifier_dataset"
    exported_image = dataset / "common_rust" / "lab" / "reliable.jpg"
    assert exported_image.read_bytes() == b"reliable-image"
    assert os.stat(exported_image).st_ino == os.stat(run / "segmented/reliable.jpg").st_ino
    assert not (dataset / "healthy").exists()
    assert summary["images_included"] == 1
    assert summary["images_excluded"] == 2
    assert summary["by_class"] == {"common_rust": 1}
    assert summary["excluded_by_status"] == {"failed": 1, "uncertain": 1}

    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == EXPORT_COLUMNS
    assert rows[0]["dataset_path"] == "common_rust/lab/reliable.jpg"
    assert rows[0]["original_environment"] == "real"
    assert json.loads((dataset / "summary.json").read_text(encoding="utf-8"))[
        "source_manifest_sha256"
    ]


def test_export_can_include_uncertain_segmentations_as_copies(tmp_path: Path) -> None:
    run = tmp_path / "inference"
    _write_inference_run(run)
    output = tmp_path / "classifier"

    summary = export_classifier_dataset(
        run,
        output_dir=output,
        reliability_statuses=("reliable", "uncertain"),
        storage_mode="copy",
    )

    assert (output / "healthy/lab/uncertain.jpg").read_bytes() == b"uncertain-image"
    assert summary["images_included"] == 2
    assert summary["excluded_by_status"] == {"failed": 1}


def test_export_rejects_unsafe_class_names_without_leaving_output(tmp_path: Path) -> None:
    run = tmp_path / "inference"
    _write_inference_run(run)
    manifest = run / "manifest.csv"
    text = manifest.read_text(encoding="utf-8").replace("common_rust", "../outside", 1)
    manifest.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="nombre de directorio seguro"):
        export_classifier_dataset(run)

    assert not (run / "classifier_dataset").exists()
    assert not (tmp_path / "outside").exists()
