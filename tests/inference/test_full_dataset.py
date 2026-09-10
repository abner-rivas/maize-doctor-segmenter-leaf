from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from src.inference.full_dataset import (
    MANIFEST_COLUMNS,
    build_summary,
    inventory_dataset,
    load_resume_manifest,
    write_gallery,
)


def _image(path: Path, color: tuple[int, int, int] = (10, 100, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color).save(path)


def test_inventory_is_recursive_and_preserves_metadata(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    first = dataset / "train" / "healthy" / "first.jpg"
    duplicate = dataset / "validation" / "healthy" / "duplicate.jpg"
    invalid = dataset / "test" / "rust" / "broken.png"
    _image(first)
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(first.read_bytes())
    invalid.parent.mkdir(parents=True)
    invalid.write_text("not an image", encoding="utf-8")

    records, summary = inventory_dataset(dataset, progress_every=0)

    assert [record.relative_path for record in records] == [
        "test/rust/broken.png",
        "train/healthy/first.jpg",
        "validation/healthy/duplicate.jpg",
    ]
    assert summary["valid_images"] == 2
    assert summary["invalid_images"] == 1
    assert summary["duplicates"] == 1
    valid = [record for record in records if record.valid]
    assert {record.split for record in valid} == {"train", "validation"}
    assert {record.class_name for record in valid} == {"healthy"}


def test_summary_keeps_failed_separate_from_runtime_errors() -> None:
    rows = [
        {
            "status": "processed",
            "reliability_status": "reliable",
            "output_mode": "segmented",
            "class": "healthy",
            "split": "",
            "mask_area_ratio": 0.4,
            "mask_area_px": 40,
            "inference_time_ms": 10,
            "rejection_reason": "",
        },
        {
            "status": "processed",
            "reliability_status": "failed",
            "output_mode": "full_image",
            "class": "rust",
            "split": "",
            "mask_area_ratio": "",
            "mask_area_px": "",
            "inference_time_ms": 20,
            "rejection_reason": "no_detection",
        },
        {
            "status": "error",
            "reliability_status": "error",
            "class": "rust",
            "split": "",
            "error": "broken",
            "relative_path": "broken.jpg",
        },
    ]

    summary = build_summary(rows, inventory_summary={}, total_wall_seconds=1.0)

    assert summary["images_processed"] == 2
    assert summary["errors"] == 1
    assert summary["reliable"] == 1
    assert summary["failed"] == 1
    assert summary["no_detection"] == 1
    assert summary["full_image"] == 1


def test_gallery_is_local_and_paginated(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    write_gallery(
        tmp_path,
        [
            {
                "image_id": "one",
                "filename": "one.jpg",
                "relative_path": "healthy/real/one.jpg",
                "class": "healthy",
                "split": "",
                "status": "processed",
                "reliability_status": "reliable",
                "original_thumbnail_path": "gallery/thumbnails/one_original.jpg",
                "original_path": "original/one.jpg",
            }
        ],
    )

    html = (gallery / "index.html").read_text(encoding="utf-8")
    data = (gallery / "data.js").read_text(encoding="utf-8")
    assert 'script src="data.js"' in html
    assert "pageSize = 100" in html
    payload = json.loads(data.removeprefix("window.GALLERY_DATA = ").removesuffix(";\n"))
    assert payload[0]["original_thumb"] == "thumbnails/one_original.jpg"
    assert payload[0]["original"] == "../original/one.jpg"


def test_resume_manifest_must_be_an_aligned_inventory_prefix(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _image(dataset / "a.jpg")
    _image(dataset / "b.jpg", (20, 90, 30))
    records, _ = inventory_dataset(dataset, progress_every=0)
    row = {
        "image_id": records[0].image_id,
        "relative_path": records[0].relative_path,
        "status": "processed",
    }
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert load_resume_manifest(tmp_path, records) == [row]

    mismatched = dict(row, image_id=records[1].image_id)
    (tmp_path / "manifest.jsonl").write_text(json.dumps(mismatched) + "\n", encoding="utf-8")
    try:
        load_resume_manifest(tmp_path, records)
    except ValueError as exc:
        assert "mismos registros" in str(exc)
    else:
        raise AssertionError("Un manifest desalineado no debe aceptarse")
