"""Tests for deterministic JPEG normalization and mutation detection."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from src.data import jpeg_normalization as jpeg_module
from src.data.jpeg_normalization import (
    JpegNormalizationError,
    canonical_jpeg_paths,
    inspect_jpeg,
    normalize_jpeg_copy,
    ultralytics_scan_hash_mutations,
    write_auxiliary_jpeg_audit,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_eoi_is_repaired_losslessly_without_touching_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    valid = tmp_path / "valid.jpg"
    derived = tmp_path / "derived.jpg"
    Image.new("RGB", (24, 16), (20, 140, 40)).save(valid, format="JPEG", quality=90)
    source.write_bytes(valid.read_bytes()[:-2] + b"\x00\x00")
    original_bytes = source.read_bytes()

    row = normalize_jpeg_copy(source, derived)

    assert source.read_bytes() == original_bytes
    assert derived.read_bytes() == original_bytes + b"\xff\xd9"
    assert row["issue"] == "missing_jpeg_eoi"
    assert row["normalization_method"] == "append_ffd9"
    assert row["original_sha256"] == _sha256(source)
    assert row["normalized_sha256"] == _sha256(derived)
    assert row["original_pixel_sha256"] == row["normalized_pixel_sha256"]
    assert row["original_mode"] == row["normalized_mode"] == "RGB"
    assert (row["width"], row["height"]) == (24, 16)
    assert row["pixel_equivalence_verified"] is True
    assert row["status"] == "normalized"
    assert inspect_jpeg(derived)["issues"] == []


def test_valid_jpeg_is_copied_byte_for_byte(tmp_path: Path) -> None:
    source = tmp_path / "source.jpeg"
    derived = tmp_path / "derived.jpeg"
    Image.new("RGB", (16, 12), "green").save(source, format="JPEG")

    row = normalize_jpeg_copy(source, derived)

    assert source.read_bytes() == derived.read_bytes()
    assert row["issue"] == "none"
    assert row["normalization_method"] == "copy_unchanged"
    assert row["pixel_equivalence_verified"] is True
    assert row["status"] == "unchanged"


def test_undecodable_jpeg_blocks_instead_of_guessing(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"\xff\xd8not-a-jpeg\xff\xd9")

    with pytest.raises(JpegNormalizationError, match="Pillow no carga"):
        normalize_jpeg_copy(source, tmp_path / "derived.jpg")


def test_reencode_path_transposes_exif_strips_metadata_and_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jpg"
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (12, 20), (30, 100, 170)).save(
        source,
        format="JPEG",
        quality=90,
        exif=exif,
    )
    real_inspect = jpeg_module.inspect_jpeg

    def normalize_with_forced_verify_issue(destination: Path) -> dict[str, object]:
        first_call = True

        def inspect(path: Path) -> dict[str, object]:
            nonlocal first_call
            result = real_inspect(path)
            if first_call:
                first_call = False
                result["verify_error"] = "SyntheticVerifyError"
                result["issues"] = [
                    "pillow_verify_error:SyntheticVerifyError"
                ]
            return result

        monkeypatch.setattr(jpeg_module, "inspect_jpeg", inspect)
        return normalize_jpeg_copy(source, destination)

    first_row = normalize_with_forced_verify_issue(first)
    second_row = normalize_with_forced_verify_issue(second)

    assert first_row["normalization_method"] == "reencode_exif_transpose_rgb_q95_444"
    assert first_row["normalized_sha256"] == second_row["normalized_sha256"]
    with Image.open(first) as normalized:
        assert normalized.mode == "RGB"
        assert normalized.size == (20, 12)
        assert not normalized.getexif()


def test_ultralytics_scan_reports_hash_mutation_only_in_temporary_copy(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    source = image_root / "leaf.jpg"
    Image.new("RGB", (16, 16), "green").save(source, format="JPEG")
    original_hash = _sha256(source)

    def mutating_checker(path: str) -> tuple[str, tuple[int, int]]:
        target = Path(path)
        target.write_bytes(target.read_bytes() + b"mutation")
        return "mutated", (16, 16)

    report = ultralytics_scan_hash_mutations(
        image_root,
        checker=mutating_checker,
        checker_version="test-double",
    )

    assert report["passed"] is False
    assert report["mutated_file_count"] == 1
    assert report["mutations"][0]["path"] == "leaf.jpg"
    assert _sha256(source) == original_hash


def test_canonical_scope_excludes_historical_and_annotation_batch_images(
    tmp_path: Path,
) -> None:
    expected: list[Path] = []
    for relative in (
        "all/images/parent.jpg",
        "images/train/train.jpg",
        "images/val/val.jpeg",
        "images/test/test.jpg",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), "green").save(path, format="JPEG")
        expected.append(path.resolve())
    for relative in (
        "annotation_batches/train/images/history.jpg",
        "test/images/legacy.jpg",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), "green").save(path, format="JPEG")

    assert canonical_jpeg_paths(tmp_path) == sorted(
        expected,
        key=lambda path: path.relative_to(tmp_path).as_posix(),
    )


def test_auxiliary_jpeg_report_is_informational_and_deterministic(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.jpg"
    Image.new("RGB", (14, 9), "green").save(valid, format="JPEG")
    auxiliary = tmp_path / "annotation_batches/train/images/history.jpg"
    auxiliary.parent.mkdir(parents=True)
    auxiliary.write_bytes(valid.read_bytes()[:-2] + b"\x00\x00")
    output = tmp_path / "manifests/auxiliary_jpeg_audit.csv"

    first = write_auxiliary_jpeg_audit(tmp_path, output)
    first_bytes = output.read_bytes()
    second = write_auxiliary_jpeg_audit(tmp_path, output)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert first == second
    assert output.read_bytes() == first_bytes
    assert first["issue_count"] == first["missing_eoi_count"] == 1
    assert first["training_gate"] is False
    assert rows[0]["path"] == "annotation_batches/train/images/history.jpg"
    assert rows[0]["issue"] == "missing_jpeg_eoi"
    assert rows[0]["training_gate"] == "False"
