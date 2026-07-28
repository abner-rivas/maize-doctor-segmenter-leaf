"""Tests for deterministic, leakage-safe segmentation splits."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from src.data import segmentation_split as split_module
from src.data.segmentation_consolidation import perceptual_hash
from src.data.segmentation_split import (
    PARENT_READY_STATUS,
    SPLITS,
    SplitGroup,
    SplitRecord,
    SplitValidationError,
    assign_groups_to_splits,
    build_split_groups,
    compute_split_fingerprint,
    materialize_split,
    normalize_roboflow_base_name,
    validate_cross_split_leakage,
    validate_pilot_leakage,
    validate_split_integrity,
    verify_parent_dataset,
    write_dataset_yaml,
)


def _record(
    index: int,
    tmp_path: Path,
    *,
    source: str = "large",
    image_hash: str | None = None,
    phash: int | None = None,
    variant: str | None = None,
    area: float = 0.2,
    instances: int = 1,
) -> SplitRecord:
    filename = f"image_{index:04d}.jpg"
    return SplitRecord(
        filename=filename,
        image_path=tmp_path / "all" / "images" / filename,
        label_path=tmp_path / "all" / "labels" / f"image_{index:04d}.txt",
        source_dataset=source,
        image_sha256=image_hash or f"{index:064x}",
        label_sha256=f"{index + 10_000:064x}",
        perceptual_hash=f"ahash64:{(phash if phash is not None else index):016x}",
        original_filename=f"capture_{index}_jpg.rf.hash{index}.jpg",
        original_base_name=f"capture_{index}",
        roboflow_variant_group=variant or f"variant_{index}",
        duplicate_group=f"exact_{index}",
        width=640,
        height=480,
        orientation="horizontal",
        resolution="small",
        instance_count=instances,
        mask_areas=tuple(area for _ in range(instances)),
        touches_border=False,
    )


def _group(record: SplitRecord) -> SplitGroup:
    group = SplitGroup(
        group_id=f"group_{record.filename}",
        records=[record],
        perceptual_cluster=f"cluster_{record.filename}",
        features=split_module._group_features([record]),
    )
    record.group_id = group.group_id
    return group


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("leaf_jpg.rf.Abc123.jpg", "leaf"),
        ("LEAF_JPEG.rf.Hash.jpeg", "leaf"),
        ("capture.png", "capture"),
        ("a.b_c_png.rf.123.PNG", "a.b_c"),
    ],
)
def test_normalize_roboflow_suffix(filename: str, expected: str) -> None:
    assert normalize_roboflow_base_name(filename) == expected


def test_exact_duplicates_and_roboflow_variants_are_grouped(tmp_path: Path) -> None:
    first = _record(1, tmp_path, image_hash="a" * 64, phash=0, variant="same")
    second = _record(2, tmp_path, image_hash="a" * 64, phash=2**20, variant="other")
    third = _record(3, tmp_path, phash=2**40, variant="same")
    groups = build_split_groups([first, second, third], perceptual_threshold=0)
    assert len(groups) == 1


def test_perceptual_variants_are_grouped_within_threshold(tmp_path: Path) -> None:
    first = _record(1, tmp_path, phash=0)
    second = _record(2, tmp_path, phash=0b1111)
    third = _record(3, tmp_path, phash=0b1_1111)
    groups = build_split_groups([first, second, third], perceptual_threshold=4)
    assert len(groups) == 1  # transitive connected component


def test_one_group_never_appears_in_multiple_splits(tmp_path: Path) -> None:
    records = [_record(index, tmp_path, variant=f"v{index // 2}") for index in range(30)]
    groups = build_split_groups(records, perceptual_threshold=-1)
    assign_groups_to_splits(
        groups,
        seed=42,
        ratios={"train": 0.7, "val": 0.15, "test": 0.15},
    )
    assert all(len({record.split for record in group.records}) == 1 for group in groups)


def test_assignment_is_deterministic_and_conserves_totals(tmp_path: Path) -> None:
    def run() -> tuple[list[tuple[str, str]], Counter[str]]:
        records = [_record(index, tmp_path, phash=1 << (index % 63)) for index in range(60)]
        groups = [_group(record) for record in records]
        assign_groups_to_splits(
            groups,
            seed=42,
            ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        )
        return (
            sorted((record.filename, record.split) for record in records),
            Counter(record.split for record in records),
        )

    first, counts = run()
    second, _ = run()
    assert first == second
    assert sum(counts.values()) == 60
    assert counts == {"train": 42, "val": 9, "test": 9}


def test_assignment_balances_rare_source_and_mask_areas(tmp_path: Path) -> None:
    records = [
        _record(
            index,
            tmp_path,
            source="corn" if index < 20 else "large",
            area=0.02 if index % 5 == 0 else 0.7 if index % 5 == 1 else 0.2,
        )
        for index in range(100)
    ]
    groups = [_group(record) for record in records]
    assign_groups_to_splits(
        groups,
        seed=42,
        ratios={"train": 0.7, "val": 0.15, "test": 0.15},
    )
    corn = Counter(record.split for record in records if record.source_dataset == "corn")
    small = Counter(record.split for record in records if record.mask_area_min < 0.05)
    assert corn == {"train": 14, "val": 3, "test": 3}
    assert small == {"train": 14, "val": 3, "test": 3}


def _create_pair(record: SplitRecord) -> None:
    record.image_path.parent.mkdir(parents=True, exist_ok=True)
    record.label_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), "green").save(record.image_path)
    record.label_path.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")
    record.image_sha256 = split_module.sha256_file(record.image_path)
    record.label_sha256 = split_module.sha256_file(record.label_path)


def test_materialization_copy_and_pair_correspondence(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    record = _record(1, source_root)
    record.split = "train"
    _create_pair(record)
    materialize_split([record], output_root, materialization="copy")
    write_dataset_yaml(output_root)
    assert record.materialization_method == "copy"
    assert not validate_split_integrity(
        [record], output_root, expected_images=1, expected_masks=1
    )


def test_materialization_hardlink_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    record = _record(1, source_root)
    record.split = "val"
    _create_pair(record)
    hardlink_root = tmp_path / "hardlink"
    materialize_split([record], hardlink_root, materialization="hardlink")
    assert record.materialization_method == "hardlink"
    assert os.stat(record.image_path).st_ino == os.stat(
        hardlink_root / "images" / "val" / record.filename
    ).st_ino

    monkeypatch.setattr(split_module.os, "link", lambda *_: (_ for _ in ()).throw(OSError()))
    fallback_root = tmp_path / "fallback"
    materialize_split([record], fallback_root, materialization="hardlink")
    assert record.materialization_method == "copy_fallback"


def test_dataset_yaml_is_portable_and_has_three_existing_paths(tmp_path: Path) -> None:
    for kind in ("images", "labels"):
        for split in SPLITS:
            (tmp_path / kind / split).mkdir(parents=True)
    write_dataset_yaml(tmp_path)
    content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
    assert "path: ." in content
    assert all(f"{split}: images/{split}" in content for split in SPLITS)
    assert "0: maize_leaf" in content
    assert "pilot" not in content


def test_cross_split_exact_group_variant_and_perceptual_leakage(tmp_path: Path) -> None:
    first = _record(1, tmp_path, image_hash="a" * 64, phash=0, variant="same")
    second = _record(2, tmp_path, image_hash="a" * 64, phash=1, variant="same")
    first.group_id = second.group_id = "same_group"
    first.split, second.split = "train", "test"
    issues = validate_cross_split_leakage([first, second], perceptual_threshold=4)
    assert Counter(issue["leakage_type"] for issue in issues) == {
        "exact_hash": 1,
        "group": 1,
        "roboflow_variant": 1,
        "perceptual_near": 1,
    }


def test_pilot_leakage_detects_exact_name_base_and_perceptual(tmp_path: Path) -> None:
    pilot_root = tmp_path / "pilot"
    pilot_image = pilot_root / "images" / "pilot.jpg"
    pilot_image.parent.mkdir(parents=True)
    Image.new("RGB", (24, 24), "green").save(pilot_image)
    pilot_hash = split_module.sha256_file(pilot_image)
    manifest = pilot_root / "manifests" / "pilot_manifest.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "pilot_image_path",
                "original_filename",
                "original_image_path",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "pilot_image_path": "images/pilot.jpg",
                "original_filename": "same_jpg.rf.pilot.jpg",
                "original_image_path": "clean/same.jpg",
            }
        )
    record = _record(1, tmp_path)
    record.filename = "pilot.jpg"
    record.original_base_name = "same"
    record.image_sha256 = pilot_hash
    record.perceptual_hash = perceptual_hash(pilot_image)
    issues, _ = validate_pilot_leakage([record], pilot_root, perceptual_threshold=4)
    assert {issue["leakage_type"] for issue in issues} == {
        "exact_hash",
        "filename",
        "original_base_name",
        "perceptual_near",
    }


def test_split_fingerprints_are_deterministic_and_membership_sensitive(tmp_path: Path) -> None:
    first, second = _record(1, tmp_path), _record(2, tmp_path)
    first.split = second.split = "train"
    baseline = compute_split_fingerprint([first, second], "train")
    assert baseline == compute_split_fingerprint([second, first], "train")
    second.split = "val"
    assert baseline != compute_split_fingerprint([first, second], "train")


def _parent_lock(tmp_path: Path, *, status: str = PARENT_READY_STATUS) -> Path:
    manifests = tmp_path / "manifests"
    manifests.mkdir(parents=True)
    lock = {
        "status": status,
        "total_images": 1155,
        "total_masks": 1224,
        "global_fingerprint": {"sha256": "expected"},
    }
    (manifests / "dataset_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    return tmp_path


def test_parent_gate_blocks_unready_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _parent_lock(tmp_path, status="blocked_by_manual_review")
    with pytest.raises(SplitValidationError, match="dataset_lock.status"):
        verify_parent_dataset(root, expected_fingerprint="expected")


def test_parent_gate_blocks_changed_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _parent_lock(tmp_path)
    monkeypatch.setattr(
        split_module,
        "dataset_fingerprint",
        lambda _: {"sha256": "changed"},
    )
    with pytest.raises(SplitValidationError, match="fingerprint actual"):
        verify_parent_dataset(root, expected_fingerprint="expected")


def test_parent_gate_accepts_ready_matching_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _parent_lock(tmp_path)
    monkeypatch.setattr(
        split_module,
        "dataset_fingerprint",
        lambda _: {"sha256": "expected"},
    )
    assert verify_parent_dataset(root, expected_fingerprint="expected")["status"] == (
        PARENT_READY_STATUS
    )


def test_parent_gate_accepts_derived_yaml_when_frozen_parent_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _parent_lock(tmp_path)
    monkeypatch.setattr(
        split_module,
        "dataset_fingerprint",
        lambda _: {"sha256": "derived"},
    )
    monkeypatch.setattr(
        split_module,
        "_frozen_parent_fingerprint",
        lambda _: "expected",
    )
    assert verify_parent_dataset(root, expected_fingerprint="expected")["status"] == (
        PARENT_READY_STATUS
    )


def test_reconstruction_assignment_and_fingerprints_repeat(tmp_path: Path) -> None:
    def run() -> tuple[list[tuple[str, str]], dict[str, str]]:
        records = [_record(index, tmp_path) for index in range(40)]
        groups = [_group(record) for record in records]
        assign_groups_to_splits(
            groups,
            seed=42,
            ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        )
        return (
            sorted((record.filename, record.split) for record in records),
            {
                split: compute_split_fingerprint(records, split) for split in SPLITS
            },
        )

    assert run() == run()
