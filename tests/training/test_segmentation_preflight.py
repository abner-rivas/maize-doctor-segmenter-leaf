"""Synthetic tests for the read-only leaf-segmentation preflight."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.data.segmentation_audit import sha256_file
from src.training import segmentation_preflight as preflight


def _write_dataset(root: Path, counts: dict[str, int] | None = None) -> Path:
    counts = counts or {"train": 1, "val": 1, "test": 1}
    root.mkdir(parents=True)
    (root / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: maize_leaf\n",
        encoding="utf-8",
    )
    manifest_rows = []
    for split, count in counts.items():
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(count):
            image = root / "images" / split / f"{split}_{index}.jpg"
            label = root / "labels" / split / f"{split}_{index}.txt"
            Image.new("RGB", (40 + index, 30), "green").save(image)
            label.write_text(
                "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n",
                encoding="utf-8",
            )
            manifest_rows.append(
                {
                    "split": split,
                    "filename": image.name,
                    "materialized_image_path": image.relative_to(root).as_posix(),
                    "materialized_label_path": label.relative_to(root).as_posix(),
                    "image_sha256": sha256_file(image),
                    "label_sha256": sha256_file(label),
                }
            )
    manifests = root / "manifests"
    manifests.mkdir(exist_ok=True)
    with (manifests / "split_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return root


def _patch_small_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preflight, "EXPECTED_IMAGES", {"train": 1, "val": 1, "test": 1}
    )
    monkeypatch.setattr(
        preflight, "EXPECTED_MASKS", {"train": 1, "val": 1, "test": 1}
    )


def test_correct_locks_and_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_dataset(tmp_path / "dataset")
    split_lock = {
        "status": "ready_for_training_preflight",
        **{f"{split}_fingerprint": "same" for split in preflight.SPLITS},
    }
    (root / "manifests" / "split_lock.json").write_text(
        json.dumps(split_lock), encoding="utf-8"
    )
    monkeypatch.setattr(
        preflight,
        "verify_parent_dataset",
        lambda _: {
            "status": "ready_for_split_generation",
            "global_fingerprint": {"sha256": "parent"},
        },
    )
    monkeypatch.setattr(preflight, "_split_digest", lambda *_: "same")
    assert preflight.verify_training_locks(root)["passed"]


def test_changed_split_fingerprint_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_dataset(tmp_path / "dataset")
    split_lock = {
        "status": "ready_for_training_preflight",
        **{f"{split}_fingerprint": "expected" for split in preflight.SPLITS},
    }
    (root / "manifests" / "split_lock.json").write_text(
        json.dumps(split_lock), encoding="utf-8"
    )
    monkeypatch.setattr(
        preflight,
        "verify_parent_dataset",
        lambda _: {
            "status": "ready_for_split_generation",
            "global_fingerprint": {"sha256": "parent"},
        },
    )
    monkeypatch.setattr(preflight, "_split_digest", lambda *_: "changed")
    with pytest.raises(preflight.PreflightError, match="Fingerprint"):
        preflight.verify_training_locks(root)


def test_cloud_payload_rejects_self_consistent_noncanonical_parent(
    tmp_path: Path,
) -> None:
    """Cambiar datos y ambos locks no puede crear un payload alternativo válido."""
    root = tmp_path / "dataset"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    alternate = "a" * 64
    (manifests / "dataset_lock.json").write_text(
        json.dumps(
            {
                "status": "ready_for_split_generation",
                "global_fingerprint": {"sha256": alternate},
            }
        ),
        encoding="utf-8",
    )
    (manifests / "split_lock.json").write_text(
        json.dumps(
            {
                "status": "ready_for_training_preflight",
                "parent_dataset_fingerprint": alternate,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(preflight.PreflightError, match="contrato congelado"):
        preflight.verify_cloud_training_payload(root)


def test_valid_dataset_yaml_and_segmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_dataset(tmp_path / "dataset")
    _patch_small_counts(monkeypatch)
    report = preflight.validate_segmentation_dataset(root)
    assert report["passed"]
    assert report["class_counts"] == {0: 3}
    assert report["bbox_mixed_count"] == 0


def test_wrong_class_name_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_dataset(tmp_path / "dataset")
    _patch_small_counts(monkeypatch)
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        yaml_path.read_text(encoding="utf-8").replace("maize_leaf", "leaf"),
        encoding="utf-8",
    )
    report = preflight.validate_segmentation_dataset(root)
    assert not report["passed"]
    assert any("names inválido" in error for error in report["errors"])


@pytest.mark.parametrize("remove_kind", ["label", "image"])
def test_missing_image_or_label_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remove_kind: str
) -> None:
    root = _write_dataset(tmp_path / "dataset")
    _patch_small_counts(monkeypatch)
    target = (
        root / "labels" / "train" / "train_0.txt"
        if remove_kind == "label"
        else root / "images" / "train" / "train_0.jpg"
    )
    target.unlink()
    report = preflight.validate_segmentation_dataset(root)
    assert not report["passed"]
    assert any("correspondencia" in error for error in report["errors"])


def test_missing_dependency_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "_version", lambda _: None)
    report = preflight.audit_candidate_model(tmp_path)
    assert report["compatibility_status"] == "blocked_by_missing_dependency"
    assert report["forward_pass"]["executed"] is False


def test_no_gpu_does_not_create_an_arbitrary_training_batch() -> None:
    config = preflight.recommended_configuration(
        {"cpu_count": 8, "cuda_available": False}
    )
    assert config["local_conservative"]["purpose"].startswith("smoke_only")
    assert config["remote_recommended"]["batch"] == -1
    assert "AutoBatch" in config["remote_recommended"]["batch_reason"]


def test_installed_version_without_candidate_config_is_incompatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "ultralytics"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        preflight, "_version", lambda name: "1.0" if name == "ultralytics" else None
    )
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(init)),
    )
    monkeypatch.setattr(
        preflight.metadata,
        "metadata",
        lambda _: {"License": "AGPL-3.0"},
    )
    report = preflight.audit_candidate_model(tmp_path)
    assert report["compatibility_status"] == "blocked_by_model_incompatibility"
    assert report["supported_segmentation_alternatives"] == []


def test_training_guard_requires_explicit_confirmation() -> None:
    with pytest.raises(preflight.PreflightError, match="CONFIRM_SEGMENTATION"):
        preflight.require_training_confirmation({})
    preflight.require_training_confirmation({"CONFIRM_SEGMENTATION_TRAINING": "1"})


def test_configuration_and_command_are_deterministic() -> None:
    hardware = {"cpu_count": 24, "cuda_available": False}
    first = preflight.recommended_configuration(hardware)
    second = preflight.recommended_configuration(hardware)
    assert first == second
    assert preflight.training_command(first) == preflight.training_command(second)
    assert "CONFIRM_SEGMENTATION_TRAINING" in preflight.training_command(first)


def test_smoke_loader_builds_finite_batch_without_training(tmp_path: Path) -> None:
    root = _write_dataset(
        tmp_path / "dataset", {"train": 4, "val": 2, "test": 2}
    )
    report = preflight.run_loader_smoke_test(root, tmp_path / "previews")
    assert report["passed"]
    assert report["image_batch_shape"] == [8, 3, 640, 640]
    assert report["mask_batch_shape"] == [8, 1, 640, 640]
    assert report["finite"]
    assert report["epochs_run"] == report["optimizer_steps"] == 0
    assert len(list((tmp_path / "previews").iterdir())) == 8


def test_preflight_safety_counters_remain_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "verify_training_locks", lambda _: {"passed": True})
    monkeypatch.setattr(
        preflight,
        "audit_environment",
        lambda _: (
            {"python": "test"},
            {"ultralytics": "installed"},
            {
                "cuda_available": True,
                "cpu_count": 8,
                "nvidia": {"vram_total_bytes": 1, "vram_free_bytes": 1},
            },
        ),
    )
    monkeypatch.setattr(
        preflight, "validate_segmentation_dataset", lambda _: {"passed": True}
    )
    monkeypatch.setattr(
        preflight,
        "audit_candidate_model",
        lambda _: {
            "compatibility_status": "locally_available",
            "weights_download_required": False,
        },
    )
    monkeypatch.setattr(
        preflight,
        "run_loader_smoke_test",
        lambda *_: {
            "passed": True,
            "image_batch_shape": [8, 3, 640, 640],
            "mask_batch_shape": [8, 1, 640, 640],
        },
    )
    summary = preflight.run_segmentation_preflight(
        project_root=tmp_path,
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "outputs",
    )
    assert summary["status"] == "ready_for_training"
    assert all(value in (False, 0) for value in summary["safety"].values())
