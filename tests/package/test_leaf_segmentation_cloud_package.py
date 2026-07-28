"""Local-only safety tests for the cloud segmentation package."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from scripts.package import build_leaf_segmentation_cloud_package as package
from src.training.segmentation_preflight import verify_cloud_training_payload

ROOT = Path(__file__).resolve().parents[2]
CLOUD = ROOT / "cloud_training"


def test_cloud_payload_fingerprints_are_valid_without_all_tree() -> None:
    report = verify_cloud_training_payload(
        ROOT / "data" / "leaf_detection" / "detector_dataset"
    )
    assert report["passed"]
    assert report["verified_manifest_rows"] == 1155


def test_every_shell_script_is_strict_and_syntax_valid() -> None:
    for path in sorted(CLOUD.glob("*.sh")):
        source = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in source
        result = subprocess.run(
            ["bash", "-n", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_shell_training_guards_precede_all_script_actions() -> None:
    cases = {
        "smoke_train.sh": "CONFIRM_SEGMENTATION_SMOKE_TRAINING",
        "train.sh": "CONFIRM_SEGMENTATION_TRAINING",
        "resume_train.sh": "CONFIRM_SEGMENTATION_TRAINING",
    }
    for script, confirmation in cases.items():
        source = (CLOUD / script).read_text(encoding="utf-8")
        guard = source.index(confirmation)
        first_action = source.index("source ")
        assert guard < first_action
        assert "exit 2" in source[guard:first_action]


def test_cloud_yaml_configs_are_deterministic_and_safe() -> None:
    smoke = yaml.safe_load((CLOUD / "configs/smoke_yolo26n_seg.yaml").read_text())
    train = yaml.safe_load((CLOUD / "configs/train_yolo26n_seg.yaml").read_text())
    validate = yaml.safe_load(
        (CLOUD / "configs/validate_yolo26n_seg.yaml").read_text()
    )
    assert smoke["epochs"] == 1 and smoke["batch"] == 2
    assert train["epochs"] == 150 and train["batch"] == -1
    assert train["seed"] == smoke["seed"] == validate["seed"] == 42
    assert train["task"] == smoke["task"] == validate["task"] == "segment"
    assert "pilot" not in str(smoke) + str(train) + str(validate)


def test_allow_list_excludes_protected_and_historical_trees() -> None:
    relatives = {
        path.relative_to(ROOT).as_posix() for path in package.collect_payload(ROOT)
    }
    assert any(
        path.startswith("data/leaf_detection/detector_dataset/images/train/")
        for path in relatives
    )
    assert not any("/all/" in f"/{path}/" for path in relatives)
    assert not any("external_sources" in path for path in relatives)
    assert not any(path.startswith("data/leaf_detection/pilot/") for path in relatives)
    assert not any(path.startswith("outputs/") for path in relatives)
    assert not any(".venv" in path or "__pycache__" in path for path in relatives)


def test_tar_metadata_is_deterministic() -> None:
    first = package.tar_info(Path("payload/file.sh"), 123, True)
    second = package.tar_info(Path("payload/file.sh"), 123, True)
    assert (first.name, first.size, first.mtime, first.uid, first.gid, first.mode) == (
        second.name,
        second.size,
        second.mtime,
        second.uid,
        second.gid,
        second.mode,
    )
    assert first.mtime == first.uid == first.gid == 0


def test_pilot_transport_manifest_is_separate() -> None:
    manifest = package.pilot_manifest(ROOT)
    assert manifest["included_in_training_package"] is False
    assert manifest["purpose"] == "external_held_out_evaluation_transport_only"
    assert manifest["file_count"] == 211
