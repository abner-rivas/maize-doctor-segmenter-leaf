"""Tests for the thin Makefile helper without GPU or training."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/package/leaf_segmentation_make.py"


def test_package_verify_rejects_bad_sha_before_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "doctor_maiz_leaf_segmentation_cloud_bad.tar.gz"
    archive.write_bytes(b"not-a-package")
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{'0' * 64}  {archive.name}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "package-verify",
            "--dataset",
            "data/leaf_detection/detector_dataset",
            "--output",
            "outputs/leaf_detection",
            "--cloud-dir",
            "cloud_training",
            "--package-dir",
            str(tmp_path),
            "--package",
            str(archive),
            "--model",
            "yolo26n-seg.pt",
            "--device",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "SHA-256 inválido" in result.stderr
    assert not list(tmp_path.glob(".tmp_leaf_cloud_verify_*"))
