"""Tests for deterministic remote-training package manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.training.package_manifest import write_training_package_manifest


class TrainingPackageManifestTests(TestCase):
    def _fixture(self, root: Path) -> Path:
        project = root / "doctor-maiz"
        files = {
            "config/dataset.yaml": "dataset:\n  classes: [healthy]\n",
            "src/module.py": "VALUE = 1\n",
            "scripts/check.py": "print('ok')\n",
            "outputs/splits/seed_42_baseline/train.csv": "image_path,label,environment\n",
            "outputs/splits/seed_42_baseline/val.csv": "image_path,label,environment\n",
            "outputs/splits/seed_42_baseline/test.csv": "image_path,label,environment\n",
            "pyproject.toml": "[project]\nname='doctor-maiz'\n",
            "Makefile": "PYTHON ?= python\n",
            ".env.example": "DATASET_ROOT=/data\n",
            "src/__pycache__/module.pyc": "cache",
            "outputs/aborted_runs/run/best.pth": "checkpoint",
        }
        for relative, content in files.items():
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return project

    def test_required_entries_hashes_and_exclusions(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = self._fixture(root)
            manifest = write_training_package_manifest(project, root / "manifest.json")
            entries = {entry["path"]: entry for entry in manifest["entries"]}

            for required in manifest["required_inputs"]:
                self.assertIn(required, entries)
                self.assertTrue(entries[required]["required"])
            source = project / "src/module.py"
            self.assertEqual(
                entries["src/module.py"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest()
            )
            self.assertFalse(any("__pycache__" in path for path in entries))
            self.assertFalse(any("aborted_runs" in path for path in entries))
            self.assertFalse(any(path.endswith(".pth") for path in entries))
            self.assertFalse(manifest["dataset_included"])

    def test_manifest_is_reproducible(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = self._fixture(root)
            first_path = root / "first.json"
            second_path = root / "second.json"

            first = write_training_package_manifest(project, first_path)
            second = write_training_package_manifest(project, second_path)

            self.assertEqual(first, second)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

    def test_missing_required_input_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            project = self._fixture(root)
            (project / ".env.example").unlink()
            with self.assertRaisesRegex(FileNotFoundError, ".env.example"):
                write_training_package_manifest(project, root / "manifest.json")
