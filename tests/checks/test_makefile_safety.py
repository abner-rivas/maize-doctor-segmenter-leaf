"""Dry-run-only tests for Makefile interpreter flexibility and training guards."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import TestCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MakefileSafetyTests(TestCase):
    def _dry_run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["make", "-n", *arguments],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_python_can_be_overridden(self) -> None:
        result = self._dry_run("PYTHON=/opt/remote/bin/python", "splits-baseline")
        self.assertEqual(result.returncode, 0)
        self.assertIn("/opt/remote/bin/python scripts/pipeline/create_splits.py", result.stdout)

    def test_default_uses_python_from_active_environment(self) -> None:
        result = self._dry_run("splits-baseline")
        self.assertEqual(result.returncode, 0)
        self.assertIn("python scripts/pipeline/create_splits.py", result.stdout)

    def test_training_is_blocked_without_confirmation(self) -> None:
        for target in ("train", "train-baselines"):
            result = self._dry_run(target)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CONFIRM_TRAINING=1", result.stderr)

    def test_training_can_only_be_rendered_with_confirmation(self) -> None:
        result = self._dry_run("train-baselines", "CONFIRM_TRAINING=1")
        self.assertEqual(result.returncode, 0)
        self.assertIn("scripts/pipeline/train_baselines.py", result.stdout)

    def test_validation_targets_are_not_blocked(self) -> None:
        for target in ("smoke-loader", "audit-dataset", "validate-splits", "training-preflight"):
            result = self._dry_run(target)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("CONFIRM_TRAINING", result.stderr)
