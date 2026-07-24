"""Tests for portable run metadata after organizing project data."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.training.common import load_run_metadata


class RunMetadataPortabilityTests(TestCase):
    def test_missing_remote_splits_path_uses_local_data_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            fallback = root / "data" / "splits" / "seed_42_baseline"
            run_dir.mkdir()
            fallback.mkdir(parents=True)
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "splits_dir": "/outputs/splits/seed_42_baseline",
                        "class_to_idx": {"healthy": 0},
                        "image_size": [224, 224],
                    }
                ),
                encoding="utf-8",
            )

            splits_dir, class_to_idx, idx_to_class, target_size = load_run_metadata(
                run_dir,
                fallback,
                ["healthy"],
                (128, 128),
            )

            self.assertEqual(splits_dir, fallback)
            self.assertEqual(class_to_idx, {"healthy": 0})
            self.assertEqual(idx_to_class, {0: "healthy"})
            self.assertEqual(target_size, (224, 224))
