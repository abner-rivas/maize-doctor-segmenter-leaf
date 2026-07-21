"""Synthetic tests for the read-only remote training preflight."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from src.training.preflight import run_training_preflight


class TrainingPreflightTests(TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        project = root / "project"
        config = project / "config" / "dataset.yaml"
        config.parent.mkdir(parents=True)
        config.write_text(
            "dataset:\n  seed: 42\n  classes: [healthy]\n"
            "baseline:\n  seed: 42\n  classes: [healthy]\n",
            encoding="utf-8",
        )
        for relative in (
            "scripts/pipeline/create_splits.py",
            "scripts/pipeline/train_baselines.py",
        ):
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("VALUE = 1\n", encoding="utf-8")

        dataset = root / "dataset"
        splits = project / "outputs" / "splits" / "seed_42_baseline"
        for index, split in enumerate(("train", "val", "test")):
            relative = Path("clean/healthy/real") / f"{split}.jpg"
            image = dataset / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 6), (30 + index * 50, 80, 20)).save(image)
            path = splits / f"{split}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("image_path", "label", "environment"))
                writer.writeheader()
                writer.writerow(
                    {
                        "image_path": relative.as_posix(),
                        "label": "healthy",
                        "environment": "real",
                    }
                )
        results = project / "outputs" / "baselines"
        output = project / "outputs" / "preflight"
        results.mkdir(parents=True)
        return {
            "project": project,
            "config": config,
            "dataset": dataset,
            "splits": splits,
            "results": results,
            "output": output,
        }

    def _run(self, paths: dict[str, Path], **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "project_root": paths["project"],
            "splits_dir": paths["splits"],
            "dataset_root": paths["dataset"],
            "config_path": paths["config"],
            "models": ["efficientnet_b0"],
            "device": "cpu",
            "check_dataset": True,
            "check_gpu": False,
            "output_dir": paths["output"],
            "results_dir": paths["results"],
        }
        arguments.update(overrides)
        return run_training_preflight(**arguments)  # type: ignore[arg-type]

    def test_valid_dataset_cpu_and_reports(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))

            report = self._run(paths)

            self.assertTrue(report["ready"])
            self.assertEqual(report["dataset"]["total_images"], 3)
            self.assertTrue((paths["output"] / "preflight_report.json").is_file())
            self.assertTrue((paths["output"] / "preflight_report.txt").is_file())
            loaded = json.loads((paths["output"] / "preflight_report.json").read_text())
            self.assertEqual(loaded["safety"]["epochs_run"], 0)
            self.assertEqual(loaded["safety"]["checkpoints_written"], 0)

    def test_missing_dataset_is_blocking(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            report = self._run(paths, dataset_root=Path(directory) / "missing")
            self.assertFalse(report["ready"])
            self.assertTrue(any("clean inexistente" in item for item in report["blockers"]))

    def test_missing_split_is_blocking(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            (paths["splits"] / "test.csv").unlink()
            report = self._run(paths)
            self.assertFalse(report["ready"])
            self.assertTrue(any("Split inexistente" in item for item in report["blockers"]))

    def test_unknown_model_is_blocking(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            report = self._run(paths, models=["unknown_model"])
            self.assertFalse(report["ready"])
            self.assertTrue(any("no registrados" in item for item in report["blockers"]))

    def test_gpu_unavailable_blocks_cuda(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            with patch(
                "src.training.preflight._gpu_information",
                return_value={"available": False, "device_count": 0},
            ):
                report = self._run(paths, device="cuda", check_gpu=True)
            self.assertFalse(report["ready"])
            self.assertTrue(any("CUDA fue solicitado" in item for item in report["blockers"]))

    def test_gpu_query_does_not_block_cpu(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            with patch(
                "src.training.preflight._gpu_information",
                return_value={"available": False, "device_count": 0},
            ):
                report = self._run(paths, device="cpu", check_gpu=True)
            self.assertTrue(report["ready"])
            self.assertTrue(report["warnings"])

    def test_unwritable_results_is_blocking(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            original_mode = paths["results"].stat().st_mode
            paths["results"].chmod(0o555)
            try:
                report = self._run(paths)
            finally:
                paths["results"].chmod(original_mode)
            self.assertFalse(report["ready"])
            self.assertTrue(any("no escribible" in item for item in report["blockers"]))

    def test_invalid_config_is_blocking(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            paths["config"].write_text("dataset: invalid\n", encoding="utf-8")
            report = self._run(paths)
            self.assertFalse(report["ready"])
            self.assertTrue(any("Configuración inválida" in item for item in report["blockers"]))

    def test_preflight_does_not_create_checkpoints_or_training_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            before = set(paths["project"].rglob("*"))
            report = self._run(paths)
            after = set(paths["project"].rglob("*"))
            created = after - before
            self.assertTrue(report["ready"])
            self.assertFalse(any(path.suffix in {".pth", ".pt", ".ckpt"} for path in created))
            self.assertFalse(any("baselines" in path.parts and path.is_file() for path in created))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
