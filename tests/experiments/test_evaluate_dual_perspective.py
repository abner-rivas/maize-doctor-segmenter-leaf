from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

from PIL import Image

from scripts.experiments.evaluate_dual_perspective import read_experiment_manifest


class DualPerspectiveManifestTests(TestCase):
    def test_manifest_resolves_images_and_optional_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "leaf.jpg"
            Image.new("RGB", (8, 8), "green").save(image_path)
            manifest = root / "cases.csv"
            manifest.write_text(
                "image_path,ground_truth,environment,multi_leaf,severe_fall_armyworm\n"
                "leaf.jpg,healthy,lab,true,0\n",
                encoding="utf-8",
            )

            cases = read_experiment_manifest(manifest)

            self.assertEqual(cases[0].image_path, image_path.resolve())
            self.assertEqual(cases[0].ground_truth, "healthy")
            self.assertTrue(cases[0].multi_leaf)
            self.assertFalse(cases[0].severe_fall_armyworm)

    def test_invalid_boolean_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8), "green").save(root / "leaf.jpg")
            manifest = root / "cases.csv"
            manifest.write_text(
                "image_path,ground_truth,environment,multi_leaf\nleaf.jpg,healthy,lab,perhaps\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "booleano"):
                read_experiment_manifest(manifest)

    def test_empty_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "cases.csv"
            manifest.write_text(
                "image_path,ground_truth,environment\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "vacío"):
                read_experiment_manifest(manifest)

    def test_runner_contains_no_training_operations(self) -> None:
        source = (
            Path(__file__).parents[2] / "scripts" / "experiments" / "evaluate_dual_perspective.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.save(", source)
        self.assertNotIn("Optimizer", source)
