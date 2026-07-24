"""Tests for standardized processing profiles and paired diagnostic outcomes."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
from PIL import Image

from scripts.experiments.compare_full_vs_manual_roi import (
    COMPARISON_COLUMNS,
    CheckpointCompatibilityError,
    build_summary,
    compare_predictions,
    load_compatible_checkpoint,
    metric_inclusion,
    resolve_device,
)
from src.data.leaf_pilot import sha256_file, write_csv_rows
from src.preprocessing.leaf_processor import (
    BASELINE_FULL,
    BASELINE_ROI,
    LeafImageProcessor,
    LeafProcessingProfile,
    LeafProcessorConfig,
)


def _config() -> dict[str, object]:
    return {
        "dataset": {
            "classes": ["healthy", "common_rust"],
            "target_size": [16, 16],
            "seed": 42,
        }
    }


def _prediction(
    pilot_id: str,
    predicted: str,
    *,
    truth: str = "healthy",
    confidence: float = 0.7,
    included: bool = True,
) -> dict[str, object]:
    return {
        "pilot_id": pilot_id,
        "image_path": f"images/{pilot_id}.jpg",
        "true_label": truth,
        "annotation_status": "annotated" if included else "ambiguous",
        "included_in_metrics": included,
        "exclusion_reason": "" if included else "annotation_status=ambiguous",
        "predicted_label": predicted,
        "confidence": confidence,
        "correct": predicted == truth,
        "loss": 0.4,
        "roi_area_ratio": 0.5,
        "roi_source": "manual",
        "fallback_used": False,
    }


class ProcessingProfileTests(TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (80, 40), (20, 100, 30))
        self.processor = LeafImageProcessor(
            LeafProcessorConfig(
                min_area_ratio=0.0,
                margin_ratio=0.0,
                target_size=(20, 20),
            )
        )

    def test_baseline_full_preserves_historical_pre_transform_image(self) -> None:
        profile = LeafProcessingProfile(BASELINE_FULL, processor=self.processor)
        historical = lambda image: (image.size, image.getpixel((0, 0)))  # noqa: E731

        result = profile.apply(
            self.image,
            stage="inference",
            normalization=historical,
        )

        self.assertIs(result.prepared.image, self.image)
        self.assertEqual(result.output, historical(self.image))
        self.assertEqual(result.prepared.metadata["processing_profile"], BASELINE_FULL)

    def test_baseline_roi_delegates_to_leaf_image_processor(self) -> None:
        wrapped_process = Mock(wraps=self.processor.process)
        self.processor.process = wrapped_process  # type: ignore[method-assign]
        profile = LeafProcessingProfile(BASELINE_ROI, processor=self.processor)

        result = profile.prepare(self.image, (20, 0, 60, 40))

        wrapped_process.assert_called_once()
        self.assertEqual(result.image.size, (20, 20))

    def test_roi_happens_before_augmentation_and_normalization(self) -> None:
        events: list[tuple[str, tuple[int, int]]] = []
        profile = LeafProcessingProfile(BASELINE_ROI, processor=self.processor)

        def augmentation(image: Image.Image) -> Image.Image:
            events.append(("augmentation", image.size))
            return image

        def normalization(image: Image.Image) -> tuple[int, int]:
            events.append(("normalization", image.size))
            return image.size

        result = profile.apply(
            self.image,
            (20, 0, 60, 40),
            stage="train",
            augmentation=augmentation,
            normalization=normalization,
        )

        self.assertEqual(events, [("augmentation", (20, 20)), ("normalization", (20, 20))])
        self.assertEqual(result.output, (20, 20))

    def test_random_augmentation_is_rejected_outside_train(self) -> None:
        profile = LeafProcessingProfile(BASELINE_ROI, processor=self.processor)

        with self.assertRaisesRegex(ValueError, "sólo están permitidas en train"):
            profile.apply(
                self.image,
                (20, 0, 60, 40),
                stage="test",
                augmentation=lambda image: image,
            )

    def test_manifest_hash_is_preserved_in_every_roi_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "roi_manifest.csv"
            manifest.write_text("pilot_id\nimage_0001\n", encoding="utf-8")
            profile = LeafProcessingProfile(
                BASELINE_ROI,
                processor=self.processor,
                roi_manifest_path=manifest,
            )

            result = profile.prepare(self.image, (20, 0, 60, 40))

            self.assertEqual(result.metadata["roi_manifest_sha256"], sha256_file(manifest))
            self.assertEqual(result.metadata["roi_manifest_path"], str(manifest.resolve()))


class ComparisonTests(TestCase):
    def test_image_0021_ambiguous_is_retained_but_excluded(self) -> None:
        included, reason = metric_inclusion(
            {
                "pilot_id": "image_0021",
                "annotation_status": "ambiguous",
                "roi_area_ratio": "",
                "notes": "area_ratio 0.092799 es menor que min_area_ratio 0.150000",
            }
        )

        self.assertFalse(included)
        self.assertEqual(reason, "annotation_status=ambiguous; roi_area_ratio=0.092799")

    def test_image_0021_does_not_enter_primary_metrics(self) -> None:
        full = [
            _prediction("image_0001", "healthy"),
            _prediction("image_0021", "common_rust", included=False),
        ]
        roi = [
            _prediction("image_0001", "healthy"),
            _prediction("image_0021", "healthy", included=False),
        ]
        comparisons = compare_predictions(full, roi)

        summary = build_summary(full, roi, comparisons, ["healthy", "common_rust"])

        self.assertEqual(summary["total_images"], 2)
        self.assertEqual(summary["included_images"], 1)
        self.assertEqual(summary["excluded_images"], 1)
        self.assertEqual(summary["baseline_full"]["images"], 1)
        self.assertEqual(summary["baseline_roi"]["images"], 1)

    def test_same_prediction_is_not_marked_changed(self) -> None:
        full = [_prediction("image_0001", "healthy")]
        roi = [_prediction("image_0001", "healthy", confidence=0.8)]

        comparison = compare_predictions(full, roi)[0]

        self.assertFalse(comparison["prediction_changed"])
        self.assertAlmostEqual(comparison["confidence_delta"], 0.1)

    def test_changed_prediction_is_marked(self) -> None:
        full = [_prediction("image_0001", "healthy")]
        roi = [_prediction("image_0001", "common_rust")]

        self.assertTrue(compare_predictions(full, roi)[0]["prediction_changed"])

    def test_corrected_and_worsened_cases_are_counted(self) -> None:
        full = [
            _prediction("image_0001", "common_rust"),
            _prediction("image_0002", "healthy"),
        ]
        roi = [
            _prediction("image_0001", "healthy"),
            _prediction("image_0002", "common_rust"),
        ]
        comparisons = compare_predictions(full, roi)

        summary = build_summary(full, roi, comparisons, ["healthy", "common_rust"])

        self.assertEqual(summary["comparison"]["corrected_errors"], 1)
        self.assertEqual(summary["comparison"]["lost_correct_predictions"], 1)
        self.assertEqual(summary["comparison"]["changed_predictions"], 2)

    def test_csv_generation_is_deterministic(self) -> None:
        rows = compare_predictions(
            [_prediction("image_0001", "healthy")],
            [_prediction("image_0001", "common_rust")],
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            write_csv_rows(first, rows, COMPARISON_COLUMNS)
            write_csv_rows(second, rows, COMPARISON_COLUMNS)

            self.assertEqual(first.read_bytes(), second.read_bytes())


class CheckpointSafetyTests(TestCase):
    def test_missing_checkpoint_fails_before_model_creation(self) -> None:
        missing = Path("/tmp/checkpoint-that-does-not-exist.pt")

        with self.assertRaises(FileNotFoundError):
            load_compatible_checkpoint(
                missing,
                "efficientnet_b0",
                _config(),
                torch.device("cpu"),
            )

    def test_incompatible_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"synthetic incompatible state")
            model = torch.nn.Linear(4, 2)
            with (
                patch(
                    "scripts.experiments.compare_full_vs_manual_roi.MODEL_REGISTRY.build",
                    return_value=model,
                ),
                patch(
                    "scripts.experiments.compare_full_vs_manual_roi.torch.load",
                    return_value={"weight": torch.ones(3, 3)},
                ),
            ):
                with self.assertRaises(CheckpointCompatibilityError):
                    load_compatible_checkpoint(
                        checkpoint,
                        "efficientnet_b0",
                        _config(),
                        torch.device("cpu"),
                    )

    def test_cpu_device_is_explicit(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_experiment_source_contains_no_training_operations(self) -> None:
        source = (
            Path(__file__).parents[2]
            / "scripts"
            / "experiments"
            / "compare_full_vs_manual_roi.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.save(", source)
        self.assertNotIn("Optimizer", source)
