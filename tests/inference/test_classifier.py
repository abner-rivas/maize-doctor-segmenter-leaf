from __future__ import annotations

import math
from unittest import TestCase

import torch
from PIL import Image

from src.inference.classifier import classify_image


class _FixedClassifier(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[1.0, 3.0]], dtype=torch.float32).repeat(inputs.shape[0], 1)


class ClassifierInferenceTests(TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (8, 8), "green")

    def test_confidence_is_the_existing_classifier_softmax(self) -> None:
        prediction = classify_image(
            _FixedClassifier(),
            self.image,
            transform=lambda _: torch.zeros((3, 8, 8)),
            idx_to_class={0: "healthy", 1: "common_rust"},
            device=torch.device("cpu"),
            top_k=2,
        )

        expected = math.exp(3.0) / (math.exp(1.0) + math.exp(3.0))
        self.assertEqual(prediction.class_name, "common_rust")
        self.assertAlmostEqual(prediction.confidence, expected, places=6)
        self.assertAlmostEqual(prediction.top_k[0].probability, prediction.confidence)
        self.assertEqual(prediction.to_metadata()["confidence"], prediction.confidence)

    def test_invalid_classifier_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "incompatible"):
            classify_image(
                torch.nn.Identity(),
                self.image,
                transform=lambda _: torch.zeros((3, 8, 8)),
                idx_to_class={0: "healthy", 1: "common_rust"},
                device=torch.device("cpu"),
            )

    def test_top_k_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mayor que cero"):
            classify_image(
                _FixedClassifier(),
                self.image,
                transform=lambda _: torch.zeros((3, 8, 8)),
                idx_to_class={0: "healthy", 1: "common_rust"},
                device=torch.device("cpu"),
                top_k=0,
            )
