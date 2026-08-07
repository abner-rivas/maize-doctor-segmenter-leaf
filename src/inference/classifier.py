"""Single source of truth for classifier probabilities at inference time."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from PIL import Image


@dataclass(frozen=True)
class RankedClassPrediction:
    """One class and its probability from the classifier softmax."""

    class_name: str
    class_index: int
    probability: float

    def to_metadata(self) -> dict[str, object]:
        return {
            "class": self.class_name,
            "class_index": self.class_index,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class ClassificationPrediction:
    """Classifier decision produced from its existing softmax probabilities."""

    class_name: str
    class_index: int
    confidence: float
    top_k: tuple[RankedClassPrediction, ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "class": self.class_name,
            "class_index": self.class_index,
            "confidence": self.confidence,
            "top_k": [prediction.to_metadata() for prediction in self.top_k],
        }


def classify_image(
    model: torch.nn.Module,
    image: Image.Image,
    *,
    transform: Callable[[Image.Image], torch.Tensor],
    idx_to_class: Mapping[int, str],
    device: torch.device,
    top_k: int = 3,
) -> ClassificationPrediction:
    """Apply the historical transform/model/softmax flow to exactly one image."""
    if not isinstance(image, Image.Image):
        raise TypeError("image debe ser una instancia de PIL.Image.Image")
    if not idx_to_class:
        raise ValueError("idx_to_class no puede estar vacío")
    if top_k <= 0:
        raise ValueError("top_k debe ser mayor que cero")

    transformed = transform(image)
    if not isinstance(transformed, torch.Tensor):
        raise TypeError("el transform de inferencia debe devolver torch.Tensor")

    with torch.inference_mode():
        logits = model(transformed.unsqueeze(0).to(device))
        if logits.ndim != 2 or logits.shape != (1, len(idx_to_class)):
            raise ValueError(
                f"salida del clasificador {tuple(logits.shape)} incompatible con "
                f"{len(idx_to_class)} clases"
            )
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    resolved_top_k = min(top_k, len(idx_to_class))
    values, indices = torch.topk(probabilities, k=resolved_top_k)
    ranked: list[RankedClassPrediction] = []
    for probability, index in zip(values.tolist(), indices.tolist(), strict=True):
        if index not in idx_to_class:
            raise ValueError(f"índice de clase {index} ausente en idx_to_class")
        ranked.append(
            RankedClassPrediction(
                class_name=idx_to_class[index],
                class_index=index,
                probability=float(probability),
            )
        )
    winner = ranked[0]
    return ClassificationPrediction(
        class_name=winner.class_name,
        class_index=winner.class_index,
        confidence=winner.probability,
        top_k=tuple(ranked),
    )
