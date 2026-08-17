"""Auditable bounding-box processing used by segmentation dataset tooling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from PIL import Image

from src.preprocessing.leaf_roi import (
    BoundingBox,
    LeafDetectionResult,
    bbox_requires_clipping,
    crop_leaf_region,
    expand_bbox,
    image_to_rgb,
    normalize_rgb_color,
    validate_bbox,
)
from src.preprocessing.letterbox import letterbox_image, validate_resample, validate_target_size

FALLBACK_ORIGINAL = "original"
FALLBACK_CENTER_CROP = "center_crop"
FALLBACK_REJECT = "reject"
SUPPORTED_FALLBACKS = frozenset(
    {FALLBACK_ORIGINAL, FALLBACK_CENTER_CROP, FALLBACK_REJECT}
)
ROI_PROCESSOR_VERSION = "1.0.0"


def _ratio(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} no puede ser booleano")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} debe estar entre 0 y 1")
    return converted


@dataclass(frozen=True)
class LeafProcessorConfig:
    margin_ratio: float = 0.08
    min_area_ratio: float = 0.15
    target_size: tuple[int, int] = (640, 640)
    padding_value: int | tuple[int, int, int] = 0
    fallback: str = FALLBACK_ORIGINAL
    preserve_aspect_ratio: bool = True
    center_crop_ratio: float = 0.8
    resample: int | Image.Resampling = Image.Resampling.BILINEAR

    def __post_init__(self) -> None:
        _ratio(self.margin_ratio, "margin_ratio")
        _ratio(self.min_area_ratio, "min_area_ratio")
        if _ratio(self.center_crop_ratio, "center_crop_ratio") <= 0.0:
            raise ValueError("center_crop_ratio debe ser mayor que cero")
        validate_target_size(self.target_size)
        normalize_rgb_color(self.padding_value)
        validate_resample(self.resample)
        if self.fallback not in SUPPORTED_FALLBACKS:
            raise ValueError(f"fallback desconocido: {self.fallback!r}")


@dataclass(frozen=True)
class LeafProcessingResult:
    original_size: tuple[int, int]
    processed_image: Image.Image | None
    detection_result: LeafDetectionResult
    clipped_bbox: BoundingBox | None
    expanded_bbox: BoundingBox | None
    fallback_bbox: BoundingBox | None
    crop_size: tuple[int, int] | None
    processed_size: tuple[int, int] | None
    fallback_used: bool
    warnings: tuple[str, ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "processor_version": ROI_PROCESSOR_VERSION,
            "original_size": list(self.original_size),
            "processed_size": list(self.processed_size) if self.processed_size else None,
            "clipped_bbox": list(self.clipped_bbox) if self.clipped_bbox else None,
            "expanded_bbox": list(self.expanded_bbox) if self.expanded_bbox else None,
            "fallback_bbox": list(self.fallback_bbox) if self.fallback_bbox else None,
            "crop_size": list(self.crop_size) if self.crop_size else None,
            "fallback_used": self.fallback_used,
            "detection": {
                "detected": self.detection_result.detected,
                "confidence": self.detection_result.confidence,
                "area_ratio": self.detection_result.area_ratio,
                "source": self.detection_result.source,
                "reason": self.detection_result.reason,
            },
            "warnings": list(self.warnings),
        }


def center_crop_bbox(image_size: tuple[int, int], ratio: float) -> BoundingBox:
    width, height = image_size
    safe_ratio = _ratio(ratio, "center_crop_ratio")
    if safe_ratio <= 0.0:
        raise ValueError("center_crop_ratio debe ser mayor que cero")
    crop_width = max(1, round(width * safe_ratio))
    crop_height = max(1, round(height * safe_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


class LeafImageProcessor:
    """Apply validation, margin, crop and optional letterbox to a manual ROI."""

    def __init__(self, config: LeafProcessorConfig | None = None) -> None:
        self.config = config or LeafProcessorConfig()

    def _resize(self, image: Image.Image) -> Image.Image:
        if self.config.preserve_aspect_ratio:
            return letterbox_image(
                image,
                self.config.target_size,
                padding_value=self.config.padding_value,
                resample=self.config.resample,
            ).image
        height, width = validate_target_size(self.config.target_size)
        return image.resize((width, height), resample=validate_resample(self.config.resample))

    def process(
        self,
        image: Image.Image,
        bbox: Sequence[int | float | str],
        *,
        confidence: float = 1.0,
        source: str = "manual",
    ) -> LeafProcessingResult:
        original = image_to_rgb(image, self.config.padding_value)
        detection = validate_bbox(
            original.width,
            original.height,
            bbox,
            self.config.min_area_ratio,
            confidence=confidence,
            source=source,
        )
        warnings: list[str] = []
        clipped = detection.bbox
        if clipped is not None:
            try:
                if bbox_requires_clipping(bbox, original.width, original.height):
                    warnings.append("bbox limitado a los bordes de la imagen")
            except ValueError:
                pass

        if detection.detected and clipped is not None:
            expanded = expand_bbox(
                clipped,
                original.width,
                original.height,
                self.config.margin_ratio,
            )
            crop = crop_leaf_region(original, expanded)
            processed = self._resize(crop)
            return LeafProcessingResult(
                original.size,
                processed,
                detection,
                clipped,
                expanded,
                None,
                crop.size,
                processed.size,
                False,
                tuple(warnings),
            )

        warnings.append(detection.reason or "ROI no válida")
        if self.config.fallback == FALLBACK_REJECT:
            return LeafProcessingResult(
                original.size,
                None,
                detection,
                clipped,
                None,
                None,
                None,
                None,
                True,
                tuple(warnings),
            )
        fallback_bbox = (
            center_crop_bbox(original.size, self.config.center_crop_ratio)
            if self.config.fallback == FALLBACK_CENTER_CROP
            else (0, 0, original.width, original.height)
        )
        crop = crop_leaf_region(original, fallback_bbox)
        processed = self._resize(crop)
        return LeafProcessingResult(
            original.size,
            processed,
            detection,
            clipped,
            None,
            fallback_bbox,
            crop.size,
            processed.size,
            True,
            tuple(warnings),
        )
