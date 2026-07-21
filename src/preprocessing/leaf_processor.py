"""High-level coordination of bbox validation, crop, fallback, and letterbox."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from PIL import Image

from src.preprocessing.leaf_roi import (
    BoundingBox,
    BoundingBoxInput,
    LeafDetectionResult,
    bbox_requires_clipping,
    crop_leaf_region,
    expand_bbox,
    image_to_rgb,
    normalize_rgb_color,
    validate_bbox,
)
from src.preprocessing.letterbox import (
    LetterboxResult,
    letterbox_image,
    validate_resample,
    validate_target_size,
)

FALLBACK_ORIGINAL = "original"
FALLBACK_CENTER_CROP = "center_crop"
FALLBACK_REJECT = "reject"
SUPPORTED_FALLBACKS = frozenset(
    {FALLBACK_ORIGINAL, FALLBACK_CENTER_CROP, FALLBACK_REJECT}
)


@dataclass(frozen=True)
class LeafProcessorConfig:
    """Validated configuration for in-memory ROI preprocessing.

    ``target_size`` uses the project's ``(height, width)`` convention. The
    center-crop fallback retains the configured proportion of each image axis.
    """

    margin_ratio: float = 0.08
    min_area_ratio: float = 0.15
    target_size: tuple[int, int] = (224, 224)
    padding_value: int | tuple[int, int, int] = 0
    fallback: str = FALLBACK_ORIGINAL
    preserve_aspect_ratio: bool = True
    center_crop_ratio: float = 0.8
    resample: int | Image.Resampling = Image.Resampling.BILINEAR

    def __post_init__(self) -> None:
        _validate_config_ratio(self.margin_ratio, "margin_ratio", allow_zero=True)
        _validate_config_ratio(self.min_area_ratio, "min_area_ratio", allow_zero=True)
        _validate_config_ratio(self.center_crop_ratio, "center_crop_ratio", allow_zero=False)
        if self.fallback not in SUPPORTED_FALLBACKS:
            supported = ", ".join(sorted(SUPPORTED_FALLBACKS))
            raise ValueError(f"fallback desconocido {self.fallback!r}; use: {supported}")
        if not isinstance(self.preserve_aspect_ratio, bool):
            raise ValueError("preserve_aspect_ratio debe ser booleano")
        validate_target_size(self.target_size)
        normalize_rgb_color(self.padding_value)
        validate_resample(self.resample)


@dataclass(frozen=True)
class FallbackResult:
    """Controlled output of one isolated fallback strategy."""

    name: str
    image: Image.Image | None
    bbox: BoundingBox | None
    rejected: bool
    reason: str


@dataclass(frozen=True)
class LeafProcessingResult:
    """Processed image plus geometry and provenance needed for later tracing."""

    processed_image: Image.Image | None
    original_bbox: tuple[object, ...] | None
    clipped_bbox: BoundingBox | None
    expanded_bbox: BoundingBox | None
    fallback_bbox: BoundingBox | None
    detection_result: LeafDetectionResult
    fallback_used: bool
    fallback: str | None
    original_size: tuple[int, int]
    crop_size: tuple[int, int] | None
    processed_size: tuple[int, int] | None
    resized_size: tuple[int, int] | None
    padding: tuple[int, int, int, int] | None
    preserve_aspect_ratio: bool
    warnings: tuple[str, ...]

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-ready processing metadata without embedding image pixels."""
        detection = self.detection_result
        return {
            "original_bbox": _json_safe_values(self.original_bbox),
            "clipped_bbox": list(self.clipped_bbox) if self.clipped_bbox is not None else None,
            "expanded_bbox": list(self.expanded_bbox) if self.expanded_bbox is not None else None,
            "fallback_bbox": list(self.fallback_bbox) if self.fallback_bbox is not None else None,
            "detection_result": {
                "detected": detection.detected,
                "bbox": list(detection.bbox) if detection.bbox is not None else None,
                "confidence": detection.confidence,
                "area_ratio": detection.area_ratio,
                "source": detection.source,
                "fallback_used": detection.fallback_used,
                "reason": detection.reason,
            },
            "fallback_used": self.fallback_used,
            "fallback": self.fallback,
            "original_size": list(self.original_size),
            "crop_size": list(self.crop_size) if self.crop_size is not None else None,
            "processed_size": (
                list(self.processed_size) if self.processed_size is not None else None
            ),
            "resized_size": list(self.resized_size) if self.resized_size is not None else None,
            "padding": list(self.padding) if self.padding is not None else None,
            "preserve_aspect_ratio": self.preserve_aspect_ratio,
            "warnings": list(self.warnings),
        }


def _validate_config_ratio(value: float, name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} debe ser un número")
    converted = float(value)
    lower_bound_valid = converted >= 0.0 if allow_zero else converted > 0.0
    if not math.isfinite(converted) or not lower_bound_valid or converted > 1.0:
        interval = "entre 0 y 1" if allow_zero else "mayor que 0 y menor o igual que 1"
        raise ValueError(f"{name} debe ser {interval}")
    return converted


def _json_safe_values(values: tuple[object, ...] | None) -> list[object] | None:
    if values is None:
        return None
    return [
        str(value) if isinstance(value, float) and not math.isfinite(value) else value
        for value in values
    ]


def _validate_center_crop_ratio(ratio: float) -> float:
    if isinstance(ratio, bool):
        raise ValueError("center_crop_ratio debe ser numérico")
    try:
        converted = float(ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("center_crop_ratio debe ser numérico") from exc
    if not math.isfinite(converted) or not 0.0 < converted <= 1.0:
        raise ValueError("center_crop_ratio debe ser mayor que 0 y menor o igual que 1")
    return converted


def center_crop_bbox(image_width: int, image_height: int, ratio: float = 0.8) -> BoundingBox:
    """Return a centered bbox retaining ``ratio`` of both image dimensions."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("la imagen debe tener dimensiones mayores que cero")
    safe_ratio = _validate_center_crop_ratio(ratio)
    crop_width = max(1, round(image_width * safe_ratio))
    crop_height = max(1, round(image_height * safe_ratio))
    x1 = (image_width - crop_width) // 2
    y1 = (image_height - crop_height) // 2
    return x1, y1, x1 + crop_width, y1 + crop_height


def apply_fallback(
    image: Image.Image,
    fallback: str,
    *,
    center_crop_ratio: float = 0.8,
    transparency_background: int | Sequence[int] = 0,
    reason: str = "no existe una región válida",
) -> FallbackResult:
    """Apply an isolated fallback without resizing for the classifier.

    ``center_crop`` is proportion-based and retains ``center_crop_ratio`` of
    both axes. ``reject`` returns ``image=None`` rather than raising.
    """
    if fallback not in SUPPORTED_FALLBACKS:
        supported = ", ".join(sorted(SUPPORTED_FALLBACKS))
        raise ValueError(f"fallback desconocido {fallback!r}; use: {supported}")
    if not isinstance(image, Image.Image):
        raise TypeError("image debe ser una instancia de PIL.Image.Image")
    if fallback == FALLBACK_REJECT:
        return FallbackResult(fallback, None, None, True, reason)
    if fallback == FALLBACK_ORIGINAL:
        copied = image_to_rgb(image, transparency_background)
        return FallbackResult(fallback, copied, (0, 0, image.width, image.height), False, reason)

    bbox = center_crop_bbox(image.width, image.height, center_crop_ratio)
    cropped = crop_leaf_region(
        image,
        bbox,
        transparency_background=transparency_background,
    )
    return FallbackResult(fallback, cropped, bbox, False, reason)


def _snapshot_bbox(bbox: BoundingBoxInput) -> tuple[object, ...] | None:
    if isinstance(bbox, (str, bytes)):
        return (bbox,)
    try:
        return tuple(bbox)
    except TypeError:
        return None


class LeafImageProcessor:
    """Coordinate validation, margin, crop, fallback, and classifier adaptation."""

    def __init__(self, config: LeafProcessorConfig | None = None) -> None:
        self.config = config or LeafProcessorConfig()

    def _adapt_image(
        self, image: Image.Image
    ) -> tuple[Image.Image, tuple[int, int], tuple[int, int, int, int]]:
        if self.config.preserve_aspect_ratio:
            result: LetterboxResult = letterbox_image(
                image,
                self.config.target_size,
                padding_value=self.config.padding_value,
                resample=self.config.resample,
            )
            return result.image, result.resized_size, result.padding

        target_height, target_width = validate_target_size(self.config.target_size)
        rgb_image = image_to_rgb(image, self.config.padding_value)
        resized = rgb_image.resize(
            (target_width, target_height),
            resample=validate_resample(self.config.resample),
        )
        return resized, resized.size, (0, 0, 0, 0)

    def _fallback_result(
        self,
        image: Image.Image,
        original_bbox: tuple[object, ...] | None,
        detection: LeafDetectionResult,
        prior_warnings: tuple[str, ...] = (),
    ) -> LeafProcessingResult:
        fallback = apply_fallback(
            image,
            self.config.fallback,
            center_crop_ratio=self.config.center_crop_ratio,
            transparency_background=self.config.padding_value,
            reason=detection.reason or "no existe una región válida",
        )
        fallback_detection = replace(detection, fallback_used=True)
        warnings = prior_warnings + (f"fallback {fallback.name}: {fallback.reason}",)
        if fallback.rejected or fallback.image is None:
            return LeafProcessingResult(
                processed_image=None,
                original_bbox=original_bbox,
                clipped_bbox=detection.bbox,
                expanded_bbox=None,
                fallback_bbox=None,
                detection_result=fallback_detection,
                fallback_used=True,
                fallback=fallback.name,
                original_size=image.size,
                crop_size=None,
                processed_size=None,
                resized_size=None,
                padding=None,
                preserve_aspect_ratio=self.config.preserve_aspect_ratio,
                warnings=warnings,
            )

        processed, resized_size, padding = self._adapt_image(fallback.image)
        return LeafProcessingResult(
            processed_image=processed,
            original_bbox=original_bbox,
            clipped_bbox=detection.bbox,
            expanded_bbox=None,
            fallback_bbox=fallback.bbox,
            detection_result=fallback_detection,
            fallback_used=True,
            fallback=fallback.name,
            original_size=image.size,
            crop_size=fallback.image.size,
            processed_size=processed.size,
            resized_size=resized_size,
            padding=padding,
            preserve_aspect_ratio=self.config.preserve_aspect_ratio,
            warnings=warnings,
        )

    def process(
        self,
        image: Image.Image,
        bbox: BoundingBoxInput,
        *,
        confidence: float = 1.0,
        source: str = "manual",
    ) -> LeafProcessingResult:
        """Process one in-memory image and return pixels plus complete tracing metadata."""
        if not isinstance(image, Image.Image):
            raise TypeError("image debe ser una instancia de PIL.Image.Image")
        original_bbox = _snapshot_bbox(bbox)
        validation_bbox = original_bbox if original_bbox is not None else bbox
        detection = validate_bbox(
            image.width,
            image.height,
            validation_bbox,
            self.config.min_area_ratio,
            confidence=confidence,
            source=source,
        )
        warnings: tuple[str, ...] = ()
        if detection.bbox is not None and bbox_requires_clipping(
            validation_bbox,
            image.width,
            image.height,
        ):
            warnings = ("bbox limitado a los bordes de la imagen",)
        if not detection.detected or detection.bbox is None:
            return self._fallback_result(image, original_bbox, detection, warnings)

        expanded = expand_bbox(
            detection.bbox,
            image.width,
            image.height,
            self.config.margin_ratio,
        )
        cropped = crop_leaf_region(
            image,
            expanded,
            transparency_background=self.config.padding_value,
        )
        processed, resized_size, padding = self._adapt_image(cropped)
        return LeafProcessingResult(
            processed_image=processed,
            original_bbox=original_bbox,
            clipped_bbox=detection.bbox,
            expanded_bbox=expanded,
            fallback_bbox=None,
            detection_result=detection,
            fallback_used=False,
            fallback=None,
            original_size=image.size,
            crop_size=cropped.size,
            processed_size=processed.size,
            resized_size=resized_size,
            padding=padding,
            preserve_aspect_ratio=self.config.preserve_aspect_ratio,
            warnings=warnings,
        )
