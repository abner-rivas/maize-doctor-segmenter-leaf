"""Reusable preprocessing components for leaf regions of interest."""

from src.preprocessing.leaf_processor import (
    FALLBACK_CENTER_CROP,
    FALLBACK_ORIGINAL,
    FALLBACK_REJECT,
    FallbackResult,
    LeafImageProcessor,
    LeafProcessingResult,
    LeafProcessorConfig,
    apply_fallback,
    center_crop_bbox,
)
from src.preprocessing.leaf_roi import (
    BoundingBox,
    InvalidBoundingBoxError,
    LeafDetectionResult,
    bbox_area,
    bbox_area_ratio,
    bbox_height,
    bbox_requires_clipping,
    bbox_width,
    clip_bbox,
    crop_leaf_region,
    expand_bbox,
    validate_bbox,
)
from src.preprocessing.letterbox import LetterboxResult, letterbox_image

__all__ = [
    "BoundingBox",
    "FALLBACK_CENTER_CROP",
    "FALLBACK_ORIGINAL",
    "FALLBACK_REJECT",
    "FallbackResult",
    "InvalidBoundingBoxError",
    "LeafDetectionResult",
    "LeafImageProcessor",
    "LeafProcessingResult",
    "LeafProcessorConfig",
    "LetterboxResult",
    "apply_fallback",
    "bbox_area",
    "bbox_area_ratio",
    "bbox_height",
    "bbox_requires_clipping",
    "bbox_width",
    "center_crop_bbox",
    "clip_bbox",
    "crop_leaf_region",
    "expand_bbox",
    "letterbox_image",
    "validate_bbox",
]
