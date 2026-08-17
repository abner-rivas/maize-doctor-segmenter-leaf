"""Reusable geometry and mask-output components for leaf segmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from src.preprocessing.roi_processor import (
    FALLBACK_CENTER_CROP,
    LeafImageProcessor,
    LeafProcessingResult,
    LeafProcessorConfig,
)

if TYPE_CHECKING:
    from src.preprocessing.segmented_leaf_processor import (
        BBOX_CROP,
        CROP_MASK_BLACK,
        CROP_MASK_LETTERBOX,
        FALLBACK_ORIGINAL,
        FALLBACK_REJECT,
        MASK_BLACK,
        LeafMaskProcessorConfig,
        SegmentedLeafProcessingResult,
        SegmentedLeafProcessor,
        mask_processor_config_from_mapping,
    )

_SEGMENTED_EXPORTS = {
    "BBOX_CROP",
    "CROP_MASK_BLACK",
    "CROP_MASK_LETTERBOX",
    "FALLBACK_ORIGINAL",
    "FALLBACK_REJECT",
    "MASK_BLACK",
    "LeafMaskProcessorConfig",
    "SegmentedLeafProcessingResult",
    "SegmentedLeafProcessor",
    "mask_processor_config_from_mapping",
}


def __getattr__(name: str) -> Any:
    """Load segmentation-aware exports lazily to avoid an import cycle."""
    if name not in _SEGMENTED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from src.preprocessing import segmented_leaf_processor

    value = getattr(segmented_leaf_processor, name)
    globals()[name] = value
    return value

__all__ = [
    "BBOX_CROP",
    "BoundingBox",
    "CROP_MASK_BLACK",
    "CROP_MASK_LETTERBOX",
    "FALLBACK_ORIGINAL",
    "FALLBACK_REJECT",
    "FALLBACK_CENTER_CROP",
    "InvalidBoundingBoxError",
    "LeafDetectionResult",
    "LeafMaskProcessorConfig",
    "LeafImageProcessor",
    "LeafProcessingResult",
    "LeafProcessorConfig",
    "LetterboxResult",
    "MASK_BLACK",
    "SegmentedLeafProcessingResult",
    "SegmentedLeafProcessor",
    "bbox_area",
    "bbox_area_ratio",
    "bbox_height",
    "bbox_requires_clipping",
    "bbox_width",
    "clip_bbox",
    "crop_leaf_region",
    "expand_bbox",
    "letterbox_image",
    "mask_processor_config_from_mapping",
    "validate_bbox",
]
