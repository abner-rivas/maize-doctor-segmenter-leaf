"""Tests for conversion from Ultralytics-like results to original-resolution masks."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from PIL import Image

from src.segmentation.leaf_segmenter import (
    LeafSegmentationError,
    UltralyticsLeafSegmenter,
    instances_from_ultralytics_result,
    rasterize_instance_polygon,
)


class _TensorLike:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def cpu(self) -> "_TensorLike":
        return self

    def tolist(self) -> list[float]:
        return self.values


@dataclass
class _Boxes:
    conf: _TensorLike
    cls: _TensorLike


@dataclass
class _Masks:
    xy: list[np.ndarray]


@dataclass
class _Result:
    masks: _Masks | None
    boxes: _Boxes | None
    orig_shape: tuple[int, int] = (50, 100)


class _Model:
    def __init__(self, result: _Result) -> None:
        self.result = result
        self.kwargs: dict[str, object] | None = None

    def predict(self, **kwargs: object) -> list[_Result]:
        self.kwargs = kwargs
        return [self.result]


class UltralyticsResultConversionTests(TestCase):
    def test_polygon_is_rasterized_at_original_resolution(self) -> None:
        polygon = np.asarray([[10, 5], [90, 5], [90, 45], [10, 45]])
        result = _Result(
            masks=_Masks([polygon]),
            boxes=_Boxes(_TensorLike([0.91]), _TensorLike([0.0])),
        )

        instances = instances_from_ultralytics_result(result, (100, 50))

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].mask.size, (100, 50))
        self.assertEqual(instances[0].mask.mode, "L")
        self.assertEqual(instances[0].bbox, (10, 5, 91, 46))
        self.assertAlmostEqual(instances[0].confidence, 0.91)
        self.assertEqual(instances[0].class_id, 0)

    def test_no_masks_is_a_controlled_empty_detection(self) -> None:
        self.assertEqual(
            instances_from_ultralytics_result(_Result(None, None), (80, 60)),
            (),
        )

    def test_mask_confidence_count_mismatch_is_rejected(self) -> None:
        polygon = np.asarray([[1, 1], [5, 1], [5, 5]])
        result = _Result(
            masks=_Masks([polygon]),
            boxes=_Boxes(_TensorLike([]), _TensorLike([])),
        )

        with self.assertRaisesRegex(LeafSegmentationError, "cantidades distintas"):
            instances_from_ultralytics_result(result, (10, 10))

    def test_out_of_bounds_polygon_is_not_silently_clipped(self) -> None:
        polygon = np.asarray([[-1, 1], [5, 1], [5, 5]])

        with self.assertRaisesRegex(LeafSegmentationError, "fuera"):
            rasterize_instance_polygon(polygon, (10, 10))

    def test_non_finite_polygon_is_rejected(self) -> None:
        polygon = np.asarray([[1, 1], [np.nan, 1], [5, 5]])

        with self.assertRaisesRegex(LeafSegmentationError, "finitos"):
            rasterize_instance_polygon(polygon, (10, 10))

    def test_wrapper_accepts_an_image_and_returns_mask_without_saving(self) -> None:
        polygon = np.asarray([[10, 5], [90, 5], [90, 45], [10, 45]])
        model = _Model(
            _Result(
                masks=_Masks([polygon]),
                boxes=_Boxes(_TensorLike([0.91]), _TensorLike([0.0])),
            )
        )
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best.pt"
            checkpoint.write_bytes(b"trusted-test-checkpoint")
            segmenter = UltralyticsLeafSegmenter(
                checkpoint,
                device="cpu",
                proposal_confidence_threshold=0.20,
            )
            segmenter._model = model

            instances = segmenter.segment(Image.new("RGB", (100, 50)))

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].mask.size, (100, 50))
        self.assertEqual(model.kwargs["save"], False)  # type: ignore[index]
        self.assertEqual(model.kwargs["retina_masks"], True)  # type: ignore[index]
        self.assertEqual(model.kwargs["conf"], 0.20)  # type: ignore[index]
        self.assertEqual(model.kwargs["iou"], 0.70)  # type: ignore[index]
        self.assertEqual(segmenter.to_metadata()["proposal_confidence_threshold"], 0.20)
        self.assertEqual(segmenter.to_metadata()["confidence_threshold"], 0.20)

    def test_legacy_confidence_alias_is_supported_but_cannot_be_combined(self) -> None:
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best.pt"
            checkpoint.write_bytes(b"trusted-test-checkpoint")

            segmenter = UltralyticsLeafSegmenter(
                checkpoint,
                confidence_threshold=0.35,
            )
            self.assertEqual(segmenter.proposal_confidence_threshold, 0.35)

            with self.assertRaisesRegex(ValueError, "no ambos"):
                UltralyticsLeafSegmenter(
                    checkpoint,
                    confidence_threshold=0.35,
                    proposal_confidence_threshold=0.20,
                )
