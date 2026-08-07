"""YOLO leaf-segmentation adapter with no classifier coupling.

The module imports Ultralytics only when real inference is requested. Tests and
the rest of the project can therefore use :class:`LeafInstance` and the
``LeafSegmenter`` protocol without installing the optional runtime.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from PIL import Image, ImageDraw

from src.preprocessing.leaf_mask import mask_area_ratio, mask_bbox
from src.preprocessing.leaf_roi import BoundingBox, image_to_rgb

EXPECTED_ULTRALYTICS_VERSION = "8.4.104"


class LeafSegmentationError(RuntimeError):
    """Raised when model output cannot be converted into auditable instances."""


@dataclass(frozen=True)
class LeafInstance:
    """One predicted maize-leaf instance in original-image coordinates."""

    mask: Image.Image
    confidence: float
    bbox: BoundingBox
    source_index: int
    class_id: int = 0

    @property
    def area_ratio(self) -> float:
        return mask_area_ratio(self.mask)


@runtime_checkable
class LeafSegmenter(Protocol):
    """Minimal interface accepted by the preprocessing pipeline."""

    def segment(self, image: Image.Image) -> Sequence[LeafInstance]:
        """Return zero or more leaf instances at exactly ``image.size``."""
        ...

    def to_metadata(self) -> dict[str, object]:
        """Return model and inference provenance without pixel data."""
        ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_confidence(value: object, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise LeafSegmentationError(
            f"confidence no numérica para instancia {index}: {value!r}"
        )
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise LeafSegmentationError(
            f"confidence no numérica para instancia {index}: {value!r}"
        ) from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise LeafSegmentationError(
            f"confidence fuera de [0, 1] para instancia {index}: {confidence!r}"
        )
    return confidence


def _class_id(value: object, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise LeafSegmentationError(
            f"class_id no numérico para instancia {index}: {value!r}"
        )
    try:
        numeric = float(value)
    except ValueError as exc:
        raise LeafSegmentationError(
            f"class_id no numérico para instancia {index}: {value!r}"
        ) from exc
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise LeafSegmentationError(
            f"class_id inválido para instancia {index}: {value!r}"
        )
    return int(numeric)


def _as_list(value: object) -> list[object]:
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    raise LeafSegmentationError(f"salida tensorial no convertible a lista: {type(value)!r}")


def rasterize_instance_polygon(
    polygon: object,
    image_size: tuple[int, int],
) -> Image.Image:
    """Rasterize one Ultralytics ``Masks.xy`` polygon without resizing.

    ``image_size`` follows Pillow's ``(width, height)`` convention. Coordinates
    are already expressed by Ultralytics in original-image pixels.
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size debe contener dimensiones positivas")
    points_array = np.asarray(polygon, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise LeafSegmentationError(
            f"polígono de máscara inválido: shape={points_array.shape!r}"
        )
    if len(points_array) < 3 or not np.isfinite(points_array).all():
        raise LeafSegmentationError("el polígono debe tener al menos 3 puntos finitos")
    if (
        (points_array[:, 0] < 0).any()
        or (points_array[:, 0] > width).any()
        or (points_array[:, 1] < 0).any()
        or (points_array[:, 1] > height).any()
    ):
        raise LeafSegmentationError("el polígono está fuera de la imagen original")

    mask = Image.new("L", image_size, 0)
    ImageDraw.Draw(mask).polygon(
        [(float(x), float(y)) for x, y in points_array],
        fill=255,
    )
    if mask.getbbox() is None:
        raise LeafSegmentationError("el polígono produjo una máscara vacía")
    return mask


def instances_from_ultralytics_result(
    result: object,
    image_size: tuple[int, int],
) -> tuple[LeafInstance, ...]:
    """Convert one Ultralytics result into original-resolution binary masks."""
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return ()

    polygons = list(getattr(masks, "xy", ()) or ())
    confidences = _as_list(getattr(boxes, "conf", None))
    classes = _as_list(getattr(boxes, "cls", None))
    if len(polygons) != len(confidences):
        raise LeafSegmentationError(
            "Ultralytics devolvió cantidades distintas de máscaras y confidences: "
            f"{len(polygons)} != {len(confidences)}"
        )
    if classes and len(classes) != len(polygons):
        raise LeafSegmentationError(
            "Ultralytics devolvió cantidades distintas de máscaras y clases"
        )

    instances: list[LeafInstance] = []
    for index, polygon in enumerate(polygons):
        mask = rasterize_instance_polygon(polygon, image_size)
        bbox = mask_bbox(mask)
        if bbox is None:
            raise LeafSegmentationError(f"máscara vacía en instancia {index}")
        class_id = _class_id(classes[index], index) if classes else 0
        instances.append(
            LeafInstance(
                mask=mask,
                confidence=_finite_confidence(confidences[index], index),
                bbox=bbox,
                source_index=index,
                class_id=class_id,
            )
        )
    return tuple(instances)


class UltralyticsLeafSegmenter:
    """Lazy, version-checked inference wrapper for the trained YOLO checkpoint."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        image_size: int = 640,
        confidence_threshold: float = 0.50,
        iou_threshold: float = 0.70,
        max_detections: int = 20,
        device: str | int | None = None,
        expected_version: str = EXPECTED_ULTRALYTICS_VERSION,
    ) -> None:
        resolved = checkpoint.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"No existe el checkpoint del segmentador: {resolved}")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size debe ser un entero positivo")
        if (
            isinstance(max_detections, bool)
            or not isinstance(max_detections, int)
            or max_detections <= 0
        ):
            raise ValueError("max_detections debe ser un entero positivo")
        for name, value in (
            ("confidence_threshold", confidence_threshold),
            ("iou_threshold", iou_threshold),
        ):
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} debe ser finito")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} debe estar entre 0 y 1")

        self.checkpoint = resolved
        self.checkpoint_sha256 = _sha256(resolved)
        self.image_size = image_size
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = max_detections
        self.device = device
        self.expected_version = expected_version
        self._model: Any | None = None
        self._runtime_version: str | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import ultralytics  # pyright: ignore[reportMissingImports]
            from ultralytics import YOLO  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise LeafSegmentationError(
                "Ultralytics no está instalado. Instale la dependencia opcional "
                "del proyecto con: pip install -e '.[segmentation]'"
            ) from exc
        runtime_version = str(ultralytics.__version__)
        if runtime_version != self.expected_version:
            raise LeafSegmentationError(
                "Versión Ultralytics incompatible con el checkpoint: "
                f"{runtime_version} != {self.expected_version}"
            )
        self._runtime_version = runtime_version
        self._model = YOLO(str(self.checkpoint))
        return self._model

    def segment(self, image: Image.Image) -> tuple[LeafInstance, ...]:
        if not isinstance(image, Image.Image):
            raise TypeError("image debe ser una instancia de PIL.Image.Image")
        normalized = image_to_rgb(image)
        model = self._load_model()
        predict_kwargs: dict[str, object] = {
            "source": normalized,
            "imgsz": self.image_size,
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "max_det": self.max_detections,
            "agnostic_nms": False,
            "retina_masks": True,
            "save": False,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device
        results = model.predict(**predict_kwargs)
        if len(results) != 1:
            raise LeafSegmentationError(
                f"se esperaba un resultado para una imagen; recibidos={len(results)}"
            )
        result = results[0]
        orig_shape = tuple(getattr(result, "orig_shape", (normalized.height, normalized.width)))
        expected_shape = (normalized.height, normalized.width)
        if orig_shape != expected_shape:
            raise LeafSegmentationError(
                f"orig_shape desalineado: {orig_shape!r} != {expected_shape!r}"
            )
        return instances_from_ultralytics_result(result, normalized.size)

    def to_metadata(self) -> dict[str, object]:
        return {
            "segmenter_model": "yolo26n-seg",
            "segmenter_checkpoint": str(self.checkpoint),
            "segmenter_checkpoint_sha256": self.checkpoint_sha256,
            "ultralytics_expected_version": self.expected_version,
            "ultralytics_runtime_version": self._runtime_version,
            "image_size": self.image_size,
            "proposal_confidence_threshold": self.confidence_threshold,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "nms": "class_aware",
            "retina_masks": True,
            "device": self.device,
        }
