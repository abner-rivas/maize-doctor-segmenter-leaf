"""Resolve and render source geometry for human segmentation review cases."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import tempfile
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.data.segmentation_audit import (
    load_coco_segmentation,
    parse_yolo_segmentation_line,
    polygon_area,
)
from src.data.segmentation_review import review_case_id, review_key

CASE_RENDER_STATUSES = {
    "rendered",
    "rendered_from_coco",
    "rendered_invalid_original",
    "no_geometry_available",
    "render_error",
}
READY_STATUS = "ready_for_human_review"
BLOCKED_STATUS = "blocked_by_preview_validation"
PANEL_SIZE = (680, 470)
SMALL_MASK_THRESHOLD = 0.001


@dataclass(frozen=True)
class ReviewGeometry:
    """One normalized polygon drawn in a review preview."""

    points: tuple[tuple[float, float], ...]
    class_id: int | None
    instance_index: int | str
    line_number: int | str
    source: str
    target: bool
    topology_issues: tuple[str, ...] = ()


def make_review_case_id(row: dict[str, str]) -> str:
    """Create a stable case identifier from the manifest's natural key."""
    return review_case_id(row)


def _roboflow_original_base(filename: str) -> str:
    stem = Path(filename).stem
    match = re.fullmatch(
        r"(?P<base>.+?)_(?:jpg|jpeg|png|bmp|tif|tiff|webp)\.rf\.[^.]+",
        stem,
        flags=re.IGNORECASE,
    )
    return match.group("base") if match else stem


def read_unique_review_cases(
    mandatory_path: Path,
    manual_path: Path,
) -> list[dict[str, str]]:
    """Read unique review cases, preferring mandatory metadata on overlap."""

    def read(path: Path, origin: str) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return [{**row, "_review_origin": origin} for row in csv.DictReader(handle)]

    unique = {review_key(row): row for row in read(manual_path, "general")}
    unique.update({review_key(row): row for row in read(mandatory_path, "mandatory")})
    return [unique[key] for key in sorted(unique)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _matching_row(
    rows: Sequence[dict[str, str]],
    case: dict[str, str],
) -> dict[str, str] | None:
    key = review_key(case)
    for row in rows:
        candidate = {
            "source_dataset": row.get("source_dataset", ""),
            "filename": Path(row.get("original_image_path", "")).name,
            "original_line_number": row.get("original_line_number", ""),
            "original_class_id": row.get("original_class_id", ""),
        }
        if review_key(candidate) == key:
            return row
    return None


def _resolve_explicit_or_source_file(
    value: str,
    external_root: Path,
    source: str,
    filename: str,
) -> Path | None:
    explicit = Path(value) if value else None
    if explicit and explicit.is_file():
        return explicit
    exact = sorted(
        path
        for path in external_root.glob(f"{source}*/**/{filename}")
        if path.is_file()
    )
    if len(exact) == 1:
        return exact[0]
    requested_base = _roboflow_original_base(filename).casefold()
    variants = sorted(
        path
        for path in external_root.glob(f"{source}*/**/*")
        if path.is_file()
        and _roboflow_original_base(path.name).casefold() == requested_base
    )
    return variants[0] if len(variants) == 1 else None


def _resolve_image(case: dict[str, str], external_root: Path) -> Path | None:
    return _resolve_explicit_or_source_file(
        case.get("original_image_path", ""),
        external_root,
        case["source_dataset"],
        case["filename"],
    )


def _resolve_label(case: dict[str, str], external_root: Path) -> Path | None:
    label_name = f"{Path(case['filename']).stem}.txt"
    return _resolve_explicit_or_source_file(
        case.get("original_label_path", ""),
        external_root,
        case["source_dataset"],
        label_name,
    )


def _parse_yolo_source(path: Path | None) -> tuple[list[dict[str, object]], int]:
    if path is None or not path.is_file():
        return [], 0
    rows: list[dict[str, object]] = []
    instance_index = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        instance_index += 1
        parsed = parse_yolo_segmentation_line(line)
        issues = tuple(str(issue["issue_type"]) for issue in parsed.issues)
        rows.append(
            {
                "line_number": line_number,
                "instance_index": instance_index,
                "class_id": parsed.class_id,
                "points": tuple(parsed.points),
                "annotation_format": parsed.annotation_format,
                "topology_issues": tuple(
                    issue
                    for issue in issues
                    if issue
                    in {
                        "self_intersection",
                        "repeated_vertex",
                        "zero_length_edge",
                        "insufficient_unique_vertices",
                        "zero_or_near_zero_area",
                        "non_simple_polygon",
                    }
                ),
                "issues": issues,
            }
        )
    return rows, instance_index


def _coco_path(external_root: Path, source: str) -> Path | None:
    preferred = (
        external_root
        / f"{source}_coco_segmentation"
        / "train"
        / "_annotations.coco.json"
    )
    if preferred.is_file():
        return preferred
    matches = sorted(
        path
        for path in external_root.glob(f"{source}*coco*/**/_annotations.coco.json")
        if path.is_file()
    )
    return matches[0] if len(matches) == 1 else None


def _match_coco_entry(
    coco: dict[str, object] | None,
    filename: str,
) -> dict[str, object] | None:
    if coco is None:
        return None
    by_filename = coco["by_filename_all"]
    exact = by_filename.get(filename, [])
    if len(exact) == 1:
        return exact[0]
    requested_base = _roboflow_original_base(filename).casefold()
    candidates = [
        entry
        for candidate_name, entries in by_filename.items()
        if _roboflow_original_base(candidate_name).casefold() == requested_base
        for entry in entries
    ]
    return candidates[0] if len(candidates) == 1 else None


def _normalized_coco_components(
    annotation: dict[str, object],
    width: int,
    height: int,
) -> list[tuple[tuple[float, float], ...]]:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list) or not segmentation:
        return []
    components = (
        segmentation
        if isinstance(segmentation[0], list)
        else [segmentation]
    )
    polygons: list[tuple[tuple[float, float], ...]] = []
    for component in components:
        if (
            not isinstance(component, list)
            or len(component) < 6
            or len(component) % 2
            or width <= 0
            or height <= 0
        ):
            continue
        points = tuple(
            (
                float(component[index]) / width,
                float(component[index + 1]) / height,
            )
            for index in range(0, len(component), 2)
        )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for point in points
            for value in point
        ):
            continue
        polygons.append(points)
    return polygons


def _recovery_metadata(row: dict[str, str] | None) -> dict[str, object]:
    if row is None:
        return {}
    evidence = row.get("recovery_evidence", "")
    try:
        metadata = json.loads(evidence) if evidence else {}
    except json.JSONDecodeError:
        metadata = {}
    for key in (
        "recovery_match_method",
        "bbox_max_abs_error",
        "bbox_iou",
        "matched_annotation_id",
    ):
        if key not in metadata and row.get(key, "") != "":
            metadata[key] = row[key]
    return metadata


def _read_consolidated_geometry(
    manifest_row: dict[str, str] | None,
) -> tuple[tuple[float, float], ...] | None:
    if manifest_row is None:
        return None
    label_value = manifest_row.get("consolidated_label_path", "")
    line_value = manifest_row.get("consolidated_line_number", "")
    if not label_value or not str(line_value).isdigit():
        return None
    label_path = Path(label_value)
    if not label_path.is_file():
        return None
    lines = [
        line
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index = int(line_value) - 1
    if not 0 <= index < len(lines):
        return None
    parsed = parse_yolo_segmentation_line(lines[index])
    return tuple(parsed.points) if parsed.valid else None


class ReviewGeometryResolver:
    """Resolve review geometry from originals before consulting derived labels."""

    def __init__(self, external_root: Path, manifests_root: Path) -> None:
        self.external_root = external_root
        self.recovered = _read_csv(manifests_root / "recovered_annotations.csv")
        self.consolidated = _read_csv(
            manifests_root / "consolidation_manifest.csv"
        )
        self._coco_cache: dict[str, dict[str, object] | None] = {}

    def _coco(self, source: str) -> dict[str, object] | None:
        if source not in self._coco_cache:
            path = _coco_path(self.external_root, source)
            self._coco_cache[source] = (
                load_coco_segmentation(path) if path is not None else None
            )
        return self._coco_cache[source]

    def resolve(self, case: dict[str, str]) -> dict[str, object]:
        """Resolve every renderable instance and identify the review target."""
        image_path = _resolve_image(case, self.external_root)
        label_path = _resolve_label(case, self.external_root)
        yolo_rows, source_instances = _parse_yolo_source(label_path)
        line_value = case.get("original_line_number", "").strip()
        target_line = int(line_value) if line_value.isdigit() else None
        class_value = case.get("original_class_id", "").strip()
        target_class = int(class_value) if class_value.lstrip("+-").isdigit() else None
        geometries: list[ReviewGeometry] = []
        target_yolo_row: dict[str, object] | None = None

        for row in yolo_rows:
            is_target = (
                int(row["line_number"]) == target_line
                if target_line is not None
                else target_class is None or row["class_id"] == target_class
            )
            if is_target:
                target_yolo_row = row
            if row["annotation_format"] != "yolo_segmentation":
                continue
            points = row["points"]
            if len(points) < 3:
                continue
            geometries.append(
                ReviewGeometry(
                    points=points,
                    class_id=row["class_id"],
                    instance_index=int(row["instance_index"]),
                    line_number=int(row["line_number"]),
                    source="yolo_original",
                    target=is_target,
                    topology_issues=row["topology_issues"],
                )
            )

        coco = self._coco(case["source_dataset"])
        coco_entry = _match_coco_entry(coco, case["filename"])
        coco_annotations = coco_entry["annotations"] if coco_entry else []
        recovered_row = _matching_row(self.recovered, case)
        consolidated_row = _matching_row(self.consolidated, case)
        recovery = _recovery_metadata(recovered_row)
        target_is_bbox = bool(
            target_yolo_row
            and target_yolo_row["annotation_format"] == "yolo_bbox"
        )
        geometry_source = "yolo_original" if geometries else "none"
        render_status = "rendered" if geometries else "no_geometry_available"
        reason = "source_yolo_polygon"

        if target_is_bbox:
            matched_id = str(recovery.get("matched_annotation_id", ""))
            annotation = next(
                (
                    item
                    for item in coco_annotations
                    if str(item.get("id", "")) == matched_id
                ),
                None,
            )
            if annotation is not None and coco_entry is not None:
                image = coco_entry["image"]
                components = _normalized_coco_components(
                    annotation,
                    int(image.get("width", 0)),
                    int(image.get("height", 0)),
                )
                for component_index, points in enumerate(components, start=1):
                    geometries.append(
                        ReviewGeometry(
                            points=points,
                            class_id=target_class,
                            instance_index=target_yolo_row["instance_index"],
                            line_number=target_line or "",
                            source="coco_original",
                            target=True,
                        )
                    )
                if components:
                    geometry_source = "coco_original"
                    render_status = "rendered_from_coco"
                    reason = f"matched_coco_annotation_id={matched_id}"
            if geometry_source != "coco_original":
                recovered_points = _read_consolidated_geometry(recovered_row)
                if recovered_points:
                    geometries.append(
                        ReviewGeometry(
                            points=recovered_points,
                            class_id=target_class,
                            instance_index=target_yolo_row["instance_index"],
                            line_number=target_line or "",
                            source="recovered_annotations",
                            target=True,
                        )
                    )
                    geometry_source = "recovered_annotations"
                    render_status = "rendered_from_coco"
                    reason = "recovered_annotations_manifest"
                else:
                    render_status = (
                        "rendered" if geometries else "no_geometry_available"
                    )
                    reason = "no_coco_match"

        if not geometries and coco_entry is not None:
            image = coco_entry["image"]
            for instance_index, annotation in enumerate(coco_annotations, start=1):
                for points in _normalized_coco_components(
                    annotation,
                    int(image.get("width", 0)),
                    int(image.get("height", 0)),
                ):
                    geometries.append(
                        ReviewGeometry(
                            points=points,
                            class_id=int(annotation.get("category_id", -1)),
                            instance_index=instance_index,
                            line_number="",
                            source="coco_original",
                            target=True,
                        )
                    )
            if geometries:
                geometry_source = "coco_original"
                render_status = "rendered_from_coco"
                reason = "source_yolo_unavailable_coco_fallback"

        if not geometries:
            consolidated_points = _read_consolidated_geometry(consolidated_row)
            if consolidated_points:
                geometries.append(
                    ReviewGeometry(
                        points=consolidated_points,
                        class_id=target_class,
                        instance_index=target_line or 1,
                        line_number=target_line or "",
                        source="consolidated",
                        target=True,
                    )
                )
                geometry_source = "consolidated"
                render_status = "rendered"
                reason = "consolidated_reference_fallback"

        topology_issues = sorted(
            {
                issue
                for geometry in geometries
                if geometry.target
                for issue in geometry.topology_issues
            }
        )
        if topology_issues:
            render_status = "rendered_invalid_original"
            geometry_source = "yolo_original"
            reason = ";".join(topology_issues)
        if render_status == "no_geometry_available":
            if label_path is None:
                reason = "missing_source_label"
            elif not label_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip():
                reason = "empty_annotation"
            elif target_is_bbox:
                reason = "no_coco_match"
            else:
                reason = "filtered_geometry"

        return {
            "image_path": image_path,
            "label_path": label_path,
            "image_found": image_path is not None,
            "source_label_found": label_path is not None,
            "coco_annotation_found": bool(coco_annotations),
            "geometry_source": geometry_source,
            "geometries": geometries,
            "total_instances_in_source": source_instances,
            "render_status": render_status,
            "reason": reason,
            "recovery_method": recovery.get("recovery_match_method", ""),
            "bbox_max_abs_error": recovery.get("bbox_max_abs_error", ""),
            "bbox_iou": recovery.get("bbox_iou", ""),
        }


def _font(size: int = 17) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fit(image: Image.Image, size: tuple[int, int], background: str) -> Image.Image:
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    canvas.paste(
        fitted,
        ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2),
    )
    return canvas


def _color(geometry: ReviewGeometry) -> tuple[int, int, int, int]:
    if not geometry.target:
        return (41, 182, 246, 85)
    if geometry.topology_issues:
        return (239, 83, 80, 115)
    if geometry.source in {"coco_original", "recovered_annotations"}:
        return (255, 202, 40, 135)
    return (0, 230, 118, 105)


def _strict_intersections(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    intersections: list[tuple[float, float]] = []
    count = len(points)
    for left in range(count):
        a, b = points[left], points[(left + 1) % count]
        for right in range(left + 1, count):
            if (
                right == left
                or (right + 1) % count == left
                or (left + 1) % count == right
            ):
                continue
            c, d = points[right], points[(right + 1) % count]
            denominator = (a[0] - b[0]) * (c[1] - d[1]) - (
                a[1] - b[1]
            ) * (c[0] - d[0])
            if abs(denominator) <= 1e-12:
                continue
            first = a[0] * b[1] - a[1] * b[0]
            second = c[0] * d[1] - c[1] * d[0]
            x = (first * (c[0] - d[0]) - (a[0] - b[0]) * second) / denominator
            y = (first * (c[1] - d[1]) - (a[1] - b[1]) * second) / denominator
            if (
                min(a[0], b[0]) < x < max(a[0], b[0])
                and min(a[1], b[1]) < y < max(a[1], b[1])
                and min(c[0], d[0]) < x < max(c[0], d[0])
                and min(c[1], d[1]) < y < max(c[1], d[1])
            ):
                intersections.append((x, y))
    return intersections


def _overlay(
    image: Image.Image,
    geometries: Sequence[ReviewGeometry],
) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for geometry in geometries:
        points = [
            (
                round(min(1.0, max(0.0, x)) * (base.width - 1)),
                round(min(1.0, max(0.0, y)) * (base.height - 1)),
            )
            for x, y in geometry.points
        ]
        color = _color(geometry)
        draw.polygon(points, fill=color)
        draw.line([*points, points[0]], fill=(*color[:3], 255), width=4)
        if geometry.target and polygon_area(geometry.points) < SMALL_MASK_THRESHOLD:
            center_x = round(sum(point[0] for point in points) / len(points))
            center_y = round(sum(point[1] for point in points) / len(points))
            draw.ellipse(
                (center_x - 12, center_y - 12, center_x + 12, center_y + 12),
                outline=(*color[:3], 255),
                width=4,
            )
        if "self_intersection" in geometry.topology_issues:
            for x, y in _strict_intersections(geometry.points):
                px = round(x * (base.width - 1))
                py = round(y * (base.height - 1))
                draw.line((px - 12, py - 12, px + 12, py + 12), fill="red", width=5)
                draw.line((px - 12, py + 12, px + 12, py - 12), fill="red", width=5)
    return Image.alpha_composite(base, layer).convert("RGB")


def _target_geometries(
    geometries: Sequence[ReviewGeometry],
) -> list[ReviewGeometry]:
    targets = [geometry for geometry in geometries if geometry.target]
    return targets or list(geometries)


def _mask(
    size: tuple[int, int],
    geometries: Sequence[ReviewGeometry],
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for geometry in geometries:
        points = [
            (
                round(min(1.0, max(0.0, x)) * (size[0] - 1)),
                round(min(1.0, max(0.0, y)) * (size[1] - 1)),
            )
            for x, y in geometry.points
        ]
        draw.polygon(points, fill=255)
        draw.line([*points, points[0]], fill=255, width=2)
    return mask


def _crop_box(
    image_size: tuple[int, int],
    geometries: Sequence[ReviewGeometry],
) -> tuple[int, int, int, int]:
    xs = [x * image_size[0] for geometry in geometries for x, _ in geometry.points]
    ys = [y * image_size[1] for geometry in geometries for _, y in geometry.points]
    if not xs or not ys:
        return (0, 0, image_size[0], image_size[1])
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    span = max(right - left, bottom - top, 24)
    margin = max(20, span * 0.8)
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    half = max(48, span / 2 + margin)
    return (
        max(0, round(center_x - half)),
        max(0, round(center_y - half)),
        min(image_size[0], round(center_x + half)),
        min(image_size[1], round(center_y + half)),
    )


def _titled(panel: Image.Image, title: str) -> Image.Image:
    result = Image.new("RGB", (panel.width, panel.height + 34), "#eeeeee")
    result.paste(panel, (0, 34))
    ImageDraw.Draw(result).text((10, 7), title, fill="#111111", font=_font(17))
    return result


def _no_geometry_panel(message: str) -> Image.Image:
    panel = Image.new("RGB", PANEL_SIZE, "#d9d9d9")
    draw = ImageDraw.Draw(panel)
    font = _font(27)
    lines = ["NO GEOMETRY AVAILABLE", message]
    y = 180
    for line in lines:
        draw.text((40, y), line, fill="#b71c1c", font=font)
        y += 48
    return panel


def render_review_preview(
    case: dict[str, str],
    resolved: dict[str, object],
    destination: Path,
) -> dict[str, object]:
    """Render four evidence panels and return measurable render statistics."""
    if resolved["render_status"] not in CASE_RENDER_STATUSES:
        raise ValueError(f"render_status inválido: {resolved['render_status']}")
    image_path = resolved["image_path"]
    if image_path is None:
        resolved["render_status"] = "render_error"
        resolved["reason"] = "missing_source_image"
        source = Image.new("RGB", (960, 640), "#bdbdbd")
    else:
        with Image.open(image_path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
    geometries = resolved["geometries"]
    targets = _target_geometries(geometries)
    original_panel = _titled(_fit(source, PANEL_SIZE, "#d9d9d9"), "1. Imagen original")

    if geometries:
        overlay = _overlay(source, geometries)
        overlay_panel = _titled(
            _fit(overlay, PANEL_SIZE, "#d9d9d9"),
            "2. Geometría original/recuperada",
        )
        target_mask = _mask(source.size, targets)
        mask_pixels = sum(target_mask.histogram()[1:])
        mask_color = Image.new("RGB", source.size, "#d0d0d0")
        highlight = Image.new("RGB", source.size, _color(targets[0])[:3])
        mask_color.paste(highlight, mask=target_mask)
        crop = _crop_box(source.size, targets)
        small = sum(polygon_area(item.points) for item in targets) < SMALL_MASK_THRESHOLD
        isolated = mask_color.crop(crop) if small else mask_color
        zoom = overlay.crop(crop)
        mask_panel = _titled(
            _fit(isolated, PANEL_SIZE, "#d0d0d0"),
            "3. Máscara aislada" + (" (ampliada)" if small else ""),
        )
        zoom_panel = _titled(
            _fit(zoom, PANEL_SIZE, "#d9d9d9"),
            "4. Zoom de la región" + (" (máscara < 0.001)" if small else ""),
        )
    else:
        mask_pixels = 0
        missing = _no_geometry_panel(str(resolved["reason"]))
        overlay_panel = _titled(missing.copy(), "2. Estado de geometría")
        mask_panel = _titled(missing.copy(), "3. Máscara aislada")
        zoom_panel = _titled(_fit(source, PANEL_SIZE, "#d9d9d9"), "4. Imagen evaluable")

    target_area = sum(polygon_area(item.points) for item in targets)
    topology = sorted(
        {
            issue
            for geometry in targets
            for issue in geometry.topology_issues
        }
    )
    issue_type = (
        "self_intersection"
        if "self_intersection" in topology
        else case.get("review_reason", "")
    )
    topology_status = (
        ";".join(topology)
        if topology
        else ("valid" if geometries else "unavailable")
    )
    instance_index = case.get("original_line_number", "").strip() or "all"
    metadata = [
        f"review_case_id={make_review_case_id(case)}",
        f"source_dataset={case['source_dataset']}",
        f"original_filename={case['filename']}",
        f"source_label_path={resolved['label_path'] or case.get('original_label_path', '')}",
        f"geometry_source={resolved['geometry_source']}",
        f"issue_type={issue_type}",
        f"class_name={case.get('original_class_name', '')}",
        f"instance_index={instance_index}",
        f"total_instances_in_source={resolved['total_instances_in_source']}",
        f"instances_rendered={len(geometries)}",
        f"polygon_area_ratio={target_area:.12g}",
        f"topology_status={topology_status}",
        f"provisional_decision={case.get('decision', '')}",
        "pool_membership=excluded_from_pool_pending_human_review",
        f"render_status={resolved['render_status']}",
    ]
    if resolved.get("recovery_method"):
        metadata.extend(
            (
                f"recovery_method={resolved['recovery_method']}",
                f"bbox_max_abs_error={resolved['bbox_max_abs_error']}",
                f"bbox_iou={resolved['bbox_iou']}",
            )
        )
    lines = [
        wrapped
        for item in metadata
        for wrapped in textwrap.wrap(item, width=112) or [""]
    ]
    panel_width = PANEL_SIZE[0]
    panel_height = PANEL_SIZE[1] + 34
    metadata_height = max(230, 14 + len(lines) * 23)
    canvas = Image.new(
        "RGB",
        (panel_width * 2, panel_height * 2 + metadata_height),
        "#f5f5f5",
    )
    for panel, position in (
        (original_panel, (0, 0)),
        (overlay_panel, (panel_width, 0)),
        (mask_panel, (0, panel_height)),
        (zoom_panel, (panel_width, panel_height)),
    ):
        canvas.paste(panel, position)
    draw = ImageDraw.Draw(canvas)
    y = panel_height * 2 + 10
    for line in lines:
        draw.text((14, y), line, fill="#111111", font=_font(16))
        y += 23
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=94, subsampling=0)
    return {
        "instances_rendered": len(geometries),
        "mask_pixels_rendered": mask_pixels,
        "polygon_area_ratio": target_area,
    }


def _replace_directory(staged: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    staged.rename(destination)
    if backup.exists():
        shutil.rmtree(backup)


def validate_review_preview_rows(
    rows: Sequence[dict[str, object]],
    *,
    expected_total: int = 35,
    expected_mandatory: int = 2,
) -> dict[str, object]:
    """Validate preview evidence without allowing known geometry to disappear."""
    mandatory = [row for row in rows if row["review_origin"] == "mandatory"]
    known_zero = [
        row
        for row in rows
        if int(row["expected_instances"]) > 0
        and int(row["instances_rendered"]) == 0
    ]
    invalid_masks = [
        row
        for row in rows
        if row["render_status"] != "no_geometry_available"
        and int(row["mask_pixels_rendered"]) <= 0
    ]
    errors = [row for row in rows if row["render_status"] == "render_error"]
    missing_previews = [
        row for row in rows if not bool(row.get("_preview_exists", False))
    ]
    mandatory_without_geometry = [
        row
        for row in mandatory
        if int(row["instances_rendered"]) == 0
        or int(row["mask_pixels_rendered"]) <= 0
    ]
    ready = (
        len(rows) == expected_total
        and len(mandatory) == expected_mandatory
        and not known_zero
        and not invalid_masks
        and not errors
        and not missing_previews
        and not mandatory_without_geometry
    )
    return {
        "global_status": READY_STATUS if ready else BLOCKED_STATUS,
        "known_geometry_with_zero_instances": len(known_zero),
        "nonempty_geometry_with_zero_mask_pixels": len(invalid_masks),
        "render_errors": len(errors),
        "missing_previews": len(missing_previews),
        "mandatory_without_visible_geometry": len(mandatory_without_geometry),
    }
def generate_review_previews(
    external_root: Path,
    manifests_root: Path,
    preview_root: Path,
    validation_path: Path,
    *,
    published_preview_root: Path | None = None,
) -> dict[str, object]:
    """Generate all unique review previews and their machine-readable validation."""
    cases = read_unique_review_cases(
        manifests_root / "mandatory_visual_review.csv",
        manifests_root / "manual_review.csv",
    )
    resolver = ReviewGeometryResolver(external_root, manifests_root)
    validation_rows: list[dict[str, object]] = []
    published_root = published_preview_root or preview_root
    preview_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".review_previews_",
        dir=preview_root.parent,
    ) as temporary:
        staging = Path(temporary)
        for case in cases:
            case_id = make_review_case_id(case)
            category = (
                "mandatory_visual_review"
                if case["_review_origin"] == "mandatory"
                else "manual_review"
            )
            final_preview = preview_root / category / f"{case_id}.jpg"
            published_preview = published_root / category / final_preview.name
            staged_preview = staging / category / final_preview.name
            try:
                resolved = resolver.resolve(case)
                expected_instances = len(resolved["geometries"])
                render = render_review_preview(case, resolved, staged_preview)
            except Exception as exc:  # pragma: no cover - defensive artifact reporting
                expected_instances = 0
                resolved = {
                    "image_found": False,
                    "source_label_found": False,
                    "coco_annotation_found": False,
                    "geometry_source": "none",
                    "total_instances_in_source": 0,
                    "render_status": "render_error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                render = {
                    "instances_rendered": 0,
                    "mask_pixels_rendered": 0,
                    "polygon_area_ratio": 0.0,
                }
            validation_rows.append(
                {
                    "review_case_id": case_id,
                    "review_origin": case["_review_origin"],
                    "source_dataset": case["source_dataset"],
                    "original_filename": case["filename"],
                    "image_found": bool(resolved["image_found"]),
                    "source_label_found": bool(resolved["source_label_found"]),
                    "coco_annotation_found": bool(
                        resolved["coco_annotation_found"]
                    ),
                    "geometry_source": resolved["geometry_source"],
                    "expected_instances": expected_instances,
                    "instances_rendered": render["instances_rendered"],
                    "mask_pixels_rendered": render["mask_pixels_rendered"],
                    "polygon_area_ratio": render["polygon_area_ratio"],
                    "render_status": resolved["render_status"],
                    "reason": resolved["reason"],
                    "preview_path": str(published_preview.resolve()),
                    "_preview_exists": staged_preview.is_file(),
                }
            )
        for category in ("manual_review", "mandatory_visual_review"):
            staged = staging / category
            staged.mkdir(exist_ok=True)
            destination = preview_root / category
            destination.parent.mkdir(parents=True, exist_ok=True)
            _replace_directory(staged, destination)

    validation = validate_review_preview_rows(validation_rows)
    mandatory = [
        row for row in validation_rows if row["review_origin"] == "mandatory"
    ]
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in validation_rows
    ]
    summary = {
        "schema_version": 1,
        "global_status": validation["global_status"],
        "total_unique_cases": len(validation_rows),
        "mandatory_cases": len(mandatory),
        "previews_generated": (
            len(validation_rows) - int(validation["missing_previews"])
        ),
        "geometry_sources": dict(
            sorted(Counter(row["geometry_source"] for row in validation_rows).items())
        ),
        "render_statuses": dict(
            sorted(Counter(row["render_status"] for row in validation_rows).items())
        ),
        **{
            key: value
            for key, value in validation.items()
            if key != "global_status"
        },
        "cases": public_rows,
    }
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
