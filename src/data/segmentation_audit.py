"""Audit external YOLO/COCO maize-leaf segmentation datasets without mutating them."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SEED = 42
SMALL_MASK_THRESHOLD = 0.05
LARGE_MASK_THRESHOLD = 0.50
BORDER_EPSILON = 1e-9
TOPOLOGY_EPSILON = 1e-12
BBOX_MATCH_TOLERANCE = 1e-5
AUDIT_CACHE_SCHEMA_VERSION = 2
PARSER_SCHEMA_VERSION = 2

SOURCE_DEFINITIONS = {
    "corn_leaf_diseases_classification": {
        "yolo_dir": "corn_leaf_diseases_classification_yolo26",
        "coco_dir": "corn_leaf_diseases_classification_coco_segmentation",
    },
    "corn": {
        "yolo_dir": "corn_yolo26",
        "coco_dir": "corn_coco_segmentation",
    },
}

SEMANTIC_ROLES = {
    "corn_leaf_diseases_classification": {
        0: ("gray_leaf_spot", "lesion", False, "exclude"),
        1: ("leaf", "full_leaf", True, "include_after_remap"),
        2: ("northern_leaf_blight", "lesion", False, "exclude"),
    },
    "corn": {
        0: ("leaf", "full_leaf", True, "include_after_remap"),
    },
}

ISSUE_COLUMNS = (
    "source",
    "label_path",
    "filename",
    "line_number",
    "field_number",
    "token",
    "issue_type",
    "detail",
    "coco_recovery_possible",
    "coco_contrast_detail",
    "recovery_match_method",
    "recovery_candidate_count",
    "bbox_max_abs_error",
    "bbox_iou",
    "class_match",
    "semantic_role_match",
    "topology_valid",
    "recovery_decision",
    "recovery_reason",
)
MISMATCH_COLUMNS = (
    "source",
    "mismatch_type",
    "stem",
    "image_path",
    "label_path",
)
DUPLICATE_COLUMNS = (
    "group_id",
    "sha256",
    "collection",
    "filename",
    "path",
    "group_size",
    "collections_in_group",
    "includes_pilot",
)


@dataclass
class PolygonParseResult:
    """Parsed YOLO segmentation row plus every validation issue."""

    class_id: int | None
    points: list[tuple[float, float]]
    valid: bool
    issues: list[dict[str, object]] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    annotation_format: str = "unknown"


@dataclass
class DatasetInventory:
    """Discovered files and metadata for one logical source."""

    source: str
    yolo_root: Path
    coco_root: Path
    image_dir: Path
    label_dir: Path
    coco_json: Path
    images: list[Path]
    labels: list[Path]
    class_names: dict[int, str]
    license_name: str


def sha256_file(path: Path) -> str:
    """Compute an exact file digest without changing the source."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    """Return polygon area using the shoelace formula."""
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def polygon_touches_border(
    points: Sequence[tuple[float, float]],
    epsilon: float = BORDER_EPSILON,
) -> dict[str, bool]:
    """Return one flag per normalized image border."""
    return {
        "touches_left": any(x <= epsilon for x, _ in points),
        "touches_right": any(x >= 1.0 - epsilon for x, _ in points),
        "touches_top": any(y <= epsilon for _, y in points),
        "touches_bottom": any(y >= 1.0 - epsilon for _, y in points),
    }


def _looks_concatenated_numeric(token: str) -> bool:
    if re.fullmatch(r"[+-]?(?:\d*\.\d+){2,}", token):
        return True
    decimal_marks = token.count(".")
    return decimal_marks > 1 and bool(re.search(r"\d", token))


def _issue(
    issue_type: str,
    detail: str,
    *,
    field_number: int | None = None,
    token: str = "",
) -> dict[str, object]:
    return {
        "issue_type": issue_type,
        "detail": detail,
        "field_number": field_number if field_number is not None else "",
        "token": token,
    }


def _cross_product(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
    epsilon: float,
) -> bool:
    orientations = (
        _cross_product(first_start, first_end, second_start),
        _cross_product(first_start, first_end, second_end),
        _cross_product(second_start, second_end, first_start),
        _cross_product(second_start, second_end, first_end),
    )
    return (
        (orientations[0] > epsilon and orientations[1] < -epsilon)
        or (orientations[0] < -epsilon and orientations[1] > epsilon)
    ) and (
        (orientations[2] > epsilon and orientations[3] < -epsilon)
        or (orientations[2] < -epsilon and orientations[3] > epsilon)
    )


def polygon_topology_issues(
    points: Sequence[tuple[float, float]],
    *,
    epsilon: float = TOPOLOGY_EPSILON,
) -> list[dict[str, object]]:
    """Return deterministic topology defects without modifying vertex order."""
    issues: list[dict[str, object]] = []
    unique_vertices = set(points)
    repeated_vertex = len(unique_vertices) < len(points)
    insufficient_unique = len(unique_vertices) < 3
    zero_length_edge = any(
        points[index] == points[(index + 1) % len(points)]
        for index in range(len(points))
    ) if points else False
    near_zero_area = len(points) >= 3 and polygon_area(points) <= epsilon

    self_intersection = False
    count = len(points)
    if count >= 4:
        for left in range(count):
            first_start = points[left]
            first_end = points[(left + 1) % count]
            for right in range(left + 1, count):
                if (
                    right == left
                    or (right + 1) % count == left
                    or (left + 1) % count == right
                ):
                    continue
                if _segments_intersect(
                    first_start,
                    first_end,
                    points[right],
                    points[(right + 1) % count],
                    epsilon,
                ):
                    self_intersection = True
                    break
            if self_intersection:
                break

    if self_intersection:
        issues.append(
            _issue("self_intersection", "Dos aristas no adyacentes se intersectan")
        )
    if repeated_vertex:
        issues.append(
            _issue("repeated_vertex", "El polígono repite al menos un vértice")
        )
    if zero_length_edge:
        issues.append(
            _issue("zero_length_edge", "El polígono contiene una arista de longitud cero")
        )
    if insufficient_unique:
        issues.append(
            _issue(
                "insufficient_unique_vertices",
                f"Sólo hay {len(unique_vertices)} vértices únicos",
            )
        )
    if near_zero_area:
        issues.append(
            _issue(
                "zero_or_near_zero_area",
                f"El área es menor o igual que {epsilon:g}",
            )
        )
    if (
        self_intersection
        or repeated_vertex
        or zero_length_edge
        or insufficient_unique
        or near_zero_area
    ):
        issues.append(
            _issue("non_simple_polygon", "La geometría no es un polígono simple")
        )
    return issues


def parse_yolo_segmentation_line(line: str) -> PolygonParseResult:
    """Parse and strictly validate one YOLO segmentation line."""
    tokens = line.strip().split()
    issues: list[dict[str, object]] = []
    if not tokens:
        return PolygonParseResult(
            class_id=None,
            points=[],
            valid=False,
            issues=[_issue("empty_line", "La línea no contiene campos")],
            tokens=[],
            annotation_format="empty",
        )

    class_id: int | None = None
    if not re.fullmatch(r"[+-]?\d+", tokens[0]):
        issues.append(
            _issue(
                "invalid_class_id",
                "El primer campo debe ser un entero",
                field_number=1,
                token=tokens[0],
            )
        )
    else:
        class_id = int(tokens[0])

    is_bbox_row = len(tokens) == 5
    if is_bbox_row:
        issues.append(
            _issue(
                "bbox_format_in_segmentation_label",
                "Fila YOLO de detección class_id x_center y_center width height",
            )
        )
    elif len(tokens) < 7:
        issues.append(
            _issue(
                "fewer_than_three_points",
                f"Se requieren al menos 7 campos y se encontraron {len(tokens)}",
            )
        )
    if not is_bbox_row and len(tokens) % 2 == 0:
        issues.append(
            _issue(
                "incomplete_coordinate_pair",
                "class_id + pares x,y debe producir un número impar de campos",
            )
        )

    coordinates: list[float | None] = []
    for field_number, token in enumerate(tokens[1:], start=2):
        try:
            value = float(token)
        except ValueError:
            issue_type = (
                "concatenated_numeric_token"
                if _looks_concatenated_numeric(token)
                else "non_numeric_token"
            )
            issues.append(
                _issue(
                    issue_type,
                    "No se puede interpretar como una coordenada",
                    field_number=field_number,
                    token=token,
                )
            )
            coordinates.append(None)
            continue
        if not math.isfinite(value):
            issues.append(
                _issue(
                    "non_finite_coordinate",
                    "NaN e infinito no son coordenadas válidas",
                    field_number=field_number,
                    token=token,
                )
            )
        elif not 0.0 <= value <= 1.0:
            issues.append(
                _issue(
                    "coordinate_out_of_range",
                    "La coordenada debe estar dentro de [0,1]",
                    field_number=field_number,
                    token=token,
                )
            )
        coordinates.append(value)

    points = [] if is_bbox_row else [
        (float(coordinates[index]), float(coordinates[index + 1]))
        for index in range(0, len(coordinates) - 1, 2)
        if coordinates[index] is not None and coordinates[index + 1] is not None
    ]
    if not is_bbox_row and len(points) < 3:
        issues.append(
            _issue(
                "insufficient_unique_vertices",
                f"Sólo hay {len(set(points))} vértices únicos",
            )
        )
    elif not is_bbox_row:
        issues.extend(polygon_topology_issues(points))
    return PolygonParseResult(
        class_id=class_id,
        points=points,
        valid=not issues,
        issues=issues,
        tokens=tokens,
        annotation_format="yolo_bbox" if is_bbox_row else "yolo_segmentation",
    )


def _read_simple_yaml_names(path: Path) -> dict[int, str]:
    """Read the small Roboflow data.yaml schema without requiring PyYAML."""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^names:\s*\[(.*)\]\s*$", text, flags=re.MULTILINE)
    if not match:
        return {}
    values = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))
    return dict(enumerate(values))


def _read_license(root: Path) -> str:
    for name in ("README.dataset.txt", "data.yaml"):
        path = root / name
        if not path.is_file():
            continue
        match = re.search(
            r"(?:License|license):\s*([^\n]+)",
            path.read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            return match.group(1).strip()
    return "unknown"


def discover_dataset_files(external_root: Path) -> dict[str, DatasetInventory]:
    """Discover the exact paired YOLO and COCO sources."""
    inventories: dict[str, DatasetInventory] = {}
    for source, definition in SOURCE_DEFINITIONS.items():
        yolo_root = external_root / definition["yolo_dir"]
        coco_root = external_root / definition["coco_dir"]
        image_dir = yolo_root / "train" / "images"
        label_dir = yolo_root / "train" / "labels"
        coco_json = coco_root / "train" / "_annotations.coco.json"
        required = (yolo_root, coco_root, image_dir, label_dir, coco_json)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Faltan rutas para {source}: {', '.join(missing)}")
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(path for path in label_dir.glob("*.txt") if path.is_file())
        inventories[source] = DatasetInventory(
            source=source,
            yolo_root=yolo_root,
            coco_root=coco_root,
            image_dir=image_dir,
            label_dir=label_dir,
            coco_json=coco_json,
            images=images,
            labels=labels,
            class_names=_read_simple_yaml_names(yolo_root / "data.yaml"),
            license_name=_read_license(yolo_root),
        )
    return inventories


def build_audit_input_fingerprint(
    inventories: dict[str, DatasetInventory],
    pilot_root: Path,
    *,
    seed: int = SEED,
) -> dict[str, object]:
    """Fingerprint every audit input with stable relative paths and SHA-256."""
    entries: list[dict[str, object]] = []

    def add(path: Path, relative_path: str, kind: str) -> None:
        entries.append(
            {
                "relative_path": relative_path,
                "kind": kind,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    for source, inventory in sorted(inventories.items()):
        for path in inventory.images:
            add(
                path,
                f"sources/{source}/yolo/images/{path.name}",
                "image",
            )
        for path in inventory.labels:
            add(
                path,
                f"sources/{source}/yolo/labels/{path.name}",
                "yolo_label",
            )
        add(
            inventory.coco_json,
            f"sources/{source}/coco/{inventory.coco_json.name}",
            "coco_json",
        )
        for name in ("data.yaml", "README.dataset.txt", "README.roboflow.txt"):
            path = inventory.yolo_root / name
            if path.is_file():
                add(path, f"sources/{source}/yolo/{name}", "metadata")

    pilot_images = sorted(
        path
        for path in (pilot_root / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for path in pilot_images:
        add(path, f"pilot/images/{path.name}", "pilot_image")

    analysis_config = {
        "cache_schema_version": AUDIT_CACHE_SCHEMA_VERSION,
        "parser_schema_version": PARSER_SCHEMA_VERSION,
        "seed": seed,
        "image_extensions": sorted(IMAGE_EXTENSIONS),
        "small_mask_threshold": SMALL_MASK_THRESHOLD,
        "large_mask_threshold": LARGE_MASK_THRESHOLD,
        "border_epsilon": BORDER_EPSILON,
        "topology_epsilon": TOPOLOGY_EPSILON,
        "bbox_match_tolerance": BBOX_MATCH_TOLERANCE,
        "source_definitions": SOURCE_DEFINITIONS,
        "semantic_roles": SEMANTIC_ROLES,
    }
    config_bytes = json.dumps(
        analysis_config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    entries.append(
        {
            "relative_path": "analysis/config.json",
            "kind": "analysis_config",
            "size_bytes": len(config_bytes),
            "sha256": config_sha256,
        }
    )
    entries.sort(key=lambda row: str(row["relative_path"]))
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["relative_path"]).encode())
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode())
        digest.update(b"\n")
    return {
        "cache_schema_version": AUDIT_CACHE_SCHEMA_VERSION,
        "parser_schema_version": PARSER_SCHEMA_VERSION,
        "analysis_config_sha256": config_sha256,
        "file_count": len(entries) - 1,
        "total_bytes": sum(
            int(entry["size_bytes"])
            for entry in entries
            if entry["kind"] != "analysis_config"
        ),
        "global_sha256": digest.hexdigest(),
        "files": entries,
    }


def audit_cache_is_current(
    cached_summary: dict[str, object],
    current_fingerprint: dict[str, object],
) -> bool:
    """Return whether a cached audit matches the current parser and all inputs."""
    return (
        cached_summary.get("cache_schema_version") == AUDIT_CACHE_SCHEMA_VERSION
        and cached_summary.get("parser_schema_version") == PARSER_SCHEMA_VERSION
        and cached_summary.get("input_fingerprint") == current_fingerprint
    )


def _orientation(width: int, height: int) -> str:
    if width > height * 1.1:
        return "horizontal"
    if height > width * 1.1:
        return "vertical"
    return "square"


def compute_image_statistics(inventory: DatasetInventory) -> list[dict[str, object]]:
    """Read image metadata and exact hashes for one source."""
    rows: list[dict[str, object]] = []
    for image_path in inventory.images:
        try:
            with Image.open(image_path) as image:
                oriented = ImageOps.exif_transpose(image)
                width, height = oriented.size
                mode = oriented.mode
                image_format = image.format or image_path.suffix.lstrip(".").upper()
        except (OSError, UnidentifiedImageError) as exc:
            rows.append(
                {
                    "source": inventory.source,
                    "filename": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "valid_image": False,
                    "image_issue": str(exc),
                    "sha256": sha256_file(image_path),
                }
            )
            continue
        rows.append(
            {
                "source": inventory.source,
                "filename": image_path.name,
                "image_path": str(image_path.resolve()),
                "valid_image": True,
                "image_issue": "",
                "width": width,
                "height": height,
                "pixel_area": width * height,
                "aspect_ratio": width / height,
                "orientation": _orientation(width, height),
                "format": image_format,
                "color_mode": mode,
                "file_size_bytes": image_path.stat().st_size,
                "sha256": sha256_file(image_path),
            }
        )
    return rows


def _label_image_maps(
    inventory: DatasetInventory,
) -> tuple[dict[str, Path], dict[str, Path]]:
    return (
        {path.stem.casefold(): path for path in inventory.images},
        {path.stem.casefold(): path for path in inventory.labels},
    )


def _polygon_statistics(
    source: str,
    filename: str,
    class_id: int | None,
    class_name: str,
    instance_index: int,
    points: Sequence[tuple[float, float]],
    width: int,
    height: int,
    valid: bool,
    issue_reason: str,
) -> dict[str, object]:
    if points:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        area = polygon_area(points)
        border = polygon_touches_border(points)
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)
    else:
        x1 = x2 = y1 = y2 = area = centroid_x = centroid_y = 0.0
        border = {
            "touches_left": False,
            "touches_right": False,
            "touches_top": False,
            "touches_bottom": False,
        }
    touches_any = any(border.values())
    if area < SMALL_MASK_THRESHOLD:
        size_category = "small"
    elif area <= LARGE_MASK_THRESHOLD:
        size_category = "medium"
    else:
        size_category = "large"
    return {
        "source": source,
        "image_id": Path(filename).stem,
        "filename": filename,
        "class_id": class_id if class_id is not None else "",
        "class_name": class_name,
        "instance_index": instance_index,
        "point_count": len(points),
        "normalized_area": area,
        "pixel_area_approx": area * width * height,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_area_ratio": max(0.0, x2 - x1) * max(0.0, y2 - y1),
        "polygon_area_ratio": area,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        **border,
        "touches_any_border": touches_any,
        "size_category": size_category,
        "valid": valid,
        "issue_reason": issue_reason,
    }


def load_yolo_dataset(
    inventory: DatasetInventory,
    image_rows: Sequence[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, list[dict[str, object]]],
]:
    """Parse YOLO labels into polygon statistics, issues and mismatches."""
    image_map, label_map = _label_image_maps(inventory)
    image_metadata = {
        Path(str(row["filename"])).stem.casefold(): row
        for row in image_rows
        if row.get("valid_image")
    }
    mismatches: list[dict[str, object]] = []
    for stem in sorted(image_map.keys() - label_map.keys()):
        mismatches.append(
            {
                "source": inventory.source,
                "mismatch_type": "image_without_label",
                "stem": stem,
                "image_path": str(image_map[stem].resolve()),
                "label_path": "",
            }
        )
    for stem in sorted(label_map.keys() - image_map.keys()):
        mismatches.append(
            {
                "source": inventory.source,
                "mismatch_type": "label_without_image",
                "stem": stem,
                "image_path": "",
                "label_path": str(label_map[stem].resolve()),
            }
        )

    polygons: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    parsed_by_filename: dict[str, list[dict[str, object]]] = defaultdict(list)
    for stem, label_path in sorted(label_map.items()):
        image_path = image_map.get(stem)
        filename = image_path.name if image_path else f"{label_path.stem}.unknown"
        metadata = image_metadata.get(stem, {})
        width = int(metadata.get("width", 0) or 0)
        height = int(metadata.get("height", 0) or 0)
        text = label_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            issues.append(
                {
                    "source": inventory.source,
                    "label_path": str(label_path.resolve()),
                    "filename": filename,
                    "line_number": "",
                    "field_number": "",
                    "token": "",
                    "issue_type": "empty_label_file",
                    "detail": "El archivo TXT no contiene polígonos",
                    "coco_recovery_possible": "",
                    "coco_contrast_detail": "",
                }
            )
            parsed_by_filename[filename] = []
            continue
        instance_index = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            instance_index += 1
            parsed = parse_yolo_segmentation_line(line)
            class_name = inventory.class_names.get(
                parsed.class_id if parsed.class_id is not None else -1,
                "unknown",
            )
            issue_reason = ";".join(
                str(issue["issue_type"]) for issue in parsed.issues
            )
            polygon_row = _polygon_statistics(
                inventory.source,
                filename,
                parsed.class_id,
                class_name,
                instance_index,
                parsed.points,
                width,
                height,
                parsed.valid,
                issue_reason,
            )
            polygon_row["label_path"] = str(label_path.resolve())
            polygon_row["line_number"] = line_number
            polygon_row["raw_line"] = line
            polygon_row["annotation_format"] = parsed.annotation_format
            polygons.append(polygon_row)
            parsed_by_filename[filename].append(
                {
                    **polygon_row,
                    "points": parsed.points,
                }
            )
            for item in parsed.issues:
                issues.append(
                    {
                        "source": inventory.source,
                        "label_path": str(label_path.resolve()),
                        "filename": filename,
                        "line_number": line_number,
                        "field_number": item["field_number"],
                        "token": item["token"],
                        "issue_type": item["issue_type"],
                        "detail": item["detail"],
                        "coco_recovery_possible": "",
                        "coco_contrast_detail": "",
                    }
                )
    return polygons, issues, mismatches, dict(parsed_by_filename)


def load_coco_segmentation(path: Path) -> dict[str, object]:
    """Load COCO JSON and expose deterministic indexes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in data.get("categories", [])}
    images = {int(item["id"]): item for item in data.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    by_filename_all: dict[str, list[dict[str, object]]] = defaultdict(list)
    for image_id, image in images.items():
        by_filename_all[str(image["file_name"])].append(
            {
                "image": image,
                "annotations": annotations_by_image.get(image_id, []),
            }
        )
    by_filename = {
        filename: entries[0]
        for filename, entries in by_filename_all.items()
        if len(entries) == 1
    }
    return {
        "path": str(path.resolve()),
        "raw": data,
        "categories": categories,
        "images": images,
        "by_filename": by_filename,
        "by_filename_all": dict(by_filename_all),
        "ambiguous_filenames": sorted(
            filename
            for filename, entries in by_filename_all.items()
            if len(entries) != 1
        ),
        "licenses": data.get("licenses", []),
        "info": data.get("info", {}),
    }


def _coco_segmentation_detail(
    annotation: dict[str, object],
    width: int,
    height: int,
) -> dict[str, object]:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        return {
            "format": "rle",
            "valid": bool(segmentation.get("counts") and segmentation.get("size")),
            "point_count": "",
            "area_ratio": float(annotation.get("area", 0.0)) / max(1, width * height),
            "component_count": 1,
            "topology_valid": False,
            "topology_issues": "rle_not_supported_for_polygon_recovery",
        }
    if not isinstance(segmentation, list) or not segmentation:
        return {
            "format": "missing",
            "valid": False,
            "point_count": 0,
            "area_ratio": 0.0,
            "component_count": 0,
            "topology_valid": False,
            "topology_issues": "missing_segmentation",
        }
    components = (
        segmentation
        if segmentation and isinstance(segmentation[0], list)
        else [segmentation]
    )
    point_count = 0
    total_area = 0.0
    valid = True
    topology_issue_types: set[str] = set()
    for component in components:
        if len(component) < 6 or len(component) % 2:
            valid = False
        points = [
            (float(component[index]) / width, float(component[index + 1]) / height)
            for index in range(0, len(component) - 1, 2)
        ]
        point_count += len(points)
        total_area += polygon_area(points)
        component_topology = polygon_topology_issues(points)
        topology_issue_types.update(
            str(issue["issue_type"]) for issue in component_topology
        )
        valid = valid and not component_topology
    return {
        "format": "polygon",
        "valid": valid,
        "point_count": point_count,
        "area_ratio": total_area,
        "component_count": len(components),
        "topology_valid": not topology_issue_types,
        "topology_issues": ";".join(sorted(topology_issue_types)),
    }


def _normalized_yolo_bbox(
    raw_line: str,
    points: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    tokens = raw_line.strip().split()
    if len(tokens) == 5:
        try:
            values = tuple(float(token) for token in tokens[1:])
        except ValueError:
            return None
        if len(values) != 4 or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in values
        ):
            return None
        return values
    if len(points) < 3:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2,
        max(xs) - min(xs),
        max(ys) - min(ys),
    )


def _normalized_coco_bbox(
    annotation: dict[str, object],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    try:
        left, top, box_width, box_height = map(float, bbox)
    except (TypeError, ValueError):
        return None
    values = (
        (left + box_width / 2) / width,
        (top + box_height / 2) / height,
        box_width / width,
        box_height / height,
    )
    if any(not math.isfinite(value) for value in values):
        return None
    return values


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    def corners(
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        center_x, center_y, width, height = bbox
        return (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        )

    first_left, first_top, first_right, first_bottom = corners(first)
    second_left, second_top, second_right, second_bottom = corners(second)
    intersection_width = max(
        0.0,
        min(first_right, second_right) - max(first_left, second_left),
    )
    intersection_height = max(
        0.0,
        min(first_bottom, second_bottom) - max(first_top, second_top),
    )
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_right - first_left) * max(
        0.0, first_bottom - first_top
    )
    second_area = max(0.0, second_right - second_left) * max(
        0.0, second_bottom - second_top
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_coco_recovery(
    *,
    source: str,
    raw_line: str,
    points: Sequence[tuple[float, float]],
    original_class_id: int,
    original_class_name: str,
    semantic_role: str,
    coco_annotations: Sequence[dict[str, object]],
    coco_categories: dict[int, str],
    width: int,
    height: int,
    image_match_unique: bool = True,
    tolerance: float = BBOX_MATCH_TOLERANCE,
) -> dict[str, object]:
    """Evaluate COCO candidates without relying on annotation order."""
    result: dict[str, object] = {
        "recovery_match_method": "unique_image_class_semantic_bbox_topology",
        "recovery_candidate_count": 0,
        "bbox_max_abs_error": "",
        "bbox_iou": "",
        "class_match": False,
        "semantic_role_match": False,
        "topology_valid": False,
        "recovery_decision": "manual_review",
        "recovery_reason": "",
        "matched_annotation_id": "",
        "matched_annotation": None,
    }
    if not image_match_unique:
        result["recovery_reason"] = "La imagen no es única en el índice COCO"
        return result
    yolo_bbox = _normalized_yolo_bbox(raw_line, points)
    if yolo_bbox is None:
        result["recovery_reason"] = "No se pudo obtener un bbox YOLO válido"
        return result

    compatible: list[dict[str, object]] = []
    all_class_matches: list[bool] = []
    all_semantic_matches: list[bool] = []
    all_topology_valid: list[bool] = []
    for annotation in coco_annotations:
        category_id = int(annotation.get("category_id", -1))
        category_name = coco_categories.get(category_id, "unknown")
        role_entry = SEMANTIC_ROLES.get(source, {}).get(original_class_id)
        class_match = bool(
            role_entry
            and str(role_entry[0]) == original_class_name
            and category_name == original_class_name
        )
        coco_semantic_role = (
            str(role_entry[1])
            if role_entry and str(role_entry[0]) == category_name
            else "unknown"
        )
        semantic_match = coco_semantic_role == semantic_role
        detail = _coco_segmentation_detail(annotation, width, height)
        topology_valid = bool(detail["valid"]) and bool(detail["topology_valid"])
        coco_bbox = _normalized_coco_bbox(annotation, width, height)
        if coco_bbox is None:
            continue
        max_error = max(
            abs(left - right) for left, right in zip(yolo_bbox, coco_bbox)
        )
        iou = _bbox_iou(yolo_bbox, coco_bbox)
        all_class_matches.append(class_match)
        all_semantic_matches.append(semantic_match)
        all_topology_valid.append(topology_valid)
        if class_match and semantic_match and topology_valid and max_error <= tolerance:
            compatible.append(
                {
                    "annotation": annotation,
                    "bbox_max_abs_error": max_error,
                    "bbox_iou": iou,
                }
            )

    result["recovery_candidate_count"] = len(compatible)
    result["class_match"] = any(all_class_matches)
    result["semantic_role_match"] = any(all_semantic_matches)
    result["topology_valid"] = any(all_topology_valid)
    if len(compatible) != 1:
        result["recovery_reason"] = (
            "No existe candidato COCO compatible"
            if not compatible
            else "Existen múltiples candidatos COCO compatibles"
        )
        return result

    match = compatible[0]
    annotation = match["annotation"]
    result.update(
        {
            "bbox_max_abs_error": match["bbox_max_abs_error"],
            "bbox_iou": match["bbox_iou"],
            "class_match": True,
            "semantic_role_match": True,
            "topology_valid": True,
            "recovery_decision": "recover_from_coco",
            "recovery_reason": (
                f"Único candidato con error bbox <= {tolerance:g}, clase, "
                "semántica y topología compatibles"
            ),
            "matched_annotation_id": annotation.get("id", ""),
            "matched_annotation": annotation,
        }
    )
    return result


def compare_yolo_coco(
    inventory: DatasetInventory,
    parsed_by_filename: dict[str, list[dict[str, object]]],
    issues: list[dict[str, object]],
    coco: dict[str, object],
) -> list[dict[str, object]]:
    """Compare image-level counts, classes, geometry and repair evidence."""
    coco_by_filename_all = coco["by_filename_all"]
    categories = coco["categories"]
    yolo_names = inventory.class_names
    rows: list[dict[str, object]] = []
    all_filenames = sorted(set(parsed_by_filename) | set(coco_by_filename_all))
    details_by_filename: dict[str, list[dict[str, object]]] = {}
    for filename in all_filenames:
        yolo_instances = parsed_by_filename.get(filename, [])
        coco_entries = coco_by_filename_all.get(filename, [])
        coco_entry = coco_entries[0] if len(coco_entries) == 1 else None
        coco_annotations = coco_entry["annotations"] if coco_entry else []
        image = coco_entry["image"] if coco_entry else {}
        width = int(image.get("width", 0) or 0)
        height = int(image.get("height", 0) or 0)
        coco_details = [
            {
                **_coco_segmentation_detail(annotation, width, height),
                "class_id": annotation.get("category_id", ""),
                "class_name": categories.get(
                    int(annotation.get("category_id", -1)), "unknown"
                ),
            }
            for annotation in coco_annotations
        ]
        details_by_filename[filename] = coco_details
        yolo_class_names = Counter(
            yolo_names.get(int(item["class_id"]), "unknown")
            for item in yolo_instances
            if item["class_id"] != ""
        )
        coco_class_names = Counter(item["class_name"] for item in coco_details)
        yolo_valid = sum(bool(item["valid"]) for item in yolo_instances)
        coco_valid = sum(bool(item["valid"]) for item in coco_details)
        rows.append(
            {
                "source": inventory.source,
                "filename": filename,
                "in_yolo": filename in parsed_by_filename,
                "in_coco": filename in coco_by_filename_all,
                "coco_image_match_unique": len(coco_entries) == 1,
                "yolo_annotation_count": len(yolo_instances),
                "coco_annotation_count": len(coco_annotations),
                "annotation_count_delta": len(yolo_instances) - len(coco_annotations),
                "yolo_valid_count": yolo_valid,
                "coco_valid_count": coco_valid,
                "yolo_classes": json.dumps(dict(sorted(yolo_class_names.items()))),
                "coco_classes": json.dumps(dict(sorted(coco_class_names.items()))),
                "class_names_match": yolo_class_names == coco_class_names,
                "yolo_point_count": sum(int(item["point_count"]) for item in yolo_instances),
                "coco_point_count": sum(
                    int(item["point_count"])
                    for item in coco_details
                    if item["point_count"] != ""
                ),
                "yolo_area_sum": sum(
                    float(item["polygon_area_ratio"]) for item in yolo_instances
                ),
                "coco_area_sum": sum(float(item["area_ratio"]) for item in coco_details),
                "coco_rle_annotations": sum(
                    item["format"] == "rle" for item in coco_details
                ),
            }
        )

    recovery_by_line: dict[tuple[str, int], dict[str, object]] = {}
    for issue in issues:
        if issue["issue_type"] == "empty_label_file":
            filename = str(issue["filename"])
            entries = coco_by_filename_all.get(filename, [])
            details = details_by_filename.get(filename, [])
            issue["coco_recovery_possible"] = False
            issue["coco_contrast_detail"] = (
                f"COCO contiene {len(details)} anotaciones en {len(entries)} imagen(es); "
                "un TXT vacío no se recupera sin correspondencia por instancia"
            )
            issue["recovery_match_method"] = "not_applicable_empty_label"
            issue["recovery_candidate_count"] = 0
            issue["recovery_decision"] = "manual_review"
            issue["recovery_reason"] = (
                "No existe una anotación YOLO individual que pueda contrastarse"
            )
            continue
        line_number = issue.get("line_number")
        if not str(line_number).isdigit():
            issue["coco_recovery_possible"] = False
            issue["recovery_decision"] = "manual_review"
            issue["recovery_reason"] = "La incidencia no identifica una línea YOLO"
            continue
        filename = str(issue["filename"])
        key = (filename, int(line_number))
        recovery = recovery_by_line.get(key)
        if recovery is None:
            yolo_instance = next(
                (
                    row
                    for row in parsed_by_filename.get(filename, [])
                    if int(row["line_number"]) == int(line_number)
                ),
                None,
            )
            entries = coco_by_filename_all.get(filename, [])
            entry = entries[0] if len(entries) == 1 else None
            if yolo_instance is None:
                recovery = {
                    "recovery_match_method": "unavailable",
                    "recovery_candidate_count": 0,
                    "bbox_max_abs_error": "",
                    "bbox_iou": "",
                    "class_match": False,
                    "semantic_role_match": False,
                    "topology_valid": False,
                    "recovery_decision": "manual_review",
                    "recovery_reason": "No se encontró la instancia YOLO",
                }
            else:
                class_id = int(yolo_instance["class_id"])
                role_entry = SEMANTIC_ROLES.get(inventory.source, {}).get(class_id)
                semantic_role = str(role_entry[1]) if role_entry else "unknown"
                recovery = evaluate_coco_recovery(
                    source=inventory.source,
                    raw_line=str(yolo_instance["raw_line"]),
                    points=yolo_instance["points"],
                    original_class_id=class_id,
                    original_class_name=str(yolo_instance["class_name"]),
                    semantic_role=semantic_role,
                    coco_annotations=entry["annotations"] if entry else [],
                    coco_categories=categories,
                    width=int(entry["image"].get("width", 0)) if entry else 0,
                    height=int(entry["image"].get("height", 0)) if entry else 0,
                    image_match_unique=len(entries) == 1,
                )
            recovery_by_line[key] = recovery

        issue.update(
            {
                column: recovery.get(column, "")
                for column in (
                    "recovery_match_method",
                    "recovery_candidate_count",
                    "bbox_max_abs_error",
                    "bbox_iou",
                    "class_match",
                    "semantic_role_match",
                    "topology_valid",
                    "recovery_decision",
                    "recovery_reason",
                )
            }
        )
        issue["coco_recovery_possible"] = (
            recovery.get("recovery_decision") == "recover_from_coco"
        )
        issue["coco_contrast_detail"] = (
            f"método={recovery.get('recovery_match_method')}; "
            f"candidatos={recovery.get('recovery_candidate_count')}; "
            f"bbox_error={recovery.get('bbox_max_abs_error')}; "
            f"bbox_iou={recovery.get('bbox_iou')}; "
            f"decisión={recovery.get('recovery_decision')}"
        )
    return rows


def find_exact_duplicates(
    image_rows: Sequence[dict[str, object]],
    pilot_images: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Find exact duplicates within candidates, across them and against the pilot."""
    entries: list[dict[str, str]] = [
        {
            "collection": str(row["source"]),
            "filename": str(row["filename"]),
            "path": str(row["image_path"]),
            "sha256": str(row["sha256"]),
        }
        for row in image_rows
        if row.get("valid_image")
    ]
    for path in sorted(pilot_images.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            entries.append(
                {
                    "collection": "retained_pilot",
                    "filename": path.name,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
            )
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        by_hash[entry["sha256"]].append(entry)

    rows: list[dict[str, object]] = []
    matrix: Counter[tuple[str, str]] = Counter()
    internal_groups: Counter[str] = Counter()
    pilot_crosses: Counter[str] = Counter()
    group_number = 0
    for digest, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        group_number += 1
        collections = sorted({entry["collection"] for entry in group})
        for collection in collections:
            count = sum(entry["collection"] == collection for entry in group)
            if count > 1:
                internal_groups[collection] += 1
        for left_index, left in enumerate(collections):
            for right in collections[left_index + 1 :]:
                matrix[(left, right)] += 1
                if "retained_pilot" in (left, right):
                    candidate = right if left == "retained_pilot" else left
                    pilot_crosses[candidate] += 1
        for entry in group:
            rows.append(
                {
                    "group_id": f"duplicate_{group_number:04d}",
                    "sha256": digest,
                    "collection": entry["collection"],
                    "filename": entry["filename"],
                    "path": entry["path"],
                    "group_size": len(group),
                    "collections_in_group": ";".join(collections),
                    "includes_pilot": "retained_pilot" in collections,
                }
            )
    summary = {
        "duplicate_groups": group_number,
        "internal_group_counts": dict(sorted(internal_groups.items())),
        "cross_source_group_counts": {
            f"{left}_vs_{right}": count
            for (left, right), count in sorted(matrix.items())
        },
        "pilot_cross_group_counts": dict(sorted(pilot_crosses.items())),
        "pilot_leakage_detected": bool(pilot_crosses),
    }
    return rows, summary


def deterministic_sample(
    rows: Sequence[dict[str, object]],
    count: int,
    seed: int = SEED,
    key: str = "filename",
) -> list[dict[str, object]]:
    """Return a stable random sample after sorting the population."""
    ordered = sorted(rows, key=lambda row: str(row.get(key, "")))
    if len(ordered) <= count:
        return list(ordered)
    return random.Random(seed).sample(ordered, count)


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    columns: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = tuple(
            dict.fromkeys(key for row in rows for key in row)
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: int, total: int) -> float:
    return 100.0 * value / total if total else 0.0


def _median_numeric(rows: Sequence[dict[str, object]], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) not in ("", None) and math.isfinite(float(row[key]))
    ]
    return median(values) if values else 0.0


def _source_decision(
    source: str,
    source_images: Sequence[dict[str, object]],
    source_polygons: Sequence[dict[str, object]],
    source_issues: Sequence[dict[str, object]],
    source_mismatches: Sequence[dict[str, object]],
    license_name: str,
    duplicate_summary: dict[str, object],
) -> dict[str, object]:
    valid = [row for row in source_polygons if row["valid"]]
    leaf_ids = {
        class_id
        for class_id, (_, role, candidate, _) in SEMANTIC_ROLES[source].items()
        if role == "full_leaf" and candidate
    }
    leaf_valid = [
        row for row in valid if row["class_id"] != "" and int(row["class_id"]) in leaf_ids
    ]
    leaf_images = {str(row["filename"]) for row in leaf_valid}
    invalid_lines = {
        (str(row["label_path"]), str(row["line_number"]))
        for row in source_issues
        if row["issue_type"] != "empty_label_file"
    }
    empty_labels = sum(row["issue_type"] == "empty_label_file" for row in source_issues)
    pilot_crosses = int(
        duplicate_summary.get("pilot_cross_group_counts", {}).get(source, 0)
    )
    if pilot_crosses:
        status = "needs_repair"
        reason = "Debe excluir duplicados exactos contra el piloto retenido"
    elif not leaf_valid:
        status = "rejected"
        reason = "No contiene polígonos válidos identificables como hoja completa"
    elif invalid_lines or empty_labels or len(leaf_images) < len(source_images):
        status = "accepted_with_filtering"
        reason = (
            "Conservar sólo la clase leaf válida, excluir lesiones y casos sin hoja válida"
        )
    else:
        status = "accepted_with_filtering"
        reason = "Conservar sólo la clase leaf y remapearla a maize_leaf"
    return {
        "source": source,
        "status": status,
        "reason": reason,
        "license": license_name,
        "images": len(source_images),
        "valid_leaf_polygons": len(leaf_valid),
        "candidate_images_with_valid_leaf": len(leaf_images),
        "invalid_annotation_lines": len(invalid_lines),
        "empty_label_files": empty_labels,
        "image_label_mismatches": len(source_mismatches),
        "pilot_duplicate_groups": pilot_crosses,
        "manual_review_required": True,
        "ready_to_consolidate": False,
    }


def _class_summary(
    inventories: dict[str, DatasetInventory],
    polygons: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source, inventory in inventories.items():
        source_rows = [row for row in polygons if row["source"] == source]
        for class_id, class_name in sorted(inventory.class_names.items()):
            class_rows = [
                row
                for row in source_rows
                if row["class_id"] != "" and int(row["class_id"]) == class_id
            ]
            role = SEMANTIC_ROLES.get(source, {}).get(
                class_id, (class_name, "unknown", False, "review")
            )
            rows.append(
                {
                    "source": source,
                    "class_id": class_id,
                    "class_name": class_name,
                    "semantic_role": role[1],
                    "candidate_for_maize_leaf": role[2],
                    "decision": role[3],
                    "reason": (
                        "Representa el contorno de la hoja completa"
                        if role[1] == "full_leaf"
                        else "Representa síntomas o lesiones, no la hoja completa"
                    ),
                    "polygon_count": len(class_rows),
                    "valid_polygon_count": sum(bool(row["valid"]) for row in class_rows),
                    "image_count": len({str(row["filename"]) for row in class_rows}),
                }
            )
    return rows


def _source_summary(
    inventories: dict[str, DatasetInventory],
    image_rows: Sequence[dict[str, object]],
    polygons: Sequence[dict[str, object]],
    issues: Sequence[dict[str, object]],
    mismatches: Sequence[dict[str, object]],
    coco_data: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source, inventory in inventories.items():
        source_images = [row for row in image_rows if row["source"] == source]
        source_polygons = [row for row in polygons if row["source"] == source]
        source_issues = [row for row in issues if row["source"] == source]
        source_mismatches = [row for row in mismatches if row["source"] == source]
        image_stems = {path.stem.casefold() for path in inventory.images}
        label_stems = {path.stem.casefold() for path in inventory.labels}
        coco_raw = coco_data[source]["raw"]
        rows.append(
            {
                "source": source,
                "yolo_images": len(inventory.images),
                "yolo_label_files": len(inventory.labels),
                "yolo_polygon_lines": len(source_polygons),
                "valid_yolo_polygons": sum(bool(row["valid"]) for row in source_polygons),
                "invalid_yolo_polygons": sum(not bool(row["valid"]) for row in source_polygons),
                "empty_label_files": sum(
                    row["issue_type"] == "empty_label_file" for row in source_issues
                ),
                "images_without_label": len(image_stems - label_stems),
                "labels_without_image": len(label_stems - image_stems),
                "unrecognized_files": sum(
                    path.is_file()
                    and path.suffix.lower() not in IMAGE_EXTENSIONS | {".txt", ".yaml"}
                    for path in inventory.yolo_root.rglob("*")
                ),
                "coco_images": len(coco_raw.get("images", [])),
                "coco_annotations": len(coco_raw.get("annotations", [])),
                "classes": len(inventory.class_names),
                "license": inventory.license_name,
                "image_extensions": ";".join(
                    sorted({path.suffix.lower() for path in inventory.images})
                ),
                "orientation_horizontal": sum(
                    row.get("orientation") == "horizontal" for row in source_images
                ),
                "orientation_vertical": sum(
                    row.get("orientation") == "vertical" for row in source_images
                ),
                "orientation_square": sum(
                    row.get("orientation") == "square" for row in source_images
                ),
                "issue_records": len(source_issues),
                "mismatch_records": len(source_mismatches),
            }
        )
    return rows


def _instance_summary(
    source: str,
    image_rows: Sequence[dict[str, object]],
    polygons: Sequence[dict[str, object]],
) -> dict[str, object]:
    counts = Counter(str(row["filename"]) for row in polygons)
    all_counts = [counts.get(str(image["filename"]), 0) for image in image_rows]
    leaf_names = {
        name
        for _, (name, role, candidate, _) in SEMANTIC_ROLES[source].items()
        if role == "full_leaf" and candidate
    }
    lesion_names = {
        name
        for _, (name, role, _, _) in SEMANTIC_ROLES[source].items()
        if role == "lesion"
    }
    leaf_counts = Counter(
        str(row["filename"]) for row in polygons if row["class_name"] in leaf_names
    )
    lesion_counts = Counter(
        str(row["filename"]) for row in polygons if row["class_name"] in lesion_names
    )
    return {
        "images_with_0_polygons": sum(value == 0 for value in all_counts),
        "images_with_1_polygon": sum(value == 1 for value in all_counts),
        "images_with_2_polygons": sum(value == 2 for value in all_counts),
        "images_with_3_or_more_polygons": sum(value >= 3 for value in all_counts),
        "max_instances": max(all_counts, default=0),
        "mean_instances": mean(all_counts) if all_counts else 0.0,
        "median_instances": median(all_counts) if all_counts else 0.0,
        "images_with_full_leaf": sum(
            leaf_counts.get(str(image["filename"]), 0) > 0 for image in image_rows
        ),
        "images_with_lesions": sum(
            lesion_counts.get(str(image["filename"]), 0) > 0 for image in image_rows
        ),
    }


def _class_instance_summary(
    inventories: dict[str, DatasetInventory],
    image_rows: Sequence[dict[str, object]],
    polygons: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source, inventory in inventories.items():
        source_images = [row for row in image_rows if row["source"] == source]
        for class_id, class_name in sorted(inventory.class_names.items()):
            class_polygons = [
                row
                for row in polygons
                if row["source"] == source
                and row["valid"]
                and row["class_id"] != ""
                and int(row["class_id"]) == class_id
            ]
            counts = Counter(str(row["filename"]) for row in class_polygons)
            all_counts = [
                counts.get(str(image["filename"]), 0) for image in source_images
            ]
            role = SEMANTIC_ROLES.get(source, {}).get(
                class_id,
                (class_name, "unknown", False, "review"),
            )
            rows.append(
                {
                    "source": source,
                    "class_id": class_id,
                    "class_name": class_name,
                    "semantic_role": role[1],
                    "valid_instances": len(class_polygons),
                    "images_with_0_instances": sum(value == 0 for value in all_counts),
                    "images_with_1_instance": sum(value == 1 for value in all_counts),
                    "images_with_2_instances": sum(value == 2 for value in all_counts),
                    "images_with_3_or_more_instances": sum(
                        value >= 3 for value in all_counts
                    ),
                    "max_instances": max(all_counts, default=0),
                    "mean_instances": mean(all_counts) if all_counts else 0.0,
                    "median_instances": median(all_counts) if all_counts else 0.0,
                }
            )
    return rows


def _manual_review_rows(
    class_rows: Sequence[dict[str, object]],
    parsed: dict[str, dict[str, list[dict[str, object]]]],
    count_per_class: int = 8,
) -> list[dict[str, object]]:
    reviews: list[dict[str, object]] = []
    for class_row in class_rows:
        source = str(class_row["source"])
        class_id = int(class_row["class_id"])
        candidates = [
            {"filename": filename, "source": source}
            for filename, items in parsed[source].items()
            if any(
                item["class_id"] != "" and int(item["class_id"]) == class_id
                for item in items
            )
        ]
        for item in deterministic_sample(
            candidates,
            count_per_class,
            SEED + class_id,
        ):
            reviews.append(
                {
                    "source": source,
                    "filename": item["filename"],
                    "class_id": class_id,
                    "class_name": class_row["class_name"],
                    "represents_full_leaf": (
                        "yes"
                        if class_row["semantic_role"] == "full_leaf"
                        else "no"
                    ),
                    "represents_lesion": (
                        "yes" if class_row["semantic_role"] == "lesion" else "no"
                    ),
                    "multiple_leaves": "unknown",
                    "annotation_quality": "unknown",
                    "background_complexity": "unknown",
                    "recommended_action": class_row["decision"],
                    "notes": "Revisión visual estratificada pendiente",
                }
            )
    return reviews


def _color_for_class(source: str, class_id: int) -> tuple[int, int, int]:
    palette = (
        (230, 57, 70),
        (29, 185, 84),
        (69, 123, 157),
        (255, 183, 3),
        (131, 56, 236),
    )
    offset = int(hashlib.sha256(source.encode()).hexdigest()[:2], 16)
    return palette[(class_id + offset) % len(palette)]


def render_segmentation_preview(
    image_path: Path,
    instances: Sequence[dict[str, object]],
    destination: Path,
    *,
    source: str,
) -> None:
    """Render normalized polygons with class labels and readable metadata."""
    with Image.open(image_path) as image:
        canvas = ImageOps.exif_transpose(image).convert("RGB")
    max_side = 900
    scale = min(1.0, max_side / max(canvas.size))
    if scale < 1:
        canvas = canvas.resize(
            (round(canvas.width * scale), round(canvas.height * scale)),
            Image.Resampling.LANCZOS,
        )
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()
    for instance in instances:
        points = [
            (round(x * canvas.width), round(y * canvas.height))
            for x, y in instance.get("points", [])
        ]
        if len(points) < 2:
            continue
        class_id = int(instance["class_id"]) if instance["class_id"] != "" else -1
        color = _color_for_class(source, class_id)
        if len(points) >= 3:
            draw.polygon(points, fill=(*color, 55), outline=(*color, 255), width=3)
        else:
            draw.line(points, fill=(*color, 255), width=3)
        anchor = points[0]
        label = (
            f"{instance['class_id']}:{instance['class_name']} "
            f"area={float(instance['polygon_area_ratio']):.3f} "
            f"pts={instance['point_count']}"
        )
        box = draw.textbbox(anchor, label, font=font)
        draw.rectangle(box, fill=(0, 0, 0, 190))
        draw.text(anchor, label, fill="white", font=font)
    caption_height = 42
    result = Image.new("RGB", (canvas.width, canvas.height + caption_height), "white")
    result.paste(canvas, (0, 0))
    caption = ImageDraw.Draw(result)
    caption.text(
        (8, canvas.height + 5),
        f"{source} | {image_path.name} | instances={len(instances)}",
        fill="black",
        font=font,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination, quality=90)


def _make_montage(images: Sequence[Path], destination: Path, title: str) -> None:
    if not images:
        return
    thumbs: list[Image.Image] = []
    for path in images:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((360, 300), Image.Resampling.LANCZOS)
            panel = Image.new("RGB", (380, 340), "white")
            panel.paste(
                thumb,
                ((panel.width - thumb.width) // 2, 24),
            )
            ImageDraw.Draw(panel).text((8, 6), path.stem[:58], fill="black")
            thumbs.append(panel)
    columns = min(3, len(thumbs))
    rows = math.ceil(len(thumbs) / columns)
    montage = Image.new("RGB", (columns * 380, rows * 340 + 40), "#f4f4f4")
    ImageDraw.Draw(montage).text((10, 10), title, fill="black")
    for index, panel in enumerate(thumbs):
        x = (index % columns) * 380
        y = 40 + (index // columns) * 340
        montage.paste(panel, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    montage.save(destination, quality=90)


def _bar_chart(
    data: Sequence[tuple[str, float]],
    title: str,
    destination: Path,
    *,
    width: int = 1000,
    height: int = 600,
) -> None:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((30, 20), title, fill="black", font=font)
    if not data:
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
        return
    left, top, right, bottom = 90, 70, width - 30, height - 130
    maximum = max(value for _, value in data) or 1.0
    gap = 12
    bar_width = max(10, (right - left - gap * (len(data) - 1)) // len(data))
    for index, (label, value) in enumerate(data):
        x1 = left + index * (bar_width + gap)
        x2 = x1 + bar_width
        y2 = bottom
        y1 = bottom - (bottom - top) * value / maximum
        color = _color_for_class(title, index)
        draw.rectangle((x1, y1, x2, y2), fill=color)
        draw.text((x1, max(top, y1 - 18)), f"{value:.2f}", fill="black", font=font)
        display = label if len(label) <= 18 else label[:16] + "…"
        draw.text((x1, bottom + 8), display, fill="black", font=font)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _histogram(
    values: Sequence[float],
    title: str,
    destination: Path,
    bins: int = 20,
) -> None:
    if not values:
        _bar_chart([], title, destination)
        return
    minimum, maximum = min(values), max(values)
    if math.isclose(minimum, maximum):
        counts = [len(values)]
        labels = [f"{minimum:.2f}"]
    else:
        step = (maximum - minimum) / bins
        counts = [0] * bins
        for value in values:
            index = min(bins - 1, int((value - minimum) / step))
            counts[index] += 1
        labels = [
            f"{minimum + index * step:.2f}" if index % max(1, bins // 5) == 0 else ""
            for index in range(bins)
        ]
    _bar_chart(list(zip(labels, map(float, counts))), title, destination)


def _scatter(
    rows: Sequence[dict[str, object]],
    x_key: str,
    y_key: str,
    title: str,
    destination: Path,
) -> None:
    width, height = 900, 650
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 20), title, fill="black")
    usable = [
        row
        for row in rows
        if row.get(x_key) not in ("", None) and row.get(y_key) not in ("", None)
    ]
    if usable:
        xs = [float(row[x_key]) for row in usable]
        ys = [float(row[y_key]) for row in usable]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        for row, x, y in zip(usable, xs, ys):
            px = 70 + 760 * (x - xmin) / max(1e-12, xmax - xmin)
            py = 570 - 500 * (y - ymin) / max(1e-12, ymax - ymin)
            source = str(row.get("source", ""))
            color = _color_for_class(source, 0)
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    draw.line((70, 570, 830, 570), fill="black", width=2)
    draw.line((70, 70, 70, 570), fill="black", width=2)
    draw.text((400, 600), x_key, fill="black")
    draw.text((10, 300), y_key, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _generate_charts(
    source_summary: Sequence[dict[str, object]],
    class_summary: Sequence[dict[str, object]],
    image_rows: Sequence[dict[str, object]],
    polygons: Sequence[dict[str, object]],
    output_dir: Path,
    public_dir: Path | None,
) -> list[str]:
    charts = output_dir / "charts"
    paths: list[Path] = []

    inventory_data: list[tuple[str, float]] = []
    for row in source_summary:
        source = str(row["source"])
        inventory_data.extend(
            [
                (f"{source[:10]} images", float(row["yolo_images"])),
                (f"{source[:10]} labels", float(row["yolo_label_files"])),
            ]
        )
    paths.append(charts / "inventory_counts.png")
    _bar_chart(inventory_data, "Imágenes y etiquetas por fuente", paths[-1])
    paths.append(charts / "inventory_differences.png")
    _bar_chart(
        [
            (
                str(row["source"])[:18],
                abs(float(row["yolo_images"]) - float(row["yolo_label_files"])),
            )
            for row in source_summary
        ],
        "Diferencia absoluta entre imágenes y etiquetas",
        paths[-1],
    )

    paths.append(charts / "class_polygon_counts.png")
    _bar_chart(
        [
            (
                f"{str(row['source'])[:8]}:{row['class_name']}",
                float(row["polygon_count"]),
            )
            for row in class_summary
        ],
        "Polígonos por clase",
        paths[-1],
    )

    valid_images = [row for row in image_rows if row.get("valid_image")]
    for key, title, filename in (
        ("width", "Histograma de ancho", "image_width_histogram.png"),
        ("height", "Histograma de alto", "image_height_histogram.png"),
        ("aspect_ratio", "Histograma de relación de aspecto", "aspect_ratio_histogram.png"),
    ):
        paths.append(charts / filename)
        _histogram([float(row[key]) for row in valid_images], title, paths[-1])
    paths.append(charts / "width_vs_height.png")
    _scatter(valid_images, "width", "height", "Ancho frente a alto", paths[-1])

    orientation_counts = Counter(str(row["orientation"]) for row in valid_images)
    paths.append(charts / "orientation_distribution.png")
    _bar_chart(
        [(key, float(value)) for key, value in sorted(orientation_counts.items())],
        "Orientación de imágenes",
        paths[-1],
    )

    valid_polygons = [row for row in polygons if row["valid"]]
    instance_counts = Counter(
        (str(row["source"]), str(row["filename"])) for row in valid_polygons
    )
    paths.append(charts / "instances_per_image.png")
    _histogram(
        [float(value) for value in instance_counts.values()],
        "Instancias válidas por imagen anotada",
        paths[-1],
    )
    paths.append(charts / "polygon_point_count.png")
    _histogram(
        [float(row["point_count"]) for row in valid_polygons],
        "Puntos por polígono",
        paths[-1],
    )
    paths.append(charts / "polygon_point_count_log.png")
    _histogram(
        [math.log10(max(1.0, float(row["point_count"]))) for row in valid_polygons],
        "log10 puntos por polígono",
        paths[-1],
    )
    paths.append(charts / "polygon_area_distribution.png")
    _histogram(
        [float(row["polygon_area_ratio"]) for row in valid_polygons],
        "Área relativa de máscaras",
        paths[-1],
    )
    area_by_source = defaultdict(list)
    area_by_class = defaultdict(list)
    for row in valid_polygons:
        area = float(row["polygon_area_ratio"])
        source = str(row["source"])
        area_by_source[source].append(area)
        area_by_class[(source, str(row["class_name"]))].append(area)
    paths.append(charts / "polygon_area_by_source.png")
    _bar_chart(
        [
            (source[:18], float(median(values)))
            for source, values in sorted(area_by_source.items())
        ],
        "Mediana del área relativa por fuente",
        paths[-1],
    )
    paths.append(charts / "polygon_area_by_class.png")
    _bar_chart(
        [
            (f"{source[:7]}:{class_name}", float(median(values)))
            for (source, class_name), values in sorted(area_by_class.items())
        ],
        "Mediana del área relativa por clase",
        paths[-1],
    )
    paths.append(charts / "bbox_vs_polygon_area.png")
    _scatter(
        valid_polygons,
        "bbox_area_ratio",
        "polygon_area_ratio",
        "Área bbox frente a área del polígono",
        paths[-1],
    )
    paths.append(charts / "mask_centroids.png")
    _scatter(
        valid_polygons,
        "centroid_x",
        "centroid_y",
        "Centroides normalizados",
        paths[-1],
    )
    border_counts = Counter(
        (
            str(row["source"]),
            bool(row["touches_any_border"]),
        )
        for row in valid_polygons
    )
    border_data = []
    for source in sorted({str(row["source"]) for row in valid_polygons}):
        total = border_counts[(source, True)] + border_counts[(source, False)]
        border_data.append(
            (source[:18], _percent(border_counts[(source, True)], total))
        )
    paths.append(charts / "border_touching_percent.png")
    _bar_chart(border_data, "Porcentaje de máscaras que tocan bordes", paths[-1])

    size_counts = Counter(str(row["size_category"]) for row in valid_polygons)
    paths.append(charts / "mask_size_categories.png")
    _bar_chart(
        [(key, float(value)) for key, value in sorted(size_counts.items())],
        "Máscaras pequeñas, medianas y grandes",
        paths[-1],
    )

    if public_dir:
        public_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, public_dir / path.name)
    return [str(path) for path in paths]


def _generate_previews(
    inventories: dict[str, DatasetInventory],
    parsed: dict[str, dict[str, list[dict[str, object]]]],
    polygons: Sequence[dict[str, object]],
    issues: Sequence[dict[str, object]],
    duplicate_rows: Sequence[dict[str, object]],
    output_dir: Path,
) -> list[str]:
    preview_root = output_dir / "previews"
    generated: list[str] = []
    image_lookup = {
        (source, path.name): path
        for source, inventory in inventories.items()
        for path in inventory.images
    }

    def render_category(
        category: str,
        source: str,
        filenames: Sequence[str],
        title: str,
        group_slug: str | None = None,
    ) -> None:
        rendered: list[Path] = []
        for index, filename in enumerate(filenames, start=1):
            image_path = image_lookup.get((source, filename))
            if image_path is None:
                continue
            destination = (
                preview_root
                / category
                / f"{source}_{index:02d}_{Path(filename).stem}.jpg"
            )
            render_segmentation_preview(
                image_path,
                parsed[source].get(filename, []),
                destination,
                source=source,
            )
            rendered.append(destination)
            generated.append(str(destination))
        if rendered:
            montage_key = source if group_slug is None else f"{source}_{group_slug}"
            montage = preview_root / category / f"{montage_key}_montage.jpg"
            _make_montage(rendered, montage, title)
            generated.append(str(montage))

    for source, inventory in inventories.items():
        filenames = [{"filename": path.name} for path in inventory.images]
        random_names = [
            str(row["filename"])
            for row in deterministic_sample(filenames, 12, SEED)
        ]
        render_category("random_samples", source, random_names, f"{source}: muestra aleatoria")

        source_polygons = [
            row for row in polygons if row["source"] == source and row["valid"]
        ]
        counts = Counter(str(row["filename"]) for row in source_polygons)
        multi = [
            {"filename": filename, "count": count}
            for filename, count in counts.items()
            if count > 1
        ]
        multi.sort(key=lambda row: (-int(row["count"]), str(row["filename"])))
        render_category(
            "multiple_instances",
            source,
            [str(row["filename"]) for row in multi[:12]],
            f"{source}: múltiples instancias",
        )

        border = sorted(
            [
                row
                for row in source_polygons
                if bool(row["touches_any_border"])
            ],
            key=lambda row: (-float(row["polygon_area_ratio"]), str(row["filename"])),
        )
        render_category(
            "border_touching",
            source,
            list(dict.fromkeys(str(row["filename"]) for row in border))[:12],
            f"{source}: máscaras en bordes",
        )
        small = sorted(
            source_polygons,
            key=lambda row: (float(row["polygon_area_ratio"]), str(row["filename"])),
        )
        large = list(reversed(small))
        render_category(
            "small_masks",
            source,
            list(dict.fromkeys(str(row["filename"]) for row in small))[:12],
            f"{source}: máscaras pequeñas",
        )
        render_category(
            "large_masks",
            source,
            list(dict.fromkeys(str(row["filename"]) for row in large))[:12],
            f"{source}: máscaras grandes",
        )
        many_points = sorted(
            source_polygons,
            key=lambda row: (-int(row["point_count"]), str(row["filename"])),
        )
        render_category(
            "many_points",
            source,
            list(dict.fromkeys(str(row["filename"]) for row in many_points))[:12],
            f"{source}: máscaras con más puntos",
        )

        source_issues = [row for row in issues if row["source"] == source]
        render_category(
            "malformed",
            source,
            list(dict.fromkeys(str(row["filename"]) for row in source_issues))[:12],
            f"{source}: etiquetas sospechosas",
        )
        for class_id, class_name in sorted(inventory.class_names.items()):
            class_files = list(
                dict.fromkeys(
                    str(row["filename"])
                    for row in source_polygons
                    if row["class_id"] != "" and int(row["class_id"]) == class_id
                )
            )
            sample = [
                str(row["filename"])
                for row in deterministic_sample(
                    [{"filename": filename} for filename in class_files],
                    8,
                    SEED + class_id,
                )
            ]
            render_category(
                "class_examples",
                source,
                sample,
                f"{source}: clase {class_id} {class_name}",
                group_slug=f"class_{class_id}_{class_name}",
            )

    duplicate_candidates = [
        row
        for row in duplicate_rows
        if row["collection"] in inventories
        and "retained_pilot" not in str(row["collections_in_group"])
    ]
    for source in inventories:
        names = list(
            dict.fromkeys(
                str(row["filename"])
                for row in duplicate_candidates
                if row["collection"] == source
            )
        )[:12]
        render_category(
            "cross_source_duplicates",
            source,
            names,
            f"{source}: duplicados entre fuentes",
        )
    return generated


def _decision_markdown(
    decisions: Sequence[dict[str, object]],
    duplicate_summary: dict[str, object],
    issues: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# Decisión de auditoría de fuentes externas",
        "",
        "Esta auditoría no entrena modelos, no corrige etiquetas y no consolida datasets.",
        "",
        "| Fuente | Estado | Imágenes candidatas con hoja válida | Acción |",
        "|---|---|---:|---|",
    ]
    for decision in decisions:
        lines.append(
            f"| {decision['source']} | {decision['status']} | "
            f"{decision['candidate_images_with_valid_leaf']} | {decision['reason']} |"
        )
    issue_counts = Counter(str(row["issue_type"]) for row in issues)
    lines.extend(
        [
            "",
            "## Problemas registrados",
            "",
        ]
    )
    for issue_type, count in sorted(issue_counts.items()):
        lines.append(f"- `{issue_type}`: {count}")
    lines.extend(
        [
            "",
            "## Duplicados",
            "",
            f"- Grupos exactos totales: {duplicate_summary['duplicate_groups']}.",
            f"- Fuga con piloto: {duplicate_summary['pilot_leakage_detected']}.",
            "",
            "## Condición metodológica",
            "",
            "Las clases de lesión deben excluirse del futuro segmentador de hoja. "
            "Toda clase de hoja aceptada deberá remapearse a `0 = maize_leaf` sólo "
            "durante una fase posterior y trazable de consolidación. El clasificador "
            "deberá entrenarse con el mismo preprocesamiento de segmentación; no es "
            "seguro aplicar segmentación únicamente durante inferencia a checkpoints "
            "entrenados con imágenes completas.",
            "",
        ]
    )
    return "\n".join(lines)


def run_external_segmentation_audit(
    project_root: Path,
    *,
    external_root: Path | None = None,
    pilot_root: Path | None = None,
    output_dir: Path | None = None,
    public_dir: Path | None = None,
    seed: int = SEED,
) -> dict[str, object]:
    """Run the complete read-only audit and write only derived outputs."""
    project_root = project_root.resolve()
    external_root = (
        external_root
        or project_root / "data" / "leaf_detection" / "external_sources"
    ).resolve()
    pilot_root = (
        pilot_root or project_root / "data" / "leaf_detection" / "pilot"
    ).resolve()
    output_dir = (
        output_dir
        or project_root / "outputs" / "leaf_detection" / "external_sources_eda"
    ).resolve()
    public_dir = (
        public_dir
        or project_root / "public" / "leaf_detection" / "external_sources_eda"
    ).resolve()
    if output_dir.exists():
        raise FileExistsError(f"La salida ya existe y no será sobrescrita: {output_dir}")
    if public_dir.exists():
        raise FileExistsError(
            f"La carpeta pública ya existe y no será sobrescrita: {public_dir}"
        )
    random.seed(seed)
    inventories = discover_dataset_files(external_root)
    input_fingerprint_before = build_audit_input_fingerprint(
        inventories,
        pilot_root,
        seed=seed,
    )

    try:
        output_dir.mkdir(parents=True)
        image_rows: list[dict[str, object]] = []
        polygons: list[dict[str, object]] = []
        issues: list[dict[str, object]] = []
        mismatches: list[dict[str, object]] = []
        parsed: dict[str, dict[str, list[dict[str, object]]]] = {}
        coco_data: dict[str, dict[str, object]] = {}
        comparisons: list[dict[str, object]] = []
        for source, inventory in inventories.items():
            source_images = compute_image_statistics(inventory)
            source_polygons, source_issues, source_mismatches, source_parsed = (
                load_yolo_dataset(inventory, source_images)
            )
            coco = load_coco_segmentation(inventory.coco_json)
            comparison = compare_yolo_coco(
                inventory,
                source_parsed,
                source_issues,
                coco,
            )
            image_rows.extend(source_images)
            polygons.extend(source_polygons)
            issues.extend(source_issues)
            mismatches.extend(source_mismatches)
            parsed[source] = source_parsed
            coco_data[source] = coco
            comparisons.extend(comparison)

        duplicate_rows, duplicate_summary = find_exact_duplicates(
            image_rows,
            pilot_root / "images",
        )
        class_rows = _class_summary(inventories, polygons)
        class_instance_rows = _class_instance_summary(
            inventories,
            image_rows,
            polygons,
        )
        source_rows = _source_summary(
            inventories,
            image_rows,
            polygons,
            issues,
            mismatches,
            coco_data,
        )
        decisions = []
        for source, inventory in inventories.items():
            decisions.append(
                _source_decision(
                    source,
                    [row for row in image_rows if row["source"] == source],
                    [row for row in polygons if row["source"] == source],
                    [row for row in issues if row["source"] == source],
                    [row for row in mismatches if row["source"] == source],
                    inventory.license_name,
                    duplicate_summary,
                )
            )
        manual_review = _manual_review_rows(class_rows, parsed)

        valid_polygons = [row for row in polygons if row["valid"]]
        source_comparison = []
        for source, inventory in inventories.items():
            source_images = [row for row in image_rows if row["source"] == source]
            source_polygons = [row for row in polygons if row["source"] == source]
            source_valid = [row for row in source_polygons if row["valid"]]
            source_issues = [row for row in issues if row["source"] == source]
            instance_stats = _instance_summary(source, source_images, source_polygons)
            source_comparison.append(
                {
                    "source": source,
                    "images": len(source_images),
                    "polygons": len(source_polygons),
                    "classes": len(inventory.class_names),
                    "images_without_label": sum(
                        row["source"] == source
                        and row["mismatch_type"] == "image_without_label"
                        for row in mismatches
                    ),
                    "empty_labels": sum(
                        row["issue_type"] == "empty_label_file"
                        for row in source_issues
                    ),
                    "invalid_polygons": sum(not bool(row["valid"]) for row in source_polygons),
                    "valid_polygons": len(source_valid),
                    "median_area": _median_numeric(source_valid, "polygon_area_ratio"),
                    "median_points": _median_numeric(source_valid, "point_count"),
                    "mean_instances_per_image": instance_stats["mean_instances"],
                    "median_instances_per_image": instance_stats["median_instances"],
                    "max_instances": instance_stats["max_instances"],
                    "border_touching_percent": _percent(
                        sum(bool(row["touches_any_border"]) for row in source_valid),
                        len(source_valid),
                    ),
                    "internal_duplicate_groups": duplicate_summary[
                        "internal_group_counts"
                    ].get(source, 0),
                    "pilot_duplicate_groups": duplicate_summary[
                        "pilot_cross_group_counts"
                    ].get(source, 0),
                    "license": inventory.license_name,
                    "visual_quality": "pending_manual_review",
                    "utility_for_leaf_segmentation": next(
                        decision["status"]
                        for decision in decisions
                        if decision["source"] == source
                    ),
                    **instance_stats,
                }
            )

        charts = _generate_charts(
            source_rows,
            class_rows,
            image_rows,
            polygons,
            output_dir,
            public_dir,
        )
        previews = _generate_previews(
            inventories,
            parsed,
            polygons,
            issues,
            duplicate_rows,
            output_dir,
        )

        _write_csv(output_dir / "source_summary.csv", source_rows)
        _write_csv(output_dir / "class_summary.csv", class_rows)
        _write_csv(
            output_dir / "class_instance_summary.csv",
            class_instance_rows,
        )
        _write_csv(output_dir / "image_statistics.csv", image_rows)
        _write_csv(output_dir / "polygon_statistics.csv", polygons)
        _write_csv(output_dir / "annotation_issues.csv", issues, ISSUE_COLUMNS)
        _write_csv(
            output_dir / "image_label_mismatches.csv",
            mismatches,
            MISMATCH_COLUMNS,
        )
        _write_csv(
            output_dir / "duplicate_report.csv",
            duplicate_rows,
            DUPLICATE_COLUMNS,
        )
        _write_csv(output_dir / "yolo_coco_comparison.csv", comparisons)
        _write_csv(output_dir / "source_comparison.csv", source_comparison)
        _write_csv(output_dir / "manual_semantic_review.csv", manual_review)
        decision_text = _decision_markdown(decisions, duplicate_summary, issues)
        (output_dir / "decision_summary.md").write_text(
            decision_text,
            encoding="utf-8",
        )

        issue_counts = Counter(str(row["issue_type"]) for row in issues)
        class_counts = {
            source: {
                str(row["class_name"]): int(row["polygon_count"])
                for row in class_rows
                if row["source"] == source
            }
            for source in inventories
        }
        coco_contrast = {
            source: {
                "images": len(coco_data[source]["raw"].get("images", [])),
                "annotations": len(coco_data[source]["raw"].get("annotations", [])),
                "categories": coco_data[source]["categories"],
                "rle_annotations": sum(
                    isinstance(annotation.get("segmentation"), dict)
                    for annotation in coco_data[source]["raw"].get("annotations", [])
                ),
                "image_count_mismatches": sum(
                    row["source"] == source
                    and int(row["annotation_count_delta"]) != 0
                    for row in comparisons
                ),
                "class_name_mismatches": sum(
                    row["source"] == source and not bool(row["class_names_match"])
                    for row in comparisons
                ),
            }
            for source in inventories
        }
        input_fingerprint_after = build_audit_input_fingerprint(
            inventories,
            pilot_root,
            seed=seed,
        )
        if input_fingerprint_before != input_fingerprint_after:
            raise RuntimeError("Las fuentes o el piloto cambiaron durante la auditoría")
        summary = {
            "schema_version": 2,
            "cache_schema_version": AUDIT_CACHE_SCHEMA_VERSION,
            "parser_schema_version": PARSER_SCHEMA_VERSION,
            "audit_name": "external_leaf_segmentation_sources_eda",
            "seed": seed,
            "project_root": str(project_root),
            "external_root": str(external_root),
            "pilot_root": str(pilot_root),
            "sources": source_rows,
            "class_counts": class_counts,
            "class_instance_summary": class_instance_rows,
            "valid_polygons": len(valid_polygons),
            "invalid_polygon_lines": sum(not bool(row["valid"]) for row in polygons),
            "bbox_format_lines": sum(
                row.get("annotation_format") == "yolo_bbox" for row in polygons
            ),
            "topologically_valid_polygon_lines": sum(
                bool(row["valid"])
                and row.get("annotation_format") == "yolo_segmentation"
                for row in polygons
            ),
            "topologically_invalid_polygon_lines": len(
                {
                    (
                        str(row["source"]),
                        str(row["label_path"]),
                        int(row["line_number"]),
                    )
                    for row in issues
                    if row["line_number"] != ""
                    and row["issue_type"]
                    in {
                        "self_intersection",
                        "repeated_vertex",
                        "zero_length_edge",
                        "insufficient_unique_vertices",
                        "zero_or_near_zero_area",
                        "non_simple_polygon",
                    }
                }
            ),
            "issue_counts": dict(sorted(issue_counts.items())),
            "malformed_token_count": sum(
                row["issue_type"]
                in {"concatenated_numeric_token", "non_numeric_token"}
                for row in issues
            ),
            "coco_recoverable_issue_records": sum(
                str(row["coco_recovery_possible"]).lower() == "true"
                for row in issues
            ),
            "coco_recoverable_invalid_lines": len(
                {
                    (str(row["source"]), str(row["label_path"]), int(row["line_number"]))
                    for row in issues
                    if str(row["coco_recovery_possible"]).lower() == "true"
                    and row["line_number"] != ""
                }
            ),
            "duplicates": duplicate_summary,
            "coco_contrast": coco_contrast,
            "decisions": decisions,
            "source_comparison": source_comparison,
            "charts": charts,
            "previews": previews,
            "manual_review_rows": len(manual_review),
            "full_dataset_duplicate_check": {
                "enabled": False,
                "executed": False,
                "control_variable": "RUN_FULL_DATASET_DUPLICATE_CHECK",
            },
            "input_fingerprint": input_fingerprint_before,
            "input_fingerprint_after": input_fingerprint_after,
            "input_files_unchanged": True,
            "training_performed": False,
            "weights_downloaded": False,
            "source_files_modified": False,
            "labels_repaired": False,
            "dataset_consolidated": False,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if public_dir and public_dir.exists():
            shutil.rmtree(public_dir)
        raise
    return summary
