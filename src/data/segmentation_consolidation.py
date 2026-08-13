"""Controlled consolidation of audited external maize-leaf segmentations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.data.jpeg_normalization import (
    IMAGE_NORMALIZATION_COLUMNS,
    normalize_jpeg_copy,
)
from src.data.segmentation_audit import (
    BBOX_MATCH_TOLERANCE,
    IMAGE_EXTENSIONS,
    SEED,
    build_audit_input_fingerprint,
    discover_dataset_files,
    evaluate_coco_recovery,
    load_coco_segmentation,
    parse_yolo_segmentation_line,
    polygon_area,
    polygon_topology_issues,
    polygon_touches_border,
    sha256_file,
)
from src.data.segmentation_review_preview import generate_review_previews

TARGET_CLASS_ID = 0
TARGET_CLASS_NAME = "maize_leaf"
VALID_DECISIONS = {
    "include",
    "include_after_remap",
    "recover_from_coco",
    "manual_review",
    "exclude_lesion",
    "exclude_invalid",
    "exclude_duplicate",
    "exclude_pilot_leakage",
    "exclude_unknown_class",
    "exclude_human_review",
    "exclude_needs_reannotation",
}
CONSOLIDATION_ARTIFACTS = (
    "consolidation_manifest.csv",
    "included_annotations.csv",
    "excluded_annotations.csv",
    "recovered_annotations.csv",
    "manual_review.csv",
    "mandatory_visual_review.csv",
    "duplicate_groups.csv",
    "image_normalization_manifest.csv",
)
REPORT_ARTIFACTS = (
    "summary.json",
    "source_flow.csv",
    "class_flow.csv",
    "exclusion_reasons.csv",
    "recovery_summary.csv",
    "duplicate_summary.csv",
    "pilot_leakage_report.csv",
    "validation_issues.csv",
    "mandatory_visual_review.csv",
)
MANIFEST_COLUMNS = (
    "source_dataset",
    "original_image_path",
    "original_label_path",
    "original_line_number",
    "original_class_id",
    "original_class_name",
    "semantic_role",
    "decision",
    "decision_reason",
    "annotation_format",
    "recovered_from_coco",
    "recovery_evidence",
    "recovery_match_method",
    "recovery_candidate_count",
    "bbox_max_abs_error",
    "bbox_iou",
    "class_match",
    "semantic_role_match",
    "topology_valid",
    "recovery_decision",
    "recovery_reason",
    "image_sha256",
    "perceptual_hash",
    "duplicate_group",
    "quality_status",
    "target_class_id",
    "target_class_name",
    "original_base_name",
    "roboflow_variant_group",
    "license",
    "consolidated_image_path",
    "consolidated_label_path",
    "consolidated_line_number",
    "consolidated_annotation_sha256",
    "polygon_point_count",
    "polygon_area_ratio",
    "touches_border",
    "notes",
    "reviewer_decision",
    "review_reason",
    "review_status",
)
REVIEW_COLUMNS = (
    "source_dataset",
    "filename",
    "original_image_path",
    "original_label_path",
    "original_line_number",
    "original_class_id",
    "original_class_name",
    "decision",
    "review_reason",
    "annotation_quality",
    "background_complexity",
    "multiple_leaves",
    "reviewer_decision",
    "review_status",
    "notes",
)
DUPLICATE_COLUMNS = (
    "duplicate_group",
    "image_sha256",
    "group_size",
    "source_dataset",
    "filename",
    "original_image_path",
    "kept",
    "decision",
    "pilot_overlap",
    "perceptual_hash",
    "roboflow_variant_group",
)
VALIDATION_COLUMNS = (
    "severity",
    "issue_type",
    "source_dataset",
    "filename",
    "path",
    "line_number",
    "detail",
)


def write_csv(
    path: Path,
    rows: Iterable[dict[str, object]],
    columns: Sequence[str],
) -> None:
    """Write a deterministic CSV with a header even when there are no rows."""
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def perceptual_hash(path: Path, size: int = 8) -> str:
    """Return a dependency-free 64-bit average hash for grouping candidates."""
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L")
        resized = gray.resize((size, size), Image.Resampling.LANCZOS)
        # tobytes() en modo L devuelve exactamente los mismos valores que el
        # deprecado getdata() (Pillow 14); el hash resultante no cambia.
        pixels = list(resized.tobytes())
    average = sum(pixels) / len(pixels)
    value = 0
    for index, pixel in enumerate(pixels):
        if pixel >= average:
            value |= 1 << index
    return f"ahash{size * size}:{value:0{size * size // 4}x}"


def roboflow_original_base(filename: str) -> str:
    """Strip the Roboflow export suffix while retaining the original basename."""
    stem = Path(filename).stem
    match = re.fullmatch(
        r"(?P<base>.+?)_(?:jpg|jpeg|png|bmp|tif|tiff|webp)\.rf\.[^.]+",
        stem,
        flags=re.IGNORECASE,
    )
    return match.group("base") if match else stem


def roboflow_variant_group(source: str, filename: str) -> str:
    """Create a stable group identifier for Roboflow variants of one original."""
    base = roboflow_original_base(filename)
    digest = hashlib.sha256(f"{source}\0{base.casefold()}".encode()).hexdigest()
    return f"rfv_{digest[:16]}"


def remap_yolo_polygon_line(line: str) -> str:
    """Validate one segmentation polygon and remap its class to maize_leaf."""
    parsed = parse_yolo_segmentation_line(line)
    if not parsed.valid:
        reasons = ",".join(str(issue["issue_type"]) for issue in parsed.issues)
        raise ValueError(f"No se puede remapear una segmentación inválida: {reasons}")
    coordinates = line.strip().split()[1:]
    return " ".join((str(TARGET_CLASS_ID), *coordinates))


def decide_annotation(
    semantic_role: str,
    *,
    known_class: bool,
    valid_polygon: bool,
    recovery_available: bool,
) -> str:
    """Apply the EDA semantic policy to one source annotation."""
    if not known_class:
        return "exclude_unknown_class"
    if semantic_role == "lesion":
        return "exclude_lesion"
    if semantic_role != "full_leaf":
        return "manual_review"
    if valid_polygon:
        return "include_after_remap"
    if recovery_available:
        return "recover_from_coco"
    return "manual_review"


def _normalized_coco_polygon(
    annotation: dict[str, object],
    width: int,
    height: int,
) -> list[tuple[float, float]] | None:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list) or len(segmentation) != 1:
        return None
    component = segmentation[0]
    if not isinstance(component, list) or len(component) < 6 or len(component) % 2:
        return None
    points = [
        (
            float(component[index]) / width,
            float(component[index + 1]) / height,
        )
        for index in range(0, len(component), 2)
    ]
    if (
        polygon_topology_issues(points)
        or any(not math.isfinite(value) for point in points for value in point)
        or any(not 0.0 <= value <= 1.0 for point in points for value in point)
    ):
        return None
    return points


def recover_yolo_annotation_from_coco(
    *,
    source: str,
    raw_line: str,
    original_class_id: int,
    original_class_name: str,
    semantic_role: str,
    coco_entry: dict[str, object] | None,
    coco_categories: dict[int, str],
    image_match_unique: bool,
    tolerance: float = BBOX_MATCH_TOLERANCE,
) -> tuple[str, dict[str, object]] | None:
    """Recover an invalid YOLO row from one uniquely supported COCO polygon."""
    parsed = parse_yolo_segmentation_line(raw_line)
    annotations = coco_entry["annotations"] if coco_entry else []
    image = coco_entry["image"] if coco_entry else {}
    recovery = evaluate_coco_recovery(
        source=source,
        raw_line=raw_line,
        points=parsed.points,
        original_class_id=original_class_id,
        original_class_name=original_class_name,
        semantic_role=semantic_role,
        coco_annotations=annotations,
        coco_categories=coco_categories,
        width=int(image.get("width", 0) or 0),
        height=int(image.get("height", 0) or 0),
        image_match_unique=image_match_unique,
        tolerance=tolerance,
    )
    if recovery["recovery_decision"] != "recover_from_coco":
        return None
    annotation = recovery.pop("matched_annotation")
    if not isinstance(annotation, dict):
        return None
    points = _normalized_coco_polygon(
        annotation,
        int(image["width"]),
        int(image["height"]),
    )
    if points is None:
        return None
    coordinate_tokens = [
        f"{coordinate:.12f}".rstrip("0").rstrip(".")
        for point in points
        for coordinate in point
    ]
    recovered = " ".join((str(TARGET_CLASS_ID), *coordinate_tokens))
    if not parse_yolo_segmentation_line(recovered).valid:
        return None
    recovery["matched_annotation_id"] = annotation.get("id", "")
    return recovered, recovery


def recover_yolo_bbox_from_coco(
    raw_line: str,
    original_class_name: str,
    coco_annotation: dict[str, object],
    coco_class_name: str,
    width: int,
    height: int,
    *,
    tolerance: float = 1e-5,
) -> tuple[str, str] | None:
    """Legacy single-candidate bbox recovery used by small compatibility tests."""
    tokens = raw_line.strip().split()
    if len(tokens) != 5 or coco_class_name != original_class_name:
        return None
    try:
        _, center_x, center_y, box_width, box_height = map(float, tokens)
    except ValueError:
        return None
    values = (center_x, center_y, box_width, box_height)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        return None
    points = _normalized_coco_polygon(coco_annotation, width, height)
    if points is None:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    coco_box = (
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2,
        max(xs) - min(xs),
        max(ys) - min(ys),
    )
    maximum_delta = max(abs(left - right) for left, right in zip(values, coco_box))
    if maximum_delta > tolerance:
        return None
    coordinate_tokens = [
        f"{coordinate:.12f}".rstrip("0").rstrip(".")
        for point in points
        for coordinate in point
    ]
    recovered = " ".join((str(TARGET_CLASS_ID), *coordinate_tokens))
    if not parse_yolo_segmentation_line(recovered).valid:
        return None
    evidence = (
        "single_candidate;same_class;topology_valid;"
        f"bbox_delta={maximum_delta:.12g};coco_polygon_points={len(points)}"
    )
    return recovered, evidence


def select_image_actions(
    rows: Sequence[dict[str, object]],
    pilot_hashes: set[str],
) -> tuple[dict[tuple[str, str], str], list[dict[str, object]]]:
    """Choose one deterministic representative per exact hash and block pilot leaks."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image_sha256"])].append(row)
    actions: dict[tuple[str, str], str] = {}
    duplicate_rows: list[dict[str, object]] = []
    for digest, group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda row: (
                str(row["source_dataset"]),
                str(row["filename"]).casefold(),
            ),
        )
        pilot_overlap = digest in pilot_hashes
        for index, row in enumerate(ordered):
            if pilot_overlap:
                action = "exclude_pilot_leakage"
            elif index == 0:
                action = "keep"
            else:
                action = "exclude_duplicate"
            key = (str(row["source_dataset"]), str(row["filename"]))
            actions[key] = action
            duplicate_rows.append(
                {
                    "duplicate_group": f"exact_{digest[:16]}",
                    "image_sha256": digest,
                    "group_size": len(group),
                    "source_dataset": row["source_dataset"],
                    "filename": row["filename"],
                    "original_image_path": row["original_image_path"],
                    "kept": action == "keep",
                    "decision": action,
                    "pilot_overlap": pilot_overlap,
                    "perceptual_hash": row["perceptual_hash"],
                    "roboflow_variant_group": row["roboflow_variant_group"],
                }
            )
    return actions, duplicate_rows


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def polygon_has_proper_self_intersection(
    points: Sequence[tuple[float, float]],
    epsilon: float = 1e-14,
) -> bool:
    """Return True when two non-adjacent polygon edges cross properly."""
    count = len(points)
    for left in range(count):
        a = points[left]
        b = points[(left + 1) % count]
        for right in range(left + 1, count):
            if (
                right == left
                or (right + 1) % count == left
                or (left + 1) % count == right
            ):
                continue
            c = points[right]
            d = points[(right + 1) % count]
            orientations = (
                _orientation(a, b, c),
                _orientation(a, b, d),
                _orientation(c, d, a),
                _orientation(c, d, b),
            )
            first_crosses = (
                orientations[0] > epsilon and orientations[1] < -epsilon
            ) or (
                orientations[0] < -epsilon and orientations[1] > epsilon
            )
            second_crosses = (
                orientations[2] > epsilon and orientations[3] < -epsilon
            ) or (
                orientations[2] < -epsilon and orientations[3] > epsilon
            )
            if first_crosses and second_crosses:
                return True
    return False


def source_files_fingerprint(paths: Sequence[Path]) -> dict[str, object]:
    """Hash the exact source files used by consolidation."""
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted({item.resolve() for item in paths}, key=lambda item: item.as_posix()):
        file_digest = sha256_file(path)
        total_bytes += path.stat().st_size
        digest.update(path.as_posix().encode())
        digest.update(b"\0")
        digest.update(file_digest.encode())
        digest.update(b"\n")
    return {
        "file_count": len(set(paths)),
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def _read_required_csv(
    path: Path,
    columns: Sequence[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Falta resultado requerido del EDA: {path}")
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} no contiene columnas: {sorted(missing)}")
    return frame


def _load_eda(eda_root: Path) -> dict[str, object]:
    required = {
        "source_summary": ("source", "license", "yolo_images"),
        "class_summary": (
            "source",
            "class_id",
            "class_name",
            "semantic_role",
            "decision",
        ),
        "polygon_statistics": (
            "source",
            "filename",
            "class_id",
            "class_name",
            "line_number",
            "raw_line",
            "valid",
            "annotation_format",
        ),
        "annotation_issues": (
            "source",
            "filename",
            "line_number",
            "issue_type",
            "coco_recovery_possible",
            "recovery_match_method",
            "recovery_candidate_count",
            "bbox_max_abs_error",
            "bbox_iou",
            "class_match",
            "semantic_role_match",
            "topology_valid",
            "recovery_decision",
            "recovery_reason",
        ),
        "duplicate_report": ("group_id", "sha256", "collection", "filename"),
        "source_comparison": ("source", "images", "utility_for_leaf_segmentation"),
        "manual_semantic_review": (
            "source",
            "filename",
            "class_id",
            "class_name",
        ),
    }
    loaded: dict[str, object] = {}
    for name, columns in required.items():
        loaded[name] = _read_required_csv(eda_root / f"{name}.csv", columns)
    summary_path = eda_root / "summary.json"
    decision_path = eda_root / "decision_summary.md"
    if not summary_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError("Faltan summary.json o decision_summary.md del EDA")
    loaded["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    loaded["decision_text"] = decision_path.read_text(encoding="utf-8")
    return loaded


def _source_paths(inventories: dict[str, object]) -> list[Path]:
    paths: list[Path] = []
    for inventory in inventories.values():
        paths.extend(inventory.images)
        paths.extend(inventory.labels)
        paths.append(inventory.coco_json)
        for name in ("data.yaml", "README.dataset.txt", "README.roboflow.txt"):
            candidate = inventory.yolo_root / name
            if candidate.is_file():
                paths.append(candidate)
    return paths


def _points_from_line(line: str) -> list[tuple[float, float]]:
    tokens = line.strip().split()[1:]
    return [
        (float(tokens[index]), float(tokens[index + 1]))
        for index in range(0, len(tokens), 2)
    ]


def _manifest_reason(decision: str, source_decision_reason: str) -> str:
    reasons = {
        "include_after_remap": "EDA aceptó full_leaf; clase remapeada a maize_leaf",
        "recover_from_coco": "Fila YOLO bbox mixta recuperada desde COCO inequívoco",
        "manual_review": "No existe segmentación recuperable inequívoca",
        "exclude_lesion": "EDA clasificó la anotación como lesión",
        "exclude_invalid": "Anotación inválida no incluida",
        "exclude_duplicate": "Duplicado exacto; se conserva representante canónico",
        "exclude_pilot_leakage": "SHA-256 coincide con el piloto retenido",
        "exclude_unknown_class": "Clase ausente de class_summary.csv",
    }
    return f"{reasons.get(decision, decision)}; source_decision={source_decision_reason}"


def _render_preview(
    image_path: Path,
    polygons: Sequence[Sequence[tuple[float, float]]],
    destination: Path,
    caption: str,
) -> None:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((900, 700), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    colors = ("#00e676", "#ffca28", "#29b6f6", "#ef5350")
    for index, points in enumerate(polygons):
        scaled = [(x * image.width, y * image.height) for x, y in points]
        if len(scaled) >= 3:
            draw.line([*scaled, scaled[0]], fill=colors[index % len(colors)], width=4)
    font = ImageFont.load_default()
    caption_height = 44
    result = Image.new("RGB", (image.width, image.height + caption_height), "white")
    result.paste(image, (0, 0))
    ImageDraw.Draw(result).text(
        (8, image.height + 8),
        caption[:150],
        fill="black",
        font=font,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination, quality=90)


def _make_montage(paths: Sequence[Path], destination: Path, title: str) -> None:
    if not paths:
        return
    panels: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            panel = image.convert("RGB")
            panel.thumbnail((360, 300), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (380, 330), "white")
            canvas.paste(panel, ((380 - panel.width) // 2, 20))
            panels.append(canvas)
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    montage = Image.new("RGB", (columns * 380, rows * 330 + 36), "#eeeeee")
    ImageDraw.Draw(montage).text((10, 10), title, fill="black")
    for index, panel in enumerate(panels):
        montage.paste(panel, ((index % columns) * 380, 36 + (index // columns) * 330))
    destination.parent.mkdir(parents=True, exist_ok=True)
    montage.save(destination, quality=90)


def _deterministic_sample(
    rows: Sequence[dict[str, object]],
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["source_dataset"]),
            str(row["filename"]).casefold(),
        ),
    )
    if len(ordered) <= count:
        return list(ordered)
    return random.Random(seed).sample(ordered, count)


def _generate_previews(
    included_images: Sequence[dict[str, object]],
    destination: Path,
    seed: int,
) -> list[Path]:
    groups: dict[str, list[dict[str, object]]] = {
        "random_samples": _deterministic_sample(included_images, 12, seed),
        "recovered_from_coco": [
            row for row in included_images if bool(row["has_recovered"])
        ][:12],
        "small_masks": _deterministic_sample(
            [row for row in included_images if float(row["minimum_area"]) < 0.05],
            12,
            seed + 1,
        ),
        "large_masks": _deterministic_sample(
            [row for row in included_images if float(row["maximum_area"]) > 0.50],
            12,
            seed + 2,
        ),
        "border_touching": _deterministic_sample(
            [row for row in included_images if bool(row["touches_border"])],
            12,
            seed + 3,
        ),
        "multiple_instances": _deterministic_sample(
            [row for row in included_images if len(row["polygons"]) > 1],
            12,
            seed + 4,
        ),
    }
    for source in sorted({str(row["source_dataset"]) for row in included_images}):
        groups[f"source_{source}"] = _deterministic_sample(
            [row for row in included_images if row["source_dataset"] == source],
            8,
            seed + len(groups),
        )
    generated: list[Path] = []
    for group_name, rows in groups.items():
        group_paths: list[Path] = []
        for index, row in enumerate(rows, start=1):
            target = destination / group_name / f"{index:02d}_{row['output_stem']}.jpg"
            _render_preview(
                Path(str(row["original_image_path"])),
                row.get("polygons", []),
                target,
                (
                    f"{row['source_dataset']} | {row['filename']} | "
                    f"instances={len(row.get('polygons', []))}"
                ),
            )
            generated.append(target)
            group_paths.append(target)
        montage = destination / group_name / "montage.jpg"
        _make_montage(group_paths, montage, group_name)
        if montage.is_file():
            generated.append(montage)
    return generated


def validate_consolidated_dataset(
    all_root: Path,
    manifest_rows: Sequence[dict[str, object]],
    pilot_hashes: set[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Validate every mandatory invariant of the materialized candidate dataset."""
    issues: list[dict[str, object]] = []

    def record(
        issue_type: str,
        detail: str,
        *,
        path: Path | None = None,
        line_number: int | str = "",
    ) -> None:
        issues.append(
            {
                "severity": "error",
                "issue_type": issue_type,
                "source_dataset": "",
                "filename": path.name if path else "",
                "path": str(path) if path else "",
                "line_number": line_number,
                "detail": detail,
            }
        )

    images = sorted(
        path
        for path in (all_root / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels = sorted((all_root / "labels").glob("*.txt"))
    image_by_stem = {path.stem: path for path in images}
    label_by_stem = {path.stem: path for path in labels}
    for stem in sorted(image_by_stem.keys() - label_by_stem.keys()):
        record("image_without_label", stem, path=image_by_stem[stem])
    for stem in sorted(label_by_stem.keys() - image_by_stem.keys()):
        record("label_without_image", stem, path=label_by_stem[stem])

    annotation_count = 0
    for label in labels:
        text = label.read_text(encoding="utf-8")
        if not text.strip():
            record("empty_label", "El TXT consolidado está vacío", path=label)
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            parsed = parse_yolo_segmentation_line(line)
            annotation_count += 1
            if parsed.class_id != TARGET_CLASS_ID:
                record(
                    "non_target_class",
                    f"class_id={parsed.class_id}",
                    path=label,
                    line_number=line_number,
                )
            for issue in parsed.issues:
                record(
                    str(issue["issue_type"]),
                    str(issue["detail"]),
                    path=label,
                    line_number=line_number,
                )

    hashes = [sha256_file(path) for path in images]
    duplicate_hashes = {
        digest for digest, count in Counter(hashes).items() if count > 1
    }
    for digest in sorted(duplicate_hashes):
        record("included_exact_duplicate", digest)
    leakage = sorted(set(hashes) & pilot_hashes)
    for digest in leakage:
        record("pilot_leakage", digest)

    included_manifest = [
        row
        for row in manifest_rows
        if row["decision"] in {"include", "include_after_remap", "recover_from_coco"}
    ]
    for row in included_manifest:
        image_path = (
            all_root / "images" / Path(str(row["consolidated_image_path"])).name
        )
        label_path = (
            all_root / "labels" / Path(str(row["consolidated_label_path"])).name
        )
        if not image_path.is_file() or not label_path.is_file():
            record(
                "manifest_correspondence_missing",
                f"{image_path};{label_path}",
                path=label_path,
            )

    summary = {
        "images": len(images),
        "labels": len(labels),
        "annotations": annotation_count,
        "errors": len(issues),
        "all_images_have_labels": image_by_stem.keys() == label_by_stem.keys(),
        "single_target_class_zero": not any(
            row["issue_type"] == "non_target_class" for row in issues
        ),
        "nonempty_labels": not any(row["issue_type"] == "empty_label" for row in issues),
        "valid_polygon_syntax": not any(
            row["issue_type"]
            in {
                "fewer_than_three_points",
                "fewer_than_three_distinct_points",
                "bbox_format_in_segmentation_label",
                "self_intersection",
                "repeated_vertex",
                "zero_length_edge",
                "insufficient_unique_vertices",
                "zero_or_near_zero_area",
                "non_simple_polygon",
                "incomplete_coordinate_pair",
                "non_numeric_token",
                "concatenated_numeric_token",
                "non_finite_coordinate",
                "coordinate_out_of_range",
                "zero_polygon_area",
            }
            for row in issues
        ),
        "zero_exact_duplicates": not duplicate_hashes,
        "zero_pilot_leakage": not leakage,
        "manifest_correspondence_complete": not any(
            row["issue_type"] == "manifest_correspondence_missing" for row in issues
        ),
    }
    summary["passed"] = not issues
    return issues, summary


def _config_safety(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    segmentation = config.get("segmentation", {})
    return {
        "model": segmentation.get("model"),
        "output_profile": segmentation.get("output_profile"),
    }


def _dataset_yaml() -> str:
    return """# Candidate pool only: train/val/test splits are intentionally absent.
path: .

candidate:
  images: all/images
  labels: all/labels

names:
  0: maize_leaf

splits_created: false
"""


def _readme(existing: str, summary: dict[str, object]) -> str:
    heading = "## Consolidación externa de segmentación"
    prefix = existing.split(heading, maxsplit=1)[0].rstrip()
    counts = summary["counts"]
    section = f"""## Consolidación externa de segmentación

`all/` contiene el pool candidato consolidado desde las dos fuentes externas
auditadas. Sólo se conserva la clase de hoja completa, remapeada a
`0 = maize_leaf`. Las lesiones se excluyen y las fuentes originales permanecen
inmutables.

- imágenes consideradas: {counts['images_considered']};
- imágenes candidatas incluidas: {counts['images_included']};
- anotaciones incluidas: {counts['annotations_included']};
- anotaciones recuperadas desde COCO: {counts['annotations_recovered']};
- duplicados exactos eliminados: {counts['duplicates_excluded']};
- fugas contra el piloto: {counts['pilot_leakage']};
- casos en cola de revisión manual: {counts['manual_review_rows']}.

Los {counts['mandatory_visual_review_rows']} casos de revisión visual obligatoria
están en `manifests/mandatory_visual_review.csv`. Una hoja autointersectada quedó
fuera del pool y la recuperación COCO extremadamente pequeña permanece marcada
como pendiente; ninguna puede entrar en futuros splits sin decisión explícita.

No existen todavía splits `train/val/test` para estas máscaras. `dataset.yaml`
describe únicamente el pool `all/`; primero deben aprobarse los previews y la
cola de revisión manual. Este conjunto todavía no se incorporó al entrenamiento
del segmentador.
"""
    return prefix + "\n\n" + section


def _report_rows(
    manifest_rows: Sequence[dict[str, object]],
    image_rows: Sequence[dict[str, object]],
    duplicate_rows: Sequence[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    source_rows: list[dict[str, object]] = []
    sources = sorted({str(row["source_dataset"]) for row in image_rows})
    for source in sources:
        source_images = [row for row in image_rows if row["source_dataset"] == source]
        source_manifest = [
            row for row in manifest_rows if row["source_dataset"] == source
        ]
        included_images = {
            str(row["consolidated_image_path"])
            for row in source_manifest
            if row["consolidated_image_path"]
        }
        source_rows.append(
            {
                "source_dataset": source,
                "images_considered": len(source_images),
                "images_included": len(included_images),
                "images_excluded": len(source_images) - len(included_images),
                "annotations_considered": sum(
                    bool(str(row["original_line_number"])) for row in source_manifest
                ),
                "annotations_included": sum(
                    row["decision"]
                    in {"include", "include_after_remap", "recover_from_coco"}
                    for row in source_manifest
                ),
                "annotations_recovered": sum(
                    row["decision"] == "recover_from_coco" for row in source_manifest
                ),
                "annotations_excluded": sum(
                    str(row["decision"]).startswith("exclude")
                    for row in source_manifest
                ),
                "manual_review": sum(
                    row["decision"] == "manual_review" for row in source_manifest
                ),
            }
        )

    class_counter: Counter[tuple[str, str, str, str]] = Counter()
    for row in manifest_rows:
        key = (
            str(row["source_dataset"]),
            str(row["original_class_name"]),
            str(row["semantic_role"]),
            str(row["decision"]),
        )
        class_counter[key] += 1
    class_rows = [
        {
            "source_dataset": key[0],
            "original_class_name": key[1],
            "semantic_role": key[2],
            "decision": key[3],
            "annotations": count,
        }
        for key, count in sorted(class_counter.items())
    ]

    exclusion_counter = Counter(
        str(row["decision"])
        for row in manifest_rows
        if str(row["decision"]).startswith("exclude")
        or row["decision"] == "manual_review"
    )
    exclusion_rows = [
        {"decision": decision, "annotations": count}
        for decision, count in sorted(exclusion_counter.items())
    ]
    recovery_counter = Counter(
        (str(row["source_dataset"]), str(row["original_class_name"]))
        for row in manifest_rows
        if row["decision"] == "recover_from_coco"
    )
    recovery_rows = [
        {
            "source_dataset": source,
            "original_class_name": class_name,
            "recovered_annotations": count,
        }
        for (source, class_name), count in sorted(recovery_counter.items())
    ]
    exact_duplicate_groups = {
        str(row["duplicate_group"])
        for row in duplicate_rows
        if int(row["group_size"]) > 1
    }
    duplicate_summary = [
        {
            "candidate_images": len(image_rows),
            "exact_groups": len(exact_duplicate_groups),
            "excluded_images": sum(
                row["decision"] == "exclude_duplicate" for row in duplicate_rows
            ),
            "pilot_overlap_images": sum(
                row["decision"] == "exclude_pilot_leakage" for row in duplicate_rows
            ),
        }
    ]
    pilot_rows = [
        row
        for row in duplicate_rows
        if row["decision"] == "exclude_pilot_leakage"
    ]
    return {
        "source_flow": source_rows,
        "class_flow": class_rows,
        "exclusion_reasons": exclusion_rows,
        "recovery_summary": recovery_rows,
        "duplicate_summary": duplicate_summary,
        "pilot_leakage_report": pilot_rows,
    }


def build_segmentation_consolidation(
    external_root: Path,
    pilot_root: Path,
    eda_root: Path,
    dataset_root: Path,
    report_root: Path,
    config_path: Path,
    *,
    seed: int = SEED,
) -> dict[str, object]:
    """Materialize the audited candidate pool without touching source files."""
    external_root = external_root.resolve()
    pilot_root = pilot_root.resolve()
    eda_root = eda_root.resolve()
    dataset_root = dataset_root.resolve()
    report_root = report_root.resolve()
    config_path = config_path.resolve()
    protected_targets = [
        dataset_root / "all",
        dataset_root / "previews",
        dataset_root / "dataset.yaml",
        report_root,
        *[dataset_root / "manifests" / name for name in CONSOLIDATION_ARTIFACTS],
    ]
    existing = [str(path) for path in protected_targets if path.exists()]
    if existing:
        raise FileExistsError(
            "La consolidación ya existe y no será sobrescrita: " + ", ".join(existing)
        )

    eda = _load_eda(eda_root)
    eda_summary = eda["summary"]
    if eda_summary.get("seed") != seed:
        raise ValueError("La semilla no coincide con summary.json del EDA")
    decisions = {
        str(row["source"]): row for row in eda_summary.get("decisions", [])
    }
    if not decisions or any(
        row.get("status") != "accepted_with_filtering" for row in decisions.values()
    ):
        raise ValueError("Las fuentes no están aceptadas con filtrado por el EDA")

    source_summary = eda["source_summary"]
    class_summary = eda["class_summary"]
    polygons = eda["polygon_statistics"]
    issues = eda["annotation_issues"]
    manual_eda = eda["manual_semantic_review"]
    inventories = discover_dataset_files(external_root)
    if set(inventories) != set(decisions):
        raise ValueError("Las fuentes descubiertas no coinciden con las decisiones del EDA")
    current_eda_fingerprint = build_audit_input_fingerprint(
        inventories,
        pilot_root,
        seed=seed,
    )
    if eda_summary.get("input_fingerprint") != current_eda_fingerprint:
        raise ValueError(
            "El EDA no corresponde a las fuentes, piloto o versión actual del parser"
        )
    coco = {
        source: load_coco_segmentation(inventory.coco_json)
        for source, inventory in inventories.items()
    }
    licenses = {
        str(row.source): str(row.license)
        for row in source_summary.itertuples()
    }
    class_rules = {
        (str(row.source), int(row.class_id)): row
        for row in class_summary.itertuples()
    }
    recovery_issue_by_key: dict[tuple[str, str, int], object] = {}
    for row in issues[issues["line_number"].notna()].itertuples():
        key = (str(row.source), str(row.filename), int(row.line_number))
        existing = recovery_issue_by_key.get(key)
        if existing is None or str(row.recovery_decision) == "recover_from_coco":
            recovery_issue_by_key[key] = row

    image_rows: list[dict[str, object]] = []
    source_paths = _source_paths(inventories)
    source_fingerprint_before = source_files_fingerprint(source_paths)
    pilot_paths = sorted(
        path
        for path in (pilot_root / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    pilot_hashes = {sha256_file(path) for path in pilot_paths}
    for source, inventory in inventories.items():
        for image_path in inventory.images:
            base = roboflow_original_base(image_path.name)
            image_rows.append(
                {
                    "source_dataset": source,
                    "filename": image_path.name,
                    "original_image_path": str(image_path.resolve()),
                    "image_sha256": sha256_file(image_path),
                    "perceptual_hash": perceptual_hash(image_path),
                    "original_base_name": base,
                    "roboflow_variant_group": roboflow_variant_group(
                        source, image_path.name
                    ),
                }
            )
    image_rows.sort(
        key=lambda row: (
            str(row["source_dataset"]),
            str(row["filename"]).casefold(),
        )
    )
    image_actions, duplicate_rows = select_image_actions(image_rows, pilot_hashes)

    with tempfile.TemporaryDirectory(
        prefix=".segmentation_consolidation_",
        dir=dataset_root.parent,
    ) as temporary:
        staging = Path(temporary)
        all_root = staging / "all"
        manifests_root = staging / "manifests"
        previews_root = staging / "previews"
        reports_root = staging / "reports"
        (all_root / "images").mkdir(parents=True)
        (all_root / "labels").mkdir(parents=True)
        manifests_root.mkdir()
        reports_root.mkdir()

        manifest_rows: list[dict[str, object]] = []
        included_images: list[dict[str, object]] = []
        image_normalization_rows: list[dict[str, object]] = []
        topology_review: list[dict[str, object]] = []
        polygons_by_image = {
            key: group.sort_values("line_number")
            for key, group in polygons.groupby(["source", "filename"])
        }
        used_output_stems: set[str] = set()

        for image_row in image_rows:
            source = str(image_row["source_dataset"])
            filename = str(image_row["filename"])
            key = (source, filename)
            inventory = inventories[source]
            label_map = {
                path.stem.casefold(): path for path in inventory.labels
            }
            original_label = label_map.get(Path(filename).stem.casefold())
            image_polygons = polygons_by_image.get(key)
            annotation_rows: list[dict[str, object]] = []
            output_lines: list[str] = []
            output_points: list[list[tuple[float, float]]] = []
            has_recovered = False

            if image_polygons is None or image_polygons.empty:
                annotation_rows.append(
                    {
                        "source_dataset": source,
                        "original_image_path": image_row["original_image_path"],
                        "original_label_path": (
                            str(original_label.resolve()) if original_label else ""
                        ),
                        "original_line_number": "",
                        "original_class_id": "",
                        "original_class_name": "",
                        "semantic_role": "unknown",
                        "decision": "manual_review",
                        "decision_reason": _manifest_reason(
                            "manual_review", str(decisions[source]["reason"])
                        ),
                        "annotation_format": "empty_label",
                        "recovered_from_coco": False,
                        "recovery_evidence": "",
                        "image_sha256": image_row["image_sha256"],
                        "perceptual_hash": image_row["perceptual_hash"],
                        "duplicate_group": (
                            f"exact_{str(image_row['image_sha256'])[:16]}"
                        ),
                        "quality_status": "manual_review_no_annotation",
                        "target_class_id": "",
                        "target_class_name": "",
                        "original_base_name": image_row["original_base_name"],
                        "roboflow_variant_group": image_row[
                            "roboflow_variant_group"
                        ],
                        "license": licenses[source],
                        "consolidated_image_path": "",
                        "consolidated_label_path": "",
                        "consolidated_line_number": "",
                        "consolidated_annotation_sha256": "",
                        "polygon_point_count": 0,
                        "polygon_area_ratio": 0.0,
                        "touches_border": False,
                        "notes": "TXT vacío o imagen sin filas de anotación",
                        "reviewer_decision": "",
                        "review_reason": "unrecoverable_empty_label",
                        "review_status": "pending",
                    }
                )
            else:
                for polygon in image_polygons.itertuples():
                    class_id = int(polygon.class_id)
                    rule = class_rules.get((source, class_id))
                    known_class = rule is not None
                    semantic_role = (
                        str(rule.semantic_role) if rule is not None else "unknown"
                    )
                    recovery_key = (source, filename, int(polygon.line_number))
                    recovery_issue = recovery_issue_by_key.get(recovery_key)
                    recovery_available = bool(
                        recovery_issue is not None
                        and str(recovery_issue.recovery_decision)
                        == "recover_from_coco"
                    )
                    decision = decide_annotation(
                        semantic_role,
                        known_class=known_class,
                        valid_polygon=bool(polygon.valid),
                        recovery_available=recovery_available,
                    )
                    consolidated_line = ""
                    evidence = ""
                    recovery_metadata: dict[str, object] = {
                        "recovery_match_method": (
                            getattr(recovery_issue, "recovery_match_method", "")
                            if recovery_issue is not None
                            else ""
                        ),
                        "recovery_candidate_count": (
                            getattr(recovery_issue, "recovery_candidate_count", "")
                            if recovery_issue is not None
                            else ""
                        ),
                        "bbox_max_abs_error": (
                            getattr(recovery_issue, "bbox_max_abs_error", "")
                            if recovery_issue is not None
                            else ""
                        ),
                        "bbox_iou": (
                            getattr(recovery_issue, "bbox_iou", "")
                            if recovery_issue is not None
                            else ""
                        ),
                        "class_match": (
                            getattr(recovery_issue, "class_match", "")
                            if recovery_issue is not None
                            else ""
                        ),
                        "semantic_role_match": (
                            getattr(recovery_issue, "semantic_role_match", "")
                            if recovery_issue is not None
                            else ""
                        ),
                        "topology_valid": (
                            getattr(recovery_issue, "topology_valid", "")
                            if recovery_issue is not None
                            else ""
                        ),
                        "recovery_decision": (
                            getattr(recovery_issue, "recovery_decision", "")
                            if recovery_issue is not None
                            else ""
                        ),
                        "recovery_reason": (
                            getattr(recovery_issue, "recovery_reason", "")
                            if recovery_issue is not None
                            else ""
                        ),
                    }
                    annotation_format = (
                        "yolo_bbox_mixed"
                        if str(polygon.annotation_format) == "yolo_bbox"
                        else "yolo_segmentation"
                    )
                    quality_status = decision
                    if decision == "include_after_remap":
                        consolidated_line = remap_yolo_polygon_line(
                            str(polygon.raw_line)
                        )
                        quality_status = "eda_accepted_pending_visual_review"
                    elif decision == "recover_from_coco":
                        entries = coco[source]["by_filename_all"].get(filename, [])
                        entry = entries[0] if len(entries) == 1 else None
                        recovered = recover_yolo_annotation_from_coco(
                            source=source,
                            raw_line=str(polygon.raw_line),
                            original_class_id=class_id,
                            original_class_name=str(polygon.class_name),
                            semantic_role=semantic_role,
                            coco_entry=entry,
                            coco_categories=coco[source]["categories"],
                            image_match_unique=len(entries) == 1,
                        )
                        if recovered is None:
                            decision = "manual_review"
                            quality_status = "manual_review_recovery_failed"
                        else:
                            consolidated_line, recovery_metadata = recovered
                            evidence = json.dumps(
                                recovery_metadata,
                                sort_keys=True,
                                ensure_ascii=False,
                            )
                            has_recovered = True
                            quality_status = (
                                "coco_recovered_pending_visual_review"
                            )

                    image_action = image_actions[key]
                    if decision in {"include_after_remap", "recover_from_coco"}:
                        if image_action == "exclude_duplicate":
                            decision = "exclude_duplicate"
                            consolidated_line = ""
                            quality_status = "excluded_exact_duplicate"
                        elif image_action == "exclude_pilot_leakage":
                            decision = "exclude_pilot_leakage"
                            consolidated_line = ""
                            quality_status = "excluded_pilot_leakage"

                    points: list[tuple[float, float]] = []
                    area = 0.0
                    touches_border = False
                    output_line_number: int | str = ""
                    annotation_digest = ""
                    if consolidated_line:
                        points = _points_from_line(consolidated_line)
                        area = polygon_area(points)
                        touches_border = any(polygon_touches_border(points).values())
                        output_lines.append(consolidated_line)
                        output_points.append(points)
                        output_line_number = len(output_lines)
                        annotation_digest = hashlib.sha256(
                            consolidated_line.encode()
                        ).hexdigest()

                    topology_issue_types = {
                        issue
                        for issue in str(polygon.issue_reason).split(";")
                        if issue
                        in {
                            "self_intersection",
                            "repeated_vertex",
                            "zero_length_edge",
                            "insufficient_unique_vertices",
                            "zero_or_near_zero_area",
                            "non_simple_polygon",
                        }
                    }
                    reviewer_decision = ""
                    review_reason = ""
                    review_status = ""
                    if semantic_role == "full_leaf" and topology_issue_types:
                        reviewer_decision = ""
                        review_reason = "topologically_invalid_full_leaf"
                        review_status = "pending"
                        raw_parsed = parse_yolo_segmentation_line(
                            str(polygon.raw_line)
                        )
                        topology_review.append(
                            {
                                "source_dataset": source,
                                "filename": filename,
                                "original_image_path": image_row[
                                    "original_image_path"
                                ],
                                "original_label_path": (
                                    str(original_label.resolve())
                                    if original_label
                                    else ""
                                ),
                                "original_line_number": int(polygon.line_number),
                                "original_class_id": class_id,
                                "original_class_name": polygon.class_name,
                                "decision": decision,
                                "review_reason": review_reason,
                                "annotation_quality": "invalid_topology",
                                "background_complexity": "unknown",
                                "multiple_leaves": "unknown",
                                "reviewer_decision": "",
                                "review_status": "pending",
                                "notes": ";".join(sorted(topology_issue_types)),
                                "polygons": [raw_parsed.points],
                            }
                        )
                    elif decision == "recover_from_coco":
                        review_reason = "coco_recovery_extremely_small_area"
                        review_status = "pending"

                    annotation_rows.append(
                        {
                            "source_dataset": source,
                            "original_image_path": image_row[
                                "original_image_path"
                            ],
                            "original_label_path": (
                                str(original_label.resolve()) if original_label else ""
                            ),
                            "original_line_number": int(polygon.line_number),
                            "original_class_id": class_id,
                            "original_class_name": str(polygon.class_name),
                            "semantic_role": semantic_role,
                            "decision": decision,
                            "decision_reason": _manifest_reason(
                                decision, str(decisions[source]["reason"])
                            ),
                            "annotation_format": annotation_format,
                            "recovered_from_coco": decision == "recover_from_coco",
                            "recovery_evidence": evidence,
                            **recovery_metadata,
                            "image_sha256": image_row["image_sha256"],
                            "perceptual_hash": image_row["perceptual_hash"],
                            "duplicate_group": (
                                f"exact_{str(image_row['image_sha256'])[:16]}"
                            ),
                            "quality_status": quality_status,
                            "target_class_id": (
                                TARGET_CLASS_ID if consolidated_line else ""
                            ),
                            "target_class_name": (
                                TARGET_CLASS_NAME if consolidated_line else ""
                            ),
                            "original_base_name": image_row[
                                "original_base_name"
                            ],
                            "roboflow_variant_group": image_row[
                                "roboflow_variant_group"
                            ],
                            "license": licenses[source],
                            "consolidated_image_path": "",
                            "consolidated_label_path": "",
                            "consolidated_line_number": output_line_number,
                            "consolidated_annotation_sha256": annotation_digest,
                            "polygon_point_count": len(points),
                            "polygon_area_ratio": area,
                            "touches_border": touches_border,
                            "notes": (
                                ";".join(sorted(topology_issue_types))
                                if topology_issue_types
                                else ""
                            ),
                            "reviewer_decision": reviewer_decision,
                            "review_reason": review_reason,
                            "review_status": review_status,
                        }
                    )

            if output_lines:
                alias = "cldc" if source == "corn_leaf_diseases_classification" else "corn"
                output_stem = f"{alias}_{str(image_row['image_sha256'])[:16]}"
                if output_stem in used_output_stems:
                    raise RuntimeError(f"Colisión de ID consolidado: {output_stem}")
                used_output_stems.add(output_stem)
                original_image = Path(str(image_row["original_image_path"]))
                image_target = all_root / "images" / (
                    output_stem + original_image.suffix.lower()
                )
                label_target = all_root / "labels" / f"{output_stem}.txt"
                normalization = normalize_jpeg_copy(
                    original_image,
                    image_target,
                    source_path=str(original_image.resolve()),
                    derived_path=str(
                        dataset_root / "all" / "images" / image_target.name
                    ),
                )
                normalized_sha256 = str(normalization["normalized_sha256"])
                image_normalization_rows.append(normalization)
                label_target.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
                final_image = dataset_root / "all/images" / image_target.name
                final_label = dataset_root / "all/labels" / label_target.name
                for row in annotation_rows:
                    row["image_sha256"] = normalized_sha256
                    row["consolidated_image_path"] = str(final_image)
                    row["consolidated_label_path"] = str(final_label)
                included_images.append(
                    {
                        **image_row,
                        "image_sha256": normalized_sha256,
                        "output_stem": output_stem,
                        "polygons": output_points,
                        "has_recovered": has_recovered,
                        "minimum_area": min(map(polygon_area, output_points)),
                        "maximum_area": max(map(polygon_area, output_points)),
                        "touches_border": any(
                            any(polygon_touches_border(points).values())
                            for points in output_points
                        ),
                    }
                )
            manifest_rows.extend(annotation_rows)

        manifest_rows.sort(
            key=lambda row: (
                str(row["source_dataset"]),
                str(row["original_image_path"]).casefold(),
                int(row["original_line_number"] or 0),
                str(row["decision"]),
            )
        )
        if any(row["decision"] not in VALID_DECISIONS for row in manifest_rows):
            raise RuntimeError("El manifiesto contiene decisiones no permitidas")

        manual_rows: list[dict[str, object]] = []
        manual_keys: set[tuple[str, str, str, str]] = set()
        for row in manual_eda.itertuples():
            key = (
                str(row.source),
                str(row.filename),
                str(row.class_id),
                "eda_stratified_visual_review",
            )
            if key in manual_keys:
                continue
            manual_keys.add(key)
            inventory = inventories[str(row.source)]
            image_path = next(
                path for path in inventory.images if path.name == str(row.filename)
            )
            label_path = next(
                (
                    path
                    for path in inventory.labels
                    if path.stem.casefold() == image_path.stem.casefold()
                ),
                None,
            )
            manual_rows.append(
                {
                    "source_dataset": row.source,
                    "filename": row.filename,
                    "original_image_path": str(image_path.resolve()),
                    "original_label_path": (
                        str(label_path.resolve()) if label_path else ""
                    ),
                    "original_line_number": "",
                    "original_class_id": row.class_id,
                    "original_class_name": row.class_name,
                    "decision": "manual_review",
                    "review_reason": "eda_stratified_visual_review",
                    "annotation_quality": getattr(row, "annotation_quality", "unknown"),
                    "background_complexity": getattr(
                        row, "background_complexity", "unknown"
                    ),
                    "multiple_leaves": getattr(row, "multiple_leaves", "unknown"),
                    "reviewer_decision": "",
                    "review_status": "pending",
                    "notes": getattr(row, "notes", ""),
                }
            )
        for row in manifest_rows:
            if row["decision"] != "manual_review":
                continue
            row_review_reason = str(
                row.get("review_reason") or "unrecoverable_annotation"
            )
            key = (
                str(row["source_dataset"]),
                Path(str(row["original_image_path"])).name,
                str(row["original_class_id"]),
                row_review_reason,
            )
            if key in manual_keys:
                continue
            manual_keys.add(key)
            manual_rows.append(
                {
                    "source_dataset": row["source_dataset"],
                    "filename": Path(str(row["original_image_path"])).name,
                    "original_image_path": row["original_image_path"],
                    "original_label_path": row["original_label_path"],
                    "original_line_number": row["original_line_number"],
                    "original_class_id": row["original_class_id"],
                    "original_class_name": row["original_class_name"],
                    "decision": "manual_review",
                    "review_reason": row_review_reason,
                    "annotation_quality": "unknown",
                    "background_complexity": "unknown",
                    "multiple_leaves": "unknown",
                    "reviewer_decision": "",
                    "review_status": "pending",
                    "notes": row["notes"],
                }
            )
        for row in topology_review:
            key = (
                str(row["source_dataset"]),
                str(row["filename"]),
                str(row["original_class_id"]),
                str(row["review_reason"]),
            )
            if key not in manual_keys:
                manual_keys.add(key)
                manual_rows.append(row)
        manual_rows.sort(
            key=lambda row: (
                str(row["source_dataset"]),
                str(row["filename"]).casefold(),
                str(row["review_reason"]),
            )
        )

        included_rows = [
            row
            for row in manifest_rows
            if row["decision"]
            in {"include", "include_after_remap", "recover_from_coco"}
        ]
        excluded_rows = [
            row
            for row in manifest_rows
            if str(row["decision"]).startswith("exclude")
        ]
        recovered_rows = [
            row for row in manifest_rows if row["decision"] == "recover_from_coco"
        ]
        mandatory_visual_rows = list(topology_review)
        for row in recovered_rows:
            mandatory_visual_rows.append(
                {
                    "source_dataset": row["source_dataset"],
                    "filename": Path(str(row["original_image_path"])).name,
                    "original_image_path": row["original_image_path"],
                    "original_label_path": row["original_label_path"],
                    "original_line_number": row["original_line_number"],
                    "original_class_id": row["original_class_id"],
                    "original_class_name": row["original_class_name"],
                    "decision": row["decision"],
                    "review_reason": row["review_reason"],
                    "annotation_quality": "extremely_small_area",
                    "background_complexity": "unknown",
                    "multiple_leaves": "unknown",
                    "reviewer_decision": "",
                    "review_status": "pending",
                    "notes": (
                        f"polygon_area_ratio={row['polygon_area_ratio']}; "
                        f"recovery_reason={row['recovery_reason']}"
                    ),
                }
            )
        write_csv(
            manifests_root / "consolidation_manifest.csv",
            manifest_rows,
            MANIFEST_COLUMNS,
        )
        write_csv(
            manifests_root / "included_annotations.csv",
            included_rows,
            MANIFEST_COLUMNS,
        )
        write_csv(
            manifests_root / "excluded_annotations.csv",
            excluded_rows,
            MANIFEST_COLUMNS,
        )
        write_csv(
            manifests_root / "recovered_annotations.csv",
            recovered_rows,
            MANIFEST_COLUMNS,
        )
        write_csv(
            manifests_root / "manual_review.csv",
            manual_rows,
            REVIEW_COLUMNS,
        )
        write_csv(
            manifests_root / "mandatory_visual_review.csv",
            mandatory_visual_rows,
            REVIEW_COLUMNS,
        )
        write_csv(
            manifests_root / "duplicate_groups.csv",
            duplicate_rows,
            DUPLICATE_COLUMNS,
        )
        write_csv(
            manifests_root / "image_normalization_manifest.csv",
            image_normalization_rows,
            IMAGE_NORMALIZATION_COLUMNS,
        )

        preview_paths = _generate_previews(
            included_images,
            previews_root,
            seed,
        )
        review_preview_summary = generate_review_previews(
            external_root,
            manifests_root,
            previews_root,
            reports_root / "review_preview_validation.json",
            published_preview_root=dataset_root / "previews",
        )
        preview_paths.extend(
            previews_root
            / (
                "mandatory_visual_review"
                if row["review_origin"] == "mandatory"
                else "manual_review"
            )
            / Path(str(row["preview_path"])).name
            for row in review_preview_summary["cases"]
        )
        shutil.copytree(previews_root, reports_root / "previews")

        validation_issues, validation_summary = validate_consolidated_dataset(
            all_root,
            manifest_rows,
            pilot_hashes,
        )
        for row in topology_review:
            validation_issues.append(
                {
                    "severity": "warning",
                    "issue_type": "polygon_topology_manual_review",
                    "source_dataset": row["source_dataset"],
                    "filename": row["filename"],
                    "path": row["original_label_path"],
                    "line_number": row["original_line_number"],
                    "detail": row["notes"],
                }
            )
        write_csv(
            reports_root / "validation_issues.csv",
            validation_issues,
            VALIDATION_COLUMNS,
        )
        write_csv(
            reports_root / "mandatory_visual_review.csv",
            mandatory_visual_rows,
            REVIEW_COLUMNS,
        )
        if not validation_summary["passed"]:
            raise RuntimeError(
                f"La validación consolidada falló: {validation_summary}"
            )

        report_rows = _report_rows(manifest_rows, image_rows, duplicate_rows)
        write_csv(
            reports_root / "source_flow.csv",
            report_rows["source_flow"],
            (
                "source_dataset",
                "images_considered",
                "images_included",
                "images_excluded",
                "annotations_considered",
                "annotations_included",
                "annotations_recovered",
                "annotations_excluded",
                "manual_review",
            ),
        )
        write_csv(
            reports_root / "class_flow.csv",
            report_rows["class_flow"],
            (
                "source_dataset",
                "original_class_name",
                "semantic_role",
                "decision",
                "annotations",
            ),
        )
        write_csv(
            reports_root / "exclusion_reasons.csv",
            report_rows["exclusion_reasons"],
            ("decision", "annotations"),
        )
        write_csv(
            reports_root / "recovery_summary.csv",
            report_rows["recovery_summary"],
            ("source_dataset", "original_class_name", "recovered_annotations"),
        )
        write_csv(
            reports_root / "duplicate_summary.csv",
            report_rows["duplicate_summary"],
            (
                "candidate_images",
                "exact_groups",
                "excluded_images",
                "pilot_overlap_images",
            ),
        )
        write_csv(
            reports_root / "pilot_leakage_report.csv",
            report_rows["pilot_leakage_report"],
            DUPLICATE_COLUMNS,
        )

        source_fingerprint_after = source_files_fingerprint(source_paths)
        if source_fingerprint_before != source_fingerprint_after:
            raise RuntimeError("Las fuentes originales cambiaron durante la consolidación")
        config_safety = _config_safety(config_path)
        if config_safety != {"model": "yolo26n-seg", "output_profile": "mask_black"}:
            raise RuntimeError(f"Configuración de seguridad inesperada: {config_safety}")

        variant_counts = Counter(
            str(row["roboflow_variant_group"]) for row in image_rows
        )
        counts = {
            "images_considered": len(image_rows),
            "images_included": len(included_images),
            "images_excluded": len(image_rows) - len(included_images),
            "manifest_rows": len(manifest_rows),
            "source_annotation_rows": sum(
                bool(str(row["original_line_number"])) for row in manifest_rows
            ),
            "annotations_included": len(included_rows),
            "annotations_excluded": len(excluded_rows),
            "annotations_recovered": len(recovered_rows),
            "manual_review_rows": len(manual_rows),
            "mandatory_visual_review_rows": len(mandatory_visual_rows),
            "duplicates_excluded": sum(
                row["decision"] == "exclude_duplicate" for row in duplicate_rows
            ),
            "pilot_leakage": sum(
                row["decision"] == "exclude_pilot_leakage"
                for row in duplicate_rows
            ),
            "roboflow_variant_groups": len(variant_counts),
            "roboflow_multi_variant_groups": sum(
                count > 1 for count in variant_counts.values()
            ),
            "previews": len(preview_paths),
            "jpeg_images": len(image_normalization_rows),
            "jpeg_images_normalized": sum(
                row["status"] == "normalized"
                for row in image_normalization_rows
            ),
            "jpeg_lossless_eoi_repairs": sum(
                row["normalization_method"] == "append_ffd9"
                for row in image_normalization_rows
            ),
            "jpeg_reencodes": sum(
                str(row["normalization_method"]).startswith("reencode_")
                for row in image_normalization_rows
            ),
        }
        summary = {
            "schema_version": 2,
            "name": "doctor_maiz_leaf_segmentation_consolidation",
            "seed": seed,
            "target_class": {
                "id": TARGET_CLASS_ID,
                "name": TARGET_CLASS_NAME,
            },
            "paths": {
                "external_root": str(external_root),
                "pilot_root": str(pilot_root),
                "eda_root": str(eda_root),
                "dataset_root": str(dataset_root),
                "report_root": str(report_root),
            },
            "eda_decisions": list(decisions.values()),
            "eda_decision_summary_sha256": sha256_file(
                eda_root / "decision_summary.md"
            ),
            "eda_input_fingerprint": current_eda_fingerprint,
            "counts": counts,
            "validation": validation_summary,
            "review_preview_validation": {
                key: value
                for key, value in review_preview_summary.items()
                if key != "cases"
            },
            "source_fingerprint_before": source_fingerprint_before,
            "source_fingerprint_after": source_fingerprint_after,
            "source_files_unchanged": True,
            "pilot_image_count": len(pilot_paths),
            "config_safety": config_safety,
            "splits_created": False,
            "training_performed": False,
            "ultralytics_installed_by_builder": False,
            "weights_downloaded": False,
            "segmentation_configured": True,
        }
        (reports_root / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (staging / "dataset.yaml").write_text(_dataset_yaml(), encoding="utf-8")
        current_readme = (
            (dataset_root / "README.md").read_text(encoding="utf-8")
            if (dataset_root / "README.md").is_file()
            else "# Dataset del detector de hojas\n"
        )
        (staging / "README.md").write_text(
            _readme(current_readme, summary),
            encoding="utf-8",
        )

        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "manifests").mkdir(parents=True, exist_ok=True)
        shutil.move(str(all_root), dataset_root / "all")
        shutil.move(str(previews_root), dataset_root / "previews")
        shutil.move(str(staging / "dataset.yaml"), dataset_root / "dataset.yaml")
        shutil.move(str(staging / "README.md"), dataset_root / "README.md")
        for name in CONSOLIDATION_ARTIFACTS:
            shutil.move(
                str(manifests_root / name),
                dataset_root / "manifests" / name,
            )
        shutil.move(str(reports_root), report_root)
    return summary
