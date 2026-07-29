"""Deterministic JPEG normalization and Ultralytics mutation checks."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image, ImageOps

JPEG_EXTENSIONS = {".jpg", ".jpeg"}
IMAGE_NORMALIZATION_COLUMNS = (
    "source_path",
    "derived_path",
    "issue",
    "normalization_method",
    "original_sha256",
    "normalized_sha256",
    "original_pixel_sha256",
    "normalized_pixel_sha256",
    "original_mode",
    "normalized_mode",
    "width",
    "height",
    "pixel_equivalence_verified",
    "status",
)
AUXILIARY_JPEG_AUDIT_COLUMNS = (
    "path",
    "issue",
    "sha256",
    "soi",
    "eoi",
    "pillow_verify_error",
    "image_load_error",
    "width",
    "height",
    "ultralytics_would_repair",
    "training_gate",
    "status",
)
CANONICAL_IMAGE_ROOTS = (
    Path("all/images"),
    Path("images/train"),
    Path("images/val"),
    Path("images/test"),
)
AUXILIARY_IMAGE_ROOTS = (
    Path("annotation_batches"),
    Path("test/images"),
)
ULTRALYTICS_VERSION = "8.4.104"


class JpegNormalizationError(RuntimeError):
    """Raised when a JPEG cannot be made canonical without changing its source."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_jpeg(path: Path) -> dict[str, object]:
    """Inspect JPEG markers and run both Pillow verification and full decoding."""
    path = path.resolve()
    if path.suffix.lower() not in JPEG_EXTENSIONS:
        raise ValueError(f"No es una ruta JPEG: {path}")
    with path.open("rb") as handle:
        soi = handle.read(2) == b"\xff\xd8"
        try:
            handle.seek(-2, os.SEEK_END)
            eoi = handle.read(2) == b"\xff\xd9"
        except OSError:
            eoi = False

    verify_error: str | None = None
    load_error: str | None = None
    image_format: str | None = None
    width: int | None = None
    height: int | None = None
    try:
        with Image.open(path) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
    except Exception as exc:  # Pillow exposes multiple corruption exception types.
        verify_error = type(exc).__name__
    try:
        with Image.open(path) as image:
            image.load()
            image_format = image.format
            width, height = image.size
    except Exception as exc:  # Full load catches truncation that verify() can miss.
        load_error = type(exc).__name__

    issues: list[str] = []
    if not soi:
        issues.append("missing_jpeg_soi")
    if not eoi:
        issues.append("missing_jpeg_eoi")
    if verify_error:
        issues.append(f"pillow_verify_error:{verify_error}")
    if load_error:
        issues.append(f"image_load_error:{load_error}")
    if image_format not in {"JPEG", None}:
        issues.append(f"unexpected_pillow_format:{image_format}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "soi": soi,
        "eoi": eoi,
        "verify_error": verify_error,
        "load_error": load_error,
        "format": image_format,
        "width": width,
        "height": height,
        "issues": issues,
        "ultralytics_would_repair": (
            image_format == "JPEG"
            and verify_error is None
            and not eoi
        ),
    }


def _pixel_snapshot(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        canonical = ImageOps.exif_transpose(image).convert("RGB")
        canonical.load()
        digest = hashlib.sha256()
        digest.update(f"{canonical.width}x{canonical.height}\0RGB\0".encode())
        digest.update(canonical.tobytes())
        return {
            "sha256": digest.hexdigest(),
            "mode": canonical.mode,
            "width": canonical.width,
            "height": canonical.height,
        }


def _reencode_jpeg(source: Path, derived: Path) -> None:
    with Image.open(source) as image:
        canonical = ImageOps.exif_transpose(image).convert("RGB")
        canonical.load()
        canonical.save(
            derived,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False,
        )


def normalize_jpeg_copy(
    source: Path,
    derived: Path,
    *,
    source_path: str | None = None,
    derived_path: str | None = None,
) -> dict[str, object]:
    """Create and validate one canonical derived JPEG without editing its source."""
    source = source.resolve()
    derived = derived.resolve()
    if source == derived:
        raise JpegNormalizationError("La fuente y la copia derivada deben ser distintas")
    original = inspect_jpeg(source)
    if original["load_error"]:
        raise JpegNormalizationError(
            f"No se puede recodificar un JPEG que Pillow no carga: {source}; "
            f"issues={original['issues']}"
        )
    if original["format"] != "JPEG":
        raise JpegNormalizationError(
            f"El sufijo es JPEG pero Pillow detectó {original['format']!r}: {source}"
        )
    if (
        not isinstance(original["width"], int)
        or not isinstance(original["height"], int)
        or original["width"] <= 0
        or original["height"] <= 0
    ):
        raise JpegNormalizationError(f"Dimensiones JPEG inválidas: {source}")

    derived.parent.mkdir(parents=True, exist_ok=True)
    issues = list(original["issues"])
    if not issues:
        shutil.copyfile(source, derived)
        method = "copy_unchanged"
    elif issues == ["missing_jpeg_eoi"]:
        shutil.copyfile(source, derived)
        with derived.open("ab") as handle:
            handle.write(b"\xff\xd9")
        method = "append_ffd9"
    else:
        _reencode_jpeg(source, derived)
        method = "reencode_exif_transpose_rgb_q95_444"

    normalized = inspect_jpeg(derived)
    if normalized["issues"]:
        raise JpegNormalizationError(
            f"La copia normalizada sigue siendo inválida: {derived}; "
            f"issues={normalized['issues']}"
        )
    original_pixels = _pixel_snapshot(source)
    normalized_pixels = _pixel_snapshot(derived)
    pixel_equivalent = original_pixels == normalized_pixels
    if method in {"copy_unchanged", "append_ffd9"} and not pixel_equivalent:
        raise JpegNormalizationError(
            "La normalización supuestamente sin pérdida cambió los píxeles: "
            f"{source} -> {derived}"
        )
    return {
        "source_path": source_path or str(source),
        "derived_path": derived_path or str(derived),
        "issue": ";".join(issues) if issues else "none",
        "normalization_method": method,
        "original_sha256": original["sha256"],
        "normalized_sha256": normalized["sha256"],
        "original_pixel_sha256": original_pixels["sha256"],
        "normalized_pixel_sha256": normalized_pixels["sha256"],
        "original_mode": original_pixels["mode"],
        "normalized_mode": normalized_pixels["mode"],
        "width": normalized["width"],
        "height": normalized["height"],
        "pixel_equivalence_verified": pixel_equivalent,
        "status": "normalized" if issues else "unchanged",
    }


def audit_jpegs(paths: Iterable[Path]) -> dict[str, object]:
    """Audit a deterministic path collection and summarize every hard JPEG issue."""
    inspections = [
        inspect_jpeg(path)
        for path in sorted(
            (path.resolve() for path in paths),
            key=lambda path: path.as_posix(),
        )
    ]
    problematic = [row for row in inspections if row["issues"]]
    return {
        "jpeg_count": len(inspections),
        "problem_count": len(problematic),
        "missing_soi_count": sum(not row["soi"] for row in inspections),
        "missing_eoi_count": sum(not row["eoi"] for row in inspections),
        "verify_error_count": sum(
            row["verify_error"] is not None for row in inspections
        ),
        "load_error_count": sum(
            row["load_error"] is not None for row in inspections
        ),
        "ultralytics_repair_count": sum(
            bool(row["ultralytics_would_repair"]) for row in inspections
        ),
        "problems": problematic,
    }


def canonical_jpeg_paths(dataset_root: Path) -> list[Path]:
    """Return only JPEGs that belong to the parent pool or materialized splits."""
    dataset_root = dataset_root.resolve()
    return sorted(
        (
            path
            for relative_root in CANONICAL_IMAGE_ROOTS
            for path in (dataset_root / relative_root).rglob("*")
            if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS
        ),
        key=lambda path: path.relative_to(dataset_root).as_posix(),
    )


def write_auxiliary_jpeg_audit(
    dataset_root: Path,
    output_path: Path,
) -> dict[str, object]:
    """Document auxiliary JPEG issues without making them a training gate."""
    dataset_root = dataset_root.resolve()
    inspected = [
        inspect_jpeg(path)
        for relative_root in AUXILIARY_IMAGE_ROOTS
        for path in sorted(
            (
                candidate
                for candidate in (dataset_root / relative_root).rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in JPEG_EXTENSIONS
            ),
            key=lambda candidate: candidate.relative_to(dataset_root).as_posix(),
        )
    ]
    rows = [
        {
            "path": Path(str(row["path"]))
            .relative_to(dataset_root)
            .as_posix(),
            "issue": "missing_jpeg_eoi",
            "sha256": row["sha256"],
            "soi": row["soi"],
            "eoi": row["eoi"],
            "pillow_verify_error": row["verify_error"] or "",
            "image_load_error": row["load_error"] or "",
            "width": row["width"],
            "height": row["height"],
            "ultralytics_would_repair": row["ultralytics_would_repair"],
            "training_gate": False,
            "status": "documented_auxiliary_issue",
        }
        for row in inspected
        if not row["eoi"]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=AUXILIARY_JPEG_AUDIT_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        "jpeg_count": len(inspected),
        "issue_count": len(rows),
        "missing_eoi_count": sum(
            row["issue"] == "missing_jpeg_eoi" for row in rows
        ),
        "verify_error_count": sum(bool(row["pillow_verify_error"]) for row in rows),
        "load_error_count": sum(bool(row["image_load_error"]) for row in rows),
        "training_gate": False,
        "report_path": str(output_path),
    }


def _load_ultralytics_checker() -> tuple[Callable[[str], tuple[str, tuple[int, int]]], str]:
    config_root = (
        Path(tempfile.gettempdir()) / "doctor_maiz_ultralytics_config"
    )
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(config_root),
    )
    import ultralytics
    from ultralytics.data.utils import check_image

    version = str(ultralytics.__version__)
    if version != ULTRALYTICS_VERSION:
        raise JpegNormalizationError(
            f"Ultralytics inesperado para el escaneo: {version} != {ULTRALYTICS_VERSION}"
        )
    return check_image, version


def ultralytics_scan_hash_mutations(
    image_root: Path,
    *,
    paths: Sequence[Path] | None = None,
    checker: Callable[[str], tuple[str, tuple[int, int]]] | None = None,
    checker_version: str | None = None,
) -> dict[str, object]:
    """Run Ultralytics' real image checker on a copy and compare every hash."""
    image_root = image_root.resolve()
    if checker is None:
        checker, checker_version = _load_ultralytics_checker()
    candidates = (
        paths
        if paths is not None
        else [
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS
        ]
    )
    image_paths = sorted(
        (path.resolve() for path in candidates),
        key=lambda path: path.relative_to(image_root).as_posix(),
    )
    with tempfile.TemporaryDirectory(prefix=".doctor_maiz_ultralytics_jpeg_scan_") as temp:
        scan_root = Path(temp) / "images"
        scan_root.mkdir()
        before = {
            path.relative_to(image_root).as_posix(): _sha256(path)
            for path in image_paths
        }
        for path in image_paths:
            target = scan_root / path.relative_to(image_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        messages: list[str] = []
        for relative in sorted(before):
            message, _ = checker(str(scan_root / relative))
            if message:
                messages.append(message)
        after = {
            relative: _sha256(scan_root / relative)
            for relative in sorted(before)
        }
    mutations = [
        {
            "path": relative,
            "before_sha256": before[relative],
            "after_sha256": after[relative],
        }
        for relative in sorted(before)
        if before[relative] != after[relative]
    ]
    return {
        "scanner": "ultralytics.data.utils.check_image",
        "ultralytics_version": checker_version,
        "scanned_file_count": len(before),
        "message_count": len(messages),
        "messages": messages,
        "mutated_file_count": len(mutations),
        "mutations": mutations,
        "passed": not mutations,
    }


def validate_jpegs_before_packaging(dataset_root: Path) -> dict[str, object]:
    """Gate only the canonical parent pool and train/val/test materializations."""
    dataset_root = dataset_root.resolve()
    paths = canonical_jpeg_paths(dataset_root)
    audit = audit_jpegs(paths)
    if audit["problem_count"]:
        raise JpegNormalizationError(
            f"JPEG inválidos antes del empaquetado: {audit['problems']}"
        )
    scan = ultralytics_scan_hash_mutations(dataset_root, paths=paths)
    if not scan["passed"]:
        raise JpegNormalizationError(
            f"Ultralytics modificó JPEG en la copia temporal: {scan['mutations']}"
        )
    return {
        "canonical_roots": [
            relative.as_posix()
            for relative in CANONICAL_IMAGE_ROOTS
            if (dataset_root / relative).is_dir()
        ],
        "audit": audit,
        "ultralytics_scan": scan,
        "passed": True,
    }
