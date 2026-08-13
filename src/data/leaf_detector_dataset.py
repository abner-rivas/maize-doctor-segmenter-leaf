"""Reproducible preparation of annotation data for the maize-leaf detector."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from src.data.leaf_pilot import read_csv_rows, require_columns, sha256_file

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
SELECTION_COLUMNS = (
    "detector_id",
    "original_image_path",
    "copied_image_path",
    "image_sha256",
    "source_label",
    "source_split",
    "environment",
    "source_dataset",
    "annotation_status",
    "selection_seed",
    "selection_reason",
    "image_width",
    "image_height",
    "aspect_ratio",
    "orientation",
    "resolution_bucket",
    "aspect_ratio_bucket",
    "manual_review_required",
    "pilot_id",
    "cvat_xml_source",
    "original_roi_x1",
    "original_roi_y1",
    "original_roi_x2",
    "original_roi_y2",
    "original_rotation_degrees",
    "roi_conversion_method",
    "roi_clipped",
    "roi_area_ratio",
    "annotation_notes",
)

CVAT_GUIDE = """# Guía de anotación para el detector de hojas

## Clase

`0 = maize_leaf`

## Regla

Anotar **todas las hojas de maíz visibles y suficientemente claras**. Una
fotografía puede y debe contener varias cajas cuando muestre varias hojas aptas.

No anotar tallos, suelo, manos, cielo ni lesiones aisladas. Evitar cajas
degeneradas o sin área. Marcar como `ambiguous` las fotografías cuya decisión
requiera una elección arbitraria y como `rejected` las imágenes imposibles de
anotar.

Esta regla sustituye, únicamente para el detector, la regla del piloto anterior
que anotaba sólo la hoja principal. Las etiquetas semánticas de la fuente no se
anotan aquí.

Las imágenes de este paquete no incluyen etiquetas: su estado inicial es
`pending` hasta completar y validar la anotación en CVAT.
"""

DATASET_YAML_TEMPLATE = """path: data/leaf_detection/detector_dataset/yolo
train: images/train
val: images/val
test: images/test

names:
  0: maize_leaf
"""

DATASET_README = """# Dataset inicial del detector de hojas

Este directorio prepara lotes de anotación para un futuro detector Ultralytics
YOLO26n. No contiene un dataset YOLO entrenable todavía: `train` y `val` no
tienen etiquetas y permanecen con `annotation_status=pending`.

- `annotation_batches/`: imágenes nuevas de los splits oficiales train y val.
- `test/`: piloto manual retenido; 99 casos anotados y un caso ambiguo.
- `manifests/`: selección reproducible, metadatos y auditoría de fugas.
- `cvat/`: paquetes separados para anotar train y val.
- `dataset.yaml.template`: plantilla que sólo debe materializarse después de
  validar las anotaciones.

El detector localiza hojas (`maize_leaf`); no clasifica enfermedades. El piloto
procede del test oficial y nunca debe entrar en train o val.
"""


@dataclass(frozen=True)
class ImageCandidate:
    """One verified split image with automatic, non-semantic diversity signals."""

    image_path: str
    label: str
    environment: str
    source_dataset: str
    source: Path
    image_sha256: str
    width: int
    height: int
    orientation: str
    resolution_bucket: str
    aspect_ratio_bucket: str

    @property
    def filename_key(self) -> str:
        return self.source.name.casefold()

    @property
    def path_key(self) -> str:
        return Path(self.image_path).as_posix().casefold()

    @property
    def diversity_key(self) -> tuple[str, str, str]:
        return self.orientation, self.resolution_bucket, self.aspect_ratio_bucket


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_audit() -> dict[str, object]:
    """Collect dependency metadata without installing packages or loading weights."""
    torch_version = _package_version("torch")
    torchvision_version = _package_version("torchvision")
    cuda_reported: str | None = None
    cuda_available: bool | None = None
    if torch_version:
        try:
            import torch

            cuda_reported = torch.version.cuda
            cuda_available = torch.cuda.is_available()
        except (ImportError, OSError):
            pass
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch_version,
        "torchvision": torchvision_version,
        "cuda_reported": cuda_reported,
        "cuda_available": cuda_available,
        "ultralytics_installed": _package_version("ultralytics"),
        "ultralytics_candidate": "8.4.104",
        "candidate_declared_python": ">=3.8",
        "candidate_declared_torch": ">=1.8.0",
        "candidate_declared_torchvision": ">=0.9.0",
        "repository_python": ">=3.11",
        "repository_torch": ">=2.2,<2.13",
        "repository_torchvision": ">=0.17,<0.28",
        "declared_constraints_compatible": True,
        "environment_ready_for_ultralytics_audit": bool(
            torch_version and torchvision_version
        ),
        "packages_installed_by_builder": False,
        "weights_downloaded_by_builder": False,
        "training_performed": False,
    }


def _image_buckets(width: int, height: int) -> tuple[str, str, str]:
    ratio = width / height
    orientation = "portrait" if ratio < 0.9 else "landscape" if ratio > 1.1 else "square"
    pixels = width * height
    resolution = "small" if pixels < 500_000 else "medium" if pixels < 2_000_000 else "large"
    aspect = "narrow" if ratio < 0.67 else "wide" if ratio > 1.5 else "moderate"
    return orientation, resolution, aspect


def _read_image_metadata(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).size
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Imagen inválida: {path}: {exc}") from exc


def scan_split_candidates(
    split_csv: Path,
    dataset_root: Path,
    *,
    excluded_paths: set[str] | None = None,
    excluded_hashes: set[str] | None = None,
    excluded_names: set[str] | None = None,
) -> tuple[list[ImageCandidate], dict[str, int]]:
    """Validate, hash and describe split rows, excluding known held-out images."""
    rows, columns = read_csv_rows(split_csv)
    require_columns(columns, ("image_path", "label", "environment"), f"split {split_csv}")
    paths = excluded_paths or set()
    hashes = excluded_hashes or set()
    names = excluded_names or set()
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    candidates: list[ImageCandidate] = []
    exclusions: Counter[str] = Counter()

    for row in rows:
        image_path = Path(row["image_path"]).as_posix()
        path_key = image_path.casefold()
        source = (
            Path(image_path)
            if Path(image_path).is_absolute()
            else dataset_root / image_path
        ).resolve()
        if source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            exclusions["unsupported_extension"] += 1
            continue
        if path_key in paths:
            exclusions["held_out_path"] += 1
            continue
        if source.name.casefold() in names:
            exclusions["held_out_filename"] += 1
            continue
        if path_key in seen_paths:
            exclusions["duplicate_path"] += 1
            continue
        if not source.is_file():
            raise FileNotFoundError(f"No existe la imagen del split: {source}")
        digest = sha256_file(source)
        if digest in hashes:
            exclusions["held_out_hash"] += 1
            continue
        if digest in seen_hashes:
            exclusions["duplicate_hash"] += 1
            continue
        width, height = _read_image_metadata(source)
        orientation, resolution, aspect = _image_buckets(width, height)
        candidates.append(
            ImageCandidate(
                image_path=image_path,
                label=row["label"],
                environment=row["environment"],
                source_dataset=row.get("source_dataset", "").strip() or "unknown",
                source=source,
                image_sha256=digest,
                width=width,
                height=height,
                orientation=orientation,
                resolution_bucket=resolution,
                aspect_ratio_bucket=aspect,
            )
        )
        seen_paths.add(path_key)
        seen_hashes.add(digest)
    return candidates, dict(sorted(exclusions.items()))


def _interleave_diversity(
    candidates: Sequence[ImageCandidate], seed: int
) -> list[ImageCandidate]:
    buckets: dict[tuple[str, str, str], list[ImageCandidate]] = defaultdict(list)
    rng = random.Random(seed)
    for candidate in candidates:
        buckets[candidate.diversity_key].append(candidate)
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
    ordered: list[ImageCandidate] = []
    keys = sorted(buckets)
    position = 0
    while True:
        added = False
        for key in keys:
            if position < len(buckets[key]):
                ordered.append(buckets[key][position])
                added = True
        if not added:
            break
        position += 1
    return ordered


def _balanced_by_class(
    candidates: Sequence[ImageCandidate], count: int, seed: int
) -> list[ImageCandidate]:
    groups: dict[str, list[ImageCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.label].append(candidate)
    for index, label in enumerate(sorted(groups)):
        groups[label] = _interleave_diversity(groups[label], seed + index)
    selected: list[ImageCandidate] = []
    position = 0
    labels = sorted(groups)
    while len(selected) < count:
        added = False
        for label in labels:
            if position < len(groups[label]) and len(selected) < count:
                selected.append(groups[label][position])
                added = True
        if not added:
            break
        position += 1
    return selected


def select_detector_candidates(
    candidates: Sequence[ImageCandidate],
    count: int,
    seed: int,
    *,
    real_fraction: float = 0.8,
) -> list[ImageCandidate]:
    """Select deterministic class and metadata diversity with controlled lab coverage."""
    if count <= 0:
        raise ValueError("count debe ser mayor que cero")
    if not 0 <= real_fraction <= 1:
        raise ValueError("real_fraction debe estar entre 0 y 1")
    if len(candidates) < count:
        raise ValueError(f"Se requieren {count} imágenes y sólo hay {len(candidates)}")

    labels = sorted({item.label for item in candidates})
    class_quotas = {
        label: count // len(labels) + (index < count % len(labels))
        for index, label in enumerate(labels)
    }
    lab_target = count - round(count * real_fraction)
    lab_by_class = {
        label: [item for item in candidates if item.label == label and item.environment == "lab"]
        for label in labels
    }
    lab_quotas = {label: 0 for label in labels}
    while sum(lab_quotas.values()) < lab_target:
        added = False
        for label in labels:
            capacity = min(class_quotas[label], len(lab_by_class[label]))
            if lab_quotas[label] < capacity and sum(lab_quotas.values()) < lab_target:
                lab_quotas[label] += 1
                added = True
        if not added:
            break

    selected: list[ImageCandidate] = []
    for index, label in enumerate(labels):
        class_candidates = [item for item in candidates if item.label == label]
        lab = [item for item in class_candidates if item.environment == "lab"]
        real = [item for item in class_candidates if item.environment == "real"]
        selected_lab = _interleave_diversity(lab, seed + 10_000 + index)[
            : lab_quotas[label]
        ]
        real_needed = class_quotas[label] - len(selected_lab)
        selected_real = _interleave_diversity(real, seed + index)[:real_needed]
        class_selected = [*selected_real, *selected_lab]
        if len(class_selected) < class_quotas[label]:
            class_hashes = {item.image_sha256 for item in class_selected}
            remaining_class = [
                item
                for item in class_candidates
                if item.image_sha256 not in class_hashes
            ]
            class_selected.extend(
                _interleave_diversity(remaining_class, seed + 20_000 + index)[
                    : class_quotas[label] - len(class_selected)
                ]
            )
        selected.extend(class_selected)

    selected_hashes = {item.image_sha256 for item in selected}
    if len(selected) < count:
        remaining = [
            item for item in candidates if item.image_sha256 not in selected_hashes
        ]
        selected.extend(_balanced_by_class(remaining, count - len(selected), seed + 20_000))
    if len(selected) != count:
        raise RuntimeError(f"Selección incompleta: {len(selected)} de {count}")
    random.Random(seed + 30_000).shuffle(selected)
    return selected


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _selection_row(
    candidate: ImageCandidate,
    detector_id: str,
    copied_path: Path,
    split: str,
    seed: int,
) -> dict[str, object]:
    reason = (
        f"source_label_environment_balance;orientation={candidate.orientation};"
        f"resolution={candidate.resolution_bucket};"
        f"aspect={candidate.aspect_ratio_bucket};manual_visual_review=pending"
    )
    return {
        "detector_id": detector_id,
        "original_image_path": candidate.image_path,
        "copied_image_path": copied_path.as_posix(),
        "image_sha256": candidate.image_sha256,
        "source_label": candidate.label,
        "source_split": split,
        "environment": candidate.environment,
        "source_dataset": candidate.source_dataset,
        "annotation_status": "pending",
        "selection_seed": seed,
        "selection_reason": reason,
        "image_width": candidate.width,
        "image_height": candidate.height,
        "aspect_ratio": f"{candidate.width / candidate.height:.6f}",
        "orientation": candidate.orientation,
        "resolution_bucket": candidate.resolution_bucket,
        "aspect_ratio_bucket": candidate.aspect_ratio_bucket,
        "manual_review_required": (
            "background_complexity;apparent_leaf_size;multiple_leaves;partial_leaves"
        ),
    }


def _copy_candidates(
    selected: Sequence[ImageCandidate],
    output_root: Path,
    split: str,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(selected, start=1):
        detector_id = f"{split}_{index:04d}"
        relative = (
            Path("annotation_batches")
            / split
            / "images"
            / f"{detector_id}{candidate.source.suffix.lower()}"
        )
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"No se sobrescribirá: {destination}")
        shutil.copy2(candidate.source, destination)
        if sha256_file(destination) != candidate.image_sha256:
            raise RuntimeError(f"Hash distinto después de copiar {destination}")
        rows.append(_selection_row(candidate, detector_id, relative, split, seed))
    return rows


def _pilot_exclusion_sets(
    imported_rows: Sequence[dict[str, str]],
) -> tuple[set[str], set[str], set[str]]:
    paths = {
        Path(row["original_image_path"]).as_posix().casefold() for row in imported_rows
    }
    hashes = {row["image_sha256"] for row in imported_rows if row.get("image_sha256")}
    names = {
        Path(row["original_image_path"]).name.casefold() for row in imported_rows
    }
    return paths, hashes, names


def _materialize_test(
    imported_rows: Sequence[dict[str, str]],
    pilot_root: Path,
    cvat_xml: Path,
    output_root: Path,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in imported_rows:
        pilot_id = row["pilot_id"]
        source = pilot_root / row["pilot_image_path"]
        suffix = source.suffix.lower()
        relative = Path("test") / "images" / f"{pilot_id}{suffix}"
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha256_file(destination)
        if digest != row["image_sha256"]:
            raise RuntimeError(f"Hash distinto en test retenido: {pilot_id}")
        width, height = _read_image_metadata(destination)
        orientation, resolution, aspect = _image_buckets(width, height)
        status = row["annotation_status"]
        if status == "annotated":
            x1 = float(row["roi_x1"])
            y1 = float(row["roi_y1"])
            x2 = float(row["roi_x2"])
            y2 = float(row["roi_y2"])
            label = output_root / "test" / "labels" / f"{pilot_id}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            label.write_text(
                f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}\n",
                encoding="utf-8",
            )
        rows.append(
            {
                "detector_id": f"test_{pilot_id}",
                "original_image_path": row["original_image_path"],
                "copied_image_path": relative.as_posix(),
                "image_sha256": digest,
                "source_label": row["label"],
                "source_split": "test",
                "environment": row["environment"],
                "source_dataset": row.get("source_dataset", "") or "unknown",
                "annotation_status": status,
                "selection_seed": seed,
                "selection_reason": "held_out_manual_pilot",
                "image_width": width,
                "image_height": height,
                "aspect_ratio": f"{width / height:.6f}",
                "orientation": orientation,
                "resolution_bucket": resolution,
                "aspect_ratio_bucket": aspect,
                "manual_review_required": (
                    "all_visible_leaves_under_new_detector_rule"
                ),
                "pilot_id": pilot_id,
                "cvat_xml_source": str(cvat_xml.resolve()),
                "original_roi_x1": row.get("roi_x1", ""),
                "original_roi_y1": row.get("roi_y1", ""),
                "original_roi_x2": row.get("roi_x2", ""),
                "original_roi_y2": row.get("roi_y2", ""),
                "original_rotation_degrees": row.get("original_rotation_degrees", ""),
                "roi_conversion_method": row.get("roi_conversion_method", ""),
                "roi_clipped": row.get("roi_clipped", ""),
                "roi_area_ratio": row.get("roi_area_ratio", ""),
                "annotation_notes": row.get("notes", ""),
            }
        )
    return rows


def _overlap(
    left: Sequence[dict[str, object]],
    right: Sequence[dict[str, object]],
    column: str,
    *,
    basename: bool = False,
) -> list[str]:
    def value(row: dict[str, object]) -> str:
        raw = str(row[column])
        return Path(raw).name.casefold() if basename else Path(raw).as_posix().casefold()

    return sorted({value(row) for row in left}.intersection(value(row) for row in right))


def build_leakage_report(
    train_rows: Sequence[dict[str, object]],
    val_rows: Sequence[dict[str, object]],
    test_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Compare original paths, filenames and SHA-256 across detector partitions."""
    partitions = {"train": train_rows, "val": val_rows, "test": test_rows}
    pairs: dict[str, object] = {}
    total = 0
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        path_overlap = _overlap(partitions[left], partitions[right], "original_image_path")
        name_overlap = _overlap(
            partitions[left], partitions[right], "original_image_path", basename=True
        )
        hash_overlap = sorted(
            {str(row["image_sha256"]) for row in partitions[left]}.intersection(
                str(row["image_sha256"]) for row in partitions[right]
            )
        )
        count = len(path_overlap) + len(name_overlap) + len(hash_overlap)
        total += count
        pairs[f"{left}_vs_{right}"] = {
            "path_overlap_count": len(path_overlap),
            "filename_overlap_count": len(name_overlap),
            "sha256_overlap_count": len(hash_overlap),
            "path_overlaps": path_overlap,
            "filename_overlaps": name_overlap,
            "sha256_overlaps": hash_overlap,
        }
    return {
        "schema_version": 1,
        "checks": ["original_image_path", "original_filename", "image_sha256"],
        "pairs": pairs,
        "total_overlap_signals": total,
        "zero_leakage": total == 0,
        "source_split_valid": {
            name: all(str(row["source_split"]) == name for row in rows)
            for name, rows in partitions.items()
        },
    }


def _write_deterministic_zip(
    path: Path,
    rows: Sequence[dict[str, object]],
    output_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        guide_info = zipfile.ZipInfo("annotation_guide.md", (2026, 1, 1, 0, 0, 0))
        guide_info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(guide_info, CVAT_GUIDE.encode("utf-8"))
        for row in sorted(rows, key=lambda item: str(item["detector_id"])):
            source = output_root / str(row["copied_image_path"])
            info = zipfile.ZipInfo(
                f"images/{source.name}",
                (2026, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())


def _distribution(rows: Sequence[dict[str, object]], column: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[column]) for row in rows).items()))


def build_detector_annotation_set(
    train_csv: Path,
    val_csv: Path,
    dataset_root: Path,
    pilot_root: Path,
    imported_annotations: Path,
    cvat_xml: Path,
    output_root: Path,
    *,
    train_count: int = 350,
    val_count: int = 75,
    seed: int = 42,
    real_fraction: float = 0.8,
) -> dict[str, object]:
    """Create detector annotation batches and retained-test artifacts without training."""
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"La salida ya existe y no será sobrescrita: {output_root}")
    imported_rows, imported_columns = read_csv_rows(imported_annotations)
    require_columns(
        imported_columns,
        (
            "pilot_id",
            "pilot_image_path",
            "original_image_path",
            "image_sha256",
            "label",
            "environment",
            "annotation_status",
            "roi_x1",
            "roi_y1",
            "roi_x2",
            "roi_y2",
            "original_rotation_degrees",
            "roi_conversion_method",
            "roi_clipped",
            "roi_area_ratio",
        ),
        "imported annotations",
    )
    if not imported_rows:
        raise ValueError("El manifiesto importado del piloto está vacío")
    if not cvat_xml.is_file():
        raise FileNotFoundError(f"No existe el XML de CVAT: {cvat_xml}")

    pilot_paths, pilot_hashes, pilot_names = _pilot_exclusion_sets(imported_rows)
    train_candidates, train_exclusions = scan_split_candidates(
        train_csv,
        dataset_root,
        excluded_paths=pilot_paths,
        excluded_hashes=pilot_hashes,
        excluded_names=pilot_names,
    )
    train_selected = select_detector_candidates(
        train_candidates, train_count, seed, real_fraction=real_fraction
    )
    train_paths = {item.path_key for item in train_selected}
    train_hashes = {item.image_sha256 for item in train_selected}
    train_names = {item.filename_key for item in train_selected}
    val_candidates, val_exclusions = scan_split_candidates(
        val_csv,
        dataset_root,
        excluded_paths=pilot_paths | train_paths,
        excluded_hashes=pilot_hashes | train_hashes,
        excluded_names=pilot_names | train_names,
    )
    val_selected = select_detector_candidates(
        val_candidates, val_count, seed, real_fraction=real_fraction
    )

    try:
        output_root.mkdir(parents=True)
        train_rows = _copy_candidates(train_selected, output_root, "train", seed)
        val_rows = _copy_candidates(val_selected, output_root, "val", seed)
        test_rows = _materialize_test(
            imported_rows, pilot_root, cvat_xml, output_root, seed
        )
        manifests = output_root / "manifests"
        _write_csv(manifests / "train_selection.csv", train_rows)
        _write_csv(manifests / "val_selection.csv", val_rows)
        _write_csv(manifests / "test_selection.csv", test_rows)
        leakage = build_leakage_report(train_rows, val_rows, test_rows)
        (manifests / "leakage_report.json").write_text(
            json.dumps(leakage, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_root / "dataset.yaml.template").write_text(
            DATASET_YAML_TEMPLATE, encoding="utf-8"
        )
        (output_root / "README.md").write_text(DATASET_README, encoding="utf-8")
        _write_deterministic_zip(
            output_root / "cvat" / "train_annotation_batch.zip",
            train_rows,
            output_root,
        )
        _write_deterministic_zip(
            output_root / "cvat" / "val_annotation_batch.zip",
            val_rows,
            output_root,
        )
        summary = {
            "schema_version": 1,
            "seed": seed,
            "real_fraction_target": real_fraction,
            "dataset_root": str(dataset_root.resolve()),
            "train_csv": str(train_csv.resolve()),
            "val_csv": str(val_csv.resolve()),
            "pilot_imported_annotations": str(imported_annotations.resolve()),
            "pilot_cvat_xml": str(cvat_xml.resolve()),
            "pilot_role": "held_out_test_only",
            "counts": {
                "train": len(train_rows),
                "val": len(val_rows),
                "test_total_documented": len(test_rows),
                "test_annotated": sum(
                    row["annotation_status"] == "annotated" for row in test_rows
                ),
                "test_ambiguous": sum(
                    row["annotation_status"] == "ambiguous" for row in test_rows
                ),
            },
            "train_distribution": {
                "source_label": _distribution(
                    train_rows, "source_label"
                ),
                "environment": _distribution(train_rows, "environment"),
                "orientation": _distribution(train_rows, "orientation"),
                "resolution_bucket": _distribution(train_rows, "resolution_bucket"),
            },
            "val_distribution": {
                "source_label": _distribution(val_rows, "source_label"),
                "environment": _distribution(val_rows, "environment"),
                "orientation": _distribution(val_rows, "orientation"),
                "resolution_bucket": _distribution(val_rows, "resolution_bucket"),
            },
            "candidate_exclusions": {
                "train": train_exclusions,
                "val": val_exclusions,
            },
            "manual_review_pending": [
                "background_complexity",
                "apparent_leaf_size",
                "multiple_leaves",
                "partial_leaves",
            ],
            "leakage_zero": leakage["zero_leakage"],
            "runtime_audit": runtime_audit(),
            "training_performed": False,
            "weights_downloaded": False,
            "labels_invented": False,
        }
        (manifests / "selection_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        if output_root.exists():
            shutil.rmtree(output_root)
        raise
    return summary
