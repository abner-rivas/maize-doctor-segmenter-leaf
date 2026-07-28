"""Deterministic, group-aware splits for the frozen maize-leaf segmenter dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps

from src.data.segmentation_audit import (
    IMAGE_EXTENSIONS,
    parse_yolo_segmentation_line,
    polygon_area,
    polygon_touches_border,
    sha256_file,
)
from src.data.segmentation_consolidation import perceptual_hash, roboflow_original_base
from src.data.segmentation_review import LOCK_MANIFESTS, dataset_fingerprint

SPLITS = ("train", "val", "test")
PARENT_READY_STATUS = "ready_for_split_generation"
SPLIT_READY_STATUS = "ready_for_training_preflight"
SPLIT_BLOCKED_STATUS = "blocked_by_split_validation"
DEFAULT_PERCEPTUAL_THRESHOLD = 4
EXPECTED_PARENT_FINGERPRINT = (
    "c087af60c2bad1c133c4ea8b14cee945405bfe4976aa80c4faf089d7a4b9e38c"
)
PARENT_DATASET_YAML = (
    "# Candidate pool only: train/val/test splits are intentionally absent.\n"
    "path: .\n\n"
    "candidate:\n"
    "  images: all/images\n"
    "  labels: all/labels\n\n"
    "names:\n"
    "  0: maize_leaf\n\n"
    "splits_created: false\n"
).encode()
MANIFEST_COLUMNS = (
    "split",
    "group_id",
    "source_dataset",
    "filename",
    "source_image_path",
    "source_label_path",
    "materialized_image_path",
    "materialized_label_path",
    "image_sha256",
    "label_sha256",
    "perceptual_hash",
    "original_filename",
    "roboflow_variant_group",
    "instance_count",
    "mask_area_min",
    "mask_area_max",
    "mask_area_mean",
    "orientation",
    "width",
    "height",
    "materialization_method",
)
GROUP_COLUMNS = (
    "group_id",
    "split",
    "group_size",
    "source_datasets",
    "original_base_names",
    "exact_hash_count",
    "perceptual_cluster",
    "assignment_reason",
    "seed",
)


class SplitValidationError(RuntimeError):
    """Raised when a protected input or generated split fails a hard gate."""


@dataclass
class SplitRecord:
    """One immutable image/label pair plus balancing and provenance metadata."""

    filename: str
    image_path: Path
    label_path: Path
    source_dataset: str
    image_sha256: str
    label_sha256: str
    perceptual_hash: str
    original_filename: str
    original_base_name: str
    roboflow_variant_group: str
    duplicate_group: str
    width: int
    height: int
    orientation: str
    resolution: str
    instance_count: int
    mask_areas: tuple[float, ...]
    touches_border: bool
    group_id: str = ""
    split: str = ""
    materialization_method: str = ""

    @property
    def mask_area_min(self) -> float:
        return min(self.mask_areas)

    @property
    def mask_area_max(self) -> float:
        return max(self.mask_areas)

    @property
    def mask_area_mean(self) -> float:
        return mean(self.mask_areas)

    @property
    def area_bins(self) -> Counter[str]:
        result: Counter[str] = Counter()
        for area in self.mask_areas:
            result["small" if area < 0.05 else "large" if area > 0.50 else "medium"] += 1
        return result


@dataclass
class SplitGroup:
    """A connected component of images that must remain in one split."""

    group_id: str
    records: list[SplitRecord]
    perceptual_cluster: str
    split: str = ""
    assignment_reason: str = "deterministic_multivariate_greedy"
    features: Counter[str] = field(default_factory=Counter)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        root_a, root_b = self.find(first), self.find(second)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def normalize_roboflow_base_name(filename: str) -> str:
    """Return a case-insensitive base before ``_jpg.rf.<hash>``-style suffixes."""
    return roboflow_original_base(filename).strip().casefold()


def perceptual_hash_value(value: str) -> int:
    """Parse the hexadecimal payload of the project's average hash."""
    try:
        return int(value.split(":", 1)[1], 16)
    except (IndexError, ValueError) as exc:
        raise SplitValidationError(f"Hash perceptual inválido: {value}") from exc


def perceptual_distance(first: str, second: str) -> int:
    """Return the Hamming distance between two equal-width average hashes."""
    return (perceptual_hash_value(first) ^ perceptual_hash_value(second)).bit_count()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _frozen_parent_fingerprint(dataset_root: Path) -> str:
    """Recompute the parent fingerprint after ``dataset.yaml`` becomes derived."""
    paths = sorted(
        [
            path
            for path in (dataset_root / "all").rglob("*")
            if path.is_file()
            and (
                path.suffix.lower() in IMAGE_EXTENSIONS
                or path.suffix.lower() == ".txt"
            )
        ]
        + [
            path
            for path in (
                dataset_root / "dataset.yaml",
                *(
                    dataset_root / "manifests" / name
                    for name in LOCK_MANIFESTS
                ),
            )
            if path.is_file()
        ],
        key=lambda path: path.relative_to(dataset_root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(dataset_root).as_posix()
        file_digest = (
            hashlib.sha256(PARENT_DATASET_YAML).hexdigest()
            if relative == "dataset.yaml"
            else sha256_file(path)
        )
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_digest.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def verify_parent_dataset(
    dataset_root: Path,
    *,
    expected_fingerprint: str = EXPECTED_PARENT_FINGERPRINT,
) -> dict[str, object]:
    """Verify the parent lock and fingerprint before writing any split artifact."""
    lock_path = dataset_root / "manifests" / "dataset_lock.json"
    if not lock_path.is_file():
        raise SplitValidationError(f"Falta el lock obligatorio: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != PARENT_READY_STATUS:
        raise SplitValidationError(
            f"dataset_lock.status={lock.get('status')!r}; se requiere {PARENT_READY_STATUS!r}"
        )
    declared = str(lock.get("global_fingerprint", {}).get("sha256", ""))
    if declared != expected_fingerprint:
        raise SplitValidationError(
            f"Fingerprint declarado inesperado: {declared}; esperado: {expected_fingerprint}"
        )
    actual = dataset_fingerprint(dataset_root)
    frozen_actual = (
        actual["sha256"]
        if actual["sha256"] == expected_fingerprint
        else _frozen_parent_fingerprint(dataset_root)
    )
    if frozen_actual != expected_fingerprint:
        raise SplitValidationError(
            "El fingerprint actual del dataset padre no coincide; no se generaron splits: "
            f"{frozen_actual} != {expected_fingerprint}"
        )
    if int(lock.get("total_images", -1)) != 1155 or int(lock.get("total_masks", -1)) != 1224:
        raise SplitValidationError("Los conteos del lock padre no son 1155 imágenes/1224 máscaras")
    return lock


def _manifest_index(dataset_root: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(dataset_root / "manifests" / "consolidation_manifest.csv")
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        consolidated = row.get("consolidated_image_path", "")
        if not consolidated:
            continue
        filename = Path(consolidated).name
        previous = index.setdefault(filename, row)
        stable = (
            "source_dataset",
            "image_sha256",
            "perceptual_hash",
            "duplicate_group",
            "original_base_name",
            "roboflow_variant_group",
        )
        if any(previous.get(column) != row.get(column) for column in stable):
            raise SplitValidationError(f"Metadatos inconsistentes para {filename}")
    return index


def _image_orientation(width: int, height: int) -> str:
    if width > height:
        return "horizontal"
    if height > width:
        return "vertical"
    return "square"


def _resolution_bucket(width: int, height: int) -> str:
    pixels = width * height
    return "small" if pixels < 500_000 else "medium" if pixels < 2_000_000 else "large"


def _parse_label(label_path: Path) -> tuple[tuple[float, ...], bool]:
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines()]
    if not lines or any(not line for line in lines):
        raise SplitValidationError(f"TXT vacío o con línea vacía: {label_path}")
    areas: list[float] = []
    touches_border = False
    for line_number, line in enumerate(lines, start=1):
        parsed = parse_yolo_segmentation_line(line)
        if not parsed.valid or parsed.class_id != 0:
            issues = ",".join(str(item["issue_type"]) for item in parsed.issues)
            raise SplitValidationError(
                f"Etiqueta inválida {label_path}:{line_number}; class={parsed.class_id}; {issues}"
            )
        areas.append(polygon_area(parsed.points))
        touches_border = touches_border or any(polygon_touches_border(parsed.points).values())
    return tuple(areas), touches_border


def load_split_records(dataset_root: Path) -> list[SplitRecord]:
    """Load and independently validate every frozen image/label pair."""
    index = _manifest_index(dataset_root)
    image_dir = dataset_root / "all" / "images"
    label_dir = dataset_root / "all" / "labels"
    images = sorted(
        (path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )
    labels_by_stem = {path.stem: path for path in label_dir.glob("*.txt")}
    if len(images) != 1155 or len(labels_by_stem) != 1155:
        raise SplitValidationError(
            f"Inventario padre inválido: {len(images)} imágenes/{len(labels_by_stem)} TXT"
        )
    records: list[SplitRecord] = []
    for image_path in images:
        label_path = labels_by_stem.get(image_path.stem)
        row = index.get(image_path.name)
        if label_path is None or row is None:
            raise SplitValidationError(f"Sin pareja o trazabilidad: {image_path.name}")
        image_digest = sha256_file(image_path)
        if image_digest != row["image_sha256"]:
            raise SplitValidationError(f"SHA-256 no coincide con consolidación: {image_path}")
        with Image.open(image_path) as image:
            width, height = ImageOps.exif_transpose(image).size
        actual_phash = perceptual_hash(image_path)
        if actual_phash != row["perceptual_hash"]:
            raise SplitValidationError(f"Hash perceptual no coincide: {image_path}")
        areas, touches_border = _parse_label(label_path)
        original_filename = Path(row["original_image_path"]).name
        records.append(
            SplitRecord(
                filename=image_path.name,
                image_path=image_path,
                label_path=label_path,
                source_dataset=row["source_dataset"],
                image_sha256=image_digest,
                label_sha256=sha256_file(label_path),
                perceptual_hash=actual_phash,
                original_filename=original_filename,
                original_base_name=normalize_roboflow_base_name(original_filename),
                roboflow_variant_group=row["roboflow_variant_group"],
                duplicate_group=row["duplicate_group"],
                width=width,
                height=height,
                orientation=_image_orientation(width, height),
                resolution=_resolution_bucket(width, height),
                instance_count=len(areas),
                mask_areas=areas,
                touches_border=touches_border,
            )
        )
    if sum(record.instance_count for record in records) != 1224:
        raise SplitValidationError("El total independiente de máscaras no coincide con 1224")
    return records


def _union_same_values(
    union_find: _UnionFind,
    records: Sequence[SplitRecord],
    key,
) -> None:
    first_by_value: dict[object, int] = {}
    for index, record in enumerate(records):
        value = key(record)
        if not value:
            continue
        if value in first_by_value:
            union_find.union(index, first_by_value[value])
        else:
            first_by_value[value] = index


def _group_features(records: Sequence[SplitRecord]) -> Counter[str]:
    features: Counter[str] = Counter()
    for record in records:
        features["images"] += 1
        features["masks"] += record.instance_count
        features[f"source:{record.source_dataset}"] += 1
        features[f"orientation:{record.orientation}"] += 1
        features[f"resolution:{record.resolution}"] += 1
        features["border_images"] += int(record.touches_border)
        features["multi_instance_images"] += int(record.instance_count > 1)
        for area_bin, count in record.area_bins.items():
            features[f"area:{area_bin}"] += count
    return features


def build_split_groups(
    records: Sequence[SplitRecord],
    *,
    perceptual_threshold: int = DEFAULT_PERCEPTUAL_THRESHOLD,
) -> list[SplitGroup]:
    """Build deterministic connected components across every leakage signal."""
    ordered = sorted(records, key=lambda record: record.filename.casefold())
    union_find = _UnionFind(len(ordered))
    _union_same_values(union_find, ordered, lambda row: row.image_sha256)
    _union_same_values(union_find, ordered, lambda row: row.duplicate_group)
    _union_same_values(union_find, ordered, lambda row: row.roboflow_variant_group)
    _union_same_values(
        union_find,
        ordered,
        lambda row: (row.source_dataset, row.original_base_name),
    )
    hash_values = [perceptual_hash_value(record.perceptual_hash) for record in ordered]
    for first in range(len(ordered)):
        for second in range(first + 1, len(ordered)):
            if (hash_values[first] ^ hash_values[second]).bit_count() <= perceptual_threshold:
                union_find.union(first, second)
    components: dict[int, list[SplitRecord]] = defaultdict(list)
    for index, record in enumerate(ordered):
        components[union_find.find(index)].append(record)
    groups: list[SplitGroup] = []
    for component in components.values():
        component.sort(key=lambda record: record.filename.casefold())
        identity = "\n".join(record.image_sha256 for record in component)
        group_id = f"grp_{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
        phashes = sorted({record.perceptual_hash for record in component})
        perceptual_digest = hashlib.sha256(chr(10).join(phashes).encode()).hexdigest()
        perceptual_cluster = f"phc_{perceptual_digest[:16]}"
        group = SplitGroup(
            group_id=group_id,
            records=component,
            perceptual_cluster=perceptual_cluster,
            features=_group_features(component),
        )
        for record in component:
            record.group_id = group_id
        groups.append(group)
    return sorted(groups, key=lambda group: group.group_id)


def compute_group_statistics(group: SplitGroup) -> dict[str, object]:
    """Return the stable public summary of one indivisible group."""
    return {
        "group_id": group.group_id,
        "split": group.split,
        "group_size": len(group.records),
        "source_datasets": "|".join(
            sorted({record.source_dataset for record in group.records})
        ),
        "original_base_names": "|".join(
            sorted({record.original_base_name for record in group.records})
        ),
        "exact_hash_count": len({record.image_sha256 for record in group.records}),
        "perceptual_cluster": group.perceptual_cluster,
        "assignment_reason": group.assignment_reason,
    }


def _target_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    raw = {split: total * ratios[split] for split in SPLITS}
    result = {split: int(raw[split]) for split in SPLITS}
    remainder = total - sum(result.values())
    order = sorted(SPLITS, key=lambda split: (-(raw[split] - result[split]), SPLITS.index(split)))
    for split in order[:remainder]:
        result[split] += 1
    return result


def _assignment_cost(
    states: Mapping[str, Counter[str]],
    totals: Counter[str],
    ratios: Mapping[str, float],
) -> float:
    weights = {
        "images": 18.0,
        "masks": 8.0,
        "border_images": 2.0,
        "multi_instance_images": 3.0,
    }
    cost = 0.0
    for feature, total in totals.items():
        if feature.startswith("source:"):
            weight = 12.0
        elif feature.startswith("area:"):
            weight = 4.0
        else:
            weight = weights.get(feature, 2.0 if ":" in feature else 1.0)
        for split in SPLITS:
            target = total * ratios[split]
            scale = max(1.0, target)
            cost += weight * (states[split][feature] - target) ** 2 / scale
    return cost


def _split_state_cost(
    state: Counter[str],
    totals: Counter[str],
    ratio: float,
) -> float:
    weights = {
        "images": 18.0,
        "masks": 8.0,
        "border_images": 2.0,
        "multi_instance_images": 3.0,
    }
    cost = 0.0
    for feature, total in totals.items():
        target = total * ratio
        scale = max(1.0, target)
        if feature.startswith("source:"):
            weight = 12.0
        elif feature.startswith("area:"):
            weight = 4.0
        else:
            weight = weights.get(feature, 2.0 if ":" in feature else 1.0)
        cost += weight * (state[feature] - target) ** 2 / scale
    return cost


def _refine_equal_size_swaps(
    groups: Sequence[SplitGroup],
    states: dict[str, Counter[str]],
    totals: Counter[str],
    ratios: Mapping[str, float],
) -> None:
    """Improve balance without changing the image count of any split."""
    for _ in range(6):
        improved = False
        for first_split_index, first_split in enumerate(SPLITS):
            for second_split in SPLITS[first_split_index + 1 :]:
                first_groups = sorted(
                    (group for group in groups if group.split == first_split),
                    key=lambda group: group.group_id,
                )
                second_groups = sorted(
                    (group for group in groups if group.split == second_split),
                    key=lambda group: group.group_id,
                )
                for first_group in first_groups:
                    pair_cost = _split_state_cost(
                        states[first_split], totals, ratios[first_split]
                    ) + _split_state_cost(
                        states[second_split], totals, ratios[second_split]
                    )
                    best: tuple[float, SplitGroup, Counter[str], Counter[str]] | None = None
                    for second_group in second_groups:
                        if (
                            first_group.split != first_split
                            or second_group.split != second_split
                            or first_group.features["images"]
                            != second_group.features["images"]
                        ):
                            continue
                        first_trial = Counter(states[first_split])
                        second_trial = Counter(states[second_split])
                        first_trial.subtract(first_group.features)
                        first_trial.update(second_group.features)
                        second_trial.subtract(second_group.features)
                        second_trial.update(first_group.features)
                        trial_cost = _split_state_cost(
                            first_trial, totals, ratios[first_split]
                        ) + _split_state_cost(
                            second_trial, totals, ratios[second_split]
                        )
                        candidate = (
                            trial_cost,
                            second_group,
                            first_trial,
                            second_trial,
                        )
                        if trial_cost + 1e-12 < pair_cost and (
                            best is None
                            or (candidate[0], candidate[1].group_id)
                            < (best[0], best[1].group_id)
                        ):
                            best = candidate
                    if best is None:
                        continue
                    _, second_group, first_trial, second_trial = best
                    first_group.split, second_group.split = second_split, first_split
                    first_group.assignment_reason += "+equal_size_swap_refinement"
                    second_group.assignment_reason += "+equal_size_swap_refinement"
                    for record in first_group.records:
                        record.split = second_split
                    for record in second_group.records:
                        record.split = first_split
                    states[first_split] = first_trial
                    states[second_split] = second_trial
                    improved = True
        if not improved:
            return


def assign_groups_to_splits(
    groups: Sequence[SplitGroup],
    *,
    seed: int,
    ratios: Mapping[str, float],
) -> list[SplitGroup]:
    """Assign complete groups with a deterministic multivariate greedy objective."""
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Las proporciones deben contener train/val/test y sumar 1")
    totals: Counter[str] = Counter()
    for group in groups:
        totals.update(group.features)
    target_images = _target_counts(totals["images"], ratios)
    states = {split: Counter() for split in SPLITS}
    seeded_rank = {
        group.group_id: hashlib.sha256(f"{seed}\0{group.group_id}".encode()).hexdigest()
        for group in groups
    }
    balancing_features = tuple(
        key
        for key in totals
        if key.startswith(("source:", "orientation:", "resolution:", "area:"))
    )

    def rarity(group: SplitGroup) -> float:
        return sum(
            group.features[key] / max(1, totals[key]) for key in balancing_features
        )

    ordered = sorted(
        groups,
        key=lambda group: (
            -group.features["images"],
            -rarity(group),
            -group.features["masks"],
            seeded_rank[group.group_id],
        ),
    )
    max_group_size = max(group.features["images"] for group in groups)
    for group in ordered:
        candidates: list[tuple[float, int, str]] = []
        for split in SPLITS:
            proposed = states[split]["images"] + group.features["images"]
            overflow = max(0, proposed - target_images[split])
            trial = {name: Counter(values) for name, values in states.items()}
            trial[split].update(group.features)
            cost = _assignment_cost(trial, totals, ratios)
            cost += 1000.0 * (overflow / max(1, max_group_size)) ** 2
            tie = int(
                hashlib.sha256(f"{seed}\0{group.group_id}\0{split}".encode()).hexdigest(),
                16,
            )
            candidates.append((cost, tie, split))
        _, _, selected = min(candidates)
        group.split = selected
        states[selected].update(group.features)
        for record in group.records:
            record.split = selected
    _refine_equal_size_swaps(groups, states, totals, ratios)
    return sorted(groups, key=lambda group: group.group_id)


def _clear_materialized_directories(dataset_root: Path) -> None:
    for kind in ("images", "labels"):
        for split in SPLITS:
            target = dataset_root / kind / split
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)


def _materialize_file(source: Path, target: Path, requested: str) -> str:
    if requested == "copy":
        shutil.copy2(source, target)
        return "copy"
    if requested != "hardlink":
        raise ValueError("materialization debe ser copy o hardlink")
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy_fallback"


def materialize_split(
    records: Sequence[SplitRecord],
    dataset_root: Path,
    *,
    materialization: str,
) -> None:
    """Materialize the known split directories, preserving the frozen ``all/`` pool."""
    _clear_materialized_directories(dataset_root)
    for record in sorted(records, key=lambda row: (row.split, row.filename.casefold())):
        image_target = dataset_root / "images" / record.split / record.filename
        label_target = dataset_root / "labels" / record.split / f"{Path(record.filename).stem}.txt"
        image_method = _materialize_file(record.image_path, image_target, materialization)
        label_method = _materialize_file(record.label_path, label_target, materialization)
        record.materialization_method = (
            image_method if image_method == label_method else f"{image_method}+{label_method}"
        )


def compute_split_fingerprint(records: Sequence[SplitRecord], split: str) -> str:
    """Hash membership and immutable file contents for one split."""
    digest = hashlib.sha256()
    for record in sorted(
        (row for row in records if row.split == split),
        key=lambda row: row.filename.casefold(),
    ):
        digest.update(record.filename.encode())
        digest.update(b"\0")
        digest.update(record.image_sha256.encode())
        digest.update(b"\0")
        digest.update(record.label_sha256.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _combined_fingerprint(records: Sequence[SplitRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: row.filename.casefold()):
        digest.update(record.split.encode())
        digest.update(b"\0")
        digest.update(record.filename.encode())
        digest.update(b"\0")
        digest.update(record.image_sha256.encode())
        digest.update(b"\0")
        digest.update(record.label_sha256.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_rows(records: Sequence[SplitRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in sorted(
        records,
        key=lambda row: (SPLITS.index(row.split), row.filename.casefold()),
    ):
        rows.append(
            {
                "split": record.split,
                "group_id": record.group_id,
                "source_dataset": record.source_dataset,
                "filename": record.filename,
                "source_image_path": f"all/images/{record.filename}",
                "source_label_path": f"all/labels/{Path(record.filename).stem}.txt",
                "materialized_image_path": f"images/{record.split}/{record.filename}",
                "materialized_label_path": (
                    f"labels/{record.split}/{Path(record.filename).stem}.txt"
                ),
                "image_sha256": record.image_sha256,
                "label_sha256": record.label_sha256,
                "perceptual_hash": record.perceptual_hash,
                "original_filename": record.original_filename,
                "roboflow_variant_group": record.roboflow_variant_group,
                "instance_count": record.instance_count,
                "mask_area_min": f"{record.mask_area_min:.12f}",
                "mask_area_max": f"{record.mask_area_max:.12f}",
                "mask_area_mean": f"{record.mask_area_mean:.12f}",
                "orientation": record.orientation,
                "width": record.width,
                "height": record.height,
                "materialization_method": record.materialization_method,
            }
        )
    return rows


def validate_cross_split_leakage(
    records: Sequence[SplitRecord],
    *,
    perceptual_threshold: int,
) -> list[dict[str, object]]:
    """Return exact, group, Roboflow and perceptual crossings between splits."""
    issues: list[dict[str, object]] = []
    signals = {
        "exact_hash": lambda row: row.image_sha256,
        "group": lambda row: row.group_id,
        "roboflow_variant": lambda row: row.roboflow_variant_group,
    }
    for issue_type, key in signals.items():
        splits_by_value: dict[str, set[str]] = defaultdict(set)
        files_by_value: dict[str, list[str]] = defaultdict(list)
        for record in records:
            splits_by_value[key(record)].add(record.split)
            files_by_value[key(record)].append(record.filename)
        for value in sorted(splits_by_value):
            if len(splits_by_value[value]) > 1:
                issues.append(
                    {
                        "leakage_type": issue_type,
                        "value": value,
                        "splits": "|".join(sorted(splits_by_value[value])),
                        "files": "|".join(sorted(files_by_value[value])),
                        "distance": "",
                    }
                )
    ordered = sorted(records, key=lambda row: row.filename.casefold())
    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1 :]:
            if first.split == second.split:
                continue
            distance = perceptual_distance(first.perceptual_hash, second.perceptual_hash)
            if distance <= perceptual_threshold:
                issues.append(
                    {
                        "leakage_type": "perceptual_near",
                        "value": f"{first.perceptual_hash}|{second.perceptual_hash}",
                        "splits": f"{first.split}|{second.split}",
                        "files": f"{first.filename}|{second.filename}",
                        "distance": distance,
                    }
                )
    return issues


def _pilot_signals(pilot_root: Path) -> list[dict[str, str]]:
    manifest_path = pilot_root / "manifests" / "pilot_manifest.csv"
    rows = _read_csv(manifest_path)
    by_name = {Path(row["pilot_image_path"]).name: row for row in rows}
    result: list[dict[str, str]] = []
    for image_path in sorted(
        (
            path
            for path in (pilot_root / "images").iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    ):
        row = by_name.get(image_path.name, {})
        original = row.get("original_filename") or Path(row.get("original_image_path", "")).name
        result.append(
            {
                "filename": image_path.name,
                "sha256": sha256_file(image_path),
                "perceptual_hash": perceptual_hash(image_path),
                "original_base_name": normalize_roboflow_base_name(original),
            }
        )
    return result


def validate_pilot_leakage(
    records: Sequence[SplitRecord],
    pilot_root: Path,
    *,
    perceptual_threshold: int,
) -> tuple[list[dict[str, object]], str]:
    """Return every exact/name/base/perceptual crossing with the held-out pilot."""
    pilot = _pilot_signals(pilot_root)
    issues: list[dict[str, object]] = []
    for record in records:
        for held_out in pilot:
            matched: list[str] = []
            if record.image_sha256 == held_out["sha256"]:
                matched.append("exact_hash")
            if record.filename.casefold() == held_out["filename"].casefold():
                matched.append("filename")
            if record.original_base_name and (
                record.original_base_name == held_out["original_base_name"]
            ):
                matched.append("original_base_name")
            distance = perceptual_distance(
                record.perceptual_hash, held_out["perceptual_hash"]
            )
            if distance <= perceptual_threshold:
                matched.append("perceptual_near")
            for match in matched:
                issues.append(
                    {
                        "leakage_type": match,
                        "split_filename": record.filename,
                        "pilot_filename": held_out["filename"],
                        "distance": distance if match == "perceptual_near" else "",
                    }
                )
    digest = hashlib.sha256()
    for row in sorted(pilot, key=lambda item: item["filename"].casefold()):
        digest.update(row["filename"].encode())
        digest.update(b"\0")
        digest.update(row["sha256"].encode())
        digest.update(b"\n")
    return issues, digest.hexdigest()


def validate_split_integrity(
    records: Sequence[SplitRecord],
    materialized_root: Path,
    *,
    expected_images: int = 1155,
    expected_masks: int = 1224,
) -> list[str]:
    """Validate materialized pairs, file hashes, YOLO syntax, totals and YAML."""
    errors: list[str] = []
    expected_by_split = Counter(record.split for record in records)
    for split in SPLITS:
        image_dir = materialized_root / "images" / split
        label_dir = materialized_root / "labels" / split
        images = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(label_dir.glob("*.txt"))
        if len(images) != expected_by_split[split] or len(labels) != expected_by_split[split]:
            errors.append(
                f"{split}: conteo materializado {len(images)}/{len(labels)} "
                f"!= {expected_by_split[split]}"
            )
        if {path.stem for path in images} != {path.stem for path in labels}:
            errors.append(f"{split}: correspondencia imagen/TXT inválida")
    for record in records:
        image = materialized_root / "images" / record.split / record.filename
        label = materialized_root / "labels" / record.split / f"{Path(record.filename).stem}.txt"
        if not image.is_file() or not label.is_file():
            errors.append(f"Falta materialización: {record.filename}")
            continue
        if sha256_file(image) != record.image_sha256 or sha256_file(label) != record.label_sha256:
            errors.append(f"Hash materializado inválido: {record.filename}")
        try:
            _parse_label(label)
        except SplitValidationError as exc:
            errors.append(str(exc))
    if (
        len(records) != expected_images
        or sum(record.instance_count for record in records) != expected_masks
    ):
        errors.append(
            f"Totales finales distintos de {expected_images} imágenes/{expected_masks} máscaras"
        )
    yaml_path = materialized_root / "dataset.yaml"
    expected_yaml = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: maize_leaf\n"
    )
    if not yaml_path.is_file() or yaml_path.read_text(encoding="utf-8") != expected_yaml:
        errors.append("dataset.yaml no coincide con la configuración portable esperada")
    return errors


def _split_summary_rows(records: Sequence[SplitRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        selected = [record for record in records if record.split == split]
        rows.append(
            {
                "split": split,
                "images": len(selected),
                "ratio": len(selected) / len(records),
                "masks": sum(record.instance_count for record in selected),
                "groups": len({record.group_id for record in selected}),
                "border_images": sum(record.touches_border for record in selected),
                "multi_instance_images": sum(record.instance_count > 1 for record in selected),
                "mean_mask_area": mean(
                    area for record in selected for area in record.mask_areas
                ),
            }
        )
    return rows


def _source_rows(records: Sequence[SplitRecord]) -> list[dict[str, object]]:
    sources = sorted({record.source_dataset for record in records})
    return [
        {
            "split": split,
            "source_dataset": source,
            "images": sum(
                record.split == split and record.source_dataset == source for record in records
            ),
        }
        for split in SPLITS
        for source in sources
    ]


def _mask_rows(records: Sequence[SplitRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        selected = [record for record in records if record.split == split]
        areas = [area for record in selected for area in record.mask_areas]
        bins = Counter(
            "small" if area < 0.05 else "large" if area > 0.50 else "medium"
            for area in areas
        )
        for area_bin in ("small", "medium", "large"):
            rows.append(
                {
                    "split": split,
                    "area_bin": area_bin,
                    "masks": bins[area_bin],
                    "ratio": bins[area_bin] / len(areas),
                    "mean_area": mean(areas),
                }
            )
    return rows


def _render_figures(
    records: Sequence[SplitRecord],
    groups: Sequence[SplitGroup],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    colors = {"train": "#3b82f6", "val": "#f59e0b", "test": "#10b981"}

    def bar_chart(name: str, title: str, ylabel: str, values: Mapping[str, float]) -> None:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(SPLITS, [values[split] for split in SPLITS], color=[colors[s] for s in SPLITS])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        figure.savefig(output / name, dpi=140)
        plt.close(figure)

    bar_chart(
        "images_per_split.png",
        "Imágenes por split",
        "imágenes",
        Counter(record.split for record in records),
    )
    bar_chart(
        "masks_per_split.png",
        "Máscaras por split",
        "máscaras",
        {
            split: sum(record.instance_count for record in records if record.split == split)
            for split in SPLITS
        },
    )
    sources = sorted({record.source_dataset for record in records})
    figure, axis = plt.subplots(figsize=(8, 4))
    bottom = [0, 0, 0]
    for source in sources:
        values = [
            sum(record.split == split and record.source_dataset == source for record in records)
            for split in SPLITS
        ]
        axis.bar(SPLITS, values, bottom=bottom, label=source)
        bottom = [left + right for left, right in zip(bottom, values, strict=True)]
    axis.set_title("Fuentes por split")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "sources_per_split.png", dpi=140)
    plt.close(figure)

    for filename, title, values_by_split, ylabel in (
        (
            "mask_area_per_split.png",
            "Área relativa de máscaras",
            {
                split: [
                    area
                    for record in records
                    if record.split == split
                    for area in record.mask_areas
                ]
                for split in SPLITS
            },
            "área relativa",
        ),
        (
            "instances_per_image.png",
            "Instancias por imagen",
            {
                split: [
                    record.instance_count for record in records if record.split == split
                ]
                for split in SPLITS
            },
            "instancias",
        ),
    ):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.boxplot([values_by_split[split] for split in SPLITS], tick_labels=SPLITS)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=140)
        plt.close(figure)

    orientations = ("horizontal", "vertical", "square")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, split in zip(axes, SPLITS, strict=True):
        counts = Counter(
            record.orientation for record in records if record.split == split
        )
        axis.bar(orientations, [counts[value] for value in orientations], color=colors[split])
        axis.set_title(split)
        axis.tick_params(axis="x", rotation=25)
    figure.suptitle("Orientación por split")
    figure.tight_layout()
    figure.savefig(output / "orientation_per_split.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    for split in SPLITS:
        selected = [record for record in records if record.split == split]
        axis.scatter(
            [record.width for record in selected],
            [record.height for record in selected],
            s=10,
            alpha=0.45,
            label=split,
        )
    axis.set_title("Resolución por split")
    axis.set_xlabel("ancho")
    axis.set_ylabel("alto")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "resolution_per_split.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist([len(group.records) for group in groups], bins=range(1, 10), align="left")
    axis.set_title("Tamaño de grupos")
    axis.set_xlabel("imágenes")
    axis.set_ylabel("grupos")
    figure.tight_layout()
    figure.savefig(output / "group_sizes.png", dpi=140)
    plt.close(figure)

    global_features = _group_features(records)
    category_keys = [
        key
        for key in sorted(global_features)
        if key.startswith(("source:", "area:", "orientation:", "resolution:"))
    ]
    figure, axis = plt.subplots(figsize=(11, 5))
    width = 0.25
    positions = list(range(len(category_keys)))
    for index, split in enumerate(SPLITS):
        selected = _group_features([record for record in records if record.split == split])
        split_size = sum(record.split == split for record in records)
        deviations = []
        for key in category_keys:
            global_rate = global_features[key] / max(1, global_features["images"])
            split_rate = selected[key] / max(1, split_size)
            deviations.append(split_rate - global_rate)
        axis.bar(
            [position + (index - 1) * width for position in positions],
            deviations,
            width=width,
            label=split,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, [key.replace(":", "\n") for key in category_keys], rotation=30)
    axis.set_title("Desviación frente a distribución global")
    axis.set_ylabel("diferencia de tasa")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "distribution_deviation.png", dpi=140)
    plt.close(figure)


def render_split_preview(record: SplitRecord, output_path: Path) -> None:
    """Render all normalized polygons as a translucent overlay."""
    with Image.open(record.image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for line in record.label_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_yolo_segmentation_line(line)
        points = [(round(x * image.width), round(y * image.height)) for x, y in parsed.points]
        draw.polygon(points, fill=(40, 220, 90, 80), outline=(20, 255, 80, 255), width=4)
    result = Image.alpha_composite(image, overlay).convert("RGB")
    result.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, quality=90)


def _render_previews(records: Sequence[SplitRecord], output: Path, seed: int) -> None:
    def choose(candidates: Sequence[SplitRecord], count: int, key: str) -> list[SplitRecord]:
        ordered = sorted(candidates, key=lambda record: record.filename.casefold())
        random.Random(f"{seed}:{key}").shuffle(ordered)
        return ordered[:count]

    categories: dict[str, list[SplitRecord]] = {
        **{
            split: choose(
                [record for record in records if record.split == split],
                12,
                split,
            )
            for split in SPLITS
        },
        "multi_instance": choose(
            [record for record in records if record.instance_count > 1],
            12,
            "multi_instance",
        ),
        "small_masks": choose(
            [record for record in records if record.mask_area_min < 0.05],
            12,
            "small_masks",
        ),
        "large_masks": choose(
            [record for record in records if record.mask_area_max > 0.50],
            12,
            "large_masks",
        ),
    }
    for category, selected in categories.items():
        for index, record in enumerate(selected, start=1):
            render_split_preview(
                record,
                output / category / f"{index:02d}_{Path(record.filename).stem}.jpg",
            )


def write_dataset_yaml(dataset_root: Path) -> None:
    """Write the portable train/val/test segmentation configuration."""
    (dataset_root / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: maize_leaf\n",
        encoding="utf-8",
    )


def write_split_artifacts(
    records: Sequence[SplitRecord],
    groups: Sequence[SplitGroup],
    *,
    dataset_root: Path,
    output_root: Path,
    parent_lock: Mapping[str, object],
    seed: int,
    ratios: Mapping[str, float],
    perceptual_threshold: int,
    pilot_root: Path | None = None,
    reproducibility: Mapping[str, object] | None = None,
    render_visuals: bool = True,
) -> dict[str, object]:
    """Validate and write all manifests, reports, figures, previews and split lock."""
    manifests = dataset_root / "manifests"
    reports = output_root
    manifest_rows = _manifest_rows(records)
    group_rows = [
        {**compute_group_statistics(group), "seed": seed} for group in groups
    ]
    _write_csv(manifests / "split_manifest.csv", manifest_rows, MANIFEST_COLUMNS)
    _write_csv(manifests / "split_groups.csv", group_rows, GROUP_COLUMNS)
    cross_issues = validate_cross_split_leakage(
        records, perceptual_threshold=perceptual_threshold
    )
    pilot_issues, pilot_fingerprint = validate_pilot_leakage(
        records,
        pilot_root or dataset_root.parent / "pilot",
        perceptual_threshold=perceptual_threshold,
    )
    integrity_errors = validate_split_integrity(records, dataset_root)
    split_fingerprints = {
        split: compute_split_fingerprint(records, split) for split in SPLITS
    }
    combined = _combined_fingerprint(records)
    counts = Counter(record.split for record in records)
    masks = {
        split: sum(record.instance_count for record in records if record.split == split)
        for split in SPLITS
    }
    actual_ratios = {split: counts[split] / len(records) for split in SPLITS}
    cross_types = Counter(str(row["leakage_type"]) for row in cross_issues)
    reproducible = bool(reproducibility and reproducibility.get("passed"))
    validation_errors = [
        *integrity_errors,
        *[f"cross_split:{row}" for row in cross_issues],
        *[f"pilot:{row}" for row in pilot_issues],
    ]
    ready = (
        not validation_errors
        and len(records) == int(parent_lock["total_images"])
        and sum(record.instance_count for record in records)
        == int(parent_lock["total_masks"])
        and reproducible
    )
    summary = {
        "schema_version": 1,
        "seed": seed,
        "target_ratios": dict(ratios),
        "actual_counts": dict(counts),
        "actual_ratios": actual_ratios,
        "mask_counts": masks,
        "group_count": len(groups),
        "perceptual_hamming_threshold": perceptual_threshold,
        "source_distribution": {
            split: dict(
                sorted(
                    Counter(
                        record.source_dataset
                        for record in records
                        if record.split == split
                    ).items()
                )
            )
            for split in SPLITS
        },
        "validation_errors": validation_errors,
        "parent_dataset_fingerprint": parent_lock["global_fingerprint"]["sha256"],
        "parent_content_equivalent": (
            len(records) == int(parent_lock["total_images"])
            and sum(record.instance_count for record in records)
            == int(parent_lock["total_masks"])
            and len({record.image_sha256 for record in records}) == len(records)
        ),
        "training_performed": False,
    }
    _write_json(manifests / "split_summary.json", summary)
    _write_json(reports / "summary.json", summary)
    _write_json(
        manifests / "split_fingerprints.json",
        {
            "algorithm": "sha256",
            "train_fingerprint": split_fingerprints["train"],
            "val_fingerprint": split_fingerprints["val"],
            "test_fingerprint": split_fingerprints["test"],
            "combined_fingerprint": combined,
            "parent_dataset_fingerprint": parent_lock["global_fingerprint"]["sha256"],
            "parent_content_equivalent": summary["parent_content_equivalent"],
        },
    )
    statistics = _split_summary_rows(records)
    _write_csv(reports / "split_statistics.csv", statistics, tuple(statistics[0]))
    source_rows = _source_rows(records)
    _write_csv(reports / "source_distribution.csv", source_rows, tuple(source_rows[0]))
    mask_rows = _mask_rows(records)
    _write_csv(reports / "mask_distribution.csv", mask_rows, tuple(mask_rows[0]))
    _write_csv(reports / "group_distribution.csv", group_rows, GROUP_COLUMNS)
    leakage_columns = ("leakage_type", "value", "splits", "files", "distance")
    _write_csv(reports / "leakage_report.csv", cross_issues, leakage_columns)
    pilot_columns = ("leakage_type", "split_filename", "pilot_filename", "distance")
    _write_csv(reports / "pilot_leakage_report.csv", pilot_issues, pilot_columns)
    if reproducibility is not None:
        _write_json(reports / "reproducibility_report.json", reproducibility)
    lock = {
        "schema_version": 1,
        "split_version": (
            f"{parent_lock['dataset_version']}-splits-seed-{seed}-{combined[:12]}"
        ),
        "parent_dataset_version": parent_lock["dataset_version"],
        "parent_dataset_fingerprint": parent_lock["global_fingerprint"]["sha256"],
        "seed": seed,
        "target_ratios": dict(ratios),
        "actual_counts": dict(counts),
        "actual_ratios": actual_ratios,
        "image_count": len(records),
        "mask_count": sum(record.instance_count for record in records),
        "mask_counts": masks,
        "train_fingerprint": split_fingerprints["train"],
        "val_fingerprint": split_fingerprints["val"],
        "test_fingerprint": split_fingerprints["test"],
        "combined_fingerprint": combined,
        "parent_content_equivalent": summary["parent_content_equivalent"],
        "pilot_fingerprint": pilot_fingerprint,
        "group_count": len(groups),
        "perceptual_hamming_threshold": perceptual_threshold,
        "cross_split_duplicate_count": cross_types["exact_hash"],
        "cross_split_group_leakage_count": cross_types["group"],
        "cross_split_roboflow_variant_count": cross_types["roboflow_variant"],
        "cross_split_perceptual_count": cross_types["perceptual_near"],
        "pilot_leakage_count": len(pilot_issues),
        "validation_errors": validation_errors,
        "reproducibility": dict(reproducibility or {}),
        "training_performed": False,
        "status": SPLIT_READY_STATUS if ready else SPLIT_BLOCKED_STATUS,
    }
    _write_json(manifests / "split_lock.json", lock)
    if render_visuals:
        _render_figures(records, groups, reports / "figures")
        _render_previews(records, reports / "previews", seed)
    return lock


def clone_records(records: Sequence[SplitRecord]) -> list[SplitRecord]:
    """Create independent mutable records for reproducibility runs."""
    return [
        SplitRecord(
            **{
                field_name: getattr(record, field_name)
                for field_name in SplitRecord.__dataclass_fields__
                if field_name not in {"group_id", "split", "materialization_method"}
            }
        )
        for record in records
    ]
