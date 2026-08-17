"""Deterministic inputs for segmentation ablations."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import yaml


class ExperimentInputError(RuntimeError):
    """Raised when an experiment input would be incomplete or leak another split."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def materialize_source_balanced_dataset(
    dataset_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Oversample each training source to the largest source count.

    Validation remains the original frozen validation directory. Test is
    deliberately omitted from the generated YAML so training cannot read it.
    """
    manifest_path = dataset_root / "manifests" / "split_manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        train_rows = [row for row in csv.DictReader(handle) if row.get("split") == "train"]
    if not train_rows:
        raise ExperimentInputError("No hay filas train en split_manifest.csv")
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        by_source[row["source_dataset"]].append(row)
    if len(by_source) < 2:
        raise ExperimentInputError("Se requieren al menos dos fuentes para balancear")
    for rows in by_source.values():
        rows.sort(key=lambda row: row["filename"])

    target = max(len(rows) for rows in by_source.values())
    sources = sorted(by_source)
    balanced: list[dict[str, str]] = []
    for index in range(target):
        for source in sources:
            rows = by_source[source]
            balanced.append(rows[index % len(rows)])

    image_paths = [(dataset_root / row["materialized_image_path"]).resolve() for row in balanced]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise ExperimentInputError(f"Faltan imágenes train: {missing[:3]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_list = output_dir / "train_source_balanced.txt"
    train_list.write_text(
        "".join(f"{path}\n" for path in image_paths),
        encoding="utf-8",
    )
    dataset_yaml = output_dir / "dataset_source_balanced.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root.resolve()),
                "train": str(train_list.resolve()),
                "val": str((dataset_root / "images" / "val").resolve()),
                "names": {0: "maize_leaf"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    input_counts = {source: len(rows) for source, rows in sorted(by_source.items())}
    output_counts = {source: target for source in sources}
    return {
        "profile": "source_balanced_corn",
        "input_train_images": len(train_rows),
        "output_train_entries": len(balanced),
        "input_source_counts": input_counts,
        "output_source_counts": output_counts,
        "duplicates_added": len(balanced) - len(train_rows),
        "train_list": str(train_list.resolve()),
        "train_list_sha256": _sha256(train_list),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "dataset_yaml_sha256": _sha256(dataset_yaml),
        "validation_split": str((dataset_root / "images" / "val").resolve()),
        "test_included": False,
        "pilot_included": False,
    }
