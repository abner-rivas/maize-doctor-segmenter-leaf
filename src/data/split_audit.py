"""Read-only validation for official train/val/test CSV manifests."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

SPLIT_NAMES = ("train", "val", "test")
SPLIT_COLUMNS = ("image_path", "label", "environment")
VALID_ENVIRONMENTS = ("lab", "real")
SPLIT_COUNT_COLUMNS = ("split", "label", "environment", "image_count")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_and_hash_image(path: Path) -> str:
    """Verify image decoding and return a hash of the exact source bytes."""
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    return hashlib.sha256(payload).hexdigest()


def load_split_configuration(config_path: Path) -> tuple[list[str], int, int | None]:
    """Load baseline classes, expected split seed, and optional class cap."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuración inexistente: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Configuración inválida: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError("Falta la sección dataset en la configuración")
    dataset = config["dataset"]
    baseline = config.get("baseline", {})
    classes = baseline.get("classes") if isinstance(baseline, dict) else None
    classes = classes or dataset.get("classes")
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(item, str) or not item.strip() for item in classes)
    ):
        raise ValueError("La lista de clases baseline/dataset es inválida")
    seed = baseline.get("seed", dataset.get("seed")) if isinstance(baseline, dict) else None
    if not isinstance(seed, int):
        raise ValueError("La semilla baseline/dataset debe ser un entero")
    max_per_class = baseline.get("max_images_per_class") if isinstance(baseline, dict) else None
    if max_per_class is not None and (not isinstance(max_per_class, int) or max_per_class < 1):
        raise ValueError("baseline.max_images_per_class debe ser un entero positivo")
    return [item.strip() for item in classes], seed, max_per_class


def _read_split(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"Split inexistente: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Split sin encabezado: {path}")
        return [dict(row) for row in reader], tuple(reader.fieldnames)


def _issue(
    issue_type: str,
    evidence: str,
    *,
    severity: str = "error",
    split: str = "",
    image_path: str = "",
) -> dict[str, str]:
    return {
        "severity": severity,
        "type": issue_type,
        "split": split,
        "image_path": image_path,
        "evidence": evidence,
    }


def _write_counts(path: Path, counts: Counter[tuple[str, str, str]]) -> None:
    rows = [
        {
            "split": split,
            "label": label,
            "environment": environment,
            "image_count": count,
        }
        for (split, label, environment), count in sorted(counts.items())
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPLIT_COUNT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def compare_split_directories(first: Path, second: Path) -> dict[str, object]:
    """Compare exact CSV hashes, order, and route sets for two generated split trees."""
    details: dict[str, dict[str, object]] = {}
    all_exact = True
    all_same_routes = True
    all_same_order = True
    for split in SPLIT_NAMES:
        first_path = first / f"{split}.csv"
        second_path = second / f"{split}.csv"
        first_rows, first_columns = _read_split(first_path)
        second_rows, second_columns = _read_split(second_path)
        first_routes = [row.get("image_path", "") for row in first_rows]
        second_routes = [row.get("image_path", "") for row in second_rows]
        exact = _sha256(first_path) == _sha256(second_path)
        same_order = first_columns == second_columns and first_rows == second_rows
        same_routes = set(first_routes) == set(second_routes)
        details[split] = {
            "first_csv_sha256": _sha256(first_path),
            "second_csv_sha256": _sha256(second_path),
            "exact_csv_match": exact,
            "same_rows_and_order": same_order,
            "same_route_set": same_routes,
        }
        all_exact = all_exact and exact
        all_same_order = all_same_order and same_order
        all_same_routes = all_same_routes and same_routes
    return {
        "checked": True,
        "first_directory": str(first.resolve()),
        "second_directory": str(second.resolve()),
        "exactly_reproducible": all_exact and all_same_order,
        "same_route_sets": all_same_routes,
        "same_order": all_same_order,
        "splits": details,
    }


def validate_splits(
    splits_dir: Path,
    dataset_root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    compare_dir: Path | None = None,
) -> dict[str, object]:
    """Validate manifests, physical files, leakage, coverage, and optional reproducibility."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"DATASET_ROOT inexistente: {dataset_root}")
    classes, expected_seed, max_per_class = load_split_configuration(config_path)
    valid_classes = set(classes)
    issues: list[dict[str, str]] = []
    counts: Counter[tuple[str, str, str]] = Counter()
    route_occurrences: dict[str, list[tuple[str, str]]] = defaultdict(list)
    hash_occurrences: dict[str, list[tuple[str, str]]] = defaultdict(list)
    split_rows: dict[str, list[dict[str, str]]] = {}
    csv_hashes: dict[str, str] = {}
    images_verified = 0

    for split in SPLIT_NAMES:
        csv_path = splits_dir / f"{split}.csv"
        rows, columns = _read_split(csv_path)
        split_rows[split] = rows
        csv_hashes[split] = _sha256(csv_path)
        if columns != SPLIT_COLUMNS:
            issues.append(
                _issue(
                    "invalid_columns",
                    f"esperadas={SPLIT_COLUMNS}; actuales={columns}",
                    split=split,
                )
            )
        for row_number, row in enumerate(rows, start=2):
            image_value = row.get("image_path", "").strip()
            label = row.get("label", "").strip()
            environment = row.get("environment", "").strip()
            if label not in valid_classes:
                issues.append(
                    _issue(
                        "invalid_label",
                        f"fila {row_number}: {label!r}",
                        split=split,
                        image_path=image_value,
                    )
                )
            if environment not in VALID_ENVIRONMENTS:
                issues.append(
                    _issue(
                        "invalid_environment",
                        f"fila {row_number}: {environment!r}",
                        split=split,
                        image_path=image_value,
                    )
                )
            counts[(split, label, environment)] += 1
            relative = Path(image_value)
            if not image_value or relative.is_absolute() or ".." in relative.parts:
                issues.append(
                    _issue(
                        "invalid_path",
                        f"fila {row_number}: la ruta debe ser relativa a DATASET_ROOT",
                        split=split,
                        image_path=image_value,
                    )
                )
                continue
            image_path = (dataset_root / relative).resolve()
            normalized_path = str(image_path)
            route_occurrences[normalized_path].append((split, image_value))
            if not image_path.is_file():
                issues.append(
                    _issue(
                        "missing_image",
                        f"fila {row_number}: {image_path}",
                        split=split,
                        image_path=image_value,
                    )
                )
                continue
            try:
                digest = _verify_and_hash_image(image_path)
            except (OSError, SyntaxError, ValueError) as exc:
                issues.append(
                    _issue(
                        "unreadable_image",
                        f"fila {row_number}: {exc}",
                        split=split,
                        image_path=image_value,
                    )
                )
                continue
            images_verified += 1
            hash_occurrences[digest].append((split, image_value))

    duplicate_route_groups = 0
    cross_split_route_groups = 0
    for path, occurrences in sorted(route_occurrences.items()):
        if len(occurrences) <= 1:
            continue
        duplicate_route_groups += 1
        occurrence_splits = {split for split, _ in occurrences}
        if len(occurrence_splits) > 1:
            cross_split_route_groups += 1
        issues.append(
            _issue(
                "duplicate_route",
                f"{path}: {occurrences}",
            )
        )

    duplicate_hash_groups = 0
    cross_split_hash_groups = 0
    for digest, occurrences in sorted(hash_occurrences.items()):
        if len(occurrences) <= 1:
            continue
        duplicate_hash_groups += 1
        occurrence_splits = {split for split, _ in occurrences}
        issue_type = "duplicate_hash"
        if len(occurrence_splits) > 1:
            cross_split_hash_groups += 1
            issue_type = "cross_split_hash_leakage"
        issues.append(_issue(issue_type, f"{digest}: {occurrences}"))

    for split in SPLIT_NAMES:
        labels = {row.get("label", "") for row in split_rows[split]}
        for missing_label in sorted(valid_classes - labels):
            issues.append(
                _issue(
                    "missing_class_coverage",
                    f"{missing_label} no aparece en {split}",
                    split=split,
                )
            )

    totals_by_label = Counter(row.get("label", "") for rows in split_rows.values() for row in rows)
    if max_per_class is not None:
        for label, count in sorted(totals_by_label.items()):
            if count > max_per_class:
                issues.append(
                    _issue(
                        "baseline_cap_exceeded",
                        f"{label}: total={count}; máximo={max_per_class}",
                    )
                )

    if compare_dir is None:
        reproducibility: dict[str, object] = {
            "checked": False,
            "expected_seed": expected_seed,
            "reason": "no se proporcionó --compare-dir",
        }
    else:
        reproducibility = compare_split_directories(splits_dir, compare_dir)
        reproducibility["expected_seed"] = expected_seed
        if not reproducibility["exactly_reproducible"]:
            issues.append(
                _issue(
                    "non_reproducible_splits",
                    f"los CSV difieren de {compare_dir}",
                )
            )

    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    report = {
        "schema_version": 1,
        "splits_directory": str(splits_dir.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "config_path": str(config_path.resolve()),
        "expected_seed": expected_seed,
        "baseline_max_images_per_class": max_per_class,
        "expected_columns": list(SPLIT_COLUMNS),
        "configured_classes": classes,
        "total_rows": sum(len(rows) for rows in split_rows.values()),
        "images_verified": images_verified,
        "rows_by_split": {split: len(split_rows[split]) for split in SPLIT_NAMES},
        "rows_by_label": dict(sorted(totals_by_label.items())),
        "csv_sha256": csv_hashes,
        "leakage": {
            "duplicate_route_groups": duplicate_route_groups,
            "cross_split_route_groups": cross_split_route_groups,
            "duplicate_hash_groups": duplicate_hash_groups,
            "cross_split_hash_groups": cross_split_hash_groups,
        },
        "error_count": len(errors),
        "warning_count": len(warnings),
        "valid": not errors,
        "issues": issues,
        "reproducibility": reproducibility,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_counts(output_dir / "split_counts.csv", counts)
    (output_dir / "split_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def split_validation_exit_code(report: dict[str, object], fail_on_error: bool) -> int:
    """Return nonzero in strict mode after retaining all validation evidence."""
    if fail_on_error and int(report.get("error_count", 0)) > 0:
        return 2
    return 0
