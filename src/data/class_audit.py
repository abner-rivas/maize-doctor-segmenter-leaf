"""Read-only audit of configured, documented, and physical dataset classes."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import yaml

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
EXPECTED_ENVIRONMENTS = ("lab", "real")
COUNT_COLUMNS = (
    "class_name",
    "environment",
    "image_count",
    "configured",
    "present_on_disk",
    "documented",
)
MISMATCH_COLUMNS = (
    "mismatch_type",
    "class_name",
    "expected_value",
    "actual_value",
    "severity",
    "evidence",
)
MISMATCH_TYPES = {
    "configured_missing_on_disk",
    "disk_class_not_configured",
    "documented_missing_on_disk",
    "disk_class_not_documented",
    "empty_class",
    "unexpected_environment",
    "invalid_file",
    "count_mismatch",
}

_DOCUMENTED_COUNT_ROW = re.compile(
    r"^\|\s*`(?P<class>[a-z][a-z0-9_]*)`\s*"
    r"\|\s*(?P<lab>[\d\s,.]+)\s*"
    r"\|\s*(?P<real>[\d\s,.]+)\s*"
    r"\|\s*(?P<total>[\d\s,.]+)\s*\|"
)


def _parse_count(value: str) -> int:
    digits = re.sub(r"\D", "", value)
    if not digits:
        raise ValueError(f"Conteo documentado inválido: {value!r}")
    return int(digits)


def load_configured_classes(config_path: Path) -> tuple[list[str], str]:
    """Load and validate canonical classes plus the configured clean directory."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuración inexistente: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"No se pudo leer la configuración: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("La configuración debe ser un objeto YAML")
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Falta la sección dataset en la configuración")
    classes = dataset.get("classes")
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(item, str) or not item.strip() for item in classes)
    ):
        raise ValueError("dataset.classes debe ser una lista no vacía de nombres")
    normalized = [item.strip() for item in classes]
    if len(normalized) != len(set(normalized)):
        raise ValueError("dataset.classes contiene nombres duplicados")
    paths = config.get("paths", {})
    clean_relative = paths.get("raw_dir", "clean") if isinstance(paths, dict) else "clean"
    if not isinstance(clean_relative, str) or not clean_relative.strip():
        raise ValueError("paths.raw_dir debe ser una ruta relativa no vacía")
    clean_path = Path(clean_relative)
    if clean_path.is_absolute() or ".." in clean_path.parts:
        raise ValueError("paths.raw_dir debe permanecer relativo a DATASET_ROOT")
    return normalized, clean_relative


def parse_documented_counts(paths: Sequence[Path]) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Extract the canonical four-column class count table from Markdown files."""
    counts: dict[str, dict[str, int]] = {}
    warnings: list[str] = []
    for path in paths:
        if not path.is_file():
            warnings.append(f"documentación inexistente: {path}")
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            warnings.append(f"no se pudo leer documentación {path}: {exc}")
            continue
        found = 0
        for line in lines:
            match = _DOCUMENTED_COUNT_ROW.match(line)
            if not match:
                continue
            class_name = match.group("class")
            parsed = {
                "lab": _parse_count(match.group("lab")),
                "real": _parse_count(match.group("real")),
                "total": _parse_count(match.group("total")),
            }
            if parsed["lab"] + parsed["real"] != parsed["total"]:
                warnings.append(
                    f"conteo documentado inconsistente para {class_name} en {path}"
                )
            previous = counts.get(class_name)
            if previous is not None and previous != parsed:
                warnings.append(
                    f"conteos documentados contradictorios para {class_name}: "
                    f"{previous} frente a {parsed}"
                )
            else:
                counts[class_name] = parsed
            found += 1
        if not found:
            warnings.append(f"sin tabla de conteos reconocible en {path}")
    return counts, warnings


def _mismatch(
    mismatch_type: str,
    class_name: str,
    expected: object,
    actual: object,
    severity: str,
    evidence: str,
) -> dict[str, object]:
    if mismatch_type not in MISMATCH_TYPES:
        raise ValueError(f"Tipo de discrepancia desconocido: {mismatch_type}")
    return {
        "mismatch_type": mismatch_type,
        "class_name": class_name,
        "expected_value": expected,
        "actual_value": actual,
        "severity": severity,
        "evidence": evidence,
    }


def _scan_clean_directory(
    clean_dir: Path,
) -> tuple[
    set[str],
    Counter[tuple[str, str]],
    list[dict[str, object]],
    list[str],
    list[str],
]:
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Directorio clean inexistente: {clean_dir}")
    present_classes: set[str] = set()
    counts: Counter[tuple[str, str]] = Counter()
    mismatches: list[dict[str, object]] = []
    ignored_files: list[str] = []
    empty_directories: list[str] = []

    for root_entry in sorted(clean_dir.iterdir(), key=lambda path: path.name.casefold()):
        relative_root = root_entry.relative_to(clean_dir).as_posix()
        if not root_entry.is_dir():
            ignored_files.append(relative_root)
            mismatches.append(
                _mismatch(
                    "invalid_file",
                    "__root__",
                    "directorio de clase",
                    root_entry.name,
                    "warning",
                    relative_root,
                )
            )
            continue
        class_name = root_entry.name
        present_classes.add(class_name)
        class_count = 0
        environment_names: set[str] = set()
        for env_entry in sorted(root_entry.iterdir(), key=lambda path: path.name.casefold()):
            relative_env = env_entry.relative_to(clean_dir).as_posix()
            if not env_entry.is_dir():
                ignored_files.append(relative_env)
                mismatches.append(
                    _mismatch(
                        "invalid_file",
                        class_name,
                        "directorio lab o real",
                        env_entry.name,
                        "warning",
                        relative_env,
                    )
                )
                continue
            environment = env_entry.name
            environment_names.add(environment)
            if environment not in EXPECTED_ENVIRONMENTS:
                mismatches.append(
                    _mismatch(
                        "unexpected_environment",
                        class_name,
                        ",".join(EXPECTED_ENVIRONMENTS),
                        environment,
                        "error",
                        relative_env,
                    )
                )
            environment_count = 0
            for file_path in sorted(env_entry.iterdir(), key=lambda path: path.name.casefold()):
                relative_file = file_path.relative_to(clean_dir).as_posix()
                supported_file = (
                    file_path.is_file()
                    and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                )
                if not supported_file:
                    ignored_files.append(relative_file)
                    mismatches.append(
                        _mismatch(
                            "invalid_file",
                            class_name,
                            ",".join(SUPPORTED_IMAGE_EXTENSIONS),
                            file_path.suffix.lower() or "sin extensión/otro directorio",
                            "warning",
                            relative_file,
                        )
                    )
                    continue
                counts[(class_name, environment)] += 1
                environment_count += 1
                class_count += 1
            if environment_count == 0:
                empty_directories.append(relative_env)
                mismatches.append(
                    _mismatch(
                        "empty_class",
                        class_name,
                        "al menos una imagen soportada",
                        0,
                        "warning",
                        f"entorno vacío: {relative_env}",
                    )
                )
        for expected_environment in EXPECTED_ENVIRONMENTS:
            if expected_environment not in environment_names:
                missing_path = f"{class_name}/{expected_environment}"
                empty_directories.append(missing_path)
                mismatches.append(
                    _mismatch(
                        "empty_class",
                        class_name,
                        f"directorio {expected_environment}",
                        "ausente",
                        "warning",
                        missing_path,
                    )
                )
        if class_count == 0:
            mismatches.append(
                _mismatch(
                    "empty_class",
                    class_name,
                    "al menos una imagen soportada en la clase",
                    0,
                    "error",
                    relative_root,
                )
            )
    return present_classes, counts, mismatches, ignored_files, empty_directories


def _write_csv(path: Path, rows: Iterable[dict[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset_classes(
    dataset_root: Path,
    config_path: Path,
    output_dir: Path,
    *,
    documentation_paths: Sequence[Path] = (),
) -> dict[str, object]:
    """Audit class alignment without writing anywhere inside DATASET_ROOT."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"DATASET_ROOT inexistente: {dataset_root}")
    configured_classes, clean_relative = load_configured_classes(config_path)
    clean_dir = dataset_root / clean_relative
    if output_dir.resolve() == dataset_root.resolve() or output_dir.resolve().is_relative_to(
        dataset_root.resolve()
    ):
        raise ValueError("La auditoría debe escribirse fuera de DATASET_ROOT")
    documented_counts, documentation_warnings = parse_documented_counts(documentation_paths)
    (
        present_classes,
        physical_counts,
        mismatches,
        ignored_files,
        empty_directories,
    ) = _scan_clean_directory(clean_dir)

    configured = set(configured_classes)
    documented = set(documented_counts)
    missing_classes = sorted(configured - present_classes)
    additional_classes = sorted(present_classes - configured)
    for class_name in missing_classes:
        mismatches.append(
            _mismatch(
                "configured_missing_on_disk",
                class_name,
                "presente",
                "ausente",
                "error",
                f"config={config_path}; clean={clean_dir}",
            )
        )
    for class_name in additional_classes:
        mismatches.append(
            _mismatch(
                "disk_class_not_configured",
                class_name,
                "ausente",
                "presente",
                "error",
                str(clean_dir / class_name),
            )
        )
    for class_name in sorted(documented - present_classes):
        mismatches.append(
            _mismatch(
                "documented_missing_on_disk",
                class_name,
                "presente",
                "ausente",
                "error",
                ", ".join(str(path) for path in documentation_paths),
            )
        )
    for class_name in sorted(present_classes - documented):
        mismatches.append(
            _mismatch(
                "disk_class_not_documented",
                class_name,
                "documentada",
                "no encontrada en tabla documental",
                "warning",
                str(clean_dir / class_name),
            )
        )

    actual_by_class = {
        class_name: sum(
            count
            for (count_class, _), count in physical_counts.items()
            if count_class == class_name
        )
        for class_name in present_classes
    }
    for class_name, expected in sorted(documented_counts.items()):
        for environment in EXPECTED_ENVIRONMENTS:
            actual = physical_counts[(class_name, environment)]
            if actual != expected[environment]:
                mismatches.append(
                    _mismatch(
                        "count_mismatch",
                        class_name,
                        expected[environment],
                        actual,
                        "error",
                        f"environment={environment}",
                    )
                )
        actual_total = actual_by_class.get(class_name, 0)
        if actual_total != expected["total"]:
            mismatches.append(
                _mismatch(
                    "count_mismatch",
                    class_name,
                    expected["total"],
                    actual_total,
                    "error",
                    "conteo total de clase",
                )
            )

    actual_total = sum(physical_counts.values())
    documented_total = sum(values["total"] for values in documented_counts.values())
    if documented_counts and actual_total != documented_total:
        mismatches.append(
            _mismatch(
                "count_mismatch",
                "__total__",
                documented_total,
                actual_total,
                "error",
                "suma de la tabla documental frente a imágenes soportadas en disco",
            )
        )

    all_classes = sorted(configured | present_classes | documented)
    all_environments = sorted(
        set(EXPECTED_ENVIRONMENTS)
        | {environment for _, environment in physical_counts}
    )
    count_rows = [
        {
            "class_name": class_name,
            "environment": environment,
            "image_count": physical_counts[(class_name, environment)],
            "configured": class_name in configured,
            "present_on_disk": class_name in present_classes,
            "documented": class_name in documented,
        }
        for class_name in all_classes
        for environment in all_environments
    ]
    mismatches.sort(
        key=lambda row: (
            str(row["severity"]),
            str(row["mismatch_type"]),
            str(row["class_name"]),
            str(row["evidence"]),
        )
    )
    errors = [str(row["evidence"]) for row in mismatches if row["severity"] == "error"]
    warnings = list(documentation_warnings)
    warnings.extend(
        str(row["evidence"]) for row in mismatches if row["severity"] == "warning"
    )
    report = {
        "schema_version": 1,
        "dataset_root": str(dataset_root.resolve()),
        "clean_directory": str(clean_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "documentation_paths": [str(path.resolve()) for path in documentation_paths],
        "configured_classes": configured_classes,
        "present_classes": sorted(present_classes),
        "documented_classes": sorted(documented),
        "missing_classes": missing_classes,
        "additional_classes": additional_classes,
        "total_images": actual_total,
        "counts_by_class": dict(sorted(actual_by_class.items())),
        "counts_by_environment": {
            environment: sum(
                count
                for (_, count_environment), count in physical_counts.items()
                if count_environment == environment
            )
            for environment in all_environments
        },
        "documented_counts": documented_counts,
        "ignored_files": ignored_files,
        "empty_directories": sorted(set(empty_directories)),
        "warnings": warnings,
        "errors": errors,
        "mismatch_count": len(mismatches),
        "critical_mismatch_count": len(errors),
        "conclusion": "class_mismatch_detected" if errors else "classes_coherent",
        "ready_for_splits": not errors,
        "split_recommendation": "do_not_generate" if errors else "eligible_after_split_validation",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "class_counts.csv", count_rows, COUNT_COLUMNS)
    _write_csv(output_dir / "class_mismatches.csv", mismatches, MISMATCH_COLUMNS)
    (output_dir / "class_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def audit_exit_code(report: dict[str, object], fail_on_mismatch: bool) -> int:
    """Fail strict mode only for mismatches classified as critical errors."""
    if fail_on_mismatch and int(report.get("critical_mismatch_count", 0)) > 0:
        return 2
    return 0
