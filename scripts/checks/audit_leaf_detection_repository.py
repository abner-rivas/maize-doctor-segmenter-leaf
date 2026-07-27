"""Inventory leaf-isolation artifacts and historical path references safely.

This check is read-only outside ``outputs/repository_audit``. It does not move or
delete artifacts; the report is intended to support an explicit human cleanup decision.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "repository_audit"
SCAN_ROOTS = ("data", "outputs", "public", "docs", "notebooks", "scripts", "src", "config")
HASH_LIMIT_BYTES = 64 * 1024 * 1024
TEXT_SUFFIXES = {
    ".csv",
    ".html",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
INVENTORY_COLUMNS = (
    "path",
    "artifact_type",
    "status",
    "reason",
    "referenced_by_code",
    "referenced_by_documentation",
    "referenced_by_config",
    "git_tracked",
    "size_bytes",
    "hash_if_file",
    "replacement_path",
    "recommended_action",
    "safe_to_move",
    "safe_to_delete",
    "notes",
)
OBSOLETE_COLUMNS = (
    "path",
    "line_number",
    "reference",
    "status",
    "replacement_path",
    "recommended_action",
    "notes",
)
MIGRATION_COLUMNS = (
    "historical_path",
    "active_path",
    "status",
    "compatibility",
    "recommended_action",
    "notes",
)

PATH_MIGRATIONS = (
    {
        "historical_path": "data/clean/",
        "active_path": "${DATASET_ROOT}/clean/",
        "status": "migrated",
        "compatibility": "Historical cleanup scripts are retained but deprecated",
        "recommended_action": "Use get_dataset_root() in active code",
        "notes": "Do not run scripts/cleanup against the repository data directory.",
    },
    {
        "historical_path": "outputs/splits/seed_42_baseline/",
        "active_path": "data/splits/seed_42_baseline/",
        "status": "migrated",
        "compatibility": "src.training.common falls back to PROJECT_DATA_ROOT",
        "recommended_action": "Use the active path in new code and documentation",
        "notes": "Historical model summaries remain immutable.",
    },
    {
        "historical_path": "/outputs/splits/seed_42_baseline",
        "active_path": "${PROJECT_DATA_ROOT}/splits/seed_42_baseline",
        "status": "historical_remote_path",
        "compatibility": "Portable fallback is implemented in load_run_metadata",
        "recommended_action": "Preserve historical summaries; never copy this path forward",
        "notes": "The path came from a remote container.",
    },
    {
        "historical_path": "outputs/leaf_detection/pilot/images/",
        "active_path": "data/leaf_detection/pilot/images/",
        "status": "migrated",
        "compatibility": "No active old directory exists",
        "recommended_action": "Use data/ for images and outputs/ only for reports",
        "notes": "",
    },
    {
        "historical_path": "outputs/leaf_detection/pilot/manifests/",
        "active_path": "data/leaf_detection/pilot/manifests/",
        "status": "migrated",
        "compatibility": "No active old directory exists",
        "recommended_action": "Use the active path",
        "notes": "",
    },
    {
        "historical_path": "outputs/leaf_detection/pilot/annotations.xml",
        "active_path": "data/leaf_detection/pilot/annotations/cvat/annotations.xml",
        "status": "migrated",
        "compatibility": "The native CVAT XML is the official source",
        "recommended_action": "Use the active path and do not alter the XML",
        "notes": "",
    },
    {
        "historical_path": "outputs/leaf_detection/pilot/packages/",
        "active_path": "data/leaf_detection/pilot/packages/",
        "status": "migrated",
        "compatibility": "Original packages are preserved",
        "recommended_action": "Keep packages in data/",
        "notes": "",
    },
    {
        "historical_path": "outputs/preflight_gpu_check/",
        "active_path": "outputs/preflight/baseline_full_existing_models_seed42_cuda/",
        "status": "deprecated_path",
        "compatibility": "Both reports remain historical evidence",
        "recommended_action": "Archive only after explicit human approval",
        "notes": "The newer preflight namespace is the active convention.",
    },
    {
        "historical_path": "outputs/dataset_audit_updated/",
        "active_path": "outputs/dataset_audit_final/",
        "status": "duplicate_copy",
        "compatibility": "All three files are byte-identical",
        "recommended_action": "Retain pending human approval for archival or deletion",
        "notes": "No automatic move or deletion was performed.",
    },
    {
        "historical_path": "data/leaf_detection/pilot/packages/pilot_images/images/",
        "active_path": "data/leaf_detection/pilot/images/",
        "status": "duplicate_copy",
        "compatibility": "100/100 image hashes are identical",
        "recommended_action": "Keep while the original package evidence is protected",
        "notes": "The ZIP and unpacked package were not modified.",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_tracked_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line for line in result.stdout.splitlines() if line}


def _iter_all_paths() -> tuple[list[Path], dict[Path, int]]:
    paths: list[Path] = []
    directory_sizes: dict[Path, int] = defaultdict(int)
    for root_name in SCAN_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        paths.append(root)
        for path in sorted(root.rglob("*")):
            if path == OUTPUT_DIR or OUTPUT_DIR in path.parents:
                continue
            paths.append(path)
            if not path.is_file():
                continue
            size = path.stat().st_size
            parent = path.parent
            while parent == root or root in parent.parents:
                directory_sizes[parent] += size
                if parent == root:
                    break
                parent = parent.parent
    for name in ("README.md", "Makefile", "pyproject.toml", "CLAUDE.md"):
        path = PROJECT_ROOT / name
        if path.exists():
            paths.append(path)
    return paths, directory_sizes


def _include_file(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    if relative.parts[0] in {"docs", "notebooks", "scripts", "src", "config", "public"}:
        return True
    if relative.parts[0] == "data":
        return path.suffix.lower() in TEXT_SUFFIXES | {".zip"}
    if relative.parts[0] == "outputs":
        return path.suffix.lower() in TEXT_SUFFIXES | {".pth", ".npy"}
    return True


def _text_corpora(paths: Iterable[Path]) -> dict[str, str]:
    corpora = {"code": "", "documentation": "", "config": ""}
    code_prefixes = ("scripts/", "src/", "tests/")
    documentation_prefixes = ("docs/", "notebooks/")
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if relative.startswith(code_prefixes) or relative in {"Makefile", "pyproject.toml"}:
            corpora["code"] += f"\n{relative}\n{text}"
        if relative.startswith(documentation_prefixes) or relative in {
            "README.md",
            "data/README.md",
        }:
            corpora["documentation"] += f"\n{relative}\n{text}"
        if relative.startswith("config/") or relative in {"Makefile", "pyproject.toml"}:
            corpora["config"] += f"\n{relative}\n{text}"
    return corpora


def _artifact_type(path: Path) -> str:
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "python_source"
    if suffix == ".ipynb":
        return "notebook"
    if suffix == ".md":
        return "documentation"
    if suffix in {".yaml", ".yml", ".toml"}:
        return "configuration"
    if suffix in {".csv", ".json", ".txt", ".xml"}:
        return "manifest_or_report"
    if suffix == ".pth":
        return "checkpoint"
    if suffix == ".zip":
        return "archive_package"
    return f"file_{suffix.lstrip('.') or 'no_extension'}"


def _classification(relative: str) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "unknown_requires_review",
        "reason": "No explicit lifecycle rule matched this artifact",
        "replacement_path": "",
        "recommended_action": "Review manually before moving or deleting",
        "safe_to_move": False,
        "safe_to_delete": False,
        "notes": "",
    }
    if relative in {"data"}:
        result.update(
            status="active_input",
            reason="PROJECT_DATA_ROOT default",
            recommended_action="Keep organized project data here",
        )
    elif relative in {"outputs"}:
        result.update(
            status="generated_output",
            reason="OUTPUT_ROOT default",
            recommended_action="Keep generated results here",
        )
    elif relative in {"public"}:
        result.update(
            status="generated_output",
            reason="Published documentation assets",
            recommended_action="Keep assets referenced by documentation",
        )
    elif relative in {"docs", "notebooks"}:
        result.update(
            status="active_documentation",
            reason="Documentation and reproducible analysis root",
            recommended_action="Keep synchronized with active decisions",
        )
    elif relative in {"scripts", "src", "config"}:
        result.update(
            status="active_source_code",
            reason="Active source or configuration root",
            recommended_action="Keep and validate",
        )
    elif "__pycache__" in relative or relative.endswith(".pyc"):
        result.update(
            status="deletion_candidate",
            reason="Generated Python bytecode; not source or scientific evidence",
            recommended_action="May be deleted after human approval",
            safe_to_move=True,
            safe_to_delete=True,
        )
    elif relative.startswith("outputs/dataset_audit_updated"):
        result.update(
            status="duplicate_copy",
            reason="Byte-identical to outputs/dataset_audit_final",
            replacement_path="outputs/dataset_audit_final/",
            recommended_action="Retain until human approves archival or deletion",
            safe_to_move=True,
            notes="No automatic cleanup was performed.",
        )
    elif relative == "data/leaf_detection/pilot/packages/pilot_images" or relative.startswith(
        "data/leaf_detection/pilot/packages/pilot_images/"
    ):
        result.update(
            status="duplicate_copy",
            reason="Unpacked package duplicates all 100 active pilot images by SHA-256",
            replacement_path="data/leaf_detection/pilot/images/",
            recommended_action="Preserve while package evidence is protected",
            notes="The package ZIP and directory were not modified.",
        )
    elif relative.startswith("outputs/preflight_gpu_check"):
        result.update(
            status="deprecated_path",
            reason="Superseded naming convention under outputs/preflight/",
            replacement_path="outputs/preflight/baseline_full_existing_models_seed42_cuda/",
            recommended_action="Archive only after human approval",
            safe_to_move=True,
            notes="Contains historical preflight evidence and must not be deleted automatically.",
        )
    elif relative.startswith("outputs/dataset_audit/"):
        result.update(
            status="historical_evidence",
            reason="Records the earlier class-mismatch diagnosis against an outdated dataset root",
            replacement_path="outputs/dataset_audit_final/",
            recommended_action="Preserve as historical evidence",
        )
    elif relative.startswith("outputs/dataset_audit_final"):
        result.update(
            status="generated_output",
            reason="Canonical successful 31,622-image class audit",
            recommended_action="Keep in place",
        )
    elif relative.startswith("outputs/baselines"):
        result.update(
            status="historical_evidence",
            reason="Official trained checkpoints and their metrics",
            recommended_action="Keep in place; never modify historical summaries",
        )
    elif relative.startswith("outputs/leaf_detection/pilot"):
        result.update(
            status="historical_evidence",
            reason="Manual ROI validation and full-vs-ROI diagnostic evidence",
            recommended_action="Keep in place",
        )
    elif relative.startswith("outputs/"):
        result.update(
            status="generated_output",
            reason="Derived report, metric, preview, audit, or preflight artifact",
            recommended_action="Keep unless a separate cleanup decision approves archival",
        )
    elif relative.startswith("data/splits/seed_42_baseline"):
        result.update(
            status="active_input",
            reason="Official reproducible seed-42 classification splits",
            recommended_action="Keep in place; do not edit manually",
        )
    elif relative.startswith("data/leaf_detection/pilot"):
        result.update(
            status="active_input",
            reason="Retained pilot, official CVAT annotation, manifests, and packages",
            recommended_action="Keep in place and out of training",
        )
    elif relative.startswith("data/leaf_detection/external_sources"):
        result.update(
            status="active_input",
            reason="Immutable external YOLO/COCO sources and original packages",
            recommended_action="Keep in place; only write derived data elsewhere",
        )
    elif relative.startswith("data/leaf_detection/detector_dataset"):
        result.update(
            status="active_input",
            reason="Prepared detector annotation batches and retained-test materialization",
            recommended_action="Keep in place; annotation batches remain pending",
        )
    elif relative.startswith("data/"):
        result.update(
            status="active_input",
            reason="Reproducible project data or its documentation",
            recommended_action="Keep under PROJECT_DATA_ROOT",
        )
    elif relative.startswith("scripts/cleanup"):
        result.update(
            status="historical_evidence",
            reason="Legacy one-off ingestion utilities with historical data/clean paths",
            recommended_action="Do not run; preserve for provenance only",
            notes="Active pipelines resolve the corpus through DATASET_ROOT.",
        )
    elif relative.startswith(("scripts/", "src/")):
        result.update(
            status="active_source_code",
            reason="Executable or reusable project source code",
            recommended_action="Keep and validate with tests/lint",
        )
    elif relative.startswith(("docs/", "notebooks/")) or relative in {
        "README.md",
        "CLAUDE.md",
    }:
        result.update(
            status="active_documentation",
            reason="Current technical documentation, ADR, or reproducible notebook",
            recommended_action="Keep synchronized with active paths",
        )
    elif relative.startswith("config/") or relative in {"Makefile", "pyproject.toml"}:
        result.update(
            status="active_source_code",
            reason="Active project configuration or build entrypoint",
            recommended_action="Keep in place",
        )
    elif relative.startswith("public/"):
        result.update(
            status="generated_output",
            reason="Published documentation asset",
            recommended_action="Keep while referenced by documentation",
        )
    return result


def _inventory_rows() -> tuple[list[dict[str, object]], list[Path]]:
    all_paths, directory_sizes = _iter_all_paths()
    included = [
        path
        for path in all_paths
        if path.is_dir() or (path.is_file() and _include_file(path))
    ]
    corpora = _text_corpora(all_paths)
    tracked = _git_tracked_paths()
    rows: list[dict[str, object]] = []
    for path in sorted(set(included)):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        lifecycle = _classification(relative)
        size = directory_sizes.get(path, path.stat().st_size if path.is_file() else 0)
        digest = ""
        notes = str(lifecycle["notes"])
        if path.is_file():
            if size <= HASH_LIMIT_BYTES:
                digest = _sha256(path)
            else:
                notes = (notes + " " if notes else "") + (
                    f"SHA-256 omitted because file exceeds {HASH_LIMIT_BYTES} bytes."
                )
        tracked_value = relative in tracked or any(
            item.startswith(f"{relative}/") for item in tracked
        )
        rows.append(
            {
                "path": relative,
                "artifact_type": _artifact_type(path),
                **lifecycle,
                "referenced_by_code": relative in corpora["code"],
                "referenced_by_documentation": relative in corpora["documentation"],
                "referenced_by_config": relative in corpora["config"],
                "git_tracked": tracked_value,
                "size_bytes": size,
                "hash_if_file": digest,
                "notes": notes,
            }
        )
    return rows, all_paths


def _obsolete_reference_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    patterns = (
        re.compile(r"(?<![/A-Z_])data/clean"),
        re.compile(r"/?outputs/splits(?:/seed_42_baseline)?"),
        re.compile(r"outputs/leaf_detection/pilot/(?:images|manifests|packages)"),
        re.compile(r"outputs/leaf_detection/pilot/annotations\.xml"),
    )
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, start=1):
            for pattern in patterns:
                for match in pattern.finditer(line):
                    reference = match.group(0)
                    if relative in {
                        "scripts/checks/audit_leaf_detection_repository.py",
                        "docs/es/decisions/adr-project-data-root-and-output-root.md",
                    }:
                        status = "intentional_migration_record"
                        action = "Preserve as the formal migration registry"
                    elif relative.startswith("scripts/cleanup/"):
                        status = "intentional_historical_reference"
                        action = "Preserve as provenance; do not execute this legacy script"
                    elif relative == "notebooks/01_eda.ipynb":
                        status = "intentional_historical_reference"
                        action = "Preserve historical notebook context"
                    elif (
                        relative.startswith("outputs/")
                        or relative == "tests/training/test_common.py"
                    ):
                        status = "intentional_historical_reference"
                        action = "Preserve; compatibility fallback is tested"
                    elif relative == "src/training/common.py":
                        status = "intentional_compatibility_reference"
                        action = "Preserve portable fallback and comment"
                    else:
                        status = "obsolete_reference_requires_correction"
                        action = "Replace with the active PROJECT_DATA_ROOT path"
                    rows.append(
                        {
                            "path": relative,
                            "line_number": number,
                            "reference": reference,
                            "status": status,
                            "replacement_path": (
                                "${PROJECT_DATA_ROOT}/splits/seed_42_baseline"
                                if "splits" in reference
                                else (
                                    "${DATASET_ROOT}/clean"
                                    if "data/clean" in reference
                                    else "data/leaf_detection/pilot/"
                                )
                            ),
                            "recommended_action": action,
                            "notes": line.strip()[:500],
                        }
                    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(
    inventory: list[dict[str, object]],
    obsolete: list[dict[str, object]],
) -> str:
    statuses = Counter(str(row["status"]) for row in inventory)
    obsolete_statuses = Counter(str(row["status"]) for row in obsolete)
    return f"""# Resumen de limpieza segura del repositorio

Fecha de auditoría: {date.today().isoformat()}.

## Alcance

Se inspeccionaron `data/`, `outputs/`, `public/`, `docs/`, `notebooks/`,
`scripts/`, `src/` y `config/`. La auditoría sólo escribió dentro de
`outputs/repository_audit/`: no movió ni eliminó artefactos, no modificó fuentes,
checkpoints, splits, el XML de CVAT, `roi_manifest.csv` ni resultados históricos.

## Inventario

- Artefactos inventariados: {len(inventory)}.
- Estados: `{json.dumps(dict(sorted(statuses.items())), ensure_ascii=False)}`.
- Referencias históricas/obsoletas registradas: {len(obsolete)}.
- Clasificación de referencias:
  `{json.dumps(dict(sorted(obsolete_statuses.items())), ensure_ascii=False)}`.

## Arquitectura activa confirmada

- Dataset grande: `DATASET_ROOT`.
- Datos derivados: `PROJECT_DATA_ROOT`, localmente `data/`.
- Resultados: `OUTPUT_ROOT`, localmente `outputs/`.
- Splits oficiales: `data/splits/seed_42_baseline/`.
- Piloto retenido: `data/leaf_detection/pilot/`.
- Fuentes externas: `data/leaf_detection/external_sources/`.
- Dataset preparado para anotación: `data/leaf_detection/detector_dataset/`.
- Checkpoints oficiales: `outputs/baselines/`.
- EDA de segmentación: `outputs/leaf_detection/external_sources_eda/`.

## Hallazgos de limpieza

1. `outputs/dataset_audit_updated/` es idéntico byte a byte a
   `outputs/dataset_audit_final/`.
2. `outputs/preflight_gpu_check/` usa una convención anterior; la convención
   vigente vive bajo `outputs/preflight/`.
3. Las 100 imágenes de `data/leaf_detection/pilot/images/` tienen copias exactas
   en el paquete descomprimido y en `detector_dataset/test/images/`. La segunda
   es una materialización de test intencional; los paquetes son evidencia
   protegida.
4. `__pycache__/` y `.pyc` son candidatos técnicos a eliminación, pero no se
   eliminaron.
5. Los `summary.json` de los modelos y preflights conservan `/outputs/splits`
   como ruta histórica remota. No se modificaron; `src.training.common` ya
   implementa el fallback portable.

## Decisión de archivo

No se archivó ninguna carpeta automáticamente. Aunque existen candidatos
seguros para mover, contienen evidencia o requieren aprobación humana. La
convención propuesta para una acción futura es
`outputs/archive/leaf_detection/<nombre>/`, siempre acompañada por
`ARCHIVE_INFO.md`.

## Candidatos no eliminados

- `outputs/dataset_audit_updated/`: copia duplicada.
- `outputs/preflight_gpu_check/`: ruta deprecada con evidencia.
- `data/leaf_detection/pilot/packages/pilot_images/`: copia desempaquetada
  protegida.
- directorios `__pycache__/` y archivos `.pyc`.

No se recomienda borrar ninguna evidencia hasta revisar el inventario y aprobar
explícitamente cada target.
"""


def main() -> None:
    inventory, all_paths = _inventory_rows()
    obsolete = _obsolete_reference_rows(all_paths)
    _write_csv(
        OUTPUT_DIR / "leaf_detection_artifact_inventory.csv",
        inventory,
        INVENTORY_COLUMNS,
    )
    (OUTPUT_DIR / "leaf_detection_artifact_inventory.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_on": date.today().isoformat(),
                "scan_roots": list(SCAN_ROOTS),
                "hash_limit_bytes": HASH_LIMIT_BYTES,
                "rows": inventory,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        OUTPUT_DIR / "obsolete_reference_report.csv",
        obsolete,
        OBSOLETE_COLUMNS,
    )
    _write_csv(
        OUTPUT_DIR / "path_migration_map.csv",
        list(PATH_MIGRATIONS),
        MIGRATION_COLUMNS,
    )
    (OUTPUT_DIR / "repository_cleanup_summary.md").write_text(
        _summary_markdown(inventory, obsolete),
        encoding="utf-8",
    )
    print(f"Inventory rows: {len(inventory)}")
    print(f"Reference rows: {len(obsolete)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
