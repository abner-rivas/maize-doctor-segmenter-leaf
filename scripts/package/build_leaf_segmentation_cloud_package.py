#!/usr/bin/env python3
"""Build and verify a deterministic allow-listed cloud segmentation package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

from src.config import PROJECT_ROOT, get_output_root
from src.data.jpeg_normalization import (
    JpegNormalizationError,
    audit_jpegs,
    validate_jpegs_before_packaging,
)
from src.training.segmentation_preflight import verify_cloud_training_payload

PACKAGE_VERSION = "v7-segmentation-improvements-7a4a5c08-seed42"
METADATA_PATHS = {
    Path("cloud_training/package_manifest.json"),
    Path("cloud_training/checksums.sha256"),
}
GENERATED_STATUS_PATHS = {
    "cloud_training/COMMIT_VERSION.txt",
    "cloud_training/pilot_transport_manifest.json",
    "cloud_training/package_manifest.json",
    "cloud_training/checksums.sha256",
}
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".venv-cloud",
    "__pycache__",
    "external_sources",
    "all",
    "pilot",
    "outputs",
    "notebooks",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git_state(root: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.splitlines()
    stable_dirty: list[str] = []
    deleted_path_count = 0
    for line in dirty:
        if "D" in line[:2]:
            deleted_path_count += 1
            continue
        if any(line.endswith(path) for path in GENERATED_STATUS_PATHS):
            continue
        stable_dirty.append(line)
    return {
        "commit": commit or "unavailable",
        "dirty_paths": sorted(stable_dirty),
        "deleted_path_count": deleted_path_count,
    }


def dataset_paths(dataset: Path) -> list[Path]:
    paths = [dataset / "dataset.yaml"]
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            paths.extend(path for path in (dataset / kind / split).iterdir() if path.is_file())
    for name in (
        "dataset_lock.json",
        "split_lock.json",
        "split_manifest.csv",
        "split_groups.csv",
        "split_fingerprints.json",
        "image_normalization_manifest.csv",
        "auxiliary_jpeg_audit.csv",
    ):
        paths.append(dataset / "manifests" / name)
    return paths


def code_paths(root: Path) -> list[Path]:
    paths: list[Path] = [
        root / "Makefile",
        root / "pyproject.toml",
        root / "README.md",
        root / "config" / "segmentation.yaml",
    ]
    for directory in ("src", "cloud_training"):
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.relative_to(root) not in METADATA_PATHS
            and path.name not in {"runtime_environment.lock", "runtime_constraints.txt"}
        )
    for relative in (
        "scripts/pipeline/leaf_segmentation_cloud_preflight.py",
        "scripts/pipeline/leaf_segmentation_downstream_metrics.py",
        "scripts/pipeline/leaf_segmentation_pilot_evaluate.py",
        "scripts/pipeline/leaf_segmentation_preflight.py",
        "scripts/package/build_leaf_segmentation_cloud_package.py",
        "scripts/package/leaf_segmentation_make.py",
        "docs/es/leaf-detection/segmentation-cloud-training.md",
        "docs/es/leaf-detection/segmentation-training-preflight.md",
        "docs/es/leaf-detection/segmentation-dataset-splits.md",
    ):
        path = root / relative
        if path.is_file():
            paths.append(path)
    return paths


def collect_payload(root: Path, dataset: Path | None = None) -> list[Path]:
    if dataset is None:
        dataset = root / "data" / "leaf_detection" / "detector_dataset"
    paths = sorted(
        {path.resolve() for path in (*dataset_paths(dataset), *code_paths(root))},
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(root)
        if path.is_symlink() or any(part in EXCLUDED_PARTS for part in relative.parts):
            raise RuntimeError(f"Ruta prohibida en payload: {relative}")
    return paths


def pilot_manifest(root: Path) -> dict[str, object]:
    pilot = root / "data" / "leaf_detection" / "pilot"
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(
            (path for path in pilot.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    ]
    return {
        "schema_version": 1,
        "purpose": "external_held_out_evaluation_transport_only",
        "included_in_training_package": False,
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }


def file_rows(root: Path, paths: Iterable[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


def tar_info(relative: Path, size: int, executable: bool) -> tarfile.TarInfo:
    info = tarfile.TarInfo(relative.as_posix())
    info.size = size
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o755 if executable else 0o644
    return info


def build_archive(
    root: Path,
    output_dir: Path,
    version: str,
    dataset_root: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    dataset_root = (
        dataset_root.resolve()
        if dataset_root is not None
        else root / "data" / "leaf_detection" / "detector_dataset"
    )
    try:
        dataset_relative = dataset_root.relative_to(root)
    except ValueError as exc:
        raise ValueError("El dataset debe estar dentro del proyecto") from exc
    locks = verify_cloud_training_payload(dataset_root)
    jpeg_validation = validate_jpegs_before_packaging(dataset_root)
    commit = git_state(root)
    (root / "cloud_training" / "COMMIT_VERSION.txt").write_text(
        f"commit={commit['commit']}\npackage_version={version}\n",
        encoding="utf-8",
    )
    write_json(
        root / "cloud_training" / "pilot_transport_manifest.json",
        pilot_manifest(root),
    )
    base_paths = collect_payload(root, dataset_root)
    rows = file_rows(root, base_paths)
    manifest = {
        "schema_version": 1,
        "package_version": version,
        "model_candidate": "yolo26n-seg.pt",
        "dataset_relative_path": dataset_relative.as_posix(),
        "expected_status": "ready_for_cloud_bootstrap",
        "parent_fingerprint": locks["parent_fingerprint"],
        "split_fingerprints": locks["split_fingerprints"],
        "git": commit,
        "payload_file_count": len(rows),
        "payload_total_bytes": sum(row["size_bytes"] for row in rows),
        "files": rows,
        "generated_metadata": [
            "cloud_training/package_manifest.json",
            "cloud_training/checksums.sha256",
        ],
        "excluded": sorted(EXCLUDED_PARTS),
        "pilot_included": False,
        "jpeg_validation": jpeg_validation,
    }
    manifest_path = root / "cloud_training" / "package_manifest.json"
    write_json(manifest_path, manifest)
    checksum_rows = [*rows, *file_rows(root, [manifest_path])]
    checksum_path = root / "cloud_training" / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in checksum_rows),
        encoding="utf-8",
    )
    final_paths = [*base_paths, manifest_path, checksum_path]
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"doctor_maiz_leaf_segmentation_cloud_{version}.tar.gz"
    package_root = Path(f"doctor_maiz_leaf_segmentation_cloud_{version}")
    with archive.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(
                    final_paths, key=lambda item: item.relative_to(root).as_posix()
                ):
                    relative = package_root / path.relative_to(root)
                    executable = path.suffix == ".sh" or path.name in {
                        "run_ultralytics.py",
                        "leaf_segmentation_cloud_preflight.py",
                    }
                    info = tar_info(relative, path.stat().st_size, executable)
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    archive_digest = sha256(archive)
    checksum_file = archive.with_suffix(archive.suffix + ".sha256")
    checksum_file.write_text(
        f"{archive_digest}  {archive.name}\n", encoding="utf-8"
    )
    manifest["archive"] = {
        "path": str(archive),
        "size_bytes": archive.stat().st_size,
        "sha256": archive_digest,
        "checksum_file": str(checksum_file),
    }
    return archive, manifest


def verify_extracted(
    archive: Path,
    source_root: Path | None = None,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(
        prefix=".tmp_leaf_cloud_verify_",
        dir=archive.parent,
    ) as temporary:
        target = Path(temporary)
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            if any(
                member.name.startswith("/")
                or ".." in Path(member.name).parts
                or member.issym()
                or member.islnk()
                for member in members
            ):
                raise RuntimeError("Miembro inseguro en tar")
            tar.extractall(target, filter="data")
        roots = list(target.iterdir())
        if len(roots) != 1:
            raise RuntimeError("Raíz inesperada")
        extracted = roots[0]
        checksum_path = extracted / "cloud_training" / "checksums.sha256"
        checked = 0
        verified_digests: dict[str, str] = {}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            path = extracted / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"Checksum inválido: {relative}")
            verified_digests[relative] = expected
            checked += 1
        forbidden = [
            relative
            for relative in ("data/leaf_detection/external_sources", "data/leaf_detection/pilot",
                             "data/leaf_detection/detector_dataset/all", "outputs", ".venv")
            if (extracted / relative).exists()
        ]
        if forbidden:
            raise RuntimeError(f"Rutas excluidas presentes: {forbidden}")
        counts: dict[str, dict[str, int]] = {}
        expected = {"train": 809, "val": 173, "test": 173}
        manifest = json.loads(
            (extracted / "cloud_training" / "package_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            manifest.get("expected_status") != "ready_for_cloud_bootstrap"
            or manifest.get("pilot_included") is not False
        ):
            raise RuntimeError("Estado o exclusión del piloto inválidos en manifiesto")
        manifest_rows = manifest.get("files")
        if not isinstance(manifest_rows, list) or len(manifest_rows) != manifest.get(
            "payload_file_count"
        ):
            raise RuntimeError("Conteo de archivos inválido en manifiesto")
        for row in manifest_rows:
            if not isinstance(row, dict):
                raise RuntimeError("Fila no válida en manifiesto")
            relative = row.get("path")
            path = extracted / str(relative)
            if (
                not isinstance(relative, str)
                or not path.is_file()
                or path.stat().st_size != row.get("size_bytes")
                or verified_digests.get(relative) != row.get("sha256")
            ):
                raise RuntimeError(f"Entrada inválida en manifiesto: {relative}")
        expected_members = {
            *(str(row["path"]) for row in manifest_rows),
            "cloud_training/package_manifest.json",
            "cloud_training/checksums.sha256",
        }
        archive_members: dict[str, tarfile.TarInfo] = {}
        for member in members:
            parts = Path(member.name).parts
            if len(parts) < 2 or not member.isfile():
                raise RuntimeError(f"Miembro no regular o sin raíz única: {member.name}")
            archive_members[Path(*parts[1:]).as_posix()] = member
        if set(archive_members) != expected_members:
            raise RuntimeError(
                "El tar contiene archivos extra o faltantes: "
                f"extra={sorted(set(archive_members) - expected_members)}, "
                f"missing={sorted(expected_members - set(archive_members))}"
            )
        for relative, member in archive_members.items():
            executable = relative.endswith(".sh") or Path(relative).name in {
                "run_ultralytics.py",
                "leaf_segmentation_cloud_preflight.py",
            }
            expected_mode = 0o755 if executable else 0o644
            if member.mode != expected_mode:
                raise RuntimeError(
                    f"Permiso inválido: {relative}: {oct(member.mode)} "
                    f"!= {oct(expected_mode)}"
                )
        source_checked = 0
        if source_root is not None:
            source_root = source_root.resolve()
            source_dataset = source_root / str(manifest["dataset_relative_path"])
            current_paths = {
                path.relative_to(source_root).as_posix()
                for path in collect_payload(source_root, source_dataset)
            }
            manifest_paths = {str(row["path"]) for row in manifest_rows}
            if current_paths != manifest_paths:
                raise RuntimeError(
                    "El payload no corresponde a la lista blanca actual: "
                    f"extra={sorted(manifest_paths - current_paths)}, "
                    f"missing={sorted(current_paths - manifest_paths)}"
                )
            for row in manifest_rows:
                relative = str(row["path"])
                source = source_root / relative
                if (
                    not source.is_file()
                    or source.stat().st_size != row["size_bytes"]
                    or sha256(source) != row["sha256"]
                ):
                    raise RuntimeError(
                        f"El paquete no corresponde al código/dataset actual: {relative}"
                    )
                source_checked += 1
        required = {
            "Makefile",
            "cloud_training/bootstrap_cloud.sh",
            "cloud_training/preflight_cloud.sh",
            "cloud_training/smoke_train.sh",
            "cloud_training/train.sh",
            "cloud_training/resume_train.sh",
            "cloud_training/validate.sh",
            "cloud_training/evaluate_test.sh",
            "cloud_training/run_ultralytics.py",
            "cloud_training/package_manifest.json",
            "cloud_training/CLOUD_READINESS_CHECKLIST.md",
            "scripts/package/build_leaf_segmentation_cloud_package.py",
            "scripts/package/leaf_segmentation_make.py",
            "scripts/pipeline/leaf_segmentation_downstream_metrics.py",
            "scripts/pipeline/leaf_segmentation_pilot_evaluate.py",
            "src/evaluation/segmentation_downstream.py",
        }
        missing = sorted(path for path in required if not (extracted / path).is_file())
        if missing:
            raise RuntimeError(f"Archivos esperados ausentes: {missing}")
        dataset = extracted / manifest["dataset_relative_path"]
        jpeg_audit = audit_jpegs(
            path
            for path in (dataset / "images").rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        )
        if jpeg_audit["problem_count"]:
            raise JpegNormalizationError(
                f"El paquete extraído contiene JPEG inválidos: {jpeg_audit['problems']}"
            )
        declared_jpeg_validation = manifest.get("jpeg_validation")
        if (
            not isinstance(declared_jpeg_validation, dict)
            or declared_jpeg_validation.get("passed") is not True
            or declared_jpeg_validation.get("ultralytics_scan", {}).get(
                "mutated_file_count"
            )
            != 0
        ):
            raise RuntimeError("Falta la validación JPEG/Ultralytics previa al paquete")
        for split, expected_count in expected.items():
            images = len(list((dataset / "images" / split).iterdir()))
            labels = len(list((dataset / "labels" / split).glob("*.txt")))
            counts[split] = {"images": images, "labels": labels}
            if images != expected_count or labels != expected_count:
                raise RuntimeError(
                    f"Conteo extraído inválido {split}: {images}/{labels}"
                )
        return {
            "passed": True,
            "members": len(members),
            "checksums_verified": checked,
            "forbidden_paths_found": forbidden,
            "dataset_counts": counts,
            "jpeg_audit": jpeg_audit,
            "source_files_verified": source_checked,
        }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    result.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "leaf_detection" / "detector_dataset",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=get_output_root() / "leaf_detection" / "packages",
    )
    result.add_argument("--version", default=PACKAGE_VERSION)
    result.add_argument("--verify-extract", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    archive, manifest = build_archive(
        args.project_root.resolve(),
        args.output_dir.resolve(),
        args.version,
        args.dataset_root.resolve(),
    )
    result = (
        verify_extracted(archive, args.project_root.resolve())
        if args.verify_extract
        else {"passed": None}
    )
    print(
        json.dumps(
            {"archive": manifest["archive"], "verification": result},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
