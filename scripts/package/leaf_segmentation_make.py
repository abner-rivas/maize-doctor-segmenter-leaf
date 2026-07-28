#!/usr/bin/env python3
"""Thin read-only/reporting helper for leaf-segmentation Makefile targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from scripts.package.build_leaf_segmentation_cloud_package import verify_extracted
from src.training.segmentation_preflight import (
    validate_segmentation_dataset,
    verify_cloud_training_payload,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def status(args: argparse.Namespace) -> None:
    manifests = args.dataset / "manifests"
    parent = load_json(manifests / "dataset_lock.json") or {}
    split = load_json(manifests / "split_lock.json") or {}
    cloud = load_json(args.output / "cloud_preflight" / "summary.json")
    smoke = load_json(args.output / "segmenter" / "smoke_summary.json")
    train = load_json(args.output / "segmenter" / "training_summary.json")
    counts = {}
    for name in ("train", "val", "test"):
        directory = args.dataset / "images" / name
        counts[name] = (
            len([path for path in directory.iterdir() if path.is_file()])
            if directory.is_dir()
            else None
        )
    experiment = args.output / "segmenter" / "yolo26n_seg_baseline"
    report = {
        "dataset_lock": parent.get("status", "missing"),
        "split_lock": split.get("status", "missing"),
        "cloud_preflight": cloud.get("status", "not-run") if cloud else "not-run",
        "smoke": smoke.get("status", "not-run") if smoke else "not-run",
        "train": train.get("status", "not-run") if train else "not-run",
        "model": args.model,
        "device": args.device,
        "counts": counts,
        "best_pt": (experiment / "weights" / "best.pt").is_file(),
        "last_pt": (experiment / "weights" / "last.pt").is_file(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def verify_locks(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            verify_cloud_training_payload(args.dataset),
            indent=2,
            sort_keys=True,
        )
    )


def verify_splits(args: argparse.Namespace) -> None:
    report = validate_segmentation_dataset(args.dataset)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


def latest_package(args: argparse.Namespace) -> Path:
    if args.package:
        return args.package.resolve()
    candidates = sorted(
        args.package_dir.glob("doctor_maiz_leaf_segmentation_cloud_*.tar.gz"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not candidates:
        raise SystemExit("No existen paquetes; use make leaf-segmentation-cloud-package")
    return candidates[-1]


def package_verify(args: argparse.Namespace) -> None:
    archive = latest_package(args)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    if not checksum_path.is_file():
        raise SystemExit(f"Falta checksum: {checksum_path}")
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    actual = sha256(archive)
    if actual != expected:
        raise SystemExit(f"SHA-256 inválido: {actual} != {expected}")
    result = verify_extracted(archive)
    print(
        json.dumps(
            {"archive": str(archive), "sha256": actual, "verification": result},
            indent=2,
            sort_keys=True,
        )
    )


def package_list(args: argparse.Namespace) -> None:
    rows = []
    for path in sorted(args.package_dir.glob("doctor_maiz_leaf_segmentation_cloud_*.tar.gz")):
        checksum_file = path.with_suffix(path.suffix + ".sha256")
        version = path.name.removeprefix(
            "doctor_maiz_leaf_segmentation_cloud_"
        ).removesuffix(".tar.gz")
        rows.append(
            {
                "package": str(path),
                "size_bytes": path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                "checksum_file": str(checksum_file),
                "sha256": (
                    checksum_file.read_text(encoding="utf-8").split()[0]
                    if checksum_file.is_file()
                    else None
                ),
                "version": version,
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))


def clean_temp(args: argparse.Namespace) -> None:
    removed = []
    for path in sorted(args.package_dir.glob(".tmp_leaf_cloud_verify_*")):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))
    print(json.dumps({"removed": removed}, indent=2))


def results(args: argparse.Namespace) -> None:
    experiment = args.output / "segmenter" / "yolo26n_seg_baseline"
    expected = [
        experiment / "weights" / "best.pt",
        experiment / "weights" / "last.pt",
        experiment / "results.csv",
        experiment / "args.yaml",
    ]
    discovered = (
        [
            path
            for path in experiment.rglob("*")
            if path.is_file()
            and (
                path.suffix.lower() in {".csv", ".json", ".log", ".png", ".yaml"}
                or path.name in {"best.pt", "last.pt"}
            )
        ]
        if experiment.is_dir()
        else []
    )
    selected = []
    for path in sorted(set(expected + discovered)):
        selected.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256(path) if path.is_file() else None,
            }
        )
    print(json.dumps({"experiment": str(experiment), "files": selected}, indent=2))


def result_checksums(args: argparse.Namespace) -> None:
    experiment = args.output / "segmenter" / "yolo26n_seg_baseline"
    candidates = [
        experiment / "weights" / "best.pt",
        experiment / "weights" / "last.pt",
        experiment / "args.yaml",
        experiment / "results.csv",
        args.cloud_dir / "runtime_environment.lock",
        *sorted(experiment.glob("*metrics*.json")),
    ]
    existing = sorted({path for path in candidates if path.is_file()})
    if not existing:
        raise SystemExit("No hay resultados para calcular checksums")
    output = args.output / "segmenter" / "result_checksums.sha256"
    output.write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in existing),
        encoding="utf-8",
    )
    print(output)


def pilot_gate(args: argparse.Namespace) -> None:
    test_summary = args.output / "segmenter_evaluation" / "test_summary.json"
    report = load_json(test_summary)
    if not report or report.get("status") != "passed":
        raise SystemExit("El test interno debe completarse antes del piloto")
    pilot = Path("data/leaf_detection/pilot")
    if not pilot.is_dir():
        raise SystemExit("Falta el piloto retenido")
    print("Pilot gate aprobado; invoque el evaluador externo autorizado.")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=(
            "status",
            "verify-locks",
            "verify-splits",
            "package-verify",
            "package-list",
            "clean-temp",
            "results",
            "checksums",
            "pilot-gate",
        ),
    )
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--cloud-dir", type=Path, required=True)
    result.add_argument("--package-dir", type=Path, required=True)
    result.add_argument("--package", type=Path)
    result.add_argument("--model", required=True)
    result.add_argument("--device", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    actions = {
        "status": status,
        "verify-locks": verify_locks,
        "verify-splits": verify_splits,
        "package-verify": package_verify,
        "package-list": package_list,
        "clean-temp": clean_temp,
        "results": results,
        "checksums": result_checksums,
        "pilot-gate": pilot_gate,
    }
    actions[args.command](args)


if __name__ == "__main__":
    main()
