#!/usr/bin/env python3
"""GPU/model preflight for the cloud-only leaf segmenter workflow."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch

from src.config import PROJECT_ROOT
from src.training.segmentation_preflight import (
    validate_segmentation_dataset,
    verify_cloud_training_payload,
)


def project_path(variable: str, default: str) -> Path:
    value = Path(os.getenv(variable, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


OUTPUT_ROOT = project_path("LEAF_SEGMENTATION_OUTPUT", "outputs/leaf_detection")
OUT = OUTPUT_ROOT / "cloud_preflight"
DATASET = project_path(
    "LEAF_SEGMENTATION_DATASET",
    "data/leaf_detection/detector_dataset",
)
MODEL_NAME = os.getenv("SEGMENTATION_MODEL", "yolo26n-seg.pt")
DEVICE = int(os.getenv("SEGMENTATION_DEVICE", "0"))


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status = "ready_for_smoke_training"
    blockers: list[str] = []
    try:
        dataset_locks = verify_cloud_training_payload(DATASET)
        dataset = validate_segmentation_dataset(DATASET)
        if not dataset["passed"]:
            raise RuntimeError(str(dataset["errors"]))
    except Exception as exc:
        dataset_locks = {"passed": False, "error": str(exc)}
        dataset = {"passed": False, "error": str(exc)}
        status = "blocked_by_dataset_change"
        blockers.append(str(exc))
    write("dataset_check.json", {**dataset, "locks": dataset_locks})

    gpu = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_compiled": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "name": None,
        "vram_free_bytes": None,
        "vram_total_bytes": None,
    }
    if gpu["cuda_available"]:
        gpu["name"] = torch.cuda.get_device_name(DEVICE)
        gpu["vram_free_bytes"], gpu["vram_total_bytes"] = torch.cuda.mem_get_info(
            DEVICE
        )
    elif status == "ready_for_smoke_training":
        status = "blocked_by_gpu"
        blockers.append("CUDA no disponible")
    write("gpu.json", gpu)
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": metadata.version("torchvision"),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        or "unavailable",
    }
    write("environment.json", environment)
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        stdout=(OUT / "dependency_lock.txt").open("w", encoding="utf-8"),
        check=True,
    )

    model_check: dict[str, object] = {
        "candidate": MODEL_NAME,
        "task": "segment",
        "constructed": False,
        "forward_executed": False,
        "segmentation_output": False,
        "ultralytics": None,
    }
    weights_manifest: dict[str, object] = {
        "model": MODEL_NAME,
        "download_allowed_in_cloud": True,
        "resolved": False,
    }
    memory: dict[str, object] = {"checked": False}
    if status == "ready_for_smoke_training":
        try:
            from ultralytics import YOLO
            from ultralytics import __version__ as ultralytics_version

            model_check["ultralytics"] = ultralytics_version
            torch.cuda.reset_peak_memory_stats(DEVICE)
            model = YOLO(MODEL_NAME)
            model_check["constructed"] = True
            candidate = Path(str(getattr(model, "ckpt_path", MODEL_NAME))).resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"Pesos no resueltos: {candidate}")
            weights_manifest.update(
                {
                    "resolved": True,
                    "path": str(candidate),
                    "sha256": sha256(candidate),
                    "size_bytes": candidate.stat().st_size,
                    "source": "Ultralytics model resolver",
                    "ultralytics_version": ultralytics_version,
                }
            )
            sample = next((DATASET / "images" / "train").iterdir())
            results = model.predict(
                source=str(sample), imgsz=640, device=DEVICE, verbose=False
            )
            model_check["forward_executed"] = True
            model_check["segmentation_output"] = bool(
                results and getattr(results[0], "masks", None) is not None
            )
            if not model_check["segmentation_output"]:
                raise RuntimeError("El forward no produjo máscaras")
            memory = {
                "checked": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(DEVICE),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(DEVICE),
                "vram_free_bytes": torch.cuda.mem_get_info(DEVICE)[0],
            }
        except ModuleNotFoundError as exc:
            status = "blocked_by_dependency"
            blockers.append(str(exc))
        except FileNotFoundError as exc:
            status = "blocked_by_weights"
            blockers.append(str(exc))
        except Exception as exc:
            status = "blocked_by_model"
            blockers.append(repr(exc))
    write("model_check.json", model_check)
    write("weights_manifest.json", weights_manifest)
    write("memory_check.json", memory)
    write(
        "summary.json",
        {
            "status": status,
            "blockers": blockers,
            "dataset_verified": bool(dataset.get("passed")),
            "gpu_verified": bool(gpu["cuda_available"]),
            "model_verified": bool(model_check["segmentation_output"]),
            "training_started": False,
            "epochs_run": 0,
            "optimizer_steps": 0,
        },
    )
    print(status)


if __name__ == "__main__":
    main()
