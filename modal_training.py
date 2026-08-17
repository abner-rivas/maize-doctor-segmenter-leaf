"""Modal control plane for the frozen DoctorMaiz leaf-segmentation package.

The 2.13 GB release archive is never added to the Modal Image. Upload it once:

    modal volume put doctor-maiz-leaf-segmentation \
      <ruta-al-paquete-del-segmentador.tar.gz> \
      /incoming/

Then invoke the independent remote functions through the segmenter Makefile.
Training functions require the literal CLI argument ``--confirm true``.
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

APP_NAME = "doctor-maiz-leaf-segmentation"
VOLUME_NAME = "doctor-maiz-leaf-segmentation"
VOLUME_MOUNT = Path("/workspace")
INCOMING_ROOT = VOLUME_MOUNT / "incoming"
PACKAGE_VERSION = "v6-segmenter-only-7a4a5c08-seed42"
PACKAGE_NAME = f"doctor_maiz_leaf_segmentation_cloud_{PACKAGE_VERSION}.tar.gz"
PACKAGE_ROOT_NAME = f"doctor_maiz_leaf_segmentation_cloud_{PACKAGE_VERSION}"
PACKAGE_SHA256 = "469b019489194929bcd32008afa5943d5f099dad4078f38dceb39f5f457d4144"
PROJECT_ROOT = VOLUME_MOUNT / f"project_{PACKAGE_VERSION}"
ARTIFACT_PROJECT_VERSION = "v4-7a4a5c08-seed42"
ARTIFACT_PROJECT_ROOT = VOLUME_MOUNT / f"project_{ARTIFACT_PROJECT_VERSION}"
SEGMENTATION_OUTPUT_ROOT = (
    ARTIFACT_PROJECT_ROOT / "outputs" / "leaf_detection"
)
EXPECTED_PARENT_FINGERPRINT = "7a4a5c083fc64b067df12bcc95ec976d5a7e3b8a585d0a090b6b3940af4d7d5c"
EXPECTED_TEST_FINGERPRINT = "046545351ce79431bb1a995dfbc7dfa44c642a18a046860ed5edb9fc0ed89c51"
EXPECTED_BEST_CHECKPOINT_SHA256 = (
    "4f66456d05d87f9e7080155eb5cd80c583f34849415ec820c950bd97f9c5ec6f"
)
PREPARED_MARKER = PROJECT_ROOT / ".modal_package_prepared.json"
PREPARE_STAGING = VOLUME_MOUNT / f".project_extracting_{PACKAGE_SHA256}"

BASE_IMAGE_TAG = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
BASE_IMAGE_DIGEST = "sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee"
BASE_IMAGE = f"{BASE_IMAGE_TAG}@{BASE_IMAGE_DIGEST}"
EXPECTED_PYTHON = "3.11"
EXPECTED_TORCH = "2.6.0"
EXPECTED_TORCHVISION = "0.21.0"
EXPECTED_ULTRALYTICS = "8.4.104"
EXPECTED_FASTER_COCO_EVAL = "1.7.2"
EXPECTED_CUDA = "12.4"
EXPECTED_CUDA_LOCAL = "cu124"
IMAGE_SYSTEM_PACKAGES = (
    "bash",
    "coreutils",
    "git",
    "libgl1",
    "libglib2.0-0",
    "make",
    "procps",
    "tar",
)
IMAGE_PYTHON_PACKAGES = (
    "filelock==3.18.0",
    f"faster-coco-eval=={EXPECTED_FASTER_COCO_EVAL}",
    "matplotlib==3.10.3",
    "numpy==1.26.4",
    "nvidia-ml-py==12.575.51",
    "opencv-python==4.11.0.86",
    "pandas==2.3.1",
    "pillow==11.2.1",
    "polars==1.31.0",
    "psutil==7.0.0",
    "python-dotenv==1.1.1",
    "pyyaml==6.0.2",
    "requests==2.32.4",
    "tqdm==4.67.1",
    "ultralytics-thop==2.0.18",
    f"ultralytics=={EXPECTED_ULTRALYTICS}",
)
IMAGE_BUILD_LOCK = Path("/opt/doctor_maiz_modal_image.lock")
IMAGE_RECIPE = {
    "base_image": BASE_IMAGE,
    "base_image_tag": BASE_IMAGE_TAG,
    "base_image_digest": BASE_IMAGE_DIGEST,
    "python": EXPECTED_PYTHON,
    "torch": EXPECTED_TORCH,
    "torchvision": EXPECTED_TORCHVISION,
    "cuda": EXPECTED_CUDA,
    "cuda_local": EXPECTED_CUDA_LOCAL,
    "system_packages": IMAGE_SYSTEM_PACKAGES,
    "python_packages": IMAGE_PYTHON_PACKAGES,
}
IMAGE_RECIPE_SHA256 = hashlib.sha256(
    json.dumps(IMAGE_RECIPE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

ALLOWED_GPUS = ("A10", "L4", "A100")
REQUESTED_GPU = os.getenv("DOCTOR_MAIZ_MODAL_GPU", "A10").strip().upper()
if REQUESTED_GPU not in ALLOWED_GPUS:
    raise ValueError(
        f"DOCTOR_MAIZ_MODAL_GPU={REQUESTED_GPU!r} no permitido; use uno de {ALLOWED_GPUS}"
    )
MINIMUM_VRAM_BYTES = 12 * 1024**3


def _validate_image_versions(
    actual: dict[str, str | None],
    python_version: tuple[int, ...],
) -> None:
    from packaging.version import InvalidVersion, Version

    expected_python = tuple(int(part) for part in EXPECTED_PYTHON.split("."))
    if python_version[:2] != expected_python:
        raise RuntimeError(
            f"python={actual.get('python')!r}; versión esperada {EXPECTED_PYTHON}.x"
        )

    expected_distributions = {
        "faster-coco-eval": EXPECTED_FASTER_COCO_EVAL,
        "torch": EXPECTED_TORCH,
        "torchvision": EXPECTED_TORCHVISION,
        "ultralytics": EXPECTED_ULTRALYTICS,
    }
    for name, expected in expected_distributions.items():
        installed = actual.get(name)
        if not installed:
            raise RuntimeError(f"{name} no reportó una versión; esperado {expected}")
        try:
            installed_release = Version(installed).release
        except InvalidVersion as exc:
            raise RuntimeError(f"{name}={installed!r} no es una versión válida") from exc
        expected_release = Version(expected).release
        if installed_release != expected_release:
            raise RuntimeError(
                f"{name}={installed!r} tiene release {installed_release}; "
                f"esperado {expected_release}"
            )

    for name in ("torch_import", "torchvision_import"):
        imported = actual.get(name)
        if not imported:
            raise RuntimeError(f"{name} no reportó una versión importada")
        try:
            local = Version(imported).local
        except InvalidVersion as exc:
            raise RuntimeError(f"{name}={imported!r} no es una versión válida") from exc
        if local != EXPECTED_CUDA_LOCAL:
            raise RuntimeError(
                f"{name}={imported!r} no contiene el sufijo local "
                f"+{EXPECTED_CUDA_LOCAL}"
            )

    if actual.get("torch_cuda") != EXPECTED_CUDA:
        raise RuntimeError(
            f"torch.version.cuda={actual.get('torch_cuda')!r}; "
            f"esperado {EXPECTED_CUDA!r}"
        )


def _validate_modal_image_versions() -> None:
    import sys
    from importlib import metadata

    import torch
    import torchvision

    actual = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": metadata.version("torch"),
        "torch_import": str(torch.__version__),
        "torchvision": metadata.version("torchvision"),
        "torchvision_import": str(torchvision.__version__),
        "ultralytics": metadata.version("ultralytics"),
        "faster-coco-eval": metadata.version("faster-coco-eval"),
        "torch_cuda": torch.version.cuda,
    }
    print("Modal image version check:", flush=True)
    for name, version in actual.items():
        print(f"  {name}: {version}", flush=True)
    _validate_image_versions(actual, tuple(sys.version_info[:3]))

modal_image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .apt_install(*IMAGE_SYSTEM_PACKAGES)
    .pip_install(*IMAGE_PYTHON_PACKAGES)
    .run_commands(
        "python -m pip check",
        f"python -m pip freeze | LC_ALL=C sort > {IMAGE_BUILD_LOCK}",
    )
    .run_function(_validate_modal_image_versions)
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "YOLO_CONFIG_DIR": "/workspace/runtime/ultralytics",
            "MPLCONFIGDIR": "/workspace/runtime/matplotlib",
        }
    )
)

app = modal.App("doctor-maiz-leaf-segmentation")
workspace = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
VOLUME_MOUNTS = {str(VOLUME_MOUNT): workspace}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _modal_object_id(handle: object) -> str | None:
    try:
        value = getattr(handle, "object_id")
    except AttributeError:
        return None
    return str(value) if value else None


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
    )


def _verify_extracted_release(root: Path) -> dict[str, Any]:
    manifest_path = root / "cloud_training" / "package_manifest.json"
    checksums_path = root / "cloud_training" / "checksums.sha256"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise RuntimeError("La extracción no contiene manifiesto/checksums cloud")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("package_version") != PACKAGE_VERSION:
        raise RuntimeError(f"Versión extraída inválida: {manifest.get('package_version')!r}")
    if manifest.get("parent_fingerprint") != EXPECTED_PARENT_FINGERPRINT:
        raise RuntimeError("El fingerprint padre del paquete no es el congelado")
    jpeg_validation = manifest.get("jpeg_validation")
    if (
        not isinstance(jpeg_validation, dict)
        or jpeg_validation.get("passed") is not True
        or jpeg_validation.get("ultralytics_scan", {}).get("mutated_file_count") != 0
    ):
        raise RuntimeError("El paquete no declara un escaneo JPEG/Ultralytics limpio")
    _run(
        ["sha256sum", "--check", "--quiet", str(checksums_path)],
        cwd=root,
    )
    return manifest


def _prepared_payload() -> dict[str, Any]:
    if not PREPARED_MARKER.is_file():
        raise RuntimeError(f"Falta {PREPARED_MARKER}; ejecute primero modal_training.py::prepare")
    payload = json.loads(PREPARED_MARKER.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "ready"
        or payload.get("package_sha256") != PACKAGE_SHA256
        or payload.get("package_version") != PACKAGE_VERSION
        or payload.get("jpeg_validation", {}).get("ultralytics_scan", {}).get(
            "mutated_file_count"
        )
        != 0
    ):
        raise RuntimeError("La extracción preparada no corresponde al paquete esperado")
    return payload


def _project_environment() -> dict[str, str]:
    environment = os.environ.copy()

    project_root = str(PROJECT_ROOT)
    current_pythonpath = environment.get("PYTHONPATH", "")
    pythonpath_entries = [
        entry for entry in current_pythonpath.split(os.pathsep) if entry
    ]
    pythonpath_entries = [
        entry for entry in pythonpath_entries if entry != project_root
    ]
    pythonpath_entries.insert(0, project_root)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    environment.update(
        {
            "PYTHON": sys.executable,
            "CLOUD_TRAINING_DIR": str(PROJECT_ROOT / "cloud_training"),
            "LEAF_SEGMENTATION_DATASET": str(
                PROJECT_ROOT / "data" / "leaf_detection" / "detector_dataset"
            ),
            "LEAF_SEGMENTATION_OUTPUT": str(SEGMENTATION_OUTPUT_ROOT),
            "SEGMENTATION_MODEL": "yolo26n-seg.pt",
            "SEGMENTATION_DEVICE": "0",
        }
    )
    return environment


def _nvidia_smi() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
    if len(values) != 5:
        raise RuntimeError(f"Salida inesperada de nvidia-smi: {completed.stdout!r}")
    return {
        "name": values[0],
        "memory_total_mib": int(values[1]),
        "memory_free_mib": int(values[2]),
        "utilization_gpu_percent": int(values[3]),
        "driver_version": values[4],
    }


def _runtime_report(operation: str, *, require_gpu: bool) -> dict[str, Any]:
    import torch
    import torchvision

    runtime_root = PROJECT_ROOT / "outputs" / "leaf_detection" / "modal_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    build_lock = IMAGE_BUILD_LOCK.read_text(encoding="utf-8") if IMAGE_BUILD_LOCK.is_file() else ""
    actual_versions = {
        "python": platform.python_version(),
        "torch": metadata.version("torch"),
        "torch_import": str(torch.__version__),
        "torchvision": metadata.version("torchvision"),
        "torchvision_import": str(torchvision.__version__),
        "ultralytics": metadata.version("ultralytics"),
        "faster-coco-eval": metadata.version("faster-coco-eval"),
        "torch_cuda": torch.version.cuda,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "checking",
        "operation": operation,
        "utc": _utc_now(),
        "app": APP_NAME,
        "volume": VOLUME_NAME,
        "volume_mount": str(VOLUME_MOUNT),
        "function_call_id": modal.current_function_call_id(),
        "input_id": modal.current_input_id(),
        "requested_gpu": REQUESTED_GPU,
        "allowed_gpus": list(ALLOWED_GPUS),
        "base_image": BASE_IMAGE,
        "base_image_tag": BASE_IMAGE_TAG,
        "base_image_digest": BASE_IMAGE_DIGEST,
        "modal_image_id": _modal_object_id(modal_image),
        "modal_volume_id": _modal_object_id(workspace),
        "image_recipe_sha256": IMAGE_RECIPE_SHA256,
        "image_build_lock_sha256": (
            hashlib.sha256(build_lock.encode()).hexdigest() if build_lock else None
        ),
        "python": actual_versions["python"],
        "python_executable": sys.executable,
        "torch": actual_versions["torch"],
        "torch_import": actual_versions["torch_import"],
        "torchvision": actual_versions["torchvision"],
        "torchvision_import": actual_versions["torchvision_import"],
        "cuda_compiled": actual_versions["torch_cuda"],
        "cudnn": torch.backends.cudnn.version(),
        "ultralytics": actual_versions["ultralytics"],
        "faster_coco_eval": actual_versions["faster-coco-eval"],
        "cuda_available": torch.cuda.is_available(),
        "gpu": None,
    }
    errors: list[str] = []
    try:
        _validate_image_versions(actual_versions, tuple(sys.version_info[:3]))
    except RuntimeError as exc:
        errors.append(str(exc))
    if require_gpu:
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() es false")
        else:
            properties = torch.cuda.get_device_properties(0)
            nvidia = _nvidia_smi()
            report["gpu"] = {
                "requested": REQUESTED_GPU,
                "received": torch.cuda.get_device_name(0),
                "total_vram_bytes": properties.total_memory,
                "free_vram_bytes": torch.cuda.mem_get_info(0)[0],
                "initial_utilization_percent": nvidia["utilization_gpu_percent"],
                "nvidia_smi": nvidia,
            }
            if REQUESTED_GPU not in str(report["gpu"]["received"]).upper():
                errors.append(
                    f"GPU recibida no coincide con {REQUESTED_GPU}: {report['gpu']['received']}"
                )
            if properties.total_memory < MINIMUM_VRAM_BYTES:
                errors.append(
                    f"VRAM insuficiente: {properties.total_memory} < {MINIMUM_VRAM_BYTES}"
                )
    report["errors"] = errors
    report["status"] = "ready" if not errors else "blocked"

    call_id = report["function_call_id"] or report["input_id"] or "unknown"
    _write_json(runtime_root / f"{operation}_{call_id}.json", report)
    _write_json(runtime_root / f"{operation}_latest.json", report)
    lock_lines = {
        "base_image": BASE_IMAGE,
        "image_recipe_sha256": IMAGE_RECIPE_SHA256,
        "python": report["python"],
        "torch": report["torch"],
        "torchvision": report["torchvision"],
        "cuda": report["cuda_compiled"],
        "cudnn": report["cudnn"],
        "ultralytics": report["ultralytics"],
        "faster_coco_eval": report["faster_coco_eval"],
        "requested_gpu": REQUESTED_GPU,
    }
    serialized_lock = "".join(f"{key}={value}\n" for key, value in lock_lines.items())
    for lock_name in (
        "runtime_environment.lock",
        "runtime_environment.modal.lock",
    ):
        (PROJECT_ROOT / "cloud_training" / lock_name).write_text(
            serialized_lock,
            encoding="utf-8",
        )
    if operation == "preflight":
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            capture_output=True,
            check=True,
        )
        (runtime_root / "pip_freeze.txt").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        if build_lock:
            (runtime_root / "image_build_pip_freeze.txt").write_text(
                build_lock,
                encoding="utf-8",
            )
    if errors:
        raise RuntimeError("; ".join(errors))
    return report


def _make(target: str, *variables: str) -> None:
    command = [
        "make",
        target,
        f"PYTHON={sys.executable}",
        "CLOUD_TRAINING_DIR=cloud_training",
        "LEAF_SEGMENTATION_DATASET=data/leaf_detection/detector_dataset",
        f"LEAF_SEGMENTATION_OUTPUT={SEGMENTATION_OUTPUT_ROOT}",
        "SEGMENTATION_MODEL=yolo26n-seg.pt",
        "SEGMENTATION_DEVICE=0",
        *variables,
    ]
    _run(command, cwd=PROJECT_ROOT, environment=_project_environment())


def _reload_workspace_before_access() -> None:
    os.chdir("/tmp")
    workspace.reload()


def _execute(
    operation: str,
    target: str,
    *,
    require_gpu: bool,
    validate_final_config: bool = False,
    variables: tuple[str, ...] = (),
) -> dict[str, Any]:
    _reload_workspace_before_access()
    _prepared_payload()
    os.chdir(PROJECT_ROOT)
    if validate_final_config:
        _require_final_training_config()
    try:
        runtime = _runtime_report(operation, require_gpu=require_gpu)
        _make(target, *variables)
    finally:
        workspace.commit()
        print(f"Volume {VOLUME_NAME} sincronizado después de {operation}", flush=True)
    return runtime


def _require_summary(
    relative_path: str,
    expected_status: str,
    *required_fields: str,
) -> dict[str, Any]:
    path = ARTIFACT_PROJECT_ROOT / relative_path
    if not path.is_file():
        raise RuntimeError(f"Falta el resumen requerido: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != expected_status:
        raise RuntimeError(
            f"Estado inesperado en {path}: {payload.get('status')!r} != {expected_status!r}"
        )
    missing = [field for field in required_fields if payload.get(field) is None]
    if missing:
        raise RuntimeError(f"Campos ausentes en {path}: {missing}")
    return payload


def _checkpoint_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(ARTIFACT_PROJECT_ROOT.resolve()):
        raise RuntimeError(f"Checkpoint fuera del proyecto persistente: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _require_final_training_config() -> Path:
    import yaml

    expected = (
        SEGMENTATION_OUTPUT_ROOT / "segmenter/configs/train_yolo26n_seg.final.yaml"
    ).resolve()
    smoke = _require_summary(
        "outputs/leaf_detection/segmenter/smoke_summary.json",
        "passed",
        "final_config",
        "final_config_sha256",
    )
    configured = Path(str(smoke["final_config"])).resolve()
    if configured != expected or not expected.is_file():
        raise RuntimeError(f"Configuración final incorrecta: {configured} != {expected}")
    if _sha256(expected) != smoke["final_config_sha256"]:
        raise RuntimeError("La configuración final cambió después del smoke")
    payload = yaml.safe_load(expected.read_text(encoding="utf-8"))
    batch = payload.get("batch") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("task") != "segment"
        or payload.get("epochs") != 150
        or not isinstance(batch, int)
        or isinstance(batch, bool)
        or batch <= 0
        or payload.get("seed") != 42
        or payload.get("deterministic") is not True
    ):
        raise RuntimeError("La configuración final de 150 épocas no es válida")
    return expected


def _require_confirmation(value: str, operation: str) -> None:
    if value != "true":
        raise RuntimeError(f"{operation} bloqueado: use --confirm true exactamente")


def _prepare_marker_payload(
    root: Path,
    archive: Path,
    manifest: dict[str, Any],
    prepare_result: str,
    jpeg_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ready",
        "prepare_result": prepare_result,
        "prepared_utc": _utc_now(),
        "package": str(archive),
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "package_sha256": PACKAGE_SHA256,
        "package_size_bytes": archive.stat().st_size,
        "package_manifest_sha256": _sha256(
            root / "cloud_training" / "package_manifest.json"
        ),
        "parent_fingerprint": manifest["parent_fingerprint"],
        "payload_file_count": manifest["payload_file_count"],
        "image_recipe_sha256": IMAGE_RECIPE_SHA256,
        "jpeg_validation": jpeg_validation,
    }


def _validate_release_jpegs(root: Path) -> dict[str, Any]:
    """Run the pinned Ultralytics checker against a temporary dataset copy."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    environment["YOLO_CONFIG_DIR"] = "/tmp/doctor_maiz_ultralytics_config"
    code = (
        "import json;"
        "from pathlib import Path;"
        "from src.data.jpeg_normalization import validate_jpegs_before_packaging;"
        f"report=validate_jpegs_before_packaging(Path({str(root)!r})/"
        "'data/leaf_detection/detector_dataset');"
        "print('__JPEG_REPORT__'+json.dumps(report, sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Falló el escaneo JPEG/Ultralytics de prepare: "
            f"stdout={completed.stdout[-2000:]!r}; stderr={completed.stderr[-2000:]!r}"
        )
    lines = [
        line.removeprefix("__JPEG_REPORT__")
        for line in completed.stdout.splitlines()
        if line.startswith("__JPEG_REPORT__")
    ]
    if len(lines) != 1:
        raise RuntimeError("El escaneo JPEG/Ultralytics no produjo un reporte único")
    report = json.loads(lines[0])
    if (
        report.get("passed") is not True
        or report.get("ultralytics_scan", {}).get("mutated_file_count") != 0
    ):
        raise RuntimeError(f"El escaneo JPEG/Ultralytics no quedó limpio: {report}")
    return report


def _remove_prepare_staging() -> None:
    if not PREPARE_STAGING.exists() and not PREPARE_STAGING.is_symlink():
        return
    if PREPARE_STAGING.is_symlink() or not PREPARE_STAGING.is_dir():
        raise RuntimeError(f"Temporal propio inseguro: {PREPARE_STAGING}")
    shutil.rmtree(PREPARE_STAGING)


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    cpu=4.0,
    memory=16384,
    timeout=2 * 3600,
)
def prepare() -> dict[str, Any]:
    """Verify and atomically prepare the frozen release inside the Volume."""
    archive = INCOMING_ROOT / PACKAGE_NAME
    if not archive.is_file():
        raise FileNotFoundError(f"Falta {archive}; súbalo una vez con modal volume put")
    actual_sha256 = _sha256(archive)
    if actual_sha256 != PACKAGE_SHA256:
        raise RuntimeError(f"SHA-256 del paquete inválido: {actual_sha256} != {PACKAGE_SHA256}")
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    if sidecar.is_file() and sidecar.read_text(encoding="utf-8").split()[0] != PACKAGE_SHA256:
        raise RuntimeError(f"Sidecar SHA-256 inconsistente: {sidecar}")

    try:
        _remove_prepare_staging()
        if PROJECT_ROOT.exists():
            if PREPARED_MARKER.is_file():
                payload = _prepared_payload()
                _verify_extracted_release(PROJECT_ROOT)
                result = {**payload, "prepare_result": "prepared_already"}
                workspace.commit()
                if not PREPARED_MARKER.is_file():
                    raise RuntimeError("El marcador desapareció después del commit")
                print(json.dumps(result, indent=2, sort_keys=True), flush=True)
                return result

            try:
                manifest = _verify_extracted_release(PROJECT_ROOT)
            except Exception as exc:
                raise RuntimeError(
                    f"Existe {PROJECT_ROOT} sin marcador y su identidad no es "
                    "verificable; no se sobrescribe"
                ) from exc
            jpeg_validation = _validate_release_jpegs(PROJECT_ROOT)
            recovered = _prepare_marker_payload(
                PROJECT_ROOT,
                archive,
                manifest,
                "prepared_recovered",
                jpeg_validation,
            )
            _write_json(PREPARED_MARKER, recovered)
            workspace.commit()
            if not PREPARED_MARKER.is_file():
                raise RuntimeError("No persistió el marcador recuperado")
            print(json.dumps(recovered, indent=2, sort_keys=True), flush=True)
            return recovered

        PREPARE_STAGING.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            roots = {Path(member.name).parts[0] for member in members}
            unsafe = [
                member.name
                for member in members
                if (
                    not member.isfile()
                    or member.name.startswith("/")
                    or ".." in Path(member.name).parts
                    or member.issym()
                    or member.islnk()
                )
            ]
            if roots != {PACKAGE_ROOT_NAME} or unsafe:
                raise RuntimeError(f"Contenido tar inválido: roots={roots}, unsafe={unsafe[:5]}")
            tar.extractall(PREPARE_STAGING, filter="data")
        extracted = PREPARE_STAGING / PACKAGE_ROOT_NAME
        manifest = _verify_extracted_release(extracted)
        jpeg_validation = _validate_release_jpegs(extracted)
        marker = _prepare_marker_payload(
            extracted,
            archive,
            manifest,
            "prepared",
            jpeg_validation,
        )
        _write_json(extracted / PREPARED_MARKER.name, marker)
        extracted.rename(PROJECT_ROOT)
        PREPARE_STAGING.rmdir()
        workspace.commit()
        if not PREPARED_MARKER.is_file():
            raise RuntimeError("No persistió el marcador de preparación")
        print(json.dumps(marker, indent=2, sort_keys=True), flush=True)
        return marker
    except Exception:
        _remove_prepare_staging()
        workspace.commit()
        raise


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    gpu=REQUESTED_GPU,
    cpu=8.0,
    memory=32768,
    timeout=3600,
)
def preflight() -> dict[str, Any]:
    """Run the dataset, runtime, weights and Segment26 CUDA preflight."""
    _execute(
        "preflight",
        "leaf-segmentation-cloud-preflight",
        require_gpu=True,
    )
    summary = _require_summary(
        "outputs/leaf_detection/cloud_preflight/summary.json",
        "ready_for_smoke_training",
        "dataset_verified",
        "gpu_verified",
        "model_verified",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    gpu=REQUESTED_GPU,
    cpu=8.0,
    memory=32768,
    timeout=2 * 3600,
)
def smoke(confirm: str = "false") -> dict[str, Any]:
    """Run one AutoBatch epoch only after ``--confirm true``."""
    _require_confirmation(confirm, "smoke")
    runtime = _execute(
        "smoke",
        "leaf-segmentation-cloud-smoke",
        require_gpu=True,
        variables=("CONFIRM_SEGMENTATION_SMOKE_TRAINING=1",),
    )
    summary = _require_summary(
        "outputs/leaf_detection/segmenter/smoke_summary.json",
        "passed",
        "save_dir",
        "selected_batch",
        "peak_vram_bytes",
        "duration_seconds",
        "final_config",
        "final_config_sha256",
    )
    weights = Path(str(summary["save_dir"])) / "weights"
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "summary": str(SEGMENTATION_OUTPUT_ROOT / "segmenter/smoke_summary.json"),
        "selected_batch": summary["selected_batch"],
        "peak_vram_bytes": summary["peak_vram_bytes"],
        "duration_seconds": summary["duration_seconds"],
        "final_config": summary["final_config"],
        "final_config_sha256": summary["final_config_sha256"],
        "checkpoints": {
            "best": _checkpoint_record(weights / "best.pt"),
            "last": _checkpoint_record(weights / "last.pt"),
        },
        "runtime": runtime,
    }
    _write_json(
        SEGMENTATION_OUTPUT_ROOT / "segmenter/modal_smoke_manifest.json",
        manifest,
    )
    workspace.commit()
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    gpu=REQUESTED_GPU,
    cpu=8.0,
    memory=32768,
    timeout=24 * 3600,
)
def train(confirm: str = "false") -> dict[str, Any]:
    """Run the frozen 150-epoch config and persist checkpoints in the Volume."""
    _require_confirmation(confirm, "train")
    _execute(
        "train",
        "leaf-segmentation-cloud-train",
        require_gpu=True,
        validate_final_config=True,
        variables=(
            "CONFIRM_SEGMENTATION_TRAINING=1",
            f"CONFIG={SEGMENTATION_OUTPUT_ROOT}/segmenter/configs/"
            "train_yolo26n_seg.final.yaml",
        ),
    )
    summary = _require_summary(
        "outputs/leaf_detection/segmenter/training_summary.json",
        "passed",
        "save_dir",
        "selected_batch",
        "peak_vram_bytes",
        "duration_seconds",
        "checkpoints",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    gpu=REQUESTED_GPU,
    cpu=8.0,
    memory=32768,
    timeout=24 * 3600,
)
def resume(confirm: str = "false") -> dict[str, Any]:
    """Resume only the exact run identified by active_run_manifest.json."""
    _require_confirmation(confirm, "resume")
    _execute(
        "resume",
        "leaf-segmentation-cloud-resume",
        require_gpu=True,
        variables=("CONFIRM_SEGMENTATION_TRAINING=1",),
    )
    manifest = _require_summary(
        "outputs/leaf_detection/segmenter/resume_manifest.json",
        "completed",
        "checkpoint",
        "checkpoint_sha256",
        "run_id",
        "save_dir",
        "checkpoints",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    gpu=REQUESTED_GPU,
    cpu=8.0,
    memory=32768,
    timeout=3 * 3600,
)
def validate() -> dict[str, Any]:
    """Evaluate the exact baseline checkpoint exclusively on retained test."""
    _execute(
        "validate",
        "leaf-segmentation-cloud-validate",
        require_gpu=True,
    )
    summary = _require_summary(
        "outputs/leaf_detection/segmenter_evaluation/test_summary.json",
        "passed",
        "requested_split",
        "evaluated_split",
        "split",
        "image_count",
        "instance_count",
        "checkpoint",
        "checkpoint_sha256",
        "metrics",
        "box_metrics",
        "mask_metrics",
        "save_dir",
    )
    if (
        summary.get("requested_split") != "test"
        or summary.get("evaluated_split") != "test"
        or summary.get("split") != "test"
        or summary.get("image_count") != 173
        or summary.get("instance_count") != 183
        or summary.get("test_fingerprint") != EXPECTED_TEST_FINGERPRINT
        or summary.get("checkpoint_sha256") != EXPECTED_BEST_CHECKPOINT_SHA256
        or summary.get("pilot_used") is not False
        or summary.get("environment_modified") is not False
        or summary.get("environment_before") != summary.get("environment_after")
        or summary.get("faster_coco_eval") != EXPECTED_FASTER_COCO_EVAL
        or Path(str(summary.get("save_dir", ""))).resolve()
        != (SEGMENTATION_OUTPUT_ROOT / "segmenter_evaluation/yolo26n_seg_test").resolve()
    ):
        raise RuntimeError(
            "La evaluación Modal no corresponde exclusivamente al test retenido"
        )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def results() -> None:
    """Print the persistent training-result inventory."""
    _execute(
        "results",
        "leaf-segmentation-cloud-results",
        require_gpu=False,
    )


@app.function(
    image=modal_image,
    volumes=VOLUME_MOUNTS,
    cpu=2.0,
    memory=4096,
    timeout=3600,
)
def checksums() -> None:
    """Write persistent hashes for the exact training artifacts."""
    _execute(
        "checksums",
        "leaf-segmentation-cloud-checksums",
        require_gpu=False,
    )
