#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib.sh"
ULTRALYTICS_VERSION="8.4.104"
OUTPUT_ROOT="${LEAF_SEGMENTATION_OUTPUT:-outputs/leaf_detection}"
CLOUD_DIR="${CLOUD_TRAINING_DIR:-cloud_training}"
BOOTSTRAP_OUT="${OUTPUT_ROOT}/cloud_bootstrap"
mkdir -p "${BOOTSTRAP_OUT}"
exec > >(tee -a "${BOOTSTRAP_OUT}/bootstrap.log") 2>&1

uname -a
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip --version
nvidia-smi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -d .venv-cloud ]] || "${PYTHON_BIN}" -m venv --system-site-packages .venv-cloud
  # shellcheck disable=SC1091
  source .venv-cloud/bin/activate
  PYTHON_BIN=python
fi

"${PYTHON_BIN}" - <<'PY'
import os
import torch
device = int(os.getenv("SEGMENTATION_DEVICE", "0"))
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("torch.cuda.is_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA no disponible: bootstrap bloqueado")
free, total = torch.cuda.mem_get_info(device)
print("gpu", torch.cuda.get_device_name(device))
print("vram_free_bytes", free)
print("vram_total_bytes", total)
PY

"${PYTHON_BIN}" - <<'PY' > "${CLOUD_DIR}/requirements/runtime_constraints.txt"
from importlib import metadata
for name in ("torch", "torchvision"):
    print(f"{name}=={metadata.version(name)}")
PY

"${PYTHON_BIN}" -m pip install --dry-run --report "${BOOTSTRAP_OUT}/pip_dry_run_report.json" \
  --constraint "${CLOUD_DIR}/requirements/runtime_constraints.txt" \
  "ultralytics==${ULTRALYTICS_VERSION}" \
  2>&1 | tee "${BOOTSTRAP_OUT}/pip_dry_run.txt"

"${PYTHON_BIN}" - "${BOOTSTRAP_OUT}/pip_dry_run_report.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
replaced = {
    item.get("metadata", {}).get("name", "").lower()
    for item in report.get("install", [])
} & {"torch", "torchvision"}
if replaced:
    raise SystemExit(f"Dry-run intenta reemplazar dependencias protegidas: {sorted(replaced)}")
PY

"${PYTHON_BIN}" -m pip install \
  --constraint "${CLOUD_DIR}/requirements/runtime_constraints.txt" \
  "ultralytics==${ULTRALYTICS_VERSION}"
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" -m pip freeze > "${BOOTSTRAP_OUT}/pip_freeze.txt"

"${PYTHON_BIN}" - "${ULTRALYTICS_VERSION}" <<'PY' > "${CLOUD_DIR}/runtime_environment.lock"
import platform
import sys
from importlib import metadata
expected = sys.argv[1]
actual = metadata.version("ultralytics")
if actual != expected:
    raise SystemExit(f"Ultralytics inesperado: {actual} != {expected}")
print(f"python={platform.python_version()}")
print(f"torch={metadata.version('torch')}")
print(f"torchvision={metadata.version('torchvision')}")
print(f"ultralytics={actual}")
PY

cat "${CLOUD_DIR}/runtime_environment.lock"
