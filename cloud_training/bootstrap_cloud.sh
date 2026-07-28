#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_ROOT="${LEAF_SEGMENTATION_OUTPUT:-outputs/leaf_detection}"
CLOUD_DIR="${CLOUD_TRAINING_DIR:-cloud_training}"
BOOTSTRAP_OUT="${OUTPUT_ROOT}/cloud_bootstrap"
VENV_DIR="${REPO_ROOT}/.venv-cloud"
REQUIREMENT_FILE="${CLOUD_DIR}/requirements/ultralytics.in"
mkdir -p "${BOOTSTRAP_OUT}"
exec > >(tee -a "${BOOTSTRAP_OUT}/bootstrap.log") 2>&1

select_python() {
  if [[ -n "${PYTHON:-}" ]] && command -v "${PYTHON}" >/dev/null 2>&1; then
    command -v "${PYTHON}"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    echo "No se encontró python3 ni python" >&2
    return 2
  fi
}

HOST_PYTHON="$(select_python)"
[[ -f "${REQUIREMENT_FILE}" ]] || {
  echo "Falta ${REQUIREMENT_FILE}" >&2
  exit 2
}
ULTRALYTICS_SPEC="$(
  while IFS= read -r line; do
    case "${line}" in
      ultralytics==*) printf '%s\n' "${line}" ;;
    esac
  done < "${REQUIREMENT_FILE}"
)"
[[ "${ULTRALYTICS_SPEC}" == ultralytics==* ]] || {
  echo "${REQUIREMENT_FILE} debe contener una única versión exacta de ultralytics" >&2
  exit 2
}
[[ "$(printf '%s\n' "${ULTRALYTICS_SPEC}" | wc -l)" -eq 1 ]] || {
  echo "${REQUIREMENT_FILE} contiene más de una versión de ultralytics" >&2
  exit 2
}
ULTRALYTICS_VERSION="${ULTRALYTICS_SPEC#ultralytics==}"

uname -a
"${HOST_PYTHON}" --version
"${HOST_PYTHON}" -m pip --version

# El proveedor debe exponer una pila CUDA funcional antes de que pip cambie nada.
nvidia-smi
"${HOST_PYTHON}" - <<'PY'
import os

import torch
import torchvision

device = int(os.getenv("SEGMENTATION_DEVICE", "0"))
print("pre_install.torch", torch.__version__)
print("pre_install.torchvision", torchvision.__version__)
print("pre_install.torch.version.cuda", torch.version.cuda)
print("pre_install.torch.cuda.is_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA no disponible antes de instalar: bootstrap bloqueado")
free, total = torch.cuda.mem_get_info(device)
print("pre_install.gpu", torch.cuda.get_device_name(device))
print("pre_install.vram_free_bytes", free)
print("pre_install.vram_total_bytes", total)
PY

if [[ -e "${VENV_DIR}" && ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "${VENV_DIR} existe pero no es un virtualenv utilizable" >&2
  exit 2
fi
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  "${HOST_PYTHON}" - "${VENV_DIR}/pyvenv.cfg" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Falta {path}")
settings = {}
for line in path.read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        settings[key.strip().lower()] = value.strip().lower()
if settings.get("include-system-site-packages") != "true":
    raise SystemExit(
        "El .venv-cloud existente no usa --system-site-packages; "
        "muévalo y vuelva a ejecutar bootstrap"
    )
PY
else
  "${HOST_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi
CLOUD_PYTHON="${VENV_DIR}/bin/python"

# Verifica que el venv realmente herede la pila del proveedor antes de resolver.
"${CLOUD_PYTHON}" - <<'PY'
import torch
import torchvision

print("venv_pre_install.torch", torch.__version__)
print("venv_pre_install.torchvision", torchvision.__version__)
if not torch.cuda.is_available():
    raise SystemExit("El .venv-cloud no puede usar CUDA")
PY

"${CLOUD_PYTHON}" - <<'PY' > "${CLOUD_DIR}/requirements/runtime_constraints.txt"
from importlib import metadata

for name in ("torch", "torchvision"):
    print(f"{name}=={metadata.version(name)}")
PY

"${CLOUD_PYTHON}" -m pip install --dry-run \
  --report "${BOOTSTRAP_OUT}/pip_dry_run_report.json" \
  --constraint "${CLOUD_DIR}/requirements/runtime_constraints.txt" \
  "${ULTRALYTICS_SPEC}" \
  2>&1 | tee "${BOOTSTRAP_OUT}/pip_dry_run.txt"

"${CLOUD_PYTHON}" - "${BOOTSTRAP_OUT}/pip_dry_run_report.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
protected = {"torch", "torchvision"}
replaced = {
    item.get("metadata", {}).get("name", "").lower().replace("_", "-").replace(".", "-")
    for item in report.get("install", [])
} & protected
if replaced:
    raise SystemExit(
        f"Dry-run intenta instalar o reemplazar dependencias protegidas: {sorted(replaced)}"
    )
PY

"${CLOUD_PYTHON}" -m pip install \
  --constraint "${CLOUD_DIR}/requirements/runtime_constraints.txt" \
  "${ULTRALYTICS_SPEC}"
"${CLOUD_PYTHON}" -m pip check
"${CLOUD_PYTHON}" -m pip freeze > "${BOOTSTRAP_OUT}/pip_freeze.txt"

"${CLOUD_PYTHON}" - "${ULTRALYTICS_VERSION}" <<'PY' \
  > "${CLOUD_DIR}/runtime_environment.lock"
import os
import platform
import sys
from importlib import metadata

import torch
import torchvision
import ultralytics

expected = sys.argv[1]
actual = metadata.version("ultralytics")
if actual != expected or ultralytics.__version__ != expected:
    raise SystemExit(f"Ultralytics inesperado: {actual}/{ultralytics.__version__} != {expected}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA dejó de estar disponible después de instalar Ultralytics")
device = int(os.getenv("SEGMENTATION_DEVICE", "0"))
free, total = torch.cuda.mem_get_info(device)
print(f"python={platform.python_version()}")
print(f"python_executable={sys.executable}")
print(f"torch={metadata.version('torch')}")
print(f"torchvision={metadata.version('torchvision')}")
print(f"torchvision_import={torchvision.__version__}")
print(f"ultralytics={actual}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu={torch.cuda.get_device_name(device)}")
print(f"vram_free_bytes={free}")
print(f"vram_total_bytes={total}")
PY

cat "${CLOUD_DIR}/runtime_environment.lock"
