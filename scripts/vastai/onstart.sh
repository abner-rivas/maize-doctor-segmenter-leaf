#!/usr/bin/env bash
# Se ejecuta dentro de la instancia vast.ai al arrancar (pasado via `vastai create instance --onstart`).
# Clona el repo, instala el proyecto, arma el .env remoto y descarga el dataset limpio.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/daiv05/corn-leaf-desease-project.git}"
REPO_BRANCH="${REPO_BRANCH:-master}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/corn-leaf-desease-project}"
REMOTE_DATASET_ROOT="${REMOTE_DATASET_ROOT:-/workspace/data}"

echo "[onstart] Clonando $REPO_URL ($REPO_BRANCH) en $WORKSPACE_DIR"
if [[ -d "$WORKSPACE_DIR/.git" ]]; then
  git -C "$WORKSPACE_DIR" pull
else
  git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKSPACE_DIR"
fi
cd "$WORKSPACE_DIR"

# Se usa venv/bin/python (no el Python global) porque es lo que el Makefile del proyecto
# espera en Linux/macOS ($(PYTHON) := venv/bin/python) — así los mismos targets `make`
# funcionan igual en local y en la instancia. Si la imagen ya trae venv/ (p.ej. una imagen
# propia construida con el Dockerfile del repo) se reutiliza tal cual.
#
# El `python3` plano no sirve: las plantillas vast.ai basadas en Ubuntu 22.04 (jammy)
# traen Python 3.10 como `python3` del sistema y agregan 3.11/3.12 vía deadsnakes bajo un
# binario con sufijo de versión (`python3.12`, etc.). Si `python3` resuelve al 3.10 del
# sistema, pip intenta resolver `requires-python = ">=3.11"` de todos modos sin fallar de
# inmediato, y termina en un backtracking severo (re-descargando wheels de CUDA de cientos
# de MB una y otra vez) en vez de dar un error claro. Por eso se busca explícitamente un
# intérprete >=3.11 en vez de asumir `python3`.
if [[ ! -x venv/bin/python ]]; then
  PYTHON_BIN=""
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      ver=$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
      if [[ "${ver%%.*}" -eq 3 && "${ver#*.}" -ge 11 ]]; then
        PYTHON_BIN="$candidate"
        break
      fi
    fi
  done
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "[onstart] ERROR: no se encontro un interprete Python >=3.11 (probado: python3.13, python3.12, python3.11, python3)." >&2
    exit 1
  fi
  echo "[onstart] Creando venv/ con $PYTHON_BIN ($("$PYTHON_BIN" --version))"
  "$PYTHON_BIN" -m venv venv
fi

venv_ver=$(venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if [[ "${venv_ver%%.*}" -ne 3 || "${venv_ver#*.}" -lt 11 ]]; then
  echo "[onstart] ERROR: venv/ existente usa Python $venv_ver, se requiere >=3.11. Borra venv/ y vuelve a correr este script." >&2
  exit 1
fi

echo "[onstart] Instalando PyTorch (CUDA 12.6, versión fija para evitar backtracking de pip)"
venv/bin/pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126

echo "[onstart] Instalando el proyecto (pip install -e .[cloud])"
venv/bin/pip install --no-cache-dir -e ".[cloud]"

echo "[onstart] Escribiendo .env remoto"
mkdir -p "$REMOTE_DATASET_ROOT"
cat > .env <<EOF
DATASET_ROOT=$REMOTE_DATASET_ROOT
HF_DATASET_REPO=${HF_DATASET_REPO:-}
HF_TOKEN=${HF_TOKEN:-}
GDRIVE_DATASET_ID=${GDRIVE_DATASET_ID:-}
EOF

echo "[onstart] Descargando dataset limpio"
venv/bin/python scripts/dataset/download_dataset.py

echo "[onstart] Listo. Conéctate por ssh y corre, por ejemplo:"
echo "  make splits-baseline && make train-baselines"
