# Imagen para entrenar en una GPU remota (p.ej. vast.ai). Python 3.11 (requisito del
# proyecto, ver pyproject.toml) + wheels de PyTorch con CUDA 12.6 desde el índice oficial
# de PyTorch — misma línea de CUDA que la plantilla vastai/pytorch:2.6.0-cuda-12.6.3-py312
# usada por scripts/vastai/launch.py. La GPU/drivers los aporta el host
# (nvidia-container-toolkit en local, o directamente el runtime de vast.ai); la imagen no
# necesita el toolkit de CUDA completo. torch/torchvision van con versión exacta (no solo
# el índice) porque sin tope pip hace backtracking entre builds de torch y sus
# dependencias nvidia-cu12, dejando el install colgado minutos buscando combinaciones.
FROM python:3.11-slim

WORKDIR /workspace/corn-leaf-desease-project

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY config ./config
COPY Makefile ./Makefile

# Se crea venv/ (no se instala en el Python global) porque el Makefile del proyecto
# resuelve $(PYTHON) como venv/bin/python en Linux/macOS y venv\Scripts\python en
# Windows: así los mismos targets de `make` funcionan igual en local y en esta imagen.
RUN python -m venv venv \
    && venv/bin/pip install --no-cache-dir --upgrade pip \
    && venv/bin/pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126 \
    && venv/bin/pip install --no-cache-dir -e ".[cloud]"

CMD ["/bin/bash"]
