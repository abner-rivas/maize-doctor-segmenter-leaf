"""Entrenamiento de baselines en GPU de Modal (https://modal.com/docs/guide).

Coexiste con scripts/vastai/. No importa funciones internas del pipeline: orquesta por
subprocess los mismos scripts CLI que corre el Makefile (splits-baseline, train-baselines),
heredando el entorno de la imagen (DATASET_ROOT=/data, OUTPUT_ROOT=/outputs) para que
get_dataset_root()/get_output_root() resuelvan a los Volumes montados.

Uso:
    modal run scripts/modal/train.py::seed_dataset            # 1 vez: dataset -> Volume
    modal run scripts/modal/train.py --models "efficientnet_b0" --epochs 30
Requiere: `pip install -e ".[cloud]"`, `modal setup`, y el secret:
    modal secret create hf HF_TOKEN=hf_xxx
"""

import subprocess
import sys
from pathlib import Path

import modal

REPO_ANCHOR = "/root"  # workdir por defecto de Modal; el código local se monta aquí
HF_DATASET_REPO = "daiv05/corn-leaf-diseases-pests-and-deficiencies"

# Volumes persistentes: dataset (seed una vez) y artefactos (splits/pesos/métricas/LIME).
dataset_vol = modal.Volume.from_name("corn-clean", create_if_missing=True)
outputs_vol = modal.Volume.from_name("corn-outputs", create_if_missing=True)

# Imagen nativa: deps horneadas; src/scripts montados en caliente (última capa, copy=False).
# .env() va ANTES de los add_local_* porque las capas copy=False deben ser las últimas.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.12.1",
        "torchvision==0.27.1",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["cloud", "xai"])
    .env(
        {
            "DATASET_ROOT": "/data",
            "OUTPUT_ROOT": "/outputs",
            "HF_DATASET_REPO": HF_DATASET_REPO,
            # Hilos del indexado de splits. Se fija explícitamente porque os.cpu_count() en el
            # contenedor reporta los cores del HOST, no la CPU asignada al contenedor: sin esto,
            # create_splits lanzaría ~32 hilos a ciegas. 8 ~= 2x los cores pedidos (cpu=4.0):
            # el indexado es I/O-bound contra el Volume, así que algo de sobre-suscripción oculta
            # la latencia de lectura sin depender del burst de CPU.
            "SPLITS_INDEX_WORKERS": "8",
        }
    )
    .add_local_dir("config", f"{REPO_ANCHOR}/config", copy=True)
    .add_local_python_source("src", "scripts")
)

app = modal.App("corn-leaf-baselines", image=image)


@app.function(
    cpu=2.0,  # descarga I/O-bound; 2 cores bastan para descompresión/hash del dataset
    volumes={"/data": dataset_vol},
    secrets=[modal.Secret.from_name("hf")],
    timeout=3600,
)
def seed_dataset() -> None:
    """Descarga el dataset limpio al Volume corn-clean. Idempotente: download_dataset.py
    salta si /data/clean ya tiene contenido."""
    subprocess.run(
        [sys.executable, "scripts/dataset/download_dataset.py"],
        check=True,
        cwd=REPO_ANCHOR,
    )
    dataset_vol.commit()


@app.function(
    gpu="A10",
    # CPU explícita (el default de Modal es 0.125 cores): garantiza cores reales para el
    # indexado paralelo de splits y el DataLoader, sin depender del burst. Alineado con
    # SPLITS_INDEX_WORKERS=8. Facturación: se cobra max(request, uso real).
    cpu=4.0,
    volumes={"/data": dataset_vol, "/outputs": outputs_vol},
    secrets=[modal.Secret.from_name("hf")],
    timeout=6 * 3600,
)
def train_baselines(models: str = "efficientnet_b0", epochs: int = 30) -> None:
    """Genera splits baseline (lazy) y entrena los baselines indicados, persistiendo
    resultados en el Volume corn-outputs."""
    dataset_vol.reload()  # ve el dataset seedeado por seed_dataset
    splits_marker = Path("/outputs/splits/seed_42_baseline/train.csv")
    if not splits_marker.exists():
        subprocess.run(
            [sys.executable, "scripts/pipeline/create_splits.py", "--baseline"],
            check=True,
            cwd=REPO_ANCHOR,
        )
        # Persistir splits aunque el training falle luego: sin este commit, un run
        # interrumpido perdería los splits recién generados y la próxima corrida
        # los regeneraría en vez de reutilizarlos (rompe la idempotencia lazy).
        outputs_vol.commit()
    subprocess.run(
        [
            sys.executable,
            "scripts/pipeline/train_baselines.py",
            "--models",
            *models.split(),
            "--baseline",
            "--epochs",
            str(epochs),
        ],
        check=True,
        cwd=REPO_ANCHOR,
    )
    outputs_vol.commit()


@app.local_entrypoint()
def main(models: str = "efficientnet_b0", epochs: int = 30) -> None:
    """Entrypoint de `modal run`: dispara train_baselines en la GPU remota."""
    train_baselines.remote(models=models, epochs=epochs)
