"""Entrenamiento de baselines en GPU de Modal (https://modal.com/docs/guide).

No importa funciones internas del pipeline: orquesta por
subprocess el mismo script CLI que corre `make train-baselines` (train_baselines.py, que
genera `PROJECT_DATA_ROOT/splits/seed_42_baseline` de forma lazy si falta), heredando el
imagen (DATASET_ROOT=/data, PROJECT_DATA_ROOT=/project-data,
OUTPUT_ROOT=/outputs) para que las tres raíces resuelvan a Volumes separados.

Uso:
    modal run scripts/modal/train.py::seed_dataset            # 1 vez: dataset -> Volume
    modal run scripts/modal/train.py --models "efficientnet_b0" --epochs 30
    modal run scripts/modal/train.py::clean_outputs            # vacía el Volume corn-outputs
Requiere: `pip install -e ".[cloud]"`, `modal setup`, y el secret:
    modal secret create hf HF_TOKEN=hf_xxx
"""

import shutil
import subprocess
import sys
from pathlib import Path

import modal

from scripts.modal._common import (
    DEFAULT_MODELS,
    REPO_ANCHOR,
    dataset_vol,
    image,
    outputs_vol,
    project_data_vol,
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
    # SPLITS_INDEX_WORKERS=24. Facturación: se cobra max(request, uso real).
    cpu=4.0,
    volumes={
        "/data": dataset_vol,
        "/project-data": project_data_vol,
        "/outputs": outputs_vol,
    },
    secrets=[modal.Secret.from_name("hf")],
    # Techo dimensionado para el peor caso `--models all` (7 baselines) x 30 epochs. Con --no-cap
    # el train baseline es ~11.5k imgs (/data/clean: healthy 8744 + common_rust 2256 +
    # fall_armyworm 4857 + nitrogen 523, split 70%), ~8x el perfil capado (~12 min/modelo).
    # Estimado ~1.5 h/modelo -> ~11 h los 7 secuenciales; 14 h dan margen para no morir por
    # timeout. El default son 3 modelos, así que sobra holgura.
    timeout=14 * 3600,
)
def train_baselines(
    models: str = DEFAULT_MODELS,
    epochs: int = 30,
    max_per_class: int = 0,
    no_cap: bool = False,
    regenerate_splits: bool = False,
    batch_size: int = 0,
    image_size: int = 0,
    learning_rate: float = 0.0,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    no_pretrained: bool = False,
    lime: bool = False,
) -> None:
    """Entrena los baselines indicados, persistiendo resultados en el Volume corn-outputs.
    Espeja `make train-baselines`/`train_baselines.py` - misma CLI, mismo comportamiento
    (incluida la generación lazy de `PROJECT_DATA_ROOT/splits/seed_42_baseline` si falta).
    """
    dataset_vol.reload()  # ve el dataset seedeado por seed_dataset

    train_args = [
        sys.executable,
        "scripts/pipeline/train_baselines.py",
        "--models",
        *models.split(),
        "--baseline",
        "--epochs",
        str(epochs),
    ]
    if no_cap:
        train_args.append("--no-cap")
    elif max_per_class:
        train_args += ["--max-per-class", str(max_per_class)]
    if regenerate_splits:
        train_args.append("--regenerate-splits")
    if batch_size:
        train_args += ["--batch-size", str(batch_size)]
    if image_size:
        train_args += ["--image-size", str(image_size)]
    if learning_rate:
        train_args += ["--learning-rate", str(learning_rate)]
    if weight_decay:
        train_args += ["--weight-decay", str(weight_decay)]
    if num_workers:
        train_args += ["--num-workers", str(num_workers)]
    if no_pretrained:
        train_args.append("--no-pretrained")
    if lime:
        train_args.append("--lime")
    subprocess.run(train_args, check=True, cwd=REPO_ANCHOR)
    project_data_vol.commit()
    outputs_vol.commit()


@app.function(volumes={"/outputs": outputs_vol}, timeout=600)
def clean_outputs() -> None:
    """Vacía modelos y reportes; los splits viven en corn-project-data."""
    outputs_root = Path("/outputs")
    for entry in outputs_root.iterdir():
        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    outputs_vol.commit()


@app.local_entrypoint()
def main(
    models: str = DEFAULT_MODELS,
    epochs: int = 30,
    max_per_class: int = 0,
    no_cap: bool = False,
    regenerate_splits: bool = False,
    batch_size: int = 0,
    image_size: int = 0,
    learning_rate: float = 0.0,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    no_pretrained: bool = False,
    lime: bool = False,
) -> None:
    """Entrypoint de `modal run`: dispara train_baselines en la GPU remota."""
    train_baselines.remote(
        models=models,
        epochs=epochs,
        max_per_class=max_per_class,
        no_cap=no_cap,
        regenerate_splits=regenerate_splits,
        batch_size=batch_size,
        image_size=image_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
        no_pretrained=no_pretrained,
        lime=lime,
    )
