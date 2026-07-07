"""Imagen y Volumes de Modal compartidos entre train.py y explain.py.

Factorizado para que ambos módulos usen exactamente la misma imagen (versión de torch,
extras instalados) y los mismos Volumes — divergir entre ellos rompería la reutilización
de checkpoints/splits generados por uno y consumidos por el otro.
"""

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
            "SPLITS_INDEX_WORKERS": "24",
        }
    )
    .add_local_dir("config", f"{REPO_ANCHOR}/config", copy=True)
    .add_local_python_source("src", "scripts")
)
