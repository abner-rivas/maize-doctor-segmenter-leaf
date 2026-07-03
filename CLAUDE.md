# CLAUDE.md - Corn Leaf Disease Project

## Reglas de datos

- **Nunca modificar `raw/`.** Es inmutable, solo fuente original.
- `clean/` es la única fuente de verdad para entrenamiento. Estructura: `clean/<clase>/{lab,real}/`.
- Los CSV de `splits/` son derivados reproducibles (`make splits` / `make splits-baseline`). No editarlos a mano. Viven en `outputs/splits/` (ver más abajo), no bajo `DATASET_ROOT`.

## Arquitectura (`src/`)

- **Único punto de entrada a imagen:** `load_and_normalize_image()` (`src/data/loader.py`) — corrección EXIF + RGB antes de cualquier transform.
- **Rutas al dataset fuente:** siempre vía `get_dataset_root()` (`src/config.py`), que falla con mensaje claro si falta `.env`. No usar la constante `DATASET_ROOT` directo (es `Path | None`). Solo cubre `raw/`/`clean/` (datos fuente).
- **Rutas a artefactos generados:** siempre vía `get_output_root()` (`src/config.py`) → `PROJECT_ROOT/outputs/` (raíz del repo, gitignored). Cubre splits, resultados de entrenamiento (pesos/metrics/LIME), reports de `dataset_summary.py` y EDA — todo lo que el pipeline *produce*, a diferencia de `get_dataset_root()` que es para lo que el pipeline *consume*.
- **Config centralizada:** `config/dataset.yaml` declara clases, `target_size`, seed y el perfil `baseline`. Los módulos lo leen; nunca hardcodear constantes de dominio.
- **Sin `sys.path.append`.** Paquete editable (`pip install -e .`); los imports `src.*` resuelven directo.
- **`MINORITY_CLASSES`** (`src/data/transforms.py`) es un `frozenset` estático que `CornDataset.__getitem__` consulta para aplicar augmentation extendido — independiente del subset/límite usado al generar los splits.
- **`target_size` es `[alto, ancho]`** (convención `(h, w)` de torchvision). El `Resize(224,224)` directo con distorsión de aspecto es intencional y consistente train/eval — no "corregirlo" a Resize+CenterCrop.
- **Balanceo de clases:** lo hace el `WeightedRandomSampler` (+ augmentation de minoritarias); la loss NO se pondera además por frecuencia — sería doble compensación.
- **Utilidades comunes de entrenamiento** (`src/training/common.py`): `resolve_model_names`, `worker_init_fn`, `select_device`, `generate_run_id`, `build_run_dir`, `update_latest_pointer`, `resolve_run_dir` — compartidas por `train.py` y `train_baselines.py`; no duplicarlas en los scripts. `worker_init_fn` deriva de `torch.initial_seed()`: nunca sembrar workers con una semilla fija — los workers renacen cada época y repetirían idéntica la secuencia de augmentation.
- **Resultados versionados por corrida:** cada entrenamiento de un modelo escribe en `outputs/<pipeline>/<modelo>/<run_id>/` (`run_id` = timestamp `YYYYMMDD_HHMMSS`), nunca sobrescribe corridas previas. `outputs/<pipeline>/<modelo>/latest.json` apunta a la corrida más reciente (`resolve_run_dir` la lee cuando no se pasa `--run`).
- Para ubicar símbolos, llamadas o impacto de cambios en `src/`, usa CodeGraph (si está disponible) en vez de grep/lectura manual.

## Pipelines

- **Datos:** `clean/<clase>/{lab,real}/` → `create_splits.py` (valida integridad PIL, deduplica por SHA-256 con escaneo `sorted()` — determinista entre máquinas —, estratifica por `label+environment`) → `outputs/splits/seed_42/` (9 clases) o `outputs/splits/seed_42_baseline/` (`--baseline`, subset de `config/dataset.yaml -> baseline:`).
- **Baselines (funcional, PyTorch):** `CornDataset` → `WeightedRandomSampler` → `DataLoader` → `MODEL_REGISTRY.build(<efficientnet_b0|efficientnet_lite0|mobilenet_v3_large>)` vía `train_baselines.py`. Pese al nombre, no es un pipeline sklearn — es DL completo, pensado para comparar arquitecturas rápido y barato.
- **Principal (`train.py`):** comparte toda la infraestructura de datos/modelos con baselines; el loop de entrenamiento está pendiente de implementar.

## Clases del dataset

Definidas en `config/dataset.yaml -> dataset.classes` (orden canónico para `class_to_idx`). Minoritarias:
`common_rust` (3.9x), `gray_leaf_spot` (7.9x), `nitrogen_deficiency` (16.8x), `phosphorus_deficiency` (14.3x), `potassium_deficiency` (32.9x).
El perfil `baseline` usa por defecto `healthy`, `common_rust`, `fall_armyworm`, `nitrogen_deficiency` (500 img/clase).

## Dataset: hosting y descarga

`clean/` (~25k imágenes) vive en Hugging Face Datasets Hub (fuente primaria) con Google Drive de respaldo;
`download_dataset.py --source auto` resuelve cuál usar. `scripts/download_datasets.sh` es un flujo distinto:
ingesta de fuentes crudas nuevas (Kaggle/Mendeley/Roboflow) hacia `raw/`, no toca `clean/`.

## Entrenamiento en GPU remota (vast.ai)

Ver guía completa en `docs/es/deployment/vast-ai.md`. Resumen: `Dockerfile` (Python 3.11 + PyTorch CUDA,
instalado en `venv/`) + `scripts/vastai/onstart.sh` (provisioning: clona, instala, descarga dataset) +
`scripts/vastai/launch.py` (wrapper sobre la CLI `vastai`: search/create/run/sync/destroy).

## Comandos frecuentes

```bash
make install                          # pip install -e ".[dev,analysis,cloud]"
make download-dataset                 # clean/ (HF Hub, fallback Google Drive)
make splits / make splits-baseline    # regenera splits CSV
make train-baselines [MODELS=<nombre>] / make train-baselines-full
make train                            # loop de entrenamiento pendiente
make summary                          # conteo de imágenes por clase/entorno
make test-loader                      # smoke check del pipeline de carga
make lint / make fmt                  # ruff check / ruff format
```

## Setup local

Ver [LOCAL.md](LOCAL.md) para levantar el proyecto (venv, `.env`, descarga del dataset).
