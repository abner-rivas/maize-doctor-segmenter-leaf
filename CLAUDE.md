# CLAUDE.md - Corn Leaf Disease Project

## Reglas de datos

- **Nunca modificar `raw/`.** Es inmutable, solo fuente original.
- `clean/` es la única fuente de verdad para entrenamiento. Estructura: `clean/<clase>/{lab,real}/`.
- Los CSV de `splits/` son derivados reproducibles (`make splits` / `make splits-baseline`). No editarlos a mano. Viven en `outputs/splits/` (ver más abajo), no bajo `DATASET_ROOT`.

## Arquitectura (`src/`)

- **Único punto de entrada a imagen:** `load_and_normalize_image()` (`src/data/loader.py`).
- **Rutas:** dataset fuente vía `get_dataset_root()`, artefactos generados vía `get_output_root()` (ambas en `src/config.py`) - nunca hardcodear paths ni usar la constante `DATASET_ROOT` directo.
- **Config centralizada:** `config/dataset.yaml` (clases, `target_size`, seed, perfil `baseline`). Nunca hardcodear constantes de dominio.
- **Sin `sys.path.append`.** Paquete editable (`pip install -e .`); los imports `src.*` resuelven directo.
- Convenciones detalladas de carga/rutas/`target_size`/clases minoritarias → skill `corn-data-pipeline`. Sampler de balanceo, utilidades de entrenamiento y versionado de runs → skill `corn-training-internals`.
- Para ubicar símbolos, llamadas o impacto de cambios en `src/`, usa CodeGraph (si está disponible) en vez de grep/lectura manual.

## Pipelines

- **Datos:** `clean/<clase>/{lab,real}/` → `create_splits.py` (valida integridad PIL, deduplica por SHA-256 con escaneo `sorted()` - determinista entre máquinas -, estratifica por `label+environment`) → `outputs/splits/seed_42/` (9 clases) o `outputs/splits/seed_42_baseline/` (`--baseline`, subset de `config/dataset.yaml -> baseline:`).
- **Baselines (funcional, PyTorch):** `CornDataset` → `WeightedRandomSampler` → `DataLoader` → `MODEL_REGISTRY.build(<efficientnet_b0|efficientnet_lite0|mobilenet_v3_large|fastvit_t8|ghostnetv2_100|shufflenet_v2_x1_0>)` vía `train_baselines.py`. Pese al nombre, no es un pipeline sklearn - es DL completo, pensado para comparar arquitecturas rápido y barato. Cada run también escribe `predictions.csv` (predicción + confianza por imagen de test), usado por `explain_report.py` para el análisis de errores.
- **Principal (`train.py`):** comparte toda la infraestructura de datos/modelos con baselines; el loop de entrenamiento está pendiente de implementar.
- **Explicabilidad (post-hoc, no acoplada al entrenamiento):** `explain_lime.py` (reporte visual LIME + Grad-CAM por imagen), `explain_report.py` (fidelidad agregada y análisis de errores, cruzando con `predictions.csv`), `scripts/checks/lime_stability.py` (auditoría manual de estabilidad de LIME). Ver sección "Explicabilidad" más abajo.

## Clases del dataset

Definidas en `config/dataset.yaml -> dataset.classes` (orden canónico para `class_to_idx`). Minoritarias:
`common_rust` (3.9x), `gray_leaf_spot` (7.9x), `nitrogen_deficiency` (16.8x), `phosphorus_deficiency` (14.3x), `potassium_deficiency` (32.9x).
El perfil `baseline` usa por defecto `healthy`, `common_rust`, `fall_armyworm`, `nitrogen_deficiency` (500 img/clase).

## Explicabilidad

Post-hoc, no acoplada al entrenamiento: `explain_lime.py` (reporte visual LIME + Grad-CAM), `explain_report.py` (fidelidad agregada / análisis de errores vía `predictions.csv`), `scripts/checks/lime_stability.py` (auditoría manual). LIME ya no corre automáticamente al entrenar (usar flag `--lime` puntual en `train_baselines.py`).

- Flujo LIME (`make explain-lime`/`explain-report`/`explain-errors`, `lime_stability.py`) → skill `corn-lime-explainability`.
- Grad-CAM (`GRADCAM_TARGET_LAYERS`, requisito al añadir modelos nuevos) → skill `corn-gradcam`.

## Dataset: hosting y descarga

`clean/` (~25k imágenes) vive en Hugging Face Datasets Hub (fuente primaria) con Google Drive de respaldo;
`download_dataset.py --source auto` resuelve cuál usar. `scripts/download_datasets.sh` es un flujo distinto:
ingesta de fuentes crudas nuevas (Kaggle/Mendeley/Roboflow) hacia `raw/`, no toca `clean/`.


## Comandos frecuentes

```bash
make install                          # pip install -e ".[dev,analysis,xai,cloud]"
make download-dataset                 # clean/ (HF Hub, fallback Google Drive)
make splits / make splits-baseline    # regenera splits CSV
make train-baselines [MODELS=<nombre> NO_CAP=1|MAX_PER_CLASS=<n>]
make train                            # loop de entrenamiento pendiente
make explain-lime [MODELS=<nombre>]   # reporte visual LIME+Grad-CAM post-hoc
make explain-report [MODELS=<nombre> SAMPLE_SIZE=<n>]  # fidelidad agregada
make explain-errors [MODELS=<nombre>] # LIME dirigido a falsos positivos/negativos
make summary                          # conteo de imágenes por clase/entorno
make test-loader                      # smoke check del pipeline de carga
make lint / make fmt                  # ruff check / ruff format
```

## Setup local

Ver [LOCAL.md](LOCAL.md) para levantar el proyecto (venv, `.env`, descarga del dataset).
