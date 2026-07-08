---
name: corn-data-pipeline
description: Use when reading or editing src/data/*, src/config.py, src/models/input_sizes.py, or config/dataset.yaml — covers image loading, dataset/output root resolution, target_size convention, and minority-class derivation for the corn leaf disease project.
---

# Corn Data Pipeline Conventions

## Carga de imágenes

Único punto de entrada: `load_and_normalize_image()` (`src/data/loader.py`) — corrección EXIF + conversión RGB antes de cualquier transform. No leer imágenes directo con PIL en otro lugar.

## Rutas

- **Dataset fuente** (`raw/`, `clean/`): siempre vía `get_dataset_root()` (`src/config.py`) — falla con mensaje claro si falta `.env`. Nunca usar la constante `DATASET_ROOT` directo (es `Path | None`).
- **Artefactos generados** (splits, pesos/metrics/LIME, reports de `dataset_summary.py`/EDA): siempre vía `get_output_root()` (`src/config.py`) → `PROJECT_ROOT/outputs/` (gitignored). Todo lo que el pipeline *produce*, a diferencia de `get_dataset_root()` (lo que *consume*).

## Config centralizada

- `config/dataset.yaml` declara clases, `target_size`, seed y el perfil `baseline`. Los módulos lo leen; nunca hardcodear constantes de dominio.
- **`target_size` es `[alto, ancho]`** (convención `(h, w)` de torchvision). El `Resize` directo con distorsión de aspecto es intencional y consistente train/eval — no "corregirlo" a Resize+CenterCrop. El tamaño por defecto (224) puede sobrescribirse por modelo vía `MODEL_INPUT_SIZES` (`src/models/input_sizes.py`; p. ej. `efficientnet_b4`→380, `fastvit_t8`→256); `train_baselines.py` auto-escala el batch a la baja para resoluciones mayores.

## Clases minoritarias

`CornDataset` las **deriva de la distribución real del split** en `compute_minority_classes` (`src/data/dataset.py`) — una clase es minoritaria si `max_count/count > augmentation.minority_ratio_threshold` (`config/dataset.yaml`, default 4.0). No hay `frozenset` estático: en un split balanceado (p. ej. el baseline de 4 clases) el conjunto queda vacío y nadie recibe augmentation asimétrico; en el dataset completo de 9 clases se recomputa.

## Packaging

Sin `sys.path.append`. Paquete editable (`pip install -e .`); los imports `src.*` resuelven directo.

Para ubicar símbolos, llamadas o impacto de cambios en `src/`, usa CodeGraph en vez de grep/lectura manual.
