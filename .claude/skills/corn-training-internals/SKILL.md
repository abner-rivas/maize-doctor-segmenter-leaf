---
name: corn-training-internals
description: Use when reading or editing src/training/*, scripts/pipeline/train.py, or scripts/pipeline/train_baselines.py — covers class-balancing via WeightedRandomSampler, shared training utilities, and run/output versioning for the corn leaf disease project.
---

# Corn Training Internals

## Balanceo de clases

Lo hace el `WeightedRandomSampler` (+ augmentation de minoritarias, ver skill `corn-data-pipeline`); la loss NO se pondera además por frecuencia — sería doble compensación. El sampler se **desactiva automáticamente** cuando el split está balanceado (`build_weighted_sampler` devuelve `None` si no hay clases minoritarias) y el `DataLoader` usa `shuffle=True`, para no reducir la cobertura por época con `replacement=True`.

## Utilidades comunes (`src/training/common.py`)

`resolve_model_names`, `worker_init_fn`, `select_device`, `generate_run_id`, `build_run_dir`, `update_latest_pointer`, `resolve_run_dir` — compartidas por `train.py` y `train_baselines.py`; no duplicarlas en los scripts.

`worker_init_fn` deriva de `torch.initial_seed()`: **nunca** sembrar workers con una semilla fija — los workers renacen cada época y repetirían idéntica la secuencia de augmentation.

## Versionado de corridas

Cada entrenamiento de un modelo escribe en `outputs/<pipeline>/<modelo>/<run_id>/` (`run_id` = timestamp `YYYYMMDD_HHMMSS`), nunca sobrescribe corridas previas. `outputs/<pipeline>/<modelo>/latest.json` apunta a la corrida más reciente (`resolve_run_dir` la lee cuando no se pasa `--run`).
