---
name: corn-lime-explainability
description: Use when running or editing scripts/pipeline/explain_lime.py, scripts/pipeline/explain_report.py, or scripts/checks/lime_stability.py, or when asked about LIME visual reports, fidelity, or error analysis for the corn leaf disease project.
---

# Corn LIME Explainability

## Flujo

LIME ya no corre automáticamente al entrenar. `train_baselines.py` mantiene el flag `--lime` (útil para encadenarlo puntualmente), pero los targets `make train-baselines`/`train-baselines-full` ya no lo pasan por defecto — entrenamiento y explicabilidad son pasos separados.

## `make explain-lime` (`scripts/pipeline/explain_lime.py`)

Reporte visual por imagen (`<run_dir>/lime_visual/`), muestreo balanceado chico (`lime.images_per_class` en `config/dataset.yaml`) o `--image` puntual. Persiste, junto al PNG, un `.json` (predicción + pesos por superpíxel) y un `.npy` (mapa de segmentos) para reanálisis sin re-ejecutar LIME.

## `make explain-report` / `make explain-errors` (`scripts/pipeline/explain_report.py`)

Fidelidad agregada sobre una muestra amplia (`lime.report_sample_size`) o, con `--errors-only`, explica específicamente las filas de `predictions.csv` donde `label != pred_label`. Requiere que el run ya tenga `predictions.csv` (lo genera `train_baselines.py`).

## `scripts/checks/lime_stability.py`

Diagnóstico manual (sin target Make) para auditar cuán estable es una explicación LIME sobre una imagen puntual corriendo varias seeds y comparando IoU/correlación.
