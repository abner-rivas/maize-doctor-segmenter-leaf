# Estado del requerimiento de detección y aislamiento de hoja

Este documento consolida el avance técnico del requerimiento DoctorMaiz. La
prueba ejecutada es un **diagnóstico de inferencia** con checkpoints históricos;
no corresponde a entrenamiento nuevo ni a un baseline oficial.

## Estado de datos y organización

El dataset fuente validado se mantiene fuera del repositorio en
`/home/desarrolloab/Documentos/ML/maize_dataset/data` y se resuelve mediante
`DATASET_ROOT`. Contiene 31 622 imágenes soportadas: 28 071 reales, 3 551 de
laboratorio y 9 clases, sin errores críticos.

La separación vigente es:

- `data/`: splits, imágenes piloto, anotaciones y manifiestos;
- `outputs/`: modelos, métricas, auditorías, previews, validaciones y
  diagnósticos;
- `scripts/`: herramientas ejecutables;
- `src/`: código reutilizable;
- `docs/`: documentación;
- `public/`: recursos publicados.

`PROJECT_DATA_ROOT` permite reubicar `data/`, `OUTPUT_ROOT` reubica resultados y
`DATASET_ROOT` apunta al dataset externo.

## Splits oficiales

`data/splits/seed_42_baseline/` contiene 10 020 filas en 9 clases, semilla 42,
sin fugas, duplicados cruzados ni errores:

| Split | Filas | SHA-256 |
|---|---:|---|
| train | 7 014 | `c96236e1a754213cdb26ad3947b5f0032baf6b18f1818812738333d16a7784dc` |
| val | 1 503 | `45c3a9ededa025ae503a723be842284f464850b58c30ac2ed4d94f9bbd7a5a5a` |
| test | 1 503 | `911e187aa462b237ca4346ae5f4f305e0afcee391f5f624a374bdefd694adb4e` |

## Piloto, CVAT y manifiesto ROI

El piloto en `data/leaf_detection/pilot/` contiene 100 imágenes reales,
seleccionadas de manera balanceada con semilla 42 y sin duplicados. CVAT usa una
sola clase, `maize_leaf`, y una caja primaria por imagen.

La fuente oficial es
`data/leaf_detection/pilot/annotations/cvat/annotations.xml`: 100 imágenes, 100
cajas, ninguna imagen sin caja y ninguna con múltiples cajas. El exportador YOLO
materializó 48 cajas sin rotación y omitió 52 rotadas. El importador
`--format cvat_xml` recuperó las 52 calculando su envolvente alineada a ejes; 36
cajas requirieron clipping.

El manifiesto final
`data/leaf_detection/pilot/manifests/roi_manifest.csv` tiene 100 filas
estructuralmente válidas: 99 `annotated`, 1 `ambiguous`, 0 `pending` y 0
`rejected`. `image_0021` permanece ambigua porque su área `0.092799` es inferior
al mínimo `0.15`; no se expandió artificialmente. Se conservaron 100 previews,
incluida su vista diagnóstica histórica.

## Diagnóstico full frente a ROI manual

La metodología, evidencia y análisis detallado están en
[Diagnóstico de imagen completa frente a ROI manual](preprocessed/manual-roi-diagnostic.md).
Se usaron únicamente estos checkpoints históricos:

- `outputs/baselines/efficientnet_b0/20260709_040040/best.pth`;
- `outputs/baselines/shufflenet_v2_x1_0/20260709_042946/best.pth`;
- `outputs/baselines/efficientnet_lite0/20260709_045817/best.pth`.

Cada modelo procesó 100 filas, incluyó 99 en métricas, excluyó `image_0021`, usó
0 fallbacks, no entrenó y no modificó checkpoints. No se usó
`outputs/aborted_runs`.

| Modelo | Full Accuracy | ROI Accuracy | Δ Accuracy | Full Macro-F1 | ROI Macro-F1 | Δ Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 0.8889 | 0.8586 | -0.0303 | 0.8827 | 0.8561 | -0.0266 |
| ShuffleNetV2-x1.0 | 0.9091 | 0.7576 | -0.1515 | 0.9064 | 0.7582 | -0.1482 |
| EfficientNet-Lite0 | 0.9091 | 0.6162 | -0.2929 | 0.9052 | 0.6101 | -0.2951 |

La reducción no demuestra que un clasificador entrenado desde el principio con
ROI vaya a rendir peor. Los checkpoints aprendieron con imágenes completas y
recibieron recorte, otra escala, letterbox, padding y menos contexto sólo durante
inferencia, provocando un cambio de distribución. Por tanto, no se debe activar
ROI en inferencia sobre estos modelos históricos.

## Fases

| Fase | Nombre | Estado |
|---:|---|---|
| 1 | Auditoría de anotaciones | Completada |
| 2 | Procesamiento ROI y letterbox | Completada |
| 3 | Herramientas del piloto | Completada |
| 3.5 | Auditoría del dataset | Completada |
| 4 | Validación y preparación remota | Completada |
| 5 | Creación del piloto real | Completada |
| 6 | Anotación manual en CVAT | Completada |
| 6.5 | Importación de cajas rotadas | Completada |
| 7 | Manifiesto ROI y previews | Completada |
| 8 | Diagnóstico full vs. ROI manual | Completada |
| 8.5 | Ablación del preprocesamiento ROI | Pendiente |
| 9 | Ampliación de anotaciones | Pendiente |
| 10 | Entrenamiento del detector | Pendiente |
| 11 | Generación de ROI para splits | Pendiente |
| 12 | Entrenamiento baseline_roi | Pendiente |
| 13 | Integración con CornDataset | Pendiente |
| 14 | Integración con predict.py | Pendiente |
| 15 | LIME, Grad-CAM y evaluación final | Pendiente |

## Hipótesis y próximos pasos

Permanecen como hipótesis no confirmadas: cambio de distribución, pérdida de
contexto, margen insuficiente, padding negro no visto, cambio de escala de
síntomas, sensibilidad por arquitectura, necesidad de reentrenar con ROI y
necesidad de conservar más extensión de hoja para deficiencias nutricionales.

Primero debe realizarse la ablación de imagen completa/ROI, resize/letterbox y
padding negro/neutro. Después, ampliar a 300–500 anotaciones, entrenar el
detector, generar y validar ROI de train/val/test, congelar sus manifiestos,
entrenar `baseline_roi` con el mismo pipeline y compararlo con `baseline_full`.
No se propone anotar manualmente las 10 020 imágenes.

## Configuración activa

```yaml
processing_profile: baseline_full

leaf_detection:
  enabled: false
```

Esta configuración no se cambió durante el diagnóstico ni durante su
documentación.
