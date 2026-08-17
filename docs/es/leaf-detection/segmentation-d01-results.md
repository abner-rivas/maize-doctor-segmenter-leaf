# Resultados de D-01 y validación sobre hojas enfermas

Estado verificado el 2026-08-17. Este documento registra la primera ablación de
mejora del segmentador, la identidad exacta del checkpoint descargado y su prueba
con el flujo de inferencia que consume el clasificador.

## Alcance

`D-01` mantiene la arquitectura `yolo26n-seg`, los splits congelados y la semilla
42 del baseline. El único cambio experimental es `mosaic=0.0`. El objetivo es
mejorar la máscara de la hoja sin usar `test` ni el piloto externo para elegir
hiperparámetros.

La evaluación sobre el dataset de enfermedades mide exclusivamente segmentación:
comprueba cuánto tejido de la hoja se conserva antes de clasificar. No predice ni
evalúa la enfermedad presente.

## Identidad reproducible

| Artefacto | Identidad |
|---|---|
| Dataset congelado | lock `7a4a5c08`; manifiesto evaluado SHA-256 `954d601ef7f4145c14f8b29ed85e6959e8d30d59f405f7d016458c7eea1034f0` |
| Paquete Modal | `doctor_maiz_leaf_segmentation_cloud_v7-segmentation-improvements-7a4a5c08-seed42.tar.gz` |
| SHA-256 del paquete | `a90f3f3089f3e628ae3212aec62fedc734358be813e3cebbfcab0495f69655b6` |
| Perfil | `cloud_training/configs/experiments/d01_mosaic0_seed42.yaml` |
| SHA-256 de configuración efectiva | `5238e44502e97313b9bf93b0beb222cfb7ae159dc239fad76987dfe684e2d0fd` |
| Checkpoint ganador | `outputs/leaf_detection/segmenter/d01_mosaic0_seed42/weights/best.pt` |
| SHA-256 de `best.pt` | `a2bf4f201ca4f5e32c349cdc66d7ac39a6b012a330b182149401e533b2ecb8ab` |
| SHA-256 de `last.pt` | `36c31bca3e97d015ab40e1fc01386baad568a53b10f1eff1c7ec7819d4229842` |

El hash local de `best.pt` coincide con el registrado por Modal. El resumen
estructurado está en
`outputs/leaf_detection/segmenter/experiment_summaries/d01_mosaic0_seed42.json`.

## Entrenamiento en Modal

La corrida se ejecutó en una GPU A10 entre
`2026-08-17T18:04:41Z` y `2026-08-17T18:29:16Z`:

- inicialización con pesos preentrenados `yolo26n-seg`;
- `imgsz=640`, batch 26 y semilla 42;
- máximo de 150 épocas y paciencia 30;
- parada temprana después de 145 épocas;
- mejor época: 115;
- duración: 1 475.71 segundos, aproximadamente 24 minutos 36 segundos;
- VRAM pico registrada: 8 610 202 112 bytes, aproximadamente 8.02 GiB;
- estado `passed`, sin errores;
- `test_included=false` y `pilot_included=false`.

### Métricas del checkpoint ganador en `val`

| Métrica | Resultado |
|---|---:|
| Mask precision | 0.99424 |
| Mask recall | 0.94309 |
| Mask mAP50 | 0.97326 |
| Mask mAP50-95 | 0.94404 |
| Box precision | 1.00000 |
| Box recall | 0.94855 |
| Box mAP50 | 0.97979 |
| Box mAP50-95 | 0.94343 |

En las curvas por época, el mejor Mask mAP50-95 pasó de `0.93806` en el
baseline de semilla 42 a `0.94399` en D-01: una mejora de `0.00593`, equivalente
a 0.59 puntos porcentuales. D-01 es el candidato actual, pero esta comparación
usa `val` y todavía no demuestra estabilidad entre semillas.

## Prueba con el dataset de enfermedades

Se seleccionaron las 150 imágenes de `val` cuya fuente es
`corn_leaf_diseases_classification`. La prueba usó el wrapper real
`UltralyticsLeafSegmenter` y el flujo `SegmentedLeafProcessor`, no el comando
batch genérico de Ultralytics.

Configuración efectiva de inferencia:

- CPU local;
- `imgsz=640`;
- máscaras `retina_masks` a resolución original;
- umbral de propuestas `0.20`;
- umbral de selección `0.50`;
- NMS IoU `0.70` y máximo 20 propuestas;
- área mínima de máscara `0.01`;
- perfil de salida `mask_black`;
- quality gate definido en `config/segmentation.yaml`.

### Resultados end-to-end

| Métrica | Resultado |
|---|---:|
| Imágenes | 150 |
| `reliable` | 150 |
| `uncertain` | 0 |
| `failed` | 0 |
| Fallbacks | 0 |
| IoU medio | 0.98122 |
| Dice medio | 0.99046 |
| Recall medio de píxel de hoja | 0.99375 |
| Precisión media de píxel de hoja | 0.98731 |
| Subsegmentación media | 0.00625 |
| Sobresegmentación media | 0.01288 |
| Tejido recortado medio | 0.62% |
| Peor tejido recortado | 11.41% |
| Duración CPU | 131.85 segundos |

Los seis casos con menor IoU se revisaron visualmente. La selección de hoja fue
coherente en todos. El peor caso es una imagen horizontal de 900×600 con fondo
complejo: recall de píxel `0.88589`, IoU `0.87546` y confianza `0.90661`. Es el
principal caso que debe vigilarse en nuevos datos difíciles.

La evidencia queda en:

- `outputs/leaf_detection/segmenter/evaluations/d01_mosaic0_seed42_disease_val_pipeline/summary.json`;
- `outputs/leaf_detection/segmenter/evaluations/d01_mosaic0_seed42_disease_val_pipeline/per_image_metrics.csv`;
- `outputs/leaf_detection/segmenter/evaluations/d01_mosaic0_seed42_disease_val_pipeline/worst_cases_comparison.jpg`;
- `outputs/leaf_detection/segmenter/evaluations/d01_mosaic0_seed42_disease_val_pipeline/review/`.

## Outputs canónicos y limpieza

Se conservan como evidencia actual:

```text
outputs/leaf_detection/segmenter/
├── d01_mosaic0_seed42/
│   ├── weights/best.pt
│   ├── weights/last.pt
│   ├── results.csv
│   └── doctor_maiz_effective_config.yaml
├── experiment_summaries/
│   └── d01_mosaic0_seed42.json
└── evaluations/
    └── d01_mosaic0_seed42_disease_val_pipeline/
        ├── summary.json
        ├── per_image_metrics.csv
        ├── worst_cases_comparison.jpg
        └── review/
```

Se eliminaron las pruebas intermedias
`evaluations/d01_mosaic0_seed42_val/` y
`evaluations/d01_mosaic0_seed42_val_conf020/`. La primera medía el modo batch de
validación y la segunda era un diagnóstico con `predict`; ninguna reproducía a
la vez el wrapper, la selección y el quality gate de producción. Son
reproducibles desde el checkpoint y el dataset congelado, pero no forman parte
del resultado oficial.

También se eliminaron:

- `outputs/leaf_detection/predictions/`, con 242 archivos y 149 MB de pruebas
  antiguas del clasificador de enfermedades, ajenas al alcance actual;
- el paquete cloud v6 `segmenter-only` y su checksum, aproximadamente 2.0 GB,
  porque fueron reemplazados por la release v7 usada para D-01;
- `outputs/leaf_detection/pilot/`, con 100 previews y dos reportes ROI de julio,
  porque no participó en el entrenamiento ni en la evaluación D-01 actual;
- `external_sources_eda/`, `detector_dataset_consolidation/` y
  `detector_dataset_splits/`, con 97 MB de reportes y previews históricos que no
  fueron usados por las pruebas D-01.

Esas copias locales no tienen recuperación directa. Las predicciones pueden
regenerarse en el proyecto del clasificador y el paquete v6 puede reconstruirse
desde su versión de código y dataset si fuera necesario.

No se eliminaron el dataset ni sus manifiestos, el preflight, la calibración, el
paquete v7 o el backup del baseline. El piloto retenido permanece íntegro bajo
`data/leaf_detection/pilot/`; sólo se eliminaron resultados regenerables bajo
`outputs/`.

## Interpretación y siguientes gates

D-01 supera el baseline en `val` y funciona correctamente sobre hojas enfermas
con el pipeline vigente. Aun así, no debe promoverse como modelo final sólo con
esta evidencia:

1. repetir la configuración ganadora con semillas 7 y 1337;
2. reportar media y desviación estándar contra el baseline multisemilla;
3. resolver el contrato de 183 anotaciones raw frente a 182 instancias efectivas;
4. ejecutar `test` una sola vez con configuración y checkpoint congelados;
5. usar después el piloto externo únicamente como auditoría cualitativa.
