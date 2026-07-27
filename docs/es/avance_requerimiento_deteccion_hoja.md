# Estado del requerimiento de detección y aislamiento de hoja

Este documento consolida el avance técnico del requerimiento DoctorMaiz. Las
ejecuciones realizadas son un diagnóstico de inferencia con checkpoints
históricos y una auditoría EDA de fuentes externas; ninguna corresponde a
entrenamiento nuevo ni a un baseline oficial.

La cronología completa está en
[Historia del aislamiento de hojas](leaf-detection/history.md) y las decisiones
formales en [`docs/es/decisions/`](decisions/adr-project-data-root-and-output-root.md).

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

## Preparación del dataset del detector

Se preparó una selección reproducible para un futuro detector Ultralytics
YOLO26n:

- 350 imágenes nuevas tomadas sólo de `train.csv`: 280 reales y 70 de
  laboratorio;
- 75 imágenes nuevas tomadas sólo de `val.csv`: 60 reales y 15 de laboratorio;
- piloto reservado como test: 99 `annotated` y `image_0021` documentada como
  `ambiguous`;
- semilla 42;
- cero cruces por ruta, nombre o SHA-256.

Train y val continúan `pending`: todavía no existen anotaciones reales para
esos lotes. Los paquetes CVAT no incluyen etiquetas inventadas. La nueva regla
del detector exige anotar todas las hojas visibles y permite varias cajas por
fotografía, a diferencia del piloto anterior, que marcaba sólo la hoja
principal.

Las 99 cajas del test conservan la regla histórica de hoja principal. Antes de
una evaluación oficial del detector deberán revisarse con la nueva regla
multihoja; de lo contrario, hojas correctamente detectadas pero no anotadas
podrían contabilizarse como falsos positivos.

Ultralytics no está instalado en el shell auditado. `ultralytics==8.4.104` se
registró únicamente como candidata compatible con los rangos declarados del
proyecto; no se instaló, no se descargó `yolo26n.pt` y no se ejecutó
entrenamiento. Detalles y comando reproducible:
[Dataset inicial del detector YOLO26n](leaf-detection/yolo26-detector-dataset.md).

## Auditoría de fuentes externas de segmentación

Se auditaron dos fuentes YOLO con sus respaldos COCO sin modificar los
originales:

- `corn_leaf_diseases_classification`: 1,003 imágenes, 14,415 líneas,
  14,395 polígonos válidos y 20 inválidas;
- `corn`: 157 imágenes, 204 polígonos válidos y un TXT vacío.

La clase `leaf` representa la hoja completa en ambas fuentes. En la primera,
`gray_leaf_spot` y `northern_leaf_blight` representan lesiones y deben
excluirse. Once líneas inválidas son bbox YOLO mezclados en el export de
segmentación; las otras nueve corresponden a ocho autointersecciones y un
vértice repetido. COCO contiene equivalentes válidos para los 11 bbox. No se
encontraron duplicados internos, entre fuentes ni contra las 100 imágenes del
piloto.

Ambas fuentes quedaron como `accepted_with_filtering`: reúnen 1,000 y 156
imágenes candidatas con hoja válida, respectivamente. Al terminar el EDA aún
requerían revisión visual, reglas de filtrado, remapeo trazable a
`0 = maize_leaf` y una consolidación derivada separada; esa consolidación se
completó en la fase siguiente. Detalles:
[Auditoría de datasets externos de segmentación](leaf-detection/external-segmentation-datasets-eda.md).

## Consolidación controlada del dataset segmentado

La decisión del EDA se materializó y luego se reconstruyó desde las fuentes sin
modificarlas:

- 1 160 imágenes consideradas;
- 1 155 imágenes definitivas;
- 1 224 polígonos definitivos, todos con `0 = maize_leaf`;
- 13 392 anotaciones de lesión excluidas;
- la recuperación COCO extremadamente pequeña excluida y enviada a
  reanotación;
- cero duplicados exactos eliminados y cero cruces contra el piloto;
- 1 094 grupos de variante original, 39 con múltiples variantes Roboflow;
- 36 filas de revisión procesadas y 35 casos humanos únicos.

El pool vive en `data/leaf_detection/detector_dataset/all/`. Sus 1 155 imágenes
tienen correspondencia 1:1 con 1 155 TXT no vacíos; todas las coordenadas son
finitas, están en `[0,1]` y forman polígonos simples de al menos tres vértices.
La hoja autointersectada, la recuperación COCO de área extremadamente pequeña
y una imagen adicional están fuera y registradas en
`reannotation_queue.csv`.

Los resultados, flujos y validaciones están en
`outputs/leaf_detection/detector_dataset_consolidation/`. No se crearon splits,
no se instaló Ultralytics, no se descargaron pesos y no se entrenó ningún
segmentador.

### Estado del gate manual

Las 36 filas de revisión están completas. Tras agrupar el caso repetido son 35
casos únicos: 16 `approved`, 16 `exclude` y 3 `needs_reannotation`, sin
contradicciones.

`data/leaf_detection/detector_dataset/manifests/dataset_lock.json` tiene
`status=ready_for_split_generation`. La reconstrucción doble desde las fuentes
produjo el mismo fingerprint
`c087af60c2bad1c133c4ea8b14cee945405bfe4976aa80c4faf089d7a4b9e38c`.
No se generaron splits.

### Corrección de previews de revisión

El renderer anterior asignaba literalmente `polygons=[]` a cada fila del
manifiesto general, por lo que mostraba la imagen con `instances=0` aunque el
TXT original contuviera geometría. Se sustituyó por resolución por caso:
YOLO original, COCO equivalente, recuperación registrada y consolidado como
último respaldo.

Se regeneraron 35 previews individuales. Treinta y tres usan YOLO original,
una usa la anotación COCO recuperada y una corresponde al TXT realmente vacío,
mostrado como `NO GEOMETRY AVAILABLE`. La recuperación de área
`4.2767428e-7` tiene zoom y 26 píxeles de máscara rasterizada; la hoja
autointersectada conserva su polígono rojo y el cruce visible. El reporte
`review_preview_validation.json` quedó `ready_for_human_review`, sin errores de
render ni geometrías conocidas con cero instancias.

No se alteraron las decisiones humanas ni el pool provisional. Por ello el
gate del dataset continúa bloqueado y no se habilitan splits ni entrenamiento.

## Fases

| Fase | Nombre | Estado |
|---:|---|---|
| 1 | Auditoría de anotaciones históricas | Completada |
| 2 | Procesamiento ROI y letterbox | Completada |
| 3 | Herramientas del piloto | Completada |
| 3.5 | Auditoría del dataset de clasificación | Completada |
| 4 | Validación y preparación remota | Completada |
| 5 | Creación del piloto real | Completada |
| 6 | Anotación manual en CVAT | Completada |
| 6.5 | Importación de cajas rotadas | Completada |
| 7 | Manifiesto ROI y previews | Completada |
| 8 | Diagnóstico full vs. ROI manual | Completada |
| 8.5 | Búsqueda de fuentes de segmentación | Completada |
| 9 | EDA de datasets externos de segmentación | Completada |
| 9.5 | Consolidación y limpieza del dataset segmentado | Completada |
| 10 | División train/val/test del segmentador | Pendiente |
| 11 | Entrenamiento del segmentador | Pendiente |
| 12 | Evaluación contra piloto retenido | Pendiente |
| 13 | Generación de máscaras para splits | Pendiente |
| 14 | Entrenamiento baseline_segmented | Pendiente |
| 15 | Comparación contra baseline_full | Pendiente |
| 16 | Integración en CornDataset y predict.py | Pendiente |
| 17 | LIME, Grad-CAM y evaluación final | Pendiente |

## Hipótesis y próximos pasos

Permanecen como hipótesis no confirmadas: cambio de distribución, pérdida de
contexto, margen insuficiente, padding negro no visto, cambio de escala de
síntomas, sensibilidad por arquitectura, necesidad de reentrenar con ROI y
necesidad de conservar más extensión de hoja para deficiencias nutricionales.

El paso inmediato es completar las 34 revisiones manuales, resolver los dos
casos visuales obligatorios y aprobar los previews. Después se crearán splits
propios del segmentador agrupando por
fuente, hash, original previo a Roboflow y `roboflow_variant_group`. La
arquitectura, licencia, versión, GPU y exportabilidad se confirmarán antes de
entrenar.

El segmentador se evaluará contra el piloto mediante precision, recall, mAP,
IoU, Dice, fallbacks y errores de selección. Sólo entonces se generarán
`baseline_bbox_roi` y `baseline_masked_roi`, y cada clasificador se entrenará
con la representación que recibirá en producción. La comparación final incluirá
`baseline_full`, métricas por clase, LIME, Grad-CAM, tamaño y latencia. No se
propone anotar manualmente las 10 020 imágenes ni usar el piloto para
entrenamiento.

## Auditoría del repositorio

La segunda revisión inventarió datos, outputs, recursos públicos, documentación,
notebooks, scripts, código y configuración. No movió ni eliminó evidencia.
Identificó una copia exacta de `dataset_audit_final`, una ruta de preflight
deprecada, copias intencionales del piloto y bytecode descartable. Los reportes
están en `outputs/repository_audit/`.

## Configuración activa

```yaml
processing_profile: baseline_full

leaf_detection:
  enabled: false
```

Esta configuración no se cambió durante el diagnóstico ni durante su
documentación.
