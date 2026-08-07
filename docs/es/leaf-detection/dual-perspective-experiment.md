# Experimento de doble perspectiva

- Fecha: 2026-08-07
- Estado: soporte implementado y smoke controlado completado
- Naturaleza: diagnóstico experimental opt-in, sin fusión y sin entrenamiento
- Clasificador del smoke: `efficientnet_b0`
- Segmentador: `yolo26n-seg`, Ultralytics 8.4.104

La funcionalidad clasifica primero la imagen completa y después intenta construir
una segunda vista independiente a partir de una hoja segmentada. Cada vista usa
el mismo transform, checkpoint, logits y `softmax` del clasificador. No se
promedian confidences, no se combinan logits y no se decide qué vista es correcta.

Codebase Memory MCP no estuvo disponible en esta sesión. Se comprobó la lista de
herramientas antes de las asignaciones principales y no aparecieron
`index_repository`, `search_graph`, `trace_path`, `query_graph`, `search_code` ni
`get_code_snippet`. El grafo se reconstruyó mediante imports, símbolos, callers,
tests, configuración, historial, `git diff` y revisión directa del código.

## 1. Pipeline reconstruido

El flujo histórico confirmado era:

```text
scripts/pipeline/predict.py
  → load_and_normalize_image()
  → CornTransformFactory.get_pipeline("inference")
  → build_model() + checkpoint state_dict
  → model(tensor)
  → torch.softmax(logits, dim=1)
  → torch.topk()
  → idx_to_class
  → Prediccion + Top-k en CLI
```

| Archivo/símbolo | Responsabilidad | Callers | Dependencias | Tests |
|---|---|---|---|---|
| `src/data/loader.py::load_and_normalize_image` | entrada única de imagen y normalización RGB | `predict`, runner experimental y pipelines históricos | Pillow | suite de loader |
| `src/data/transforms.py::CornTransformFactory` | resize, tensor y normalización de inferencia | entrenamiento, evaluación, `predict`, experimentos | Torchvision, YAML | tests de transforms/dataset |
| `src/models::build_model` | resolver arquitectura registrada | `predict`, entrenamiento y experimentos | registry de modelos | tests de modelos |
| `src/inference/classifier.py::classify_image` | transform, logits, único `softmax`, top-k, clase y confidence | `predict`, doble perspectiva y runner | Torch, PIL | `test_classifier.py` |
| `src/preprocessing/segmented_leaf_processor.py` | gates, selección, máscara, perfil y fallback | CLI individual, doble perspectiva | segmentador y utilidades de máscara | tests de preprocessing |
| `src/inference/dual_perspective.py` | orden full-first, estados, disponibilidad y agreement | `predict`, runner experimental | classifier callback y procesador | `test_dual_perspective.py` |
| `src/evaluation/dual_perspective.py` | métricas por vista y subconjunto | runner experimental | scikit-learn | tests de evaluación |
| `scripts/experiments/evaluate_dual_perspective.py` | ejecución por manifiesto, trazabilidad y outputs | CLI manual opt-in | todos los anteriores | tests del manifiesto y guard de no entrenamiento |

El porcentaje histórico se generaba en `predict.py` mediante
`torch.softmax(logits, dim=1)`. Esa lógica fue extraída a `classify_image()` y
ahora es el único punto nuevo que transforma logits en probabilidades para ambas
perspectivas. No existe un segundo sistema de porcentajes.

## 2. Integración realizada

Se añadieron cuatro piezas desacopladas:

1. `ClassificationPrediction` y `RankedClassPrediction`, que representan la
   salida del clasificador existente.
2. `classify_dual_perspective()`, que siempre ejecuta `full_image` antes de llamar
   al segmentador.
3. estados conservadores `reliable`, `uncertain` y `failed`.
4. un runner read-only que consume un manifiesto pequeño y emite `cases.csv`,
   `structured_results.json` y `summary.json`.

La activación individual es explícita:

```bash
python scripts/pipeline/predict.py \
  --model efficientnet_b0 \
  --checkpoint outputs/baselines/efficientnet_b0/<run>/best.pth \
  --image /ruta/hoja.jpg \
  --dual-perspective
```

La configuración conserva:

```yaml
leaf_detection:
  dual_perspective:
    enabled: false
    segmented_profile: mask_black
    reject_multiple_eligible: true
```

`--dual-perspective` no se puede combinar con el modo single-view
`--leaf-profile`. Así se evita confundir una entrada sustituida con dos resultados
independientes.

## 3. Imagen completa

`full_image` recibe siempre la imagen devuelta por `load_and_normalize_image()`.
Se clasifica antes de intentar segmentación y conserva:

- el checkpoint clasificador histórico;
- el `class_to_idx` del `summary.json` del run;
- el tamaño de entrada del checkpoint;
- el transform de inferencia histórico;
- logits, `softmax`, argmax y top-k existentes.

Un error del segmentador, una no detección o una máscara inválida ocurren después
de obtener esta predicción y no pueden eliminarla.

## 4. Hoja segmentada

`segmented_leaf` sólo se clasifica cuando la evaluación de segmentación es
`reliable`. El flujo es:

```text
imagen original
  → UltralyticsLeafSegmenter
  → select_target_leaf
  → gates de confidence, clase, máscara y área
  → máscara binaria original
  → fondo RGB(0, 0, 0)
  → evaluación de confiabilidad
  → mismo classify_image() del full_image
```

Si está disponible, la clase y confidence proceden del mismo `softmax` que la
vista completa. La confidence del segmentador se conserva por separado dentro de
`segmentation`; nunca se presenta como confidence de enfermedad.

## 5. Política de fallback

La política implementada es full-first:

```text
clasificar full_image
  → intentar segmentación
  → evaluar evidencia
      reliable  → clasificar segmented_leaf
      uncertain → segmented_leaf.available = false
      failed    → segmented_leaf.available = false
  → conservar full_image en todos los casos
```

El fallback `original` del procesador no se reclasifica como si fuera una hoja
segmentada. También se capturan excepciones del segmentador y se registran como
`segmentation_error`, preservando la predicción completa. Un fallo posterior al
clasificar la vista segmentada queda como `segmented_classification_error` y no
elimina `full_image`.

Razones normalizadas:

- `no_detection`;
- `low_segmentation_confidence`;
- `invalid_mask`;
- `no_reliable_instance`;
- `ambiguous_multiple_eligible_leaves`;
- `segmentation_error`.

## 6. Estados de segmentación

| Estado | Regla |
|---|---|
| `reliable` | sin fallback, imagen procesada, máscara, confidence, instancia seleccionada y exactamente una instancia elegible |
| `uncertain` | más de una instancia pasa los gates existentes y la selección sería multihoja |
| `failed` | no detección, todas las propuestas rechazadas, máscara inválida, evidencia incompleta o excepción |

La decisión no depende sólo de confidence: usa número de propuestas, elegibilidad,
validez de máscara, área, selección, fallback y trazas del selector. Los thresholds,
gates y fórmula de selección existentes no se modificaron.

## 7. Multihoja

La validación previa registró 20/30 selecciones correctas, 7/30 ambiguas y 3/30
incorrectas. Por ello, cuando hay más de una instancia elegible el nuevo soporte
marca `uncertain`, conserva la razón y no clasifica la vista segmentada.

Esto es intencionalmente más conservador que el procesador, que continúa
seleccionando una instancia para fines de inspección. No se cambió su score
`0.45 × área relativa + 0.35 × centro + 0.20 × confidence`.

Existe una limitación observada: `fall_armyworm_multi_desease_real_67220087.jpg`
produce una única instancia elegible y pasa como `reliable`, aunque la revisión
visual de la Parte 2 calificó su máscara como incorrecta por subsegmentación. La
política sólo puede evaluar la evidencia expuesta por el modelo; no detecta por
sí misma errores semánticos sin ground truth o reglas nuevas. Este hallazgo impide
interpretar `reliable` como garantía de máscara correcta.

## 8. Fall Armyworm

Los seis fallos severos conocidos se ejecutaron con el threshold oficial 0.50:

| Archivo | Estado | Razón | full_image | Confidence full | segmented_leaf |
|---|---|---|---|---:|---|
| `50022046` | failed | no_detection | fall_armyworm | 0.999998 | no disponible |
| `89203456` | failed | no_detection | fall_armyworm | 0.999844 | no disponible |
| `94767583` | failed | no_detection | fall_armyworm | 0.999999 | no disponible |
| `30018448` | failed | no_detection | fall_armyworm | 0.999986 | no disponible |
| `32286052` | failed | no_detection | fall_armyworm | 0.999991 | no disponible |
| `75510763` | failed | no_detection | fall_armyworm | 0.999761 | no disponible |

No se bajó ningún threshold. Los seis casos demuestran el fallback requerido:
la vista segmentada no está disponible y la clasificación completa continúa.
Que el clasificador acierte los seis casos de esta selección dirigida no permite
extrapolar accuracy general.

## 9. Salida estructurada

La salida incluye top-k por vista, trazabilidad del segmentador y una marca
explícita de no fusión:

```json
{
  "experimental": true,
  "full_image": {
    "class": "common_rust",
    "confidence": 0.999689
  },
  "segmented_leaf": {
    "available": true,
    "class": "common_rust",
    "confidence": 0.995578
  },
  "segmentation": {
    "status": "reliable",
    "reason": null,
    "number_of_instances": 1,
    "eligible_instances": 1
  },
  "agreement": true,
  "fusion_applied": false
}
```

En fallback, `segmented_leaf.available` es `false`, clase y confidence no existen,
`agreement` es `null` y la razón permanece visible.

## 10. Agreement / disagreement

`agreement` se calcula sólo cuando ambas vistas tienen predicción:

- `true`: misma clase;
- `false`: clases diferentes;
- `null`: segunda vista no disponible.

No hay voting, ranking de vistas, promedio de confidence, suma de logits ni
meta-clasificador. Los desacuerdos se conservan como observaciones. El smoke real
tuvo cuatro acuerdos y ningún desacuerdo; los caminos `agreement=false` y
`agreement=null` también están cubiertos por tests unitarios.

## 11. Domain shift

El clasificador histórico se entrenó con imágenes completas. Aplicarlo a una hoja
con fondo negro cambia la distribución de entrada. Por ello:

- una confidence segmentada mayor no demuestra superioridad;
- comparar valores de confidence entre vistas no calibra su confiabilidad;
- el experimento B es diagnóstico, no un nuevo baseline oficial;
- deben analizarse exactitud y errores con ground truth, no sólo confidence.

La advertencia se imprime en CLI y queda persistida en cada salida estructurada y
en la procedencia del resumen.

## 12. Preparación de dataset segmentado

No se generó un dataset segmentado masivo. El mecanismo futuro queda diseñado
como derivación reproducible:

```text
clean/ + splits congelados
  → manifest de derivación con la misma pertenencia train/val/test
  → checkpoint del segmentador + SHA-256 + versión Ultralytics
  → máscara + perfil versionado
  → data/segmented_classification/<derivation_id>/
  → manifest por split + lock + fingerprints
```

Cada fila deberá registrar source path y SHA-256, label, environment, split
original, checkpoint y SHA-256, thresholds, perfil, estado, razón, bbox, área,
output path y SHA-256. Los originales permanecen inmutables y los CSV oficiales
de splits no se editan.

Las filas `uncertain` y `failed` deben permanecer en el manifest aunque no tengan
imagen segmentada. Antes de entrenar C se debe congelar una política explícita:
entrenar/evaluar sobre el subconjunto reliable y comparar A sobre exactamente ese
mismo subconjunto, además de reportar cobertura y el sistema end-to-end con
fallback. No se deben mover ejemplos entre splits para compensar fallos.

## 13. Diseño experimental

| Experimento | Clasificador | Entrada | Propósito |
|---|---|---|---|
| A | histórico | imagen completa | referencia histórica |
| B | histórico | hoja segmentada reliable | medir domain shift y señal diagnóstica, no superioridad |
| C | futuro, entrenado con derivación segmentada | hoja segmentada compatible | comparación científicamente relevante frente a A |

El runner implementado permite A vs B mediante un manifiesto pequeño:

```bash
python scripts/experiments/evaluate_dual_perspective.py \
  --manifest /ruta/cases.csv \
  --model efficientnet_b0 \
  --checkpoint outputs/baselines/efficientnet_b0/<run>/best.pth \
  --output-dir outputs/experiments/dual/<run_id> \
  --device cpu
```

Columnas mínimas: `image_path,ground_truth,environment`. Tags opcionales:
`multi_leaf,severe_fall_armyworm`. El output no se sobrescribe y persiste hashes
del manifiesto y checkpoints. El script no contiene `backward`, optimizer,
`torch.save` ni operaciones de entrenamiento.

## 14. Métricas

Por perspectiva se mantienen:

- Accuracy;
- macro Precision, Recall y F1;
- confusion matrix;
- precision, recall, F1 y support por clase;
- accuracy por environment `real`/`lab`.

Se añadieron:

- segmentation coverage sobre todas las imágenes;
- view agreement/disagreement rate sobre casos con ambas vistas;
- accuracy de ambas vistas cuando agreement es true o false;
- accuracy full cuando la segmentación falla;
- resultados en casos multihoja;
- resultados en Fall Armyworm severo.

Cada métrica conserva su denominador. En particular, agreement no usa como
denominador los casos sin segunda vista. No se calcula una métrica fusionada.

## 15. Tests

Cobertura nueva:

1. confidence igual al `softmax` existente;
2. segmentación confiable con predicciones iguales;
3. segmentación confiable con predicciones diferentes;
4. no detección;
5. confidence de segmentación baja;
6. multihoja ambigua;
7. máscara inválida;
8. excepción del segmentador;
9. `full_image` ejecutada primero y siempre conservada;
10. agreement true, false y null;
11. ausencia de fusión;
12. métricas y denominadores;
13. manifest válido, vacío y booleanos inválidos;
14. ausencia de operaciones de entrenamiento en el runner.

Resultados de cierre:

- suite completa: 405 tests y 3 subtests aprobados en 10.70 s;
- Ruff en todos los archivos afectados: aprobado;
- Pyright en todos los archivos afectados: 0 errores y 0 advertencias;
- `git diff --check`: aprobado;
- carga de configuración y `--help` del runner: aprobados.

El chequeo global `ruff check .` conserva 45 errores preexistentes y ajenos a
esta fase, limitados a `fiftyone/import_datasets.py` y
`notebooks/01_eda.ipynb`. No se modificaron esos archivos. El baseline real sin
flags y con `--leaf-profile baseline_full` produjo exactamente la misma clase,
confidence visible y top-k.

## 16. Smoke experimental

Se ejecutó el runner real en CPU con:

- 12 imágenes dirigidas: 1 lab y 11 real;
- classifier checkpoint SHA-256
  `6ff2e7b0cf5a5a19fdcc9ff1b87964707dbbeecceb5d579b702c4786b81b5534`;
- segmenter checkpoint SHA-256
  `4f66456d05d87f9e7080155eb5cd80c583f34849415ec820c950bd97f9c5ec6f`;
- manifest SHA-256
  `35fae839eb4f20ac6a15a82ddd51d132019e9de5acd756d61891aad3f0eca894`.

| Resultado | Valor |
|---|---:|
| full_image disponible | 12/12 |
| reliable / segmented_leaf disponible | 4/12 |
| uncertain | 2/12 |
| failed | 6/12 |
| segmentation coverage | 33.33% |
| agreement entre vistas disponibles | 4/4 |
| disagreement | 0/4 |
| fall_armyworm severo con fallback | 6/6 |

Las dos filas `uncertain` fueron los casos conocidos de nitrogen deficiency con
tres hojas elegibles y gray leaf spot con dos. Las cuatro vistas disponibles
fueron dos common rust, el Fall Armyworm pequeño `63568813` y `67220087`.

En esta muestra, ambas vistas acertaron todos sus casos disponibles y full_image
acertó 12/12. Esto no es una estimación general: la muestra es pequeña, dirigida,
desbalanceada, cubre sólo cuatro de nueve clases y contiene seis fallos conocidos.
El Macro F1 sobre las nueve clases tampoco es interpretable como resultado final
porque cinco clases tienen support cero.

Artefactos:

```text
outputs/leaf_detection/validation_real_pipeline/dual_perspective_smoke/
  manifest_12.csv
  runner_smoke_12/cases.csv
  runner_smoke_12/structured_results.json
  runner_smoke_12/summary.json
  common_rust_lab.json
  common_rust_lab/
```

## 17. Limitaciones

1. `reliable` significa que la evidencia pasa reglas automáticas, no que la
   máscara sea semánticamente correcta; `67220087` lo demuestra.
2. La policy multihoja no puede reconocer múltiples hojas si el segmentador sólo
   devuelve una propuesta elegible.
3. El clasificador histórico está fuera de distribución sobre fondo negro.
4. Las probabilities no están recalibradas para la vista segmentada.
5. El smoke es dirigido y no sirve para comparar rendimiento poblacional.
6. No existe todavía ground truth externo de máscaras para medir IoU/Dice.
7. El label duplicado 183→182 continúa pendiente de corrección versionada antes
   del test oficial del segmentador.
8. No se entrenó el experimento C y no existe evidencia para activar la función
   por defecto o usarla en producción.

## 18. Próximo experimento recomendado

El siguiente paso recomendado es congelar un manifiesto de evaluación A vs B
estratificado por clase y environment, con tags manuales de multihoja, daño
severo, blur y oclusión. Debe incluir suficientes desacuerdos y medir cobertura,
accuracy condicionada y errores visuales, manteniendo el test oficial retenido.

Después, crear una derivación piloto versionada sólo para train/val, entrenar un
clasificador C sobre esa representación y definir su modelo usando val. Una vez
cerrado el protocolo, evaluar A y C una sola vez sobre la misma membresía de test
reliable, reportando por separado cobertura y fallback. Hasta entonces:

**DOBLE PERSPECTIVA LISTA PARA EXPERIMENTOS CONTROLADOS; NO LISTA PARA PRODUCCIÓN
NI ACTIVACIÓN POR DEFECTO.**
