# Validación real del pipeline de segmentación

- Fecha: 2026-08-07
- Alcance: inferencia real en CPU con el checkpoint oficial configurado
- Dataset fuente: sólo lectura
- Restricciones respetadas: no se reentrenó, no se modificaron imágenes, labels,
  splits, clases, pesos, CUDA ni checkpoints clasificadores

## Resumen ejecutivo

El pipeline real carga el checkpoint, genera máscaras a resolución original,
selecciona una instancia, preserva exactamente los píxeles de la hoja y escribe
el exterior como RGB `(0, 0, 0)`. Los siete smoke cases generaron seis máscaras y
un fallback controlado; todas las comprobaciones automáticas de píxel y metadata
pasaron.

La evaluación visual revela límites relevantes. En los 30 casos históricamente
multiinstancia hay 20 selecciones correctas, 7 ambiguas y 3 incorrectas. Los seis
fallos `fall_armyworm` sí producen propuestas a `conf=0.001`, pero su confidence
máxima está entre `0.0015` y `0.0482`, y la mayoría no contiene una lámina foliar
completa. El test interno 183→182 contiene dos polígonos casi idénticos de una
misma hoja; Ultralytics deduplica correctamente uno, pero el label congelado debe
corregirse mediante el flujo de versionado del dataset antes de ejecutar la
evaluación oficial.

Veredicto: **LISTO PARA EXPERIMENTOS CONTROLADOS**. No está listo para activación
por defecto ni para producción.

## 1. Estado confirmado de la implementación

| Componente | Responsabilidad | Quién lo llama | Qué utiliza | Configuración | Tests | Origen |
|---|---|---|---|---|---|---|
| `leaf_segmenter` | cargar YOLO, inferir y convertir `Masks.xy` a máscaras originales | `segmented_leaf_processor`, `preprocess_leaf`, `predict` | Ultralytics, `leaf_mask`, `leaf_roi` | checkpoint, versión, imgsz, conf, IoU, max detections, device | `test_leaf_segmenter.py` | nuevo |
| `leaf_mask` | validar binario, área, bbox, centroide y aplicar fondo exacto | segmentador y procesador segmentado | NumPy, Pillow, `leaf_roi` | color de fondo | `test_leaf_mask.py` | nuevo |
| `segmented_leaf_processor` | gates, score, selección, perfiles, fallback, debug y metadata | `preprocess_leaf`, `predict` | segmentador, `leaf_mask`, `leaf_roi`, `letterbox` | bloque `leaf_detection.segmentation` | `test_segmented_leaf_processor.py` | nuevo |
| `leaf_roi` | geometría bbox, clipping, crop y normalización RGB | pipeline ROI histórico y módulos nuevos | Pillow | min area/margen en config histórica | `test_leaf_roi.py` y relacionados | histórico |
| `letterbox` | resize proporcional y padding | `leaf_processor` y perfil `crop_mask_letterbox` | `leaf_roi` | target size/padding | `test_letterbox.py` | histórico |
| `leaf_processor` | pipeline bbox histórico `baseline_full`/`baseline_roi` | manifests y validaciones ROI | `leaf_roi`, `letterbox` | `processing_profile`, `leaf_detection` histórico | tests ROI históricos | histórico |
| `predict` | clasificación de una imagen; rama segmentada opt-in | CLI | loader, transforms, clasificador y, sólo si se pide, procesador segmentado | `--leaf-profile` | suite y smoke real | histórico modificado |
| `preprocess_leaf` | bundle auditable de una imagen | CLI manual | loader y pipeline segmentado | flags + `dataset.yaml` | smoke real | nuevo |
| configuración | defaults y checkpoint | ambos CLI | `config/dataset.yaml` | `baseline_full`, `enabled=false`, bloque segmentación | carga en tests | histórico ampliado |

`baseline_full` continúa siendo el default en `predict.py`. La rama que crea el
segmentador está protegida por:

```python
if args.leaf_profile != BASELINE_FULL:
    ...
```

Sin `--leaf-profile`, no se carga YOLO ni se altera la imagen antes del transform
histórico.

## 2. Grafo relevante obtenido con Codebase Memory MCP

Codebase Memory MCP **no estuvo disponible en esta sesión**. Se comprobó la lista
de herramientas y recursos MCP; sólo se expuso `codex_apps`. Por tanto, no se
atribuyen resultados ficticios a ese MCP. Como fallback se reconstruyó el grafo
mediante imports, callers, símbolos, `git diff`, historial y lectura del código:

```text
load_and_normalize_image
  ├─ preprocess_leaf
  │    └─ SegmentedLeafProcessor
  │         ├─ LeafSegmenter / UltralyticsLeafSegmenter
  │         │    ├─ leaf_mask: mask_bbox, mask_area_ratio
  │         │    └─ leaf_roi: image_to_rgb
  │         ├─ leaf_mask: validación + apply_leaf_mask
  │         ├─ leaf_roi: crop_leaf_region
  │         └─ letterbox: perfil crop_mask_letterbox
  └─ predict
       ├─ baseline_full ───────────────→ CornTransformFactory → clasificador
       └─ perfil experimental → SegmentedLeafProcessor → transform → clasificador

pipeline ROI histórico
  leaf_processor
    ├─ leaf_roi
    └─ letterbox
```

Callers externos nuevos encontrados: únicamente `predict.py`,
`preprocess_leaf.py` y los tests. No aparecieron cambios o callers nuevos en
`CornDataset`, DataLoader, transforms, splits, entrenamiento, evaluación ni
normalización.

## 3. Checkpoint real utilizado

| Campo | Valor |
|---|---|
| Modelo | `yolo26n-seg` |
| Run | `doctor_maiz_yolo26n_seg_baseline_v4-7a4a5c08-seed42` |
| Checkpoint activo | `outputs/leaf_detection/models/doctor_maiz_leaf_segmenter_best.pt` |
| Backup canónico | `outputs/backups/doctor_maiz_yolo26n_seg_baseline_v4-7a4a5c08-seed42/weights/best.pt` |
| `last.pt` | `outputs/backups/doctor_maiz_yolo26n_seg_baseline_v4-7a4a5c08-seed42/weights/last.pt` |
| Tamaño activo | 6,546,902 bytes |
| Fecha activa | 2026-07-29 13:20:05 -0600 |
| SHA-256 activo/best | `4f66456d05d87f9e7080155eb5cd80c583f34849415ec820c950bd97f9c5ec6f` |
| SHA-256 last | `b355074c7fb9afccf05db7e97535188f37ad5b5b5bbcc27ccf1a7d3e2f79a197` |
| Clase | `0: maize_leaf` |

El activo y el backup `best.pt` son idénticos. La configuración del run confirma
150 épocas, seed 42, determinismo, imgsz 640, batch 26, IoU 0.70 y modelo
`yolo26n-seg.pt`. El checkpoint fue despojado para distribución (`epoch=-1`), por
lo que la época exacta de `best.pt` no se deduce del archivo. En `results.csv`, el
máximo de mask mAP50-95 es `0.93806` en época 87 y la última época registra
`0.92791`.

## 4. Entorno utilizado

| Componente | Versión/estado |
|---|---|
| Python | 3.12.3 |
| Torch | 2.12.1+cu130 |
| Torchvision | 0.27.1+cu130 |
| Ultralytics | 8.4.104 |
| CUDA de build | 13.0 |
| GPU disponible | no; inferencia CPU |

Se instaló exactamente el extra configurado `ultralytics==8.4.104` dentro de
`.venv`. Torch, Torchvision y CUDA no se actualizaron. `pip check` terminó sin
dependencias rotas. Las dependencias nuevas fueron las requeridas por
Ultralytics: OpenCV, Polars, `nvidia-ml-py` y `ultralytics-thop`.

## 5. Smoke test

Se procesaron siete imágenes sin modificar las fuentes:

| Condición | Imagen | Resultado | Conf. | Área máscara | Instancias |
|---|---|---|---:|---:|---:|
| Lab / una hoja | `common_rust_maize_desease_lab_18227491.jpg` | máscara | 0.97294 | 0.68036 | 1 |
| Real / multihoja | `nitrogen_deficiency_corn_leaf_roboflow_real_14931860.jpg` | máscara | 0.95972 | 0.17251 | 3 |
| Fondo complejo | `gray_leaf_spot_maize_field_real_78936724.jpg` | máscara parcial | 0.97408 | 0.25576 | 2 |
| Resolución 48×48 / hoja grande | `fall_armyworm_maize_africa_real_63568813.jpg` | máscara | 0.66448 | 0.89844 | 1 |
| Alta resolución / parcial | `common_rust_maize_field_real_18958685.jpg` | máscara | 0.97771 | 0.66397 | 1 |
| Daño severo / varias hojas | `fall_armyworm_maize_africa_real_94767583.jpg` | fallback original | — | — | 0 |
| Close-up larva/hoja | `fall_armyworm_multi_desease_real_67220087.jpg` | máscara deficiente | 0.64923 | 0.54189 | 1 |

Comprobaciones automáticas sobre los bundles:

- 7/7 contienen los archivos debug requeridos;
- 6/6 máscaras son binarias y conservan resolución;
- 6/6 tienen exterior exactamente RGB `(0, 0, 0)`;
- 6/6 conservan los píxeles interiores idénticos a la fuente;
- 7/7 registran checkpoint, SHA-256 y versión runtime;
- 1/1 no detección usa fallback explícito y advertencia.

La integración completa `mask_black → transform → efficientnet_b0` también
terminó y produjo metadata antes de clasificar.

Artefactos:

- `outputs/leaf_detection/validation_real_pipeline/smoke/`
- `outputs/leaf_detection/validation_real_pipeline/smoke_validation.json`
- `outputs/leaf_detection/validation_real_pipeline/classifier_integration/`

## 6. Resultados visuales

Cada bundle incluye `original.jpg`, `mask.png`, `overlay.jpg`,
`masked_black.png`, `crop.png`, `comparison.jpg` y `metadata.json` cuando hay
máscara. El caso fallback conserva original/comparison/metadata sin inventar una
máscara.

Hallazgos visuales:

- las hojas aisladas, verticales, horizontales y parcialmente visibles suelen
  quedar bien alineadas;
- la máscara de una imagen 48×48 es gruesa pero espacialmente coherente;
- el fondo negro no presenta ringing porque se guarda como PNG;
- `gray_leaf_spot_maize_field_real_78936724.jpg` se divide en dos instancias que
  son mitades de la misma hoja;
- `fall_armyworm_multi_desease_real_67220087.jpg` sufre subsegmentación severa;
- fotografías completamente ocupadas por hoja producen máscaras cercanas a 1.0
  o fallback conservador si son exactamente 1.0.

## 7. Análisis de los seis fallos fall_armyworm

El script original que produjo `dataset_sample_20` no está en el repositorio. El
reporte contiene 240 filas, 234 detecciones, 30 multiinstancia y confidence
mínima almacenada 0.106634; no permite recuperar el threshold exacto, seed o
todos los argumentos. La inferencia diagnóstica actual se ejecutó con generación
de propuestas a `0.001`, conservando 0.50 como gate durante la investigación.

| Imagen | Tipo de fallo | Conf. máxima | Características visuales | Posible causa | Acción recomendada |
|---|---|---:|---|---|---|
| `…50022046.jpg` | debajo de threshold | 0.02071 | planta completa, varias hojas superpuestas, fondo vegetal | identidad objetivo ambigua y dominio multihoja | anotar hoja objetivo y agregar plantas completas multihoja |
| `…89203456.jpg` | debajo de threshold | 0.03591 | 121×121, larva/cogollo, blur, sin lámina completa | imagen fuera del concepto “hoja completa” | hard negative o ruta explícita para cogollo |
| `…94767583.jpg` | debajo de threshold | 0.03786 | 3888×5184, varias hojas perforadas, sombras y suelo | daño, solapamiento y fondo complejo | anotar cada hoja dañada y añadir casos de sombra/oclusión |
| `…30018448.jpg` | debajo de threshold | 0.02059 | cogollo destruido, residuos, oclusión fuerte | silueta foliar casi ausente | definir si cogollo pertenece al alcance; fallback si no |
| `…32286052.jpg` | debajo de threshold | 0.00153 | 224×224, desenfoque severo, close-up sin hoja distinguible | calidad insuficiente | quality gate y hard negative |
| `…75510763.jpg` | debajo de threshold | 0.04818 | rotación, triángulos negros, cogollo destruido | artefacto geométrico y dominio no representado | recolectar rotaciones/cogollo; no bajar threshold globalmente |

No hay ground truth de segmentación para estas seis imágenes externas. Por ello
no se puede afirmar un error de anotación de máscara; sí hay un problema de
selección de dominio, porque varias imágenes clasificatorias no contienen una
hoja segmentable bajo la definición del modelo.

## 8. Evaluación de los 30 casos multihoja

Los 30 casos se definieron a partir del reporte histórico a threshold bajo. Al
repetir con el pipeline configurado a 0.50:

- 17 casos producen una instancia;
- 8 producen dos;
- 1 produce tres;
- 4 no producen instancias;
- 5 usan fallback: cuatro sin propuestas y uno por máscara exacta completa;
- sólo 9 siguen siendo multiinstancia a 0.50.

La regla aplicada es:

```text
score = 0.45 × área_relativa + 0.35 × proximidad_al_centro
      + 0.20 × confidence
```

| # | Imagen | Inst. @0.50 | Seleccionada | Regla/evidencia | Revisión |
|---:|---|---:|---:|---|---|
| 1 | `common_rust_maize_field_real_12375054.jpg` | 1 | 0 | score 0.948; área 0.497 | **CORRECTA** |
| 2 | `common_rust_maize_field_real_20965795.jpg` | 1 | 0 | score 0.957; área 0.791 | **CORRECTA** |
| 3 | `common_rust_maize_field_real_73258109.jpg` | 1 | 0 | score 0.987; área 0.725 | **CORRECTA** |
| 4 | `fall_armyworm_maize_africa_real_37622117.jpg` | 0 | — | sin instancias | **AMBIGUA** |
| 5 | `fall_armyworm_maize_africa_real_68583594.jpg` | 1 | 0 | score 0.902; área 0.291 | **AMBIGUA** |
| 6 | `fall_armyworm_maize_africa_real_95289884.jpg` | 0 | — | sin instancias | **AMBIGUA** |
| 7 | `fall_armyworm_multi_desease_real_67220087.jpg` | 1 | 0 | score 0.854; área 0.542 | **INCORRECTA** |
| 8 | `gray_leaf_spot_cropdg_real_39978786.jpg` | 0 | — | sin instancias | **INCORRECTA** |
| 9 | `gray_leaf_spot_cropdg_real_68684767.jpg` | 1 | 0 | score 0.897; área 0.737 | **CORRECTA** |
| 10 | `gray_leaf_spot_cropdg_real_92680687.jpg` | 1 | 0 | score 0.963; área 0.820 | **CORRECTA** |
| 11 | `gray_leaf_spot_maize_field_real_28853442.jpg` | 2 | 0 | score 0.940; área 0.588 | **CORRECTA** |
| 12 | `gray_leaf_spot_maize_field_real_39881527.jpg` | 1 | 0 | score 0.979; área 0.772 | **CORRECTA** |
| 13 | `gray_leaf_spot_maize_field_real_78936724.jpg` | 2 | 0 | score 0.896; área 0.256 | **INCORRECTA** |
| 14 | `healthy_cropdg_real_236.jpg` | 1 | — | máscara exacta completa; fallback | **AMBIGUA** |
| 15 | `healthy_maize_africa_v1.2_real_116.jpg` | 1 | 0 | score 0.962; área 0.998 | **CORRECTA** |
| 16 | `healthy_maize_desease_v1.1_real_52.jpg` | 1 | 0 | score 0.922; área 0.418 | **CORRECTA** |
| 17 | `lethal_necrosis_multi_desease_real_68128670.jpeg` | 2 | 0 | score 0.847; área 0.585 | **AMBIGUA** |
| 18 | `lethal_necrosis_multi_desease_real_86292252.jpeg` | 0 | — | sin instancias | **AMBIGUA** |
| 19 | `nitrogen_deficiency_corn_leaf_roboflow_real_14931860.jpg` | 3 | 0 | score 0.929; área 0.173 | **CORRECTA** |
| 20 | `nitrogen_deficiency_corn_leaf_roboflow_real_24828346.jpg` | 1 | 0 | score 0.935; área 0.091 | **CORRECTA** |
| 21 | `nitrogen_deficiency_corn_leaf_roboflow_real_55265158.jpg` | 1 | 0 | score 0.957; área 0.174 | **CORRECTA** |
| 22 | `nitrogen_deficiency_corn_leaf_roboflow_real_61428696.jpg` | 1 | 0 | score 0.732; área 0.043 | **CORRECTA** |
| 23 | `nitrogen_deficiency_corn_leaf_roboflow_real_78861979.jpg` | 2 | 0 | score 0.936; área 0.190 | **CORRECTA** |
| 24 | `nitrogen_deficiency_corn_leaf_roboflow_real_90808646.jpg` | 2 | 1 | score 0.854; área 0.101 | **AMBIGUA** |
| 25 | `northern_corn_leaf_blight_corn_leaf_roboflow_real_19639356.jpg` | 1 | 0 | score 0.951; área 0.162 | **CORRECTA** |
| 26 | `northern_corn_leaf_blight_maize_africa_real_11294654.jpg` | 2 | 0 | score 0.887; área 0.360 | **CORRECTA** |
| 27 | `northern_corn_leaf_blight_maize_africa_real_89409035.jpg` | 1 | 0 | score 0.949; área 0.507 | **CORRECTA** |
| 28 | `phosphorus_deficiency_corn_leaf_roboflow_real_20041937.jpg` | 2 | 0 | score 0.973; área 0.270 | **CORRECTA** |
| 29 | `phosphorus_deficiency_corn_leaf_roboflow_real_24944108.jpg` | 1 | 0 | score 0.903; área 0.099 | **CORRECTA** |
| 30 | `potassium_deficiency_corn_leaf_roboflow_real_31998367.jpg` | 2 | 0 | score 0.965; área 0.139 | **CORRECTA** |

Resultado: **20 correctas, 7 ambiguas, 3 incorrectas**. No se cambió la
fórmula. Los fallos no señalan un único problema del score: uno es no detección,
uno es subsegmentación y otro requiere fusionar dos propuestas de una misma hoja.

Evidencia completa:

- `multi_case_summary.csv`: resultado por imagen;
- `multi_instance_metrics.csv`: confidence, área, centro, score y bbox;
- `multi_visual_review.csv`: clasificación y justificación;
- paneles numerados y salidas seleccionadas.

Todo vive en `outputs/leaf_detection/validation_real_pipeline/multi_leaf/`.

## 9. Problema del test interno 183 → 182

El split congelado contiene 173 imágenes y 183 líneas de anotación. El único
caso deduplicado es:

```text
images/test/cldc_ec40ec2d7da5243e.jpg
labels/test/cldc_ec40ec2d7da5243e.txt
```

Procedencia: `nlb_317_jpg.rf.feHnFniAxSkTohP48732.jpg`, fuente
`corn_leaf_diseases_classification`. El label contiene dos polígonos de clase
`maize_leaf` con 169 y 171 puntos.

| Comprobación | Resultado |
|---|---:|
| Polígonos textualmente iguales | no |
| Bbox normalizado de ambos | `[0, 0, 0.9996693, 0.9996693]` |
| Área máscara 1 | 0.40709 |
| Área máscara 2 | 0.40835 |
| IoU entre máscaras | 0.99432 |
| Máscara 1 cubierta por 2 | 99.87 % |
| Máscara 2 cubierta por 1 | 99.56 % |

Visualmente ambas anotaciones cubren la misma hoja. Ultralytics 8.4.104 convierte
cada polígono a bbox mediante `segments2boxes()` y ejecuta
`np.unique(class_id + bbox)`. Como los bbox son idénticos, conserva uno y reporta
un duplicate label removed.

Conclusiones:

- la deduplicación es correcta para este caso;
- el problema está en la anotación/consolidación, no en el validador;
- no hay fuga: `leakage_report.json` registra cero overlaps entre splits;
- no debe editarse el split a mano;
- la corrección debe hacerse en la fuente/manifiesto, regenerar locks y crear una
  nueva versión/fingerprint;
- hasta entonces, el test oficial permanece bloqueado y no se publican métricas
  calculadas silenciosamente con 182 objetos.

Artefactos: `outputs/leaf_detection/validation_real_pipeline/test_duplicate/`.

## 10. Fallbacks

| Entrada | Estado | Acción | Warning/error | Salida/metadata |
|---|---|---|---|---|
| sin detección real | 0 instancias | fallback `original` | `fallback original: sin instancias` | imagen original, `fallback_used=true` |
| propuestas de baja confidence | candidatos rechazados | fallback `original` | razón por candidato + fallback | conserva número de instancias y trazas |
| máscara vacía/corrupta | gate de máscara falla | rechaza instancia; fallback | `máscara corrupta: máscara vacía` | no propaga máscara inválida |
| máscara pequeña | área menor a 0.01 | rechaza instancia; fallback | `máscara demasiado pequeña` | original o `None` según fallback |
| máscara casi completa | `0.98 ≤ área < 1` | acepta | warning explícito | máscara procesada |
| máscara exacta completa | área 1.0 | rechaza como degenerada | razón explícita | fallback conservador |
| múltiples hojas | más de una elegible | aplica score determinista | warning con índice elegido | metadata de todos los candidatos |
| fallback `reject` | ninguna elegible | no produce entrada | razón explícita | `processed_image=None` |

Los 29 tests reales/sintéticos asociados pasan. El pipeline no continúa
silenciosamente: todo descarte aparece en `selection`, `warnings`,
`fallback_used` y `fallback_reason`.

## 11. Trazabilidad

Después de la validación, cada resultado registra:

- ruta de imagen fuente;
- modelo y ruta de checkpoint;
- SHA-256 del checkpoint;
- versión Ultralytics esperada y runtime;
- imgsz, IoU, NMS, max detections y device;
- threshold de generación de propuestas;
- threshold de selección;
- número de instancias y trazas por instancia;
- confidence, área, centro, score, bbox e índice seleccionado;
- perfil, fallback, warnings y versión del procesador;
- tamaños original y final.

El SHA del checkpoint se calcula al construir `UltralyticsLeafSegmenter` y fue
contrastado con `sha256sum`. `smoke_validation.json` agrega SHA-256 de cada imagen
fuente para esta validación; el metadata ordinario conserva la ruta, pero todavía
no incluye el hash de la fuente.

Se detectó y corrigió una ambigüedad: antes, `--confidence` cambiaba el threshold
de propuestas pero el selector seguía usando 0.50 sin registrar ambos valores.
Ahora se publican `proposal_confidence_threshold` y
`selection_confidence_threshold`, y el override explícito se aplica a ambos. El
default permanece en 0.50.

## 12. Regresiones

Comprobaciones ejecutadas:

- sin `--leaf-profile` frente a `--leaf-profile baseline_full`: mismo modelo,
  predicción, top-k y probabilidades;
- ruta experimental real hasta el clasificador: aprobada;
- tests específicos: 29 aprobados;
- suite completa: 386 tests y 3 subtests aprobados en 10.28 s;
- Ruff en los archivos afectados: aprobado;
- Pyright en los archivos afectados: 0 errores y 0 advertencias;
- `git diff --check`: aprobado;
- revisión de callers posterior al ajuste.

El chequeo global `ruff check .` conserva 45 errores preexistentes y ajenos a
este cambio, limitados a `fiftyone/import_datasets.py` y
`notebooks/01_eda.ipynb`; no se modificaron esos archivos.

No se modificaron transforms, Dataset, DataLoader, splits, entrenamiento,
evaluación, helpers de paths ni normalización. `leaf_detection.enabled` continúa
en `false` y `baseline_full` continúa como default.

## 13. Problemas encontrados

### Alta

1. Tres de 30 casos históricamente multiinstancia producen una selección
   incorrecta a 0.50.
2. Seis casos `fall_armyworm` están fuera o en el borde del concepto visual
   entrenado; bajar globalmente el threshold no está justificado.
3. El test congelado contiene una anotación duplicada semánticamente y requiere
   corrección versionada antes de evaluación oficial.
4. Siete de 30 casos carecen de identidad de hoja objetivo inequívoca.

### Media

1. La prueba externa histórica no tiene script, seed ni argumentos completos.
2. El debug estándar sólo muestra la instancia seleccionada; para evaluar
   multihoja hubo que generar paneles adicionales.
3. El metadata ordinario no incluye SHA-256 de la imagen fuente.
4. No existen métricas IoU/Dice externas porque las 240 imágenes no tienen ground
   truth de segmentación.

### Baja

1. En este sandbox Ultralytics intentó escribir settings bajo un directorio home
   no escribible; se usó `/tmp`. No afectó inferencia ni artefactos.
2. La advertencia NVML es esperada porque no hay GPU disponible.

## 14. Cambios realizados, si fueron necesarios

No se reimplementó ningún componente. Sólo se corrigió la trazabilidad observada
durante la prueba real:

1. `mask_processor_config_from_mapping()` acepta un override explícito de
   confidence de selección.
2. `preprocess_leaf.py --confidence` aplica el mismo valor a generación y
   selección.
3. metadata distingue threshold de propuestas y de selección.
4. se añadieron tests para el override y ambos campos.

No se cambiaron el default 0.50, IoU, max detections, pesos del score, gates,
fallbacks, perfiles ni comportamiento histórico.

Se generaron únicamente artefactos nuevos bajo:

```text
outputs/leaf_detection/validation_real_pipeline/
```

## 15. Recomendación

### LISTO PARA EXPERIMENTOS CONTROLADOS

La mecánica del pipeline está validada: carga real, alineación, negro exacto,
inmutabilidad, fallbacks, metadata e integración con classifier funcionan. Esto
permite comparar `baseline_full` frente a `mask_black` sobre un protocolo
congelado y, preferiblemente, reentrenar clasificadores con la misma
representación usada en inferencia.

No debe activarse por defecto. Antes de producción se requiere:

1. corregir y versionar el label duplicado del test;
2. cerrar una evaluación oficial una sola vez;
3. anotar identidad objetivo en multihoja;
4. resolver los tres errores visuales y repetir la revisión;
5. construir un test externo con máscaras ground truth;
6. recolectar cogollo, larva, daño severo, blur, sombras y oclusiones;
7. medir el clasificador entrenado con máscaras, no sólo aplicarlas a checkpoints
   históricos.
