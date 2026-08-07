# Auditoría y fortalecimiento del Reliability Gate

Fecha de ejecución: 2026-08-07. Esta auditoría es experimental y de sólo lectura sobre
`DATASET_ROOT`; no entrenó modelos ni modificó imágenes, labels, clases, splits o checkpoints.

## 1. Estado de Codebase Memory MCP

La configuración de `codebase-memory-mcp` ya era válida y única en
`~/.codex/config.toml`; el ejecutable `0.9.0` y su cache configurado también existían. El
conector no fue hidratado como herramienta nativa en esta sesión, por lo que no había un
proyecto cargado al comienzo. Se utilizó el mismo servidor mediante su CLI MCP, con cache
aislado en `/tmp/doctor-maiz-codebase-memory`, sin cambiar la configuración del usuario.

El índice final contiene 3,417 nodos y 14,920 aristas. Se ejecutaron `search_graph`,
`trace_path`, `get_code_snippet`, `detect_changes` e `index_repository` antes y después de
los cambios. Por tanto, Codebase Memory sí estuvo operativo para descubrimiento e impacto;
la única limitación fue la ausencia del conector nativo en la lista de herramientas de la
sesión activa.

## 2. Grafo relevante del quality gate

El grafo confirmó este flujo:

```text
predict.main / evaluate_dual_perspective.run_experiment
  -> classify_dual_perspective
     -> SegmentedLeafProcessor.process
        -> select_target_leaf
     -> assess_segmentation
        -> _quality_gate_metrics
           -> mask_geometry
        -> SegmentationAssessment
```

Los callers directos de `assess_segmentation` son `classify_dual_perspective` y el nuevo
runner de auditoría. Los consumidores a dos saltos son el CLI `predict`, el experimento de
doble perspectiva y sus tests. `mask_geometry` sólo alimenta el gate y su test directo; no
altera la máscara, el selector ni la imagen procesada. El grafo no encuentra rutas hacia
entrenamiento, creación de splits, labels, clases o checkpoints.

## 3. Reliability Gate actual

El gate anterior queda reproducido por `assess_segmentation_legacy` para que la comparación
sea ejecutable y no una reconstrucción manual.

| Variable/criterio | Dónde se calcula | Cómo afectaba el estado | Configuración | Tests |
|---|---|---|---|---|
| `fallback_used` / `processed_image` | `SegmentedLeafProcessor.process` | fallback o ausencia de imagen → `failed` | `segmentation.fallback` | fallback, no detection, máscara vacía |
| número elegible | `select_target_leaf` y `selection_traces` | más de una elegible → `uncertain`; distinta de una → `failed` | `reject_multiple_eligible: true` en el gate anterior | multihoja ambigua |
| confidence | segmentador y selector | debajo de `0.50` no llega como elegible; todas bajas → `failed` | `leaf_detection.confidence_threshold` | confidence baja |
| `mask_area_ratio` | `SegmentedLeafProcessingResult` | sólo descartaba área menor que `0.01` o exactamente `1.0`; no auditaba completitud | `min_mask_area_ratio`, `near_full_warning_ratio` | máscara pequeña, exacta completa |
| instancia seleccionada y máscara | `select_target_leaf` | evidencia ausente → `failed`; presente → `reliable` | pesos área/centro/confidence | selección determinista |
| geometría de calidad | no existía | no afectaba el estado | no existía | no existía |

La metadata anterior registraba confidence, área, bbox, instancias, selección, fallback y
warnings, pero no una decisión geométrica ni una lista estructurada de razones del gate.

## 4. Análisis del caso 67220087

Artefactos: imagen
`clean/fall_armyworm/real/fall_armyworm_multi_desease_real_67220087.jpg`, panel de auditoría
`visuals/007_multi_07_67220087.jpg` y fila `multi_07_67220087` en `audit_metrics.csv`.

La causalidad anterior fue exacta:

```text
confidence seleccionada = 0.649225 >= 0.50
instancias elegibles     = 1
selected_instance        = 0
mask_area_ratio          = 0.541893 (mayor que 0.01 y distinta de 1.0)
fallback                 = false
mask + processed_image   = presentes
=> supera todos los checks anteriores
=> reliable
```

La inspección visual muestra que la máscara sigue el tejido alrededor de la larva, abre un
vacío grande y elimina parte sustancial de la lámina. Las señales medibles son bbox
`[1, 0, 189, 216]`, `bbox_area_ratio=0.809311`,
`mask_bbox_ratio=0.669572` y `normalized_perimeter=9.970047`. Área o confidence por sí
solas no detectan el error; la combinación de máscara grande, bbox poco lleno y contorno
complejo sí separa este caso de las máscaras GOOD grandes observadas.

## 5. Conjunto de auditoría

El manifest versionado
[`segmentation_reliability_audit_v1.csv`](../../../scripts/experiments/manifests/segmentation_reliability_audit_v1.csv)
selecciona 42 imágenes existentes: los 30 casos multihoja revisados, 6 fallos severos
conocidos y 6 controles adicionales. Contiene 7 casos `lab`, 35 `real` y las 9 clases; sus
tags, que pueden solaparse, incluyen multihoja, daño severo, blur, oclusión, hoja parcial,
fondo complejo, hoja pequeña y hoja grande.

La muestra contiene segmentación correcta, subsegmentación, fuga menor/sobresegmentación,
multihoja correcta/ambigua/incorrecta, tamaños extremos, parcialidad, blur, oclusión, daño
severo, fondo complejo y no detección. Incluye obligatoriamente `67220087`. El SHA-256 del
manifest usado fue `732a62e0e64d37cfcde5341539dad7f51c2eded1c2e7b2b6aefa7399588c63ed`.

## 6. Taxonomía de calidad de máscaras

- `GOOD`: el resultado segmentado es suficientemente fiel para clasificación.
- `AMBIGUOUS`: no es inequívocamente incorrecto, pero no debería aprobarse de forma
  automática.
- `BAD`: la máscara o la no-detección es incorrecta frente a un objetivo visual claro.

El recuento es 24 `GOOD`, 12 `AMBIGUOUS` y 6 `BAD`. Es una etiqueta de auditoría del
segmentador, no una clase agrícola y no modifica el ground truth del dataset.

## 7. Variables analizadas

Se registran confidence umbral y seleccionada, área de máscara, área de bbox, llenado
mask/bbox, número de propuestas/elegibles, instancia seleccionada, área relativa, cercanía
al centro, score y margen entre las dos mejores instancias. También se calculan ancho,
alto, aspecto, centro normalizado, contactos de borde, componentes 4-conectados, proporción
del componente principal, área, perímetro por aristas, perímetro/área y
perímetro/raíz-área.

Las métricas son exactas en coordenadas de la imagen y sólo usan NumPy/Pillow ya presentes.
Los componentes se obtienen con runs por fila, sin SciPy, OpenCV ni una dependencia nueva.
No se implementó solidez/convex hull: requería más complejidad y las variables simples ya
mostraron suficiente evidencia. `proposal_confidence` y `selection_confidence` coinciden
en este selector porque no existe una segunda confidence aprendida; el score de selección
se registra por separado y no se inventa un valor.

## 8. Distribuciones encontradas

Hay geometría para 29 resultados; los otros 13 son fallos sin máscara utilizable.

| Señal | GOOD | AMBIGUOUS | BAD | Conclusión |
|---|---:|---:|---:|---|
| `mask_area_ratio`, mediana (rango) | 0.502 (0.043–0.998) | 0.291 (0.101–0.585) | 0.399 (0.256–0.542) | solapa demasiado; no usar sola |
| `mask_bbox_ratio`, mediana (rango) | 0.776 (0.243–1.000) | 0.756 (0.259–0.995) | 0.806 (0.670–0.942) | no separa globalmente |
| perímetro normalizado, mediana (máx.) | 5.379 (11.877) | 8.313 (11.428) | 7.458 (9.970) | no separa globalmente |
| confidence seleccionada, mínimo | 0.575 | 0.553 | 0.649 | confidence alta no implica calidad |
| margen multihoja, mínimo/máximo | 0.369/0.743 | 0.096/0.151 | 0.302/0.302 | separación observada útil |

Fragmentación tampoco es una regla segura: hubo hasta 95 componentes pequeños en una
máscara GOOD por agujeros/ruido rasterizado. En cambio, entre las 12 máscaras GOOD con área
al menos `0.50`, el menor llenado de bbox fue `0.750741` y el mayor perímetro normalizado
fue `7.783603`; `67220087` cruza ambos límites en la dirección sospechosa.

## 9. Reliability Precision actual

Con el gate anterior hubo 20 `reliable`: 18 eran `GOOD` y 2 no lo eran
(`multi_05_68583594` AMBIGUOUS y `multi_07_67220087` BAD).

```text
Reliability Precision anterior = 18 / 20 = 0.90 (90.00%)
```

## 10. Segmentation Coverage actual

```text
Segmentation Coverage anterior = 20 / 42 = 0.476190 (47.62%)
```

Los estados eran 20 `reliable`, 9 `uncertain` y 13 `failed`.

## 11. Problemas del gate actual

El gate confundía evidencia disponible con evidencia correcta: una única propuesta válida
por confidence y área se promovía a `reliable` sin analizar completitud. A la vez, rechazaba
las 9 selecciones multihoja sin mirar su margen; seis de esas máscaras eran GOOD y tenían
una instancia claramente dominante. Confidence, área, aspect ratio, contactos de borde,
componentes o perímetro aislados no separan las etiquetas y habrían creado regresiones.

## 12. Nuevo gate propuesto

La lógica implementada es ordenada y transparente:

```text
si no hay detección/fallback/evidencia completa:
    failed
si hay varias elegibles y reject_multiple_eligible está activo:
    uncertain: ambiguous_multiple_eligible_leaves
si hay varias elegibles y margen < 0.33:
    uncertain: ambiguous_instance_score_margin
si mask_area_ratio >= 0.999:
    uncertain: excessive_mask_area_ratio
si área >= 0.50 Y mask_bbox_ratio < 0.70 Y perímetro normalizado > 8.0:
    uncertain: suspicious_large_mask_geometry
en otro caso:
    reliable
```

No existe quality score agregado ni pesos nuevos. La política configurada permite una
multihoja sólo cuando el margen es suficiente; el override estricto anterior continúa
disponible con `reject_multiple_eligible: true`.

## 13. Justificación de thresholds

- `min_multi_instance_score_margin=0.33` queda entre el mayor BAD/AMBIGUOUS observado
  (`0.301883`) y el menor GOOD (`0.369172`).
- `large_mask_area_ratio=0.50` define el estrato de máscaras que ya ocupa al menos la mitad
  de la imagen; no es un criterio de rechazo aislado.
- `min_large_mask_bbox_ratio=0.70` queda por debajo del mínimo GOOD grande (`0.750741`).
- `max_large_mask_normalized_perimeter=8.0` queda por encima del máximo GOOD grande
  (`7.783603`). Sólo la conjunción de las tres señales geométricas rechaza.
- `max_mask_area_ratio=0.999` conserva el caso GOOD casi completo (`0.997577`) y marca como
  sospechosa una fuga prácticamente total; la máscara exactamente completa ya era
  degenerada para el selector.

Los límites se redondearon hacia afuera del rango GOOD para conservar margen. Son hipótesis
auditables de una muestra pequeña y deben recalibrarse con un conjunto ciego más amplio;
no son reglas botánicas universales ni fueron elegidos sólo para el ID `67220087`.

## 14. Implementación

- [`dual_perspective.py`](../../../src/inference/dual_perspective.py): configuración,
  comparación legacy, nuevo gate, razones y metadata.
- [`leaf_mask.py`](../../../src/preprocessing/leaf_mask.py): geometría determinista y
  componentes sin dependencias adicionales.
- [`segmentation_reliability.py`](../../../src/evaluation/segmentation_reliability.py):
  Reliability Precision, coverage, distribuciones y subgrupos.
- [`audit_segmentation_reliability.py`](../../../scripts/experiments/audit_segmentation_reliability.py):
  runner no destructivo, CSV/JSON, predicciones y paneles.
- [`dataset.yaml`](../../../config/dataset.yaml): thresholds centralizados y feature aún
  desactivada por defecto.

`SegmentationAssessment.to_metadata()` contiene `quality_gate.version`, `reasons`,
`metrics` y `thresholds`. Las razones no están dispersas en el CLI. El output estructurado
conserva además toda la metadata previa del procesador.

## 15. Resultado de 67220087

El resultado medido cambia de `legacy_status=reliable` a
`proposed_status=uncertain`, con razón `suspicious_large_mask_geometry`. `full_image`
permanece disponible como `fall_armyworm` con confidence `0.99999547`;
`segmented_prediction`, `segmented_confidence` y `agreement` quedan nulos. El clasificador
se llamó una sola vez en el test de regresión de la máscara real, por lo que la vista
rechazada no se clasifica.

## 16. Reliability Precision nueva

El gate propuesto produjo 25 `reliable`: 24 `GOOD` y 1 `AMBIGUOUS`.

```text
Reliability Precision nueva = 24 / 25 = 0.96 (96.00%)
```

## 17. Segmentation Coverage nueva

```text
Segmentation Coverage nueva = 25 / 42 = 0.595238 (59.52%)
```

La cobertura aumentó porque seis multihoja GOOD pasaron con margen claro, aunque la máscara
BAD de `67220087` fue retirada. Los estados finales son 25 `reliable`, 4 `uncertain` y 13
`failed`; no se bajó el umbral de proposals `0.50`.

## 18. Comparación antes vs después

La precisión aumenta 6.00 puntos porcentuales y la cobertura 11.90 puntos. Los false
reliable bajan de 2 a 1 y las máscaras GOOD rechazadas de 6 a 0. Esto no se obtuvo marcando
todo como `uncertain`: el gate distingue multihoja clara de multihoja ambigua y agrega una
protección geométrica conservadora.

## 19. Resultados por subgrupos

| Subgrupo | N | Precision antes → después | Coverage antes → después | Observación |
|---|---:|---:|---:|---|
| Lab | 7 | 100.00% → 100.00% | 57.14% → 57.14% | sin regresión |
| Real | 35 | 87.50% → 95.24% | 45.71% → 60.00% | elimina `67220087` y recupera multihoja GOOD |
| Multihoja | 31 | 87.50% → 95.24% | 51.61% → 67.74% | margen separa 6 GOOD de 3 no-GOOD |
| Fondo complejo | 29 | 92.31% → 94.74% | 44.83% → 65.52% | mejora con un false reliable restante |
| Oclusión | 14 | 50.00% → 75.00% | 28.57% → 28.57% | `67220087` sale; N pequeño |
| Hoja parcial | 20 | 71.43% → 83.33% | 35.00% → 30.00% | prioriza precisión |
| Fall armyworm severo | 11 | 33.33% → 50.00% | 27.27% → 18.18% | sigue siendo el subgrupo más débil |
| Blur | 4 | 100.00% → 100.00% | 25.00% → 25.00% | tres no-detecciones |
| Hoja pequeña | 5 | 100.00% → 100.00% | 40.00% → 40.00% | no se castiga por área baja |

Las precisiones con pocos `reliable` son descriptivas, no estimaciones poblacionales.

## 20. False reliable restantes

Queda uno: `multi_05_68583594`, etiquetado `AMBIGUOUS`. La causa es
`multihoja/hoja objetivo ambigua`: visualmente varias hojas convergen, pero a confidence
`0.50` sólo sobrevive una propuesta (`0.556769`), por lo que no existe segundo score para
calcular margen. Su área `0.291305`, llenado `0.755818` y perímetro `8.312526` no justifican
una regla geométrica sin rechazar GOOD dañadas o pequeñas. Es la máxima prioridad de la
siguiente auditoría; no se añadió un threshold de confidence estrecho que sobreajustara el
caso.

## 21. Tests

Se cubren: máscara correcta, no detección, confidence baja, subsegmentación evidente,
máscara excesivamente grande, multihoja ambigua, multihoja clara, máscara vacía, fixture de
la máscara real `67220087`, disponibilidad permanente de `full_image`, ausencia de llamada
al clasificador segmentado cuando el estado no es reliable y metadata completa del gate.
También hay tests de geometría exacta/componentes y de métricas de auditoría.

Resultado: 33 tests dirigidos pasaron; la suite completa terminó con 415 tests y 3 subtests
pasando. Pyright sobre los archivos afectados reportó 0 errores/0 warnings, Ruff y el check
de formato pasaron sobre los 7 archivos Python modificados, y `git diff --check` no reportó
errores.

## 22. Smoke real ampliado

El smoke final procesó las 42 imágenes, frente a 12 del smoke anterior, con
`efficientnet_b0` y el mismo segmentador/checkpoints ya existentes. Generó 42 paneles,
`audit_metrics.csv`, `structured_results.json`, `summary.json` y `false_reliable.csv` bajo
`outputs/leaf_detection/validation_real_pipeline/reliability_gate_audit_v1/`.

Todas las 42 imágenes obtuvieron `full_image`; 25 obtuvieron vista segmentada. La accuracy
full fue 38/42 (`90.48%`) y la segmentada 15/25 (`60.00%`); hubo 13/25 acuerdos (`52.00%`).
La caída de clasificación segmentada confirma el domain shift ya documentado: una máscara
GOOD no implica que el clasificador histórico, entrenado con imágenes completas, esté
calibrado para fondo negro. No se aplicó fusión ni se compararon confidences para decidir.

## 23. Impacto sobre doble perspectiva

Codebase Memory reindexó el estado modificado y trazó los callers/callees hasta profundidad
3. `classify_dual_perspective` sigue clasificando `full_image` antes de intentar
segmentación. Sólo llama al clasificador sobre `processed_image` cuando
`assessment.status is RELIABLE`. `predict.main` y `evaluate_dual_perspective.run_experiment`
reciben la nueva política mediante `DualPerspectiveConfig.from_mapping`; sus contratos de
resultado, agreement nulo y `fusion_applied=false` permanecen.

El procesador, el selector, el adaptador Ultralytics y los thresholds de proposals no
cambiaron. El único helper nuevo aguas abajo es `mask_geometry`. El runner de auditoría usa
el loader oficial, `_build_leaf_processor`, el mismo mapeo de config y SHA-256 de manifest,
config y checkpoints. No hay rutas de impacto hacia entrenamiento o datasets oficiales.

## 24. Limitaciones

La muestra es intencionalmente difícil pero pequeña (42), con sólo 7 casos lab y una única
revisión visual por caso. Los thresholds se validaron sobre la misma población usada para
derivarlos, no sobre un holdout ciego. La geometría no puede reconocer por sí sola la
identidad semántica de una hoja cuando sólo sobrevive una propuesta; de ahí el false
reliable restante. No se midió IoU contra máscaras humanas. El desempeño de clasificación
segmentada no evalúa pureza del gate debido al domain shift del clasificador histórico.

## 25. Recomendación para la siguiente fase

Congelar `quality_gate.version=1.0.0` y auditarlo sobre un conjunto ciego mayor, con doble
revisión visual y más oclusión/fall armyworm severo. Priorizar ejemplos como
`multi_05_68583594` y estudiar evidencia de proposals descartadas sin bajar el threshold.
Sólo después recalibrar límites o generar un dataset experimental; cada fila debe conservar
razones, métricas, versión y panel, y los casos no-GOOD deben pasar por revisión humana. No
activar segmentación ni fusión por defecto.

| MÉTRICA | ANTES | DESPUÉS |
|---|---:|---:|
| Reliability Precision | 90.00% (18/20) | 96.00% (24/25) |
| Segmentation Coverage | 47.62% (20/42) | 59.52% (25/42) |
| Reliable total | 20 | 25 |
| False reliable | 2 | 1 |
| Uncertain | 9 | 4 |
| Failed | 13 | 13 |
| Good masks rechazadas | 6 | 0 |

Respuesta final: **PARCIALMENTE**. El gate es reproducible y mejora simultáneamente
Reliability Precision y Coverage, distingue la multihoja clara y bloquea la
subsegmentación de `67220087`. Sin embargo, `96%` deja 1 false reliable entre 25 aceptadas,
el subgrupo severo sigue débil y el smoke muestra fuerte domain shift. Es suficiente para
construir un dataset experimental con trazabilidad y revisión humana, pero todavía no para
generarlo de forma automática y tratar todas sus máscaras como ground truth confiable.
