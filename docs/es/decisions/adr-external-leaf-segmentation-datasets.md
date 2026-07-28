# ADR: selección de datasets externos de segmentación de hoja

- Estado: aceptada con filtrado y consolidación candidata materializada
- Fecha: 2026-07-27
- Alcance: dos fuentes externas YOLO/COCO

## Fuentes evaluadas

| Fuente | Imágenes | Anotaciones YOLO | Licencia |
|---|---:|---:|---|
| `corn_leaf_diseases_classification` | 1,003 | 14,415 | CC BY 4.0 |
| `corn` | 157 | 204 | CC BY 4.0 |

Cada fuente conserva su exportación YOLO y un respaldo COCO bajo
`data/leaf_detection/external_sources/`.

## Razón de selección

La fuente grande aporta volumen, campo, fondos, orientaciones y escalas
variadas. La fuente pequeña tiene una semántica más simple de una sola clase de
hoja y sirve como contraste entre dominios. Ninguna se consideró lista sin
auditoría.

## Hallazgos

### Fuente grande

- `gray_leaf_spot`: lesión, excluir;
- `leaf`: hoja completa, conservar tras remapeo;
- `northern_leaf_blight`: lesión, excluir;
- 14,395 polígonos válidos y 20 líneas inválidas;
- 11 inválidas son bbox mezclados; 8 son autointersecciones y una tiene un
  vértice repetido;
- 1,000 imágenes contienen al menos una hoja completa topológicamente válida;
- mediana de área de `leaf`: `0.460460`;
- cero duplicados internos, cruzados o contra el piloto.

Fortaleza: diversidad y 1,000 candidatas válidas. Limitación: mezcla masiva de lesiones,
11 filas en formato YOLO bbox dentro del export de segmentación y necesidad de
revisión semántica.

### Fuente pequeña

- una clase `leaf`, de hoja completa;
- 204 polígonos válidos;
- una imagen tiene TXT vacío y cero anotaciones en COCO;
- 156 imágenes contienen al menos una hoja válida;
- mediana de área: `0.280577`;
- `71.57 %` de las máscaras toca bordes;
- cero duplicados internos, cruzados o contra el piloto.

Fortaleza: semántica simple. Limitación: bajo volumen, menor diversidad,
recortes frecuentes y un caso no recuperable desde COCO.

## Decisión

Ambas fuentes quedan como `accepted_with_filtering`, no como datasets listos:

- conservar sólo polígonos de hoja completa;
- excluir lesiones;
- remapear después a `0 = maize_leaf`;
- recuperar desde COCO sólo con trazabilidad y evidencia;
- excluir o revisar casos sin hoja;
- mantener licencia y procedencia por imagen;
- completar la revisión manual antes de crear splits o entrenar;
- mantener el piloto fuera del entrenamiento.

El total provisional corregido es 1,156 imágenes candidatas sin duplicados exactos entre
fuentes. No se afirma que aumentarán la precisión del clasificador.

## Resultado de la consolidación

La decisión se aplicó de forma trazable en
`data/leaf_detection/detector_dataset/all/`:

- 1 160 imágenes consideradas y 1 156 incluidas;
- 1 226 anotaciones finales `0 = maize_leaf`;
- 13 392 anotaciones de lesión excluidas;
- una anotación `leaf` recuperada desde COCO por imagen única, clase, rol,
  bbox y topología, sin depender del índice;
- un TXT vacío enviado a revisión manual;
- cero duplicados exactos y cero fugas contra el piloto;
- 1 094 grupos de variante original, 39 con más de una variante Roboflow.

La cola manual tiene 34 filas: 32 casos visuales estratificados heredados del
EDA, el TXT vacío y una hoja autointersectada. Esta última no se materializó en
el pool. Ella y la recuperación COCO extremadamente pequeña figuran en
`mandatory_visual_review.csv` con estado pendiente y no pueden entrar en
futuros splits sin decisión explícita.

La auditoría usa caché y parser con esquema 2. El fingerprint global incluye
SHA-256 individuales de fuentes y piloto, además de la configuración del
análisis; cualquier cambio invalida automáticamente la reutilización del EDA.

## Gate posterior de revisión

Al procesar las tablas no se encontró ninguna decisión humana completada. Los
dos casos obligatorios conservan `reviewer_decision` vacío y
`review_status=pending`; las 34 filas del manifiesto general también están
pendientes. Son 35 casos únicos porque la hoja topológicamente inválida figura
en ambas tablas.

Se creó `manifests/dataset_lock.json` con
`status=blocked_by_manual_review`. No se reconstruyó un supuesto dataset
definitivo: hacerlo habría requerido inventar decisiones. El pool de 1 156
imágenes permanece provisional y su fingerprint es
`4c84416487929e05749765f3574cba87d606cde956356422542b79f7532bdd2c`.

## Evidencia visual corregida

El generador inicial construía las filas generales de preview con
`polygons=[]`; no resolvía `original_label_path` y por ello reportaba
`instances=0` para geometrías existentes fuera del consolidado. El generador
actual resuelve cada caso desde YOLO original, COCO, el manifiesto de
recuperación y finalmente el consolidado.

Los 35 casos únicos tienen preview individual. El reporte
`review_preview_validation.json` registra 33 casos desde YOLO, uno desde COCO y
un TXT vacío como `no_geometry_available`; no hay errores ni máscaras
conocidas con cero píxeles. Esto deja las previews listas para evaluación
humana, pero no cambia el gate: las decisiones siguen pendientes y
`dataset_lock.json` continúa `blocked_by_manual_review`.

## Criterios aplicados y siguiente fase

- fuente, nombre original y SHA-256;
- clase original y remapeada;
- fuente de anotación y método de reparación;
- licencia y estado de calidad;
- agrupación por imagen original previa a Roboflow;
- splits propios sin fuga;
- aprobación de la muestra manual estratificada.

## Decisión posterior: splits propios

El 2026-07-28 se ejecutó la fase prevista sobre el padre definitivo
`c087af60c2bad1c133c4ea8b14cee945405bfe4976aa80c4faf089d7a4b9e38c`.
Se unieron relaciones por fuente, nombre original, `roboflow_variant_group`,
SHA-256 y hash perceptual con distancia Hamming menor o igual a 4. Los 1 035
componentes resultantes se asignaron de forma determinista con semilla 42 y
balance de imágenes, máscaras, fuente, área, orientación, resolución, bordes y
múltiples instancias.

El resultado es train/val/test con 809/173/173 imágenes y 858/183/183 máscaras.
La distribución de fuentes es 109/23/23 para `corn` y 700/150/150 para
`corn_leaf_diseases_classification`. No hay grupos, duplicados, variantes
Roboflow ni vecinos perceptuales cruzados; tampoco cruces con el piloto. El
estado final es `ready_for_training_preflight`. Esta decisión no entrenó un
segmentador ni alteró las fuentes.

## Evidencia

- `notebooks/02_leaf_segmentation_external_sources_eda.ipynb`;
- `outputs/leaf_detection/external_sources_eda/`;
- `data/leaf_detection/detector_dataset/manifests/consolidation_manifest.csv`;
- `outputs/leaf_detection/detector_dataset_consolidation/`;
- `outputs/leaf_detection/detector_dataset_consolidation/review_preview_validation.json`;
- `outputs/leaf_detection/detector_dataset_splits/`;
- `docs/es/leaf-detection/segmentation-dataset-splits.md`;
- `docs/es/leaf-detection/external-segmentation-datasets-eda.md`.
