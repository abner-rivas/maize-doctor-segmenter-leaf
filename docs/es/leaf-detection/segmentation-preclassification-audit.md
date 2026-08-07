# Auditoría del segmentador y diseño de segmentación previa a clasificación

- Fecha de auditoría: 2026-08-07
- Rama revisada: `test-yolo`
- Alcance: artefactos locales, código, configuración, métricas, overlays y
  manifiestos presentes en el repositorio
- Restricciones respetadas: no se entrenó, no se modificó `raw/`, `clean/`,
  splits, pesos ni imágenes originales

## Resumen ejecutivo

El segmentador `yolo26n-seg` muestra un ajuste fuerte en `val`, pero todavía no
cuenta con una evaluación oficial aprobada sobre el test interno ni con métricas
de píxel sobre el dominio externo de Doctor Maíz. La prueba visual externa de
240 imágenes alcanza detección en 234 (`97.5 %`) con el umbral bajo empleado,
pero seis casos `fall_armyworm` no producen máscara, 30 imágenes producen más
de una instancia y todavía no existía una estrategia para elegir la hoja
objetivo.

La conclusión es **parcialmente listo**: se puede integrar de forma opt-in para
generar máscaras, depurar visualmente y preparar experimentos controlados. No se
debe activar automáticamente sobre clasificadores históricos ni declarar listo
un dataset enmascarado definitivo.

# Auditoría del segmentador

## Estado actual

El flujo histórico activo permanece:

```text
imagen completa
→ resize directo
→ normalización
→ clasificador baseline_full
```

En `config/dataset.yaml` se conserva:

```yaml
processing_profile: baseline_full

leaf_detection:
  enabled: false
```

Antes de este requerimiento sí existían:

- dataset de segmentación consolidado y congelado;
- splits agrupados y reproducibles;
- runner cloud de entrenamiento/evaluación;
- un modelo entrenado y sus artefactos principales;
- overlays de una prueba externa sobre imágenes del clasificador;
- infraestructura histórica de bbox, crop y letterbox;
- cálculo offline de IoU, Dice y subsegmentación, pero sin predicciones crudas
  disponibles para ejecutarlo sobre el modelo actual.

No existían:

- adaptador reutilizable del checkpoint para inferencia en `src/`;
- salida de máscara binaria en resolución original;
- selección de una instancia objetivo;
- aplicación reusable de fondo negro;
- política de fallback específica para máscaras;
- bundles de debug con máscara, resultado y metadata;
- activación A/B en la inferencia del clasificador.

## Modelo actual

| Campo | Valor comprobado |
|---|---|
| Arquitectura | `yolo26n-seg` |
| Tarea | segmentación de instancias |
| Clase | `0 = maize_leaf` |
| Ultralytics | `8.4.104` |
| Checkpoint activo | `outputs/leaf_detection/models/doctor_maiz_leaf_segmenter_best.pt` |
| Backup `best.pt` | `outputs/backups/doctor_maiz_yolo26n_seg_baseline_v4-7a4a5c08-seed42/weights/best.pt` |
| SHA-256 de ambos | `4f66456d05d87f9e7080155eb5cd80c583f34849415ec820c950bd97f9c5ec6f` |
| `last.pt` | presente, SHA-256 distinto al mejor |
| Tamaño de entrada | `640` |
| Épocas solicitadas/completadas | `150/150` |
| Patience | `30`; no activó parada temprana |
| Batch efectivo | `26` |
| Optimizer | `auto` |
| Seed | `42` |
| Determinista | `true` |
| AMP | `true` |
| IoU configurado durante train/val | `0.7` |
| NMS | por clase; `agnostic_nms=false` |
| Máximo de detecciones train/val | `300` |

El `best.pt` fue despojado por Ultralytics: dentro del checkpoint `epoch=-1` y
`best_fitness=None`. Falta `training_summary.json` en la copia local, por lo
que no puede probarse el número exacto de época del checkpoint. En
`results.csv`, la mayor media de bbox/mask mAP50-95 aparece en la época 136 y el
máximo mask mAP50-95 en la 87.

Augmentations efectivas registradas: HSV, `translate=0.1`, `scale=0.5`, flip
horizontal `0.5`, mosaic `1.0`, `close_mosaic=10`, RandAugment y erasing `0.4`.
No se usaron rotaciones (`degrees=0`), perspectiva, shear, mixup, cutmix ni
copy-paste.

## Implementación, entrenamiento e inferencia localizados

- Entrenamiento/evaluación: `cloud_training/run_ultralytics.py`.
- Config de train: `cloud_training/configs/train_yolo26n_seg.yaml`.
- Config de test: `cloud_training/configs/validate_yolo26n_seg.yaml`.
- Guards: `cloud_training/train.sh`, `validate.sh`, `evaluate_test.sh`.
- Evaluación piloto prevista:
  `scripts/pipeline/leaf_segmentation_pilot_evaluate.py`.
- Métricas downstream previstas:
  `src/evaluation/segmentation_downstream.py` y
  `scripts/pipeline/leaf_segmentation_downstream_metrics.py`.
- EDA: `notebooks/02_leaf_segmentation_external_sources_eda.ipynb`.
- Documentación previa: `docs/es/leaf-detection/` y ADR bajo
  `docs/es/decisions/`.

La prueba de 240 imágenes no conserva su script de creación. Ésta es una brecha
de reproducibilidad: sólo quedan overlays y un CSV resumen.

## Test localizado

### 1. Validación del entrenamiento

Los artefactos están bajo:

```text
outputs/backups/doctor_maiz_yolo26n_seg_baseline_v4-7a4a5c08-seed42/
```

Contiene `results.csv`, curvas de bbox/mask, matriz de confusión, args, config
efectiva, `best.pt` y `last.pt`.

### 2. Test interno congelado

El contrato exige 173 imágenes, 183 instancias, fingerprint
`04654535…89c51` y el SHA exacto de `best.pt`. La ejecución está bloqueada:
Ultralytics 8.4.104 deduplica dos polígonos distintos con el mismo bbox en
`cldc_ec40ec2d7da5243e.txt` y observa 182 instancias. El runner aborta en vez de
publicar métricas inconsistentes.

No existe:

```text
outputs/leaf_detection/segmenter_evaluation/test_summary.json
```

Por tanto, no deben presentarse las métricas de `val` como métricas de test.

### 3. Prueba externa `dataset_sample_20`

Los artefactos están en:

```text
outputs/leaf_detection/predictions/dataset_sample_20/
```

Contiene 240 overlays y `prediction_report.csv`. El nombre “sample_20” refleja
20 imágenes por combinación clase/entorno disponible:

- 180 reales;
- 60 de laboratorio;
- 9 clases con entorno real;
- `common_rust`, `gray_leaf_spot` y `northern_corn_leaf_blight` también con
  entorno lab.

El reporte conserva detecciones, cantidad de máscaras y confidences. No
conserva máscaras binarias, polígonos TXT, bbox tabulares, semilla, versión,
hash de checkpoint ni parámetros completos. La menor confidence guardada es
aproximadamente `0.106`, así que el umbral efectivo fue menor o igual a ese
valor; no es posible probar el número exacto.

## Dataset utilizado

### Dataset del segmentador

| Split | Imágenes | Instancias | Fuente grande | Fuente `corn` |
|---|---:|---:|---:|---:|
| train | 809 | 858 | 700 | 109 |
| val | 173 | 183 | 150 | 23 |
| test | 173 | 183 | 150 | 23 |

Totales: 1,155 imágenes, 1,224 máscaras y 1,035 grupos. Los locks reportan cero
fugas exactas, por nombre, grupo, variante Roboflow, hash perceptual y contra el
piloto de 100 imágenes.

Sólo 44/1,155 imágenes son multiinstancia (`3.8 %`). En test interno hay siete.
La máscara media por imagen en test va de `0.1115` a `0.9935`; el test no mide
hojas realmente pequeñas. La fuente `corn` tiene `71.57 %` de polígonos tocando
borde, mientras que la fuente grande tiene `9.07 %`.

### Dataset de la prueba externa

- 240 imágenes, todas encontradas y legibles;
- 57 resoluciones;
- 138 cuadradas, 56 portrait y 46 landscape;
- área desde 2,304 píxeles (`48×48`) hasta 20,155,392 píxeles;
- mediana de 409,600 píxeles;
- diversidad de campo, suelo, vegetación, varias hojas, fondos lab, guantes y
  tarjetas.

No existe solapamiento SHA-256 entre esas 240 imágenes y los tres splits del
segmentador. Respecto a los splits baseline del clasificador: 112 pertenecen a
train, 20 a val, 14 a test y 94 no entraron al cap baseline. Es una prueba de
generalización visual del segmentador, no un test oficial del clasificador.

## Métricas disponibles

### Última época de `val`

| Métrica | Bbox | Máscara |
|---|---:|---:|
| Precision | 0.9829 | 0.9773 |
| Recall | 0.9448 | 0.9393 |
| mAP50 | 0.9662 | 0.9617 |
| mAP50-95 | 0.9298 | 0.9279 |

Máximos observados para máscara:

- mAP50: `0.9708`, época 80;
- mAP50-95: `0.9381`, época 87;
- la curva F1 de máscara alcanza aproximadamente `0.96` a confidence `0.713`
  en el dominio de `val`.

### Interpretación para Doctor Maíz

- Precision alta: pocas instancias extra en el dominio de validación. No evita
  los falsos positivos externos observados sobre tarjetas y guantes.
- Recall alto: se detectan la mayoría de instancias anotadas. No mide cuántos
  píxeles sintomáticos se recortan.
- mAP50 alto: las máscaras suelen coincidir a un IoU permisivo.
- mAP50-95 alto: el contorno continúa funcionando con criterios más estrictos.
- Ninguna de esas métricas decide si se seleccionó la hoja que el usuario
  pretendía diagnosticar.

Para este proyecto, la métrica crítica es recall de píxel de la hoja objetivo:
perder una lesión, borde seco o zona clorótica es irreversible. Una máscara
algo amplia sólo conserva parte del problema de fondo ya presente en
`baseline_full`.

### Prueba externa de 240 imágenes

| Grupo | Imágenes | Con alguna detección | Sin detección | Multiinstancia |
|---|---:|---:|---:|---:|
| Total | 240 | 234 (`97.5 %`) | 6 | 30 |
| Lab | 60 | 60 (`100 %`) | 0 | 3 |
| Real | 180 | 174 (`96.7 %`) | 6 | 27 |
| `fall_armyworm` | 20 | 14 (`70 %`) | 6 | 4 |

Aplicar retrospectivamente el umbral configurado `0.50` dejaría candidato en
222/240 (`92.5 %`) y sólo 9/20 `fall_armyworm`. No debe cambiarse el umbral con
este cálculo: no hay verdad de referencia para saber si las detecciones bajas
son correctas. De hecho, se observó una máscara lab útil a confidence `0.12` y
falsos positivos de objetos a `0.11–0.13`.

## Resultados visuales

La revisión cubrió el contact sheet de los 240 overlays y comparaciones a
resolución original de casos límite.

### Tipo A — Segmentación correcta

Es el patrón mayoritario. Hojas únicas de lab y campo conservan forma, lesiones
y bordes con poco fondo. También hay buenos resultados con sombras, fondos de
vegetación e iluminación variable.

### Tipo B — Sobresegmentación o falso positivo

En `gray_leaf_spot_maize_field_real_78936724.jpg` el modelo detecta las dos
partes de la hoja con confidence alta, pero también tarjetas verdes y guantes
con confidence `0.11–0.13`. Un umbral elimina estos falsos positivos, aunque no
resuelve la calibración externa.

### Tipo C — Subsegmentación

Los overlays no permiten medir tejido perdido por píxel. Visualmente, hojas
rasgadas, compuestas o superpuestas pueden dividirse en instancias separadas.
Si el pipeline conservara sólo una, la salida quedaría subsegmentada aunque
cada máscara individual sea geométricamente razonable.

### Tipo D — Hoja incorrecta seleccionada

No era evaluable: la prueba dibuja todas las instancias y no implementaba
selección. Ésta es una limitación crítica, no evidencia de que la selección
funcione.

### Tipo E — Varias hojas detectadas

Ocurre en 30 imágenes, con hasta cuatro detecciones. En campo la hoja
diagnóstica suele ser la mayor y más centrada, pero existen hojas secundarias
con confidence similar.

### Tipo F — No detección

Los seis casos son `fall_armyworm`:

- `fall_armyworm_corn_leaf_roboflow_real_50022046.jpg`;
- `fall_armyworm_maize_africa_real_89203456.jpg`;
- `fall_armyworm_maize_africa_real_94767583.jpg`;
- `fall_armyworm_multi_desease_real_30018448.jpg`;
- `fall_armyworm_multi_desease_real_32286052.jpg`;
- `fall_armyworm_multi_desease_real_75510763.jpg`.

Predominan cogollos, tejido muy destruido, oclusión, primeros planos y regiones
que ya no presentan la silueta de una lámina individual completa.

### Tipo G — Máscara fragmentada

Una hoja extendida en lab puede quedar dividida en dos instancias de alta
confidence. No se aplicará unión morfológica: unir instancias cercanas también
podría fusionar hojas distintas.

## Errores detectados

1. Test oficial sin resumen aprobado.
2. Prueba externa no reproducible completamente.
3. Ausencia de máscaras/polígonos crudos para calcular métricas de píxel.
4. Seis no detecciones concentradas en una morfología concreta.
5. Multihoja externa más frecuente que en train.
6. Falsos positivos de objetos planos/coloridos a baja confidence.
7. Fragmentación de hojas rasgadas o presentadas en varias piezas.
8. No había selector de hoja objetivo.
9. No había fallback estructurado.
10. El checkpoint local no conserva el resumen de entrenamiento que identificaría
    directamente la mejor época.

## Posibles causas

- Brecha de dominio: el modelo aprendió principalmente “hoja completa”, no
  cogollo o tejido destruido.
- Pocos ejemplos multihoja: `3.8 %` del dataset frente a `12.5 %` de la prueba
  externa con múltiples predicciones.
- Test interno sin hojas pequeñas: área media mínima `0.1115`.
- Dos fuentes de entrenamiento y un dominio Doctor Maíz mucho más heterogéneo.
- Confidence calibrada en `val` no transferible directamente a campo/lab
  externo.
- La definición de instancia separa hojas/piezas; el objetivo downstream puede
  requerir una identidad diagnóstica adicional.

## Mejoras recomendadas

- Resolver primero el contrato de 183/182 sin alterar test ni usarlo para
  tunear.
- Anotar una muestra externa estratificada con máscara objetivo e identidad de
  hoja seleccionada.
- Medir recall de píxel, IoU, Dice, subsegmentación, tejido recortado,
  fallbacks y selección correcta.
- Ampliar cogollos, daño severo, hojas ocluidas y multihoja antes de probar un
  modelo mayor.
- Añadir hard negatives anotados: guantes, tarjetas, suelo y hojas de otras
  especies.
- Ajustar un factor por experimento usando `val` externo o un nuevo conjunto de
  desarrollo; no el test congelado.

## Riesgos

- pérdida de síntoma al seleccionar o recortar;
- selección de hoja secundaria;
- domain shift por fondo negro;
- uso accidental del perfil experimental con checkpoints históricos;
- artefactos negros correlacionados con clase/fuente;
- confianza alta no equivale a identidad correcta;
- caché de máscaras obsoleta tras cambiar checkpoint o config;
- licencia AGPL de Ultralytics y atribución CC BY 4.0 de datos.

## ¿Está listo para integrarse con el clasificador?

**Parcialmente.** Está listo para una integración opt-in, debug y generación
controlada de candidatos. No está listo para activación automática ni para
alimentar por defecto los clasificadores existentes.

# Diseño — Segmentación previa a clasificación

## Problema

El clasificador puede utilizar suelo, cielo, manos, plantas, tarjetas u otras
hojas como atajos. Un bbox reduce área, pero conserva fondo rectangular. Una
máscara permite conservar sólo los píxeles de la hoja.

## Objetivo

Producir una imagen RGB trazable donde:

```text
mask == hoja  → píxel RGB original
mask == fondo → RGB(0, 0, 0)
```

sin tocar la fuente y sin activar el cambio en producción.

## Flujo actual

```text
load_and_normalize_image
→ resize directo
→ tensor/normalización
→ clasificador
```

## Flujo propuesto

```text
load_and_normalize_image
→ UltralyticsLeafSegmenter
→ máscaras binarias a resolución original
→ validación de candidatos
→ selección de hoja objetivo
→ mask_black / crop / letterbox configurable
→ transform histórico del clasificador
→ clasificador
```

`load_and_normalize_image()` sigue siendo el único punto de entrada desde
archivo. El segmentador recibe la imagen RGB ya corregida por EXIF.

## Selección de hoja objetivo

Primero se aplican gates:

- clase `maize_leaf`;
- confidence mínima;
- máscara válida, no vacía y con resolución exacta;
- área mínima del `1 %`;
- máscara no exactamente igual al `100 %`.

Después:

```text
score = 0.45 × área relativa al candidato mayor
      + 0.35 × proximidad del centroide al centro
      + 0.20 × confidence
```

La fórmula surge de los overlays:

- la hoja diagnóstica de campo suele ser la mayor;
- el centro ayuda contra hojas periféricas;
- confidence pesa menos porque hubo una hoja correcta a `0.12` y objetos
  incorrectos a `0.11–0.13`;
- no se penaliza tocar borde: existen muchos casos válidos así.

Los pesos son configurables y cada candidato conserva sus componentes y razón
de rechazo. La fórmula no resuelve hojas divididas en piezas; esos casos deben
marcarse en la evaluación de selección.

## Generación de máscara

Se rasteriza `Ultralytics Masks.xy`, cuyas coordenadas están expresadas en la
imagen original, directamente con Pillow. No se redimensiona `masks.data` ni se
usa interpolación. Se rechazan polígonos fuera de frame, no finitos o con menos
de tres puntos.

La máscara final es `L`, exactamente `(width, height)`, y sólo contiene
`0/255`. `apply_leaf_mask()` también acepta `bool`, `0/1` y `0/255`; rechaza
valores intermedios que delatarían una interpolación bilinear accidental.

## Fondo negro

La salida es una nueva imagen `RGB` uint8. Los píxeles de hoja se copian y el
resto recibe `(0, 0, 0)` exacto. La imagen fuente nunca se modifica.

## Crop / Letterbox

| Estrategia | Cambio | Uso recomendado |
|---|---|---|
| A `baseline_full` | ninguno | única segura para checkpoints históricos |
| B `mask_black` | elimina fondo, conserva frame/escala | primer experimento controlado futuro |
| C `bbox_crop` | cambia contexto y escala, conserva fondo rectangular | baseline de ablación |
| D `crop_mask_black` | elimina fondo y cambia escala | segundo experimento |
| E `crop_mask_letterbox` | D + aspecto preservado + padding | experimento posterior, entrenado consistentemente |

La estrategia B aísla una sola variable. D/E no deben ser el primer ensayo
porque mezclan fondo, crop, escala, padding y relación de aspecto.

## Manejo de múltiples hojas

- Se conservan todas las trazas.
- Se elige una por score explícito.
- Se emite warning cuando hay más de una elegible.
- No se unen máscaras automáticamente.
- Una evaluación futura debe anotar `selected_instance` esperado.

## Manejo de errores

| Caso | Política |
|---|---|
| Sin detección | warning + fallback original |
| Baja confidence | candidato rechazado; fallback si no queda otro |
| Múltiples hojas | score + warning + metadata completa |
| Área menor a 1 % | candidato rechazado |
| Área ≥98 % pero <100 % | aceptar con warning |
| Área exactamente 100 % | considerar degenerada y fallback |
| Máscara vacía/corrupta/desalineada | rechazar sin resize silencioso |
| Fallback `reject` explícito | no entregar imagen al clasificador |

Se acepta una máscara casi completa porque el dataset contiene anotaciones
válidas de hasta `99.72 %`. El umbral de área no basta por sí solo para declarar
un error.

## Refinamiento de máscara

No se implementa apertura, cierre, relleno de agujeros ni componente principal.
Los tests actuales no prueban que esas operaciones resuelvan un problema sin
arriesgar bordes, puntas o lesiones. La selección de instancia ya evita mezclar
componentes de hojas distintas.

## Riesgo de domain shift

El diagnóstico ROI histórico ya aporta evidencia:

| Modelo | Macro-F1 full | Macro-F1 ROI sólo en inferencia |
|---|---:|---:|
| EfficientNet-B0 | 0.8827 | 0.8561 |
| ShuffleNetV2-x1.0 | 0.9064 | 0.7582 |
| EfficientNet-Lite0 | 0.9052 | 0.6101 |

Por tanto:

```text
train original → inferencia enmascarada
```

no es una ruta segura. La comparación válida es:

```text
train original → test original
train mask_black → test mask_black
train crop_mask_letterbox → test crop_mask_letterbox
```

Si se materializa un dataset, debe vivir bajo `data/` como derivado versionado,
nunca en `clean/`, y registrar fuente, hash de checkpoint, hash de config,
perfil, máscara, bbox y fallback.

## Integración con clasificador

`predict.py` conserva `--leaf-profile baseline_full` por defecto. Los perfiles
de máscara son opt-in, muestran un warning de domain shift y pueden guardar un
bundle de debug. No se conectó el segmentador a `CornDataset` ni al entrenamiento
porque todavía no existe un clasificador entrenado consistentemente.

## Debug y trazabilidad

Por imagen se puede guardar:

```text
original.jpg
mask.png
overlay.jpg
masked_black.png
crop.png
comparison.jpg
metadata.json
```

`masked_black.png` usa PNG deliberadamente: JPEG convertiría el negro exacto en
ringing no cero. El bundle no sobrescribe un directorio anterior.

Metadata incluida:

- ruta fuente;
- modelo, checkpoint y SHA-256;
- versión Ultralytics esperada/efectiva;
- `imgsz`, confidence, IoU, NMS y máximo de instancias;
- cantidad de instancias;
- instancia seleccionada;
- confidence, bbox y área de máscara;
- score y componentes por candidato;
- estrategia, fallback, warnings y versión del procesador.

## Tests

Se cubren:

1. hoja claramente visible;
2. múltiples hojas;
3. fondo complejo;
4. hoja parcial tocando borde;
5. hoja pequeña;
6. sin hoja;
7. máscara vacía/corrupta;
8. máscara exacta y casi completa;
9. resoluciones distintas;
10. horizontal/vertical;
11. baja confidence;
12. crop y letterbox;
13. fallback reject;
14. bundle debug y no overwrite;
15. conversión de una salida Ultralytics simulada.

También se verifica negro exacto, conservación de píxeles, alineación, bbox
half-open, dtype uint8 y no mutación.

## Criterios de aceptación

- [x] interfaz de segmentador para una imagen;
- [x] máscara binaria a resolución original;
- [x] selector explícito multihoja;
- [x] fondo negro exacto;
- [x] salida consumible por el transform del clasificador;
- [x] fuente inmutable;
- [x] debug visual y metadata;
- [x] fallbacks;
- [x] perfiles A/B desacoplados;
- [x] domain shift documentado;
- [x] tests unitarios normales/límite;
- [x] smoke local con el checkpoint real: completado en la validación real de la
  Parte 2, con Ultralytics 8.4.104 y seis máscaras verificadas más un fallback;
- [ ] test interno oficial aprobado;
- [ ] IoU/Dice/recall de píxel externo;
- [ ] experimento de clasificadores reentrenados.

# Plan de implementación

| Archivo | Acción | Justificación | Riesgo |
|---|---|---|---|
| `src/segmentation/leaf_segmenter.py` | crear | adaptador YOLO y polígonos→máscara | medio |
| `src/preprocessing/leaf_mask.py` | crear | máscara binaria y fondo exacto | bajo |
| `src/preprocessing/segmented_leaf_processor.py` | crear | selección, perfiles, fallback y debug | medio |
| `scripts/pipeline/preprocess_leaf.py` | crear | inspección individual reproducible | bajo |
| `scripts/pipeline/predict.py` | modificar | activación opt-in para inferencia | medio |
| `config/dataset.yaml` | modificar | configuración centralizada | bajo |
| `pyproject.toml` | modificar | dependencia opcional fijada | bajo |
| tests de preprocessing/segmentation | crear | normal y casos límite | bajo |
| este documento | crear | auditoría, diseño y backlog | bajo |

# Implementación

## `src/segmentation/leaf_segmenter.py`

- Cambio: `LeafInstance`, protocolo `LeafSegmenter`, rasterización y wrapper
  `UltralyticsLeafSegmenter`.
- Motivo: desacoplar YOLO del clasificador y mantener máscaras alineadas.
- Cómo probar: `pytest tests/segmentation/test_leaf_segmenter.py`.

## `src/preprocessing/leaf_mask.py`

- Cambio: validación binaria, bbox, centroide, área y `apply_leaf_mask()`.
- Motivo: una sola implementación exacta y reusable.
- Cómo probar: `pytest tests/preprocessing/test_leaf_mask.py`.

## `src/preprocessing/segmented_leaf_processor.py`

- Cambio: selector, perfiles, fallback, metadata, overlay y bundle debug.
- Motivo: interfaz de alto nivel `processor.process(image)`.
- Cómo probar:
  `pytest tests/preprocessing/test_segmented_leaf_processor.py`.

## `scripts/pipeline/preprocess_leaf.py`

- Cambio: CLI para una imagen y bundle visual.
- Motivo: auditoría manual sin tocar dataset.
- Cómo probar:

```bash
python scripts/pipeline/preprocess_leaf.py \
  --image /ruta/hoja.jpg \
  --profile mask_black
```

## `scripts/pipeline/predict.py`

- Cambio: `--leaf-profile`, checkpoint, device y debug opcionales.
- Motivo: permitir A/B sin cambiar default.
- Cómo probar: primero `--leaf-profile baseline_full`; usar máscara sólo como
  diagnóstico hasta reentrenar.

## Configuración y dependencia

- Cambio: subconfig `leaf_detection.segmentation`, `enabled=false` intacto.
- Cambio: extra opcional `segmentation = ["ultralytics==8.4.104"]`.
- Cómo probar: `pip install -e '.[segmentation]'` en un entorno compatible y
  ejecutar el CLI individual. La instalación no se ejecutó automáticamente.

# ¿Dónde podemos mejorar el segmentador?

## 1. Problemas encontrados en los tests

### CRÍTICA

- Falta test oficial aprobado y máscaras externas con verdad de referencia.
- Falta medir selección correcta en 30 casos multihoja.

### ALTA

- 6/20 no detecciones en morfología `fall_armyworm` severamente dañada.
- Fragmentación de hojas rasgadas o presentadas en piezas.
- Falsos positivos sobre guantes/tarjetas a baja confidence.

### MEDIA

- Confidence externa no calibrada.
- Prueba de 240 imágenes sin script/seed/config completa.

## 2. Problemas relacionados con dataset

### CRÍTICA

- Ningún conjunto externo anotado representa directamente el target downstream.

### ALTA

- Sólo 44 imágenes multiinstancia.
- Test interno sin hojas pequeñas.
- Poca cobertura explícita de cogollo, tejido destruido y oclusión severa.

### MEDIA

- Dos fuentes dominan el entrenamiento; falta dominio de cámara Doctor Maíz.
- Debe estratificarse iluminación, escala, fondo y porcentaje visible.

## 3. Problemas relacionados con anotaciones

### ALTA

- La semántica “todas las hojas visibles” del segmentador debe separarse de
  “hoja que el usuario quiere diagnosticar”. Se necesita un `target_leaf_id`.
- Tres casos siguen en cola de reanotación.

### MEDIA

- Revisar criterios en solapamientos, hojas rasgadas y límites secos.
- Auditar si se incluye tallo/vaina y mantener un único criterio.

No se recomienda unir instancias o cerrar agujeros hasta contar errores
etiquetados que lo justifiquen.

## 4. Problemas relacionados con entrenamiento

### ALTA

- La limitación principal parece ser cobertura de dominio, no capacidad: las
  curvas de train/val convergen con métricas altas. Priorizar datos antes de
  `yolo26s-seg`.

### MEDIA

- Tras ampliar datos, probar una sola ablación por corrida: rotación moderada,
  copy-paste multihoja o escala; elegir sólo con val.
- Ejecutar varias semillas antes de congelar configuración.
- Conservar `training_summary.json` y best epoch en el backup local.

### BAJA

- Modelo mayor sólo si aparece subajuste verificable tras corregir datos.

## 5. Problemas relacionados con inferencia

### CRÍTICA

- Preservar siempre config y predicciones crudas; el test externo actual no lo
  hizo.

### ALTA

- Calibrar confidence contra recall de píxel y fallback externo, no sólo F1 de
  `val`.
- Evaluar score de selección contra identidad objetivo anotada.

### MEDIA

- Medir sensibilidad a `imgsz` en hojas pequeñas.
- Evaluar IoU/NMS en solapamientos sin cambiar ambos factores a la vez.

## 6. Casos que todavía fallan

- cogollo en primer plano;
- hoja destruida por insecto sin contorno completo;
- hoja parcialmente oculta por otras hojas/tallo;
- hojas superpuestas con confidence similar;
- una hoja diagnóstica dividida en piezas;
- objetos de laboratorio verdes/blancos;
- hoja muy pequeña, no cubierta por el test interno;
- selección ambigua cuando la mayor hoja no es la señalada por el usuario.

## 7. Qué imágenes sería conveniente conseguir

### CRÍTICA

- Los seis patrones de fallo `fall_armyworm`, con máscaras completas y
  `target_leaf_id`.
- Múltiples hojas donde una persona marque explícitamente cuál diagnosticar.
- Pares con todas las hojas segmentadas y una hoja objetivo seleccionada.

### ALTA

- cogollos desde arriba y lateral;
- daño severo, perforaciones y tejido faltante;
- hojas superpuestas/cruzadas;
- hojas parcialmente fuera de cuadro;
- oclusión por tallo, mazorca, mano u otra hoja;
- hoja ocupando `<5 %`, `5–15 %`, `15–50 %` y `>90 %` del frame;
- fondos de suelo, maleza y otras especies;
- guantes, tarjetas, etiquetas y herramientas como hard negatives;
- iluminación solar dura, contraluz, sombra profunda y baja luz;
- desenfoque moderado y movimiento de captura móvil.

### MEDIA

- distintas cámaras/teléfonos, compresión y distancias;
- mañana/mediodía/tarde;
- hoja mojada, polvo, reflejos y manchas no patológicas;
- etapas vegetativas distintas;
- healthy y todas las enfermedades/deficiencias, evitando que fondo o encuadre
  queden correlacionados con clase.

### BAJA

- más hojas únicas centradas sobre fondo uniforme: ya es el caso mejor cubierto.

## 8. Qué nuevas bases de datos serían útiles

### ALTA

Buscar datasets que incluyan simultáneamente:

- maíz real de campo, no sólo hojas recortadas;
- segmentación por instancia de cada hoja;
- multihoja, cogollo y oclusión;
- resolución original y licencia compatible;
- identidad estable entre variantes para evitar fugas;
- anotaciones COCO polygons/RLE o YOLO-seg auditables.

Un dataset de clasificación de enfermedades sin máscaras sólo sirve como fuente
de imágenes para anotar; no mejora directamente el segmentador. Un dataset de
otras especies puede servir para hard negatives, pero no sustituye contornos de
maíz.

### MEDIA

- datasets de segmentación de órganos de plantas con solapamientos;
- datasets agrícolas móviles con iluminación extrema;
- datasets con secuencias de la misma planta, agrupables antes del split.

## 9. Qué datos propios sería conveniente recolectar

### CRÍTICA

- Un set Doctor Maíz de 300–500 fotos nuevas, agrupado por planta/sesión, con
  todas las hojas y `target_leaf_id`.
- Un test externo separado que nunca entre a entrenamiento.

### ALTA

- Campaña dirigida a los seis fallos, no muestreo aleatorio genérico.
- Por cada escena: una toma cercana, media y amplia; horizontal y vertical.
- Metadata no sensible: dispositivo, hora, iluminación, distancia aproximada,
  etapa del cultivo y porcentaje visible.

### MEDIA

- Doble anotación en 10–20 % para medir consistencia de bordes y selección.
- Capturas negativas sin hoja objetivo para evaluar rechazo.

## 10. Prioridad de cada mejora

| Prioridad | Mejora | Evidencia |
|---|---|---|
| CRÍTICA | resolver/evaluar test y crear verdad externa | no existe test summary ni métricas de píxel |
| CRÍTICA | anotar identidad de hoja objetivo | 30 casos multihoja, selector no evaluado |
| ALTA | recolectar cogollo/daño severo | 6/6 no detecciones concentran ese patrón |
| ALTA | aumentar multihoja/oclusiones | 3.8 % train frente a 12.5 % externo |
| ALTA | hard negatives lab/campo | falsos positivos sobre tarjeta/guante |
| ALTA | entrenar clasificador con misma representación | caída ROI histórica |
| MEDIA | calibrar confidence/IoU/imgsz | val y dominio externo divergen |
| MEDIA | reanotación y acuerdo entre anotadores | tres casos pendientes y criterios complejos |
| MEDIA | multi-semilla/ablaciones | separar efecto de datos e hiperparámetros |
| BAJA | aumentar tamaño del modelo | no hay evidencia actual de subajuste |

# Conclusión final

> Con el segmentador actual, ¿es recomendable comenzar a utilizar máscaras para
> alimentar el clasificador o primero debemos mejorar el segmentador?

Es recomendable **comenzar a generar y auditar máscaras en un pipeline
experimental**, y preparar el entrenamiento consistente del clasificador. No es
recomendable alimentar automáticamente los clasificadores históricos ni activar
la ruta en producción.

La evidencia es mixta: `val` es fuerte y 234/240 imágenes externas tienen
detección, pero el test oficial está bloqueado, faltan métricas de píxel, seis
casos morfológicos críticos fallan y no se ha validado la selección multihoja.
Por eso deben avanzar en paralelo dos trabajos: integración opt-in y mejora
dirigida del segmentador/dataset. La activación final requiere primero cerrar
esas evaluaciones y entrenar el clasificador con la misma representación que
recibirá en inferencia.
