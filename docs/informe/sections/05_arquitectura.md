# Arquitectura del sistema

El diseño se organiza en capas que siguen el ciclo de vida de una máscara. `src/data` audita, consolida, revisa y divide el dataset; `src/training` valida locks, entorno y carga; `cloud_training` y `modal_training.py` controlan la ejecución remota; `src/segmentation` adapta Ultralytics y decide confiabilidad; `src/preprocessing` convierte máscaras en salidas; `src/evaluation` calcula métricas downstream; y `src/inference` ejecuta y exporta el corpus completo. Los scripts bajo `scripts/` son fachadas CLI, mientras que el `Makefile` expone objetivos con guards.

![El flujo separa preparación de datos, entrenamiento, inferencia, selección y publicación de la salida.](figures/pipeline_architecture.png)

## Adaptador de inferencia

`UltralyticsLeafSegmenter` valida que el checkpoint exista, calcula su SHA-256 y carga el modelo de forma perezosa. Antes de crear `YOLO`, exige que la versión de runtime sea exactamente `8.4.104`; una diferencia produce un error explícito. Cada imagen se normaliza a RGB y se envía con `imgsz=640`, confianza de propuesta 0.20, IoU NMS 0.70, máximo 20 detecciones, NMS por clase y `retina_masks=True`. El wrapper comprueba además que Ultralytics devuelva un único resultado y que su forma original coincida con la imagen.

El checkpoint operativo C-01 pesa 6,546,902 bytes, tiene SHA-256 `4f66456d05d87f9e7080155eb5cd80c583f34849415ec820c950bd97f9c5ec6f` y, cargado con el runtime fijado, declara tarea `segment`, escala `n`, una clase `maize_leaf` y 3,053,055 parámetros. La familia YOLO26 soporta cabezas para segmentación de instancias [@jocher2026yolo26], pero el contrato real del proyecto está definido por ese archivo y su configuración, no por métricas genéricas de COCO.

## Procesador y salida

`SegmentedLeafProcessor.process` invoca el segmentador, selecciona una instancia, genera máscara binaria a resolución original, aplica fondo negro y calcula el bbox. Admite cuatro perfiles: máscara a tamaño original, recorte rectangular, recorte con fondo negro y recorte con letterbox. El perfil vigente es `mask_black`; por lo tanto, conserva las dimensiones de entrada y neutraliza el exterior de la hoja con `[0,0,0]`.

Si no existe una instancia elegible, `_fallback` devuelve la imagen original y registra la razón. Cuando hay una máscara, el resultado contiene original, máscara, imagen enmascarada, crop, bbox, confianza, cantidad de instancias, índice elegido, trazas de score, estrategia, warnings, procedencia y metadata del segmentador. Esta estructura permite que el quality gate decida después sin repetir inferencia.

## Ejecución batch

`run_full_dataset` impide combinar modos incompatibles, no sobrescribe una salida existente y sólo reanuda un run sin `summary.json`. Al reanudar, compara dataset, inventario y procedencia con el run previo. Cada imagen se procesa dentro de un bloque de aislamiento: una excepción crea un archivo de error y no detiene el corpus. El pipeline guarda original enlazado, visualización de propuestas, máscara, overlay, imagen segmentada o fallback, thumbnails y un JSON por muestra. Al final genera manifiestos CSV/JSONL, resumen, galería y metadata de ejecución.

::: evidence Separación de responsabilidades
La decisión `reliable`, `uncertain` o `failed` no la toma directamente YOLO. Surge después de convertir propuestas, aplicar el selector y evaluar geometría y ambigüedad. Esto hace posible auditar por separado modelo, política de selección y política de publicación.
:::

