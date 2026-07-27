# Historia del aislamiento de hojas

Este documento reúne la evolución técnica del requerimiento. Distingue
decisiones históricas, configuración activa y trabajo futuro para que los
resultados no se interpreten fuera de contexto.

## Cronología

1. **Problema de fondo.** Se observó el riesgo de que el clasificador atendiera
   suelo, cielo, manos, tallos, sombras, otras hojas, fondos de laboratorio y
   objetos cercanos.
2. **Diseño ROI.** Se implementaron validación de bounding boxes, clipping,
   margen, recorte, letterbox y fallbacks.
3. **Piloto retenido.** Se seleccionaron 100 imágenes reales del test oficial,
   balanceadas y sin duplicados, con semilla 42.
4. **Anotación CVAT.** Se anotó una hoja principal `maize_leaf` por imagen.
5. **Problema del exportador.** El export YOLO produjo 48 TXT y omitió 52 cajas
   rotadas.
6. **Recuperación desde XML.** El XML nativo de CVAT permitió convertir las 52
   cajas rotadas; 36 necesitaron clipping.
7. **Manifiesto ROI.** Se materializaron 100 filas válidas: 99 `annotated` y
   `image_0021` como `ambiguous`, con área `0.092799`.
8. **Diagnóstico histórico.** EfficientNet-B0, ShuffleNetV2-x1.0 y
   EfficientNet-Lite0 se evaluaron con imagen completa y ROI manual. No hubo
   entrenamiento.
9. **Caída observada.** Macro-F1 cambió `0.8827→0.8561`,
   `0.9064→0.7582` y `0.9052→0.6101`, respectivamente.
10. **Interpretación.** El recorte, escala, letterbox, padding y pérdida de
    contexto sólo en inferencia introdujeron cambio de distribución.
11. **Decisión de seguridad.** No se activó ROI en los checkpoints históricos;
    `baseline_full` y `leaf_detection.enabled=false` continuaron vigentes.
12. **Preparación con boxes.** Se seleccionaron 350 imágenes de train y 75 de
    val para posible anotación multihoja; permanecen `pending`.
13. **Búsqueda de segmentación.** Se exploraron máscaras para reducir el fondo
    residual que conserva una caja rectangular.
14. **Fuentes externas.** Se incorporaron, como fuentes inmutables para
    auditoría, datasets de 1,003 y 157 imágenes con respaldos COCO.
15. **EDA reproducible.** El notebook validó clases, sintaxis, geometría,
    duplicados, contraste YOLO/COCO y calidad visual.
16. **Decisión del EDA.** Ambas fuentes quedaron `accepted_with_filtering`; se
    conserva sólo `leaf`.
17. **Corrección metodológica.** El EDA distinguió 11 bbox mezclados de los
    polígonos, invalidó 8 autointersecciones y un vértice repetido, y sustituyó
    la caché débil por fingerprints SHA-256 versionados de 2 428 archivos.
18. **Consolidación controlada.** Se materializaron 1 156 imágenes y 1 226
    máscaras `0 = maize_leaf`; una máscara se recuperó desde COCO y 13 392
    lesiones se excluyeron.
19. **Protección y revisión.** No hubo duplicados exactos ni cruces con las 100
    imágenes del piloto. Quedaron 34 filas para revisión manual. La hoja
    autointersectada no entró al pool y, junto con la recuperación COCO
    extremadamente pequeña, requiere una decisión explícita.
20. **Gate manual.** Ninguna fila tenía una decisión humana completada. El
    `dataset_lock.json` quedó como `blocked_by_manual_review`, con dos casos
    obligatorios y 35 casos únicos pendientes.
21. **Corrección de previews.** El renderer general descartaba la geometría al
    construir cada caso con `polygons=[]`. Las 35 previews se regeneraron desde
    las anotaciones originales: 33 YOLO, una COCO y un TXT realmente vacío
    señalado de forma explícita. La validación quedó
    `ready_for_human_review`.
22. **Próxima fase.** Completar la revisión humana; después reconstruir
    desde las fuentes originales y crear splits
    propios agrupados por original, hash, fuente y variante Roboflow.
23. **Evaluación futura.** El segmentador se evaluará contra el piloto retenido.
24. **Clasificación adaptada.** Se entrenarán comparativamente
    `baseline_full`, `baseline_bbox_roi` y `baseline_masked_roi` con
    representaciones consistentes entre entrenamiento e inferencia.

## Qué validó el piloto

El piloto no demostró una mejora de clasificación. Sí validó el costo y la
trazabilidad de anotación, la importación CVAT, geometría de cajas rotadas,
clipping, margen, letterbox, manifiestos, previews y evaluación pareada.

La caída de los modelos históricos tampoco descartó el aislamiento: demostró
que el cambio de representación posterior al entrenamiento no es seguro.

## Por qué segmentación

Una caja incluye regiones rectangulares ajenas a la hoja. Una máscara puede
seguir su contorno, generar un bbox ajustado y sustituir el fondo de manera
controlada. Esto permitirá separar tres preguntas:

- si quitar fondo ayuda;
- si basta con un bbox derivado de la máscara;
- si el clasificador aprende correctamente una hoja con fondo neutral.

YOLO segmentará hojas; el clasificador continuará diagnosticando.

## Estado de artefactos

### Activos

- `data/splits/seed_42_baseline/`;
- `data/leaf_detection/pilot/`;
- `data/leaf_detection/external_sources/`;
- `data/leaf_detection/detector_dataset/`;
- `outputs/baselines/`;
- `outputs/leaf_detection/external_sources_eda/`;
- `outputs/leaf_detection/detector_dataset_consolidation/`;
- código en `src/preprocessing/`, `src/data/` y `scripts/`.

### Evidencia histórica

- `outputs/leaf_detection/pilot/validation/`;
- `outputs/leaf_detection/pilot/previews/`;
- `outputs/leaf_detection/pilot/diagnostic_experiment/`;
- tres corridas oficiales bajo `outputs/baselines/`;
- `outputs/dataset_audit/`, que conserva el diagnóstico anterior del corpus.

### Deprecados o duplicados, no eliminados

- `outputs/preflight_gpu_check/`: convención anterior;
- `outputs/dataset_audit_updated/`: copia exacta de `dataset_audit_final/`;
- `data/leaf_detection/pilot/packages/pilot_images/`: copia desempaquetada del
  piloto, protegida como parte del paquete;
- `__pycache__/` y `.pyc`: candidatos técnicos a eliminación.

Ninguna de esas rutas se movió o borró automáticamente. El inventario está en
`outputs/repository_audit/`.

## Pipeline activo

```text
imagen completa
→ transformación histórica
→ clasificador baseline_full
```

La configuración activa es:

```yaml
processing_profile: baseline_full

leaf_detection:
  enabled: false
```

## Pipeline previsto

```text
imagen
→ segmentador de hoja
→ selección de máscara principal
→ baseline_bbox_roi o baseline_masked_roi
→ clasificador entrenado con la misma representación
```

## Próximo experimento

El trabajo inmediato no es entrenar: es completar las 34 revisiones manuales y
aprobar los previews del consolidado, incluidos los dos casos obligatorios de
`mandatory_visual_review.csv`. Después:

1. crear splits del segmentador agrupando hash, fuente, secuencia, nombre base
   y `roboflow_variant_group`;
2. confirmar arquitectura, licencia, versión, GPU y exportabilidad;
3. entrenar el segmentador;
4. evaluar precision, recall, mAP, IoU, Dice, fallbacks y errores de selección;
5. generar variantes full, bbox y máscara;
6. entrenar y comparar clasificadores adaptados;
7. ejecutar análisis por clase, LIME, Grad-CAM, tamaño y latencia.

## Decisiones pendientes

- aprobación de los 32 casos estratificados;
- tratamiento final de la imagen vacía de `corn`;
- revisión de la advertencia topológica incluida como candidata;
- reglas finales de split por grupos de variantes;
- modelo y licencia del segmentador;
- fondo neutral, margen y selección de máscara;
- umbrales de aceptación en el piloto;
- decisión sobre la ruta de boxes preparada pero aún no anotada.

## Registros de decisión

- [Separación data/outputs](../decisions/adr-project-data-root-and-output-root.md)
- [Resultado del diagnóstico ROI](../decisions/adr-manual-roi-diagnostic-result.md)
- [Estrategia de segmentación](../decisions/adr-leaf-instance-segmentation-strategy.md)
- [Datasets externos](../decisions/adr-external-leaf-segmentation-datasets.md)
