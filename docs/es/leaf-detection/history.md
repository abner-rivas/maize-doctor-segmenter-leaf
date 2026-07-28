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
22. **Cierre del dataset padre.** Las decisiones humanas se completaron y el
    pool se reconstruyó con 1 155 imágenes, 1 224 máscaras y fingerprint
    `c087af60c2bad1c133c4ea8b14cee945405bfe4976aa80c4faf089d7a4b9e38c`.
23. **Splits reproducibles.** Se formaron 1 035 grupos por procedencia,
    original, variante Roboflow, SHA-256 y cercanía perceptual. Con semilla 42
    quedaron 809/173/173 imágenes y 858/183/183 máscaras.
24. **Gate de entrenamiento.** Cero fugas exactas, grupales, Roboflow,
    perceptuales o contra el piloto; la reconstrucción doble fue idéntica y
    `split_lock.status=ready_for_training_preflight`.
25. **Evaluación futura.** El segmentador se evaluará contra el piloto retenido.
26. **Clasificación adaptada.** Se entrenarán comparativamente
    `baseline_full`, `baseline_bbox_roi` y `baseline_masked_roi` con
    representaciones consistentes entre entrenamiento e inferencia.
27. **Preflight del segmentador.** Locks, fingerprints, dataset y batch 4/2/2
    pasaron. El entorno tiene PyTorch 2.12.1+cu130, pero no CUDA utilizable,
    pesos locales ni Ultralytics; quedó `blocked_by_missing_dependency` con
    cero entrenamiento y cero descargas.
28. **Paquete cloud-ready.** Se creó un payload determinista con lista blanca,
    bootstrap compatible con el PyTorch remoto, preflight GPU/modelo, smoke y
    entrenamiento autorizados por guards separados, reanudación manual y
    evaluación val/test. `all/`, fuentes, piloto e históricos quedan fuera.

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
- `outputs/leaf_detection/detector_dataset_splits/`;
- `outputs/leaf_detection/training_preflight/`;
- `outputs/leaf_detection/packages/`;
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

Las revisiones manuales, los splits agrupados, el preflight local y el paquete
cloud están completos. El trabajo inmediato, en la máquina remota:

1. ejecutar bootstrap y preflight GPU/modelo (verifica `yolo26n-seg` con
   `ultralytics==8.4.104`, licencia, pesos y forward);
2. ejecutar el smoke de una época y revisar batch, VRAM y velocidad;
3. autorizar y entrenar el baseline del segmentador;
4. evaluar precision, recall, mAP, IoU, Dice, fallbacks y errores de selección;
5. generar variantes full, bbox y máscara;
6. entrenar y comparar clasificadores adaptados;
7. ejecutar análisis por clase, LIME, Grad-CAM, tamaño y latencia.

## Decisiones pendientes

- confirmación remota de soporte y licencia de `yolo26n-seg` en
  `ultralytics==8.4.104`;
- política de `cache` (false frente a disk) tras medir el DataLoader en el
  smoke;
- fondo neutral, margen y selección de máscara principal;
- umbrales de aceptación provisionales y evaluación del piloto (sus cajas usan
  la regla histórica de hoja principal: revisión cualitativa primero);
- decisión sobre la ruta de boxes preparada pero aún no anotada
  (`annotation_batches/`: 350 train + 75 val siguen `pending`);
- tratamiento futuro de los tres casos en `reannotation_queue.csv`.

## Registros de decisión

- [Separación data/outputs](../decisions/adr-project-data-root-and-output-root.md)
- [Resultado del diagnóstico ROI](../decisions/adr-manual-roi-diagnostic-result.md)
- [Estrategia de segmentación](../decisions/adr-leaf-instance-segmentation-strategy.md)
- [Datasets externos](../decisions/adr-external-leaf-segmentation-datasets.md)
