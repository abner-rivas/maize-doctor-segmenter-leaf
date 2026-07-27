# ADR: segmentación de hoja como siguiente estrategia

- Estado: aceptada para experimentación
- Fecha: 2026-07-27
- Alcance: aislamiento de hoja posterior al diagnóstico ROI

## Contexto

Los bounding boxes validaron selección, clipping, margen, recorte y letterbox,
pero una caja rectangular puede conservar suelo, tallos, otras hojas y regiones
vacías, especialmente cuando la hoja es diagonal.

El diagnóstico ROI tampoco fue una comparación de entrenamiento justa: sólo
cambió la representación durante inferencia. Se necesita una estrategia que
permita experimentar con aislamiento preciso y entrenar el clasificador desde
el inicio sobre esa misma representación.

## Decisión

La siguiente línea de investigación será segmentación de instancias de hoja:

```text
imagen
→ segmentador de hoja
→ selección de máscara principal
→ procesamiento estandarizado
→ clasificador entrenado con esa representación
```

La máscara permitirá comparar tres variantes:

1. `baseline_full`: imagen completa;
2. `baseline_bbox_roi`: bounding box derivado de la máscara;
3. `baseline_masked_roi`: hoja segmentada con fondo neutral.

La segmentación no clasifica enfermedades y no reemplaza el clasificador.

## Razones

- sigue el contorno real de la hoja;
- reduce fondo residual frente a una caja;
- permite un bbox más ajustado;
- admite fondo neutral reproducible;
- reduce atajos visuales potenciales;
- permite aislar el efecto de forma, fondo, escala y padding.

## Alternativas consideradas

- Mantener sólo bounding boxes: útil como baseline, pero conserva fondo.
- Aplicar ROI a checkpoints existentes: descartado por cambio de distribución.
- Anotar manualmente las 10,020 imágenes: costo innecesario antes de validar un
  segmentador.
- Usar fuentes externas sin auditoría: rechazado por clases de lesión,
  geometría, duplicados y licencias.

## Riesgos

- máscaras incorrectas o selección de la hoja equivocada;
- pérdida de contexto diagnóstico;
- errores en hojas superpuestas;
- sesgo por fuente;
- fuga entre variantes exportadas por Roboflow;
- dependencia del fondo neutral y la escala;
- costo y licencia de la arquitectura de segmentación elegida.

## Salvaguardas

- piloto de 100 imágenes completamente retenido;
- auditoría YOLO/COCO previa;
- remapeo único `0 = maize_leaf`;
- agrupación por hash y nombre base antes de dividir;
- evaluación del segmentador antes de generar ROI masivos;
- reentrenamiento de clasificadores con representación consistente.

## Estado

No existe todavía un segmentador entrenado. La arquitectura concreta, la
versión de Ultralytics, GPU, exportabilidad y licencia siguen pendientes de
validación.
