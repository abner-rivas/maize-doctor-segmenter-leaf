# Historia técnica del segmentador

Este documento resume únicamente la evolución del proyecto de segmentación de hojas.

1. Se preparó un piloto retenido de 100 imágenes y se anotó una hoja principal por
   imagen en CVAT.
2. El export YOLO omitió cajas rotadas; el XML nativo permitió recuperar 52 y validar
   clipping, margen, recorte, letterbox, manifiestos y previews.
3. Se exploraron dos fuentes externas con anotaciones YOLO/COCO: 1 003 y 157 imágenes.
4. El EDA separó máscaras de hoja de etiquetas de lesión, detectó bbox mezclados,
   autointersecciones, vértices repetidos y sustituyó cachés débiles por fingerprints
   SHA-256.
5. La consolidación materializó 1 155 imágenes y 1 224 máscaras de clase única
   `0 = maize_leaf`, después de revisión humana y normalización JPEG trazable.
6. Se formaron 1 035 grupos indivisibles por procedencia, variante, SHA-256 y cercanía
   perceptual. Los splits finales son 809/173/173 imágenes.
7. El preflight validó locks, fingerprints, geometría, carga por batches y configuración
   portable antes de permitir entrenamiento.
8. Se construyó un paquete cloud determinista con guards independientes para bootstrap,
   smoke, entrenamiento, reanudación y test.
9. Se entrenó el baseline `yolo26n-seg`; la evaluación final quedó bloqueada al detectar
   que Ultralytics deduplicaba dos polígonos con la misma clase y bbox.
10. La inferencia se complementó con selección determinista de instancia, generación de
    máscara, perfiles de salida y un quality gate auditable.
11. Se construyó la release cloud v7 y se ejecutó `D-01` en una A10 con
    `mosaic=0.0`; la parada temprana seleccionó la época 115 y el checkpoint
    alcanzó Mask mAP50-95 `0.94404` en `val`.
12. El checkpoint D-01 se descargó y verificó por SHA-256. El pipeline real se
    evaluó sobre las 150 imágenes de `val` procedentes del dataset de enfermedades:
    150 `reliable`, IoU medio `0.98122`, Dice `0.99046` y cero fallbacks.
13. Se retiraron dos evaluaciones diagnósticas intermedias, predicciones antiguas
    del clasificador, outputs antiguos del piloto, previews/reportes de
    preparación del dataset y el paquete cloud v6 reemplazado. Se dejó una única
    evaluación end-to-end y el paquete v7 como evidencia actual. `test` y el
    piloto externo permanecen sin consumir.

El piloto externo nunca participa en train, val ni test interno. Las fuentes, el padre
consolidado, los locks y los artefactos de entrenamiento mantienen responsabilidades
separadas y verificables.
