# Evaluación downstream con ground truth

Las métricas downstream comparan la unión de polígonos verdaderos con la máscara seleccionada, rasterizadas a una resolución común. Para cada imagen se calcula intersección, unión, área real, área predicha y contacto de la máscara verdadera con los bordes. IoU divide intersección entre unión; Dice pondera dos veces la intersección sobre la suma de áreas [@dice1945].

El criterio prioritario es `leaf_pixel_recall`: fracción del tejido foliar real conservado por la máscara. Su complemento es `under_segmentation_ratio`. La precisión de píxel mide qué proporción de la máscara predicha pertenece realmente a la hoja; el exceso se expresa también como `over_segmentation_ratio`, relativo al área real. Esta asimetría responde al uso downstream: quitar hoja puede borrar lesiones, mientras añadir algo de fondo suele preservar la señal foliar.

La evaluación D-01 seleccionó las 150 imágenes de validación procedentes de `corn_leaf_diseases_classification`. Ejecutó `UltralyticsLeafSegmenter` y `SegmentedLeafProcessor`, no el validador batch genérico. Se usaron CPU, `imgsz=640`, máscaras a resolución original, propuesta 0.20, selección 0.50, IoU NMS 0.70, máximo 20 propuestas, área mínima 0.01, `mask_black` y el gate activo.

Table: Resultado end-to-end de D-01 en 150 imágenes de validación.

| Métrica | Resultado |
|---|---:|
| `reliable` / `uncertain` / `failed` | 150 / 0 / 0 |
| Fallbacks | 0 |
| IoU medio | 0.98122 |
| Dice medio | 0.99046 |
| Recall medio de píxel foliar | 0.99375 |
| Precisión media de píxel foliar | 0.98731 |
| Subsegmentación media | 0.00625 |
| Sobresegmentación media | 0.01288 |
| Tejido recortado medio | 0.62% |
| Peor tejido recortado | 11.41% |
| Duración CPU | 131.85 s |

![Las métricas downstream de D-01 son altas en las 150 imágenes de validación evaluadas.](figures/downstream_metrics.png)

Los seis peores IoU se revisaron visualmente y mantuvieron una selección coherente. El caso más débil era horizontal, de 900×600, con fondo complejo: confianza 0.90661, recall 0.88589 e IoU 0.87546. Su 11.41% de tejido perdido explica por qué la media alta no elimina la necesidad de examinar colas.

Esta prueba aporta evidencia con ground truth, pero tiene tres límites. Proviene de validación y pudo influir en el desarrollo; sólo representa 150 imágenes de la fuente grande; y usa D-01 con una semilla. No sustituye el test interno ni el piloto externo. Además, la fuente pequeña `corn`, caracterizada por bordes frecuentes, no está representada en esta evaluación específica.

::: risk No es evaluación diagnóstica
Las 150 imágenes contienen hojas enfermas, pero el experimento sólo mide máscaras. No predice la enfermedad ni demuestra que el clasificador mejore después de segmentar.
:::

