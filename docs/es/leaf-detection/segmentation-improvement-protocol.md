# Protocolo de mejora del segmentador

Este protocolo convierte los hallazgos del baseline en experimentos reproducibles.
No reutiliza `test` para seleccionar hiperparámetros y no incorpora el piloto externo
retenido al entrenamiento.

## Cambios ya calibrados

- inferencia: propuestas `0.20`, selección `0.50`;
- en `val`, los umbrales de selección `0.20`, `0.30`, `0.40` y `0.50`
  produjeron el mismo recall de píxel (`0.97987`) y Dice (`0.97636`) en las
  166 imágenes de una hoja; `0.60` empeoró ambos;
- gate geométrico: `large_mask_area_ratio=0.25` y
  `min_large_mask_bbox_ratio=0.80`;
- en la auditoría humana congelada de 42 imágenes, el gate calibrado conserva
  las 24 máscaras `GOOD` y reduce falsos confiables de 1 a 0. Es una estimación
  sobre la misma muestra usada para calibrar y debe confirmarse con datos nuevos.

Reproducción:

```bash
make leaf-segmentation-calibrate-quality-gate
make leaf-segmentation-calibrate-selection
```

## Ablaciones

Los perfiles están en `cloud_training/configs/experiments/`. El orden recomendado
es `D-01`, `D-02`, `D-02B`, `D-03`, `D-06`, `D-05` y, sólo con pesos verificados,
`D-04`. Cada perfil conserva el split y genera su propio manifiesto y resumen.

Ejemplo, después del smoke cloud y con autorización explícita de coste:

```bash
CONFIG=cloud_training/configs/experiments/d01_mosaic0_seed42.yaml \
CONFIRM_SEGMENTATION_TRAINING=1 make leaf-segmentation-cloud-train
```

`D-02B` y `D-04` requieren un smoke específico para confirmar el batch. `D-04`
también requiere registrar un archivo suministrado y verificable llamado
`yolo26s-seg.pt`; el repositorio no lo descarga implícitamente.

Después de elegir el ganador únicamente con `val`, se repite con semillas 7,
42 y 1337. El baseline existente cubre la semilla 42; los perfiles E-01 cubren
7 y 1337. Si gana otra ablación, se copian sus hiperparámetros a esos dos
perfiles antes de ejecutarlos. Se reportan media y desviación estándar; la mejora
debe superar la variación entre semillas.

## Datos difíciles nuevos

La siguiente ronda requiere 300 imágenes reales nuevas, con etiquetas que pueden
solaparse:

- al menos 80 con daño severo de cogollero;
- al menos 50 con oclusión;
- al menos 50 con hoja parcial tocando el borde;
- al menos 50 con hoja pequeña;
- al menos 50 con fondo complejo;
- al menos 30 con varias hojas.

Se usa `scripts/experiments/manifests/hard_example_intake_template.csv`. Cada imagen
debe tener procedencia/licencia o consentimiento, hash, sesión de captura y una
máscara YOLO-seg revisada por una segunda persona. No se aceptan máscaras generadas
como verdad de referencia. Las sesiones de captura se separan por grupo antes de
asignar 80% a train y 20% a val; ningún archivo o duplicado perceptual puede entrar
desde el test interno o el piloto externo.

La anotación incluye todas las hojas de maíz diagnósticamente visibles. Los casos
ambiguos se marcan para revisión y no se fuerzan. Tras integrar y congelar la nueva
versión del dataset, se repiten baseline y ganador; sólo se promueve un modelo si
mejora recall/under-segmentation en los subgrupos difíciles sin degradar la fuente
grande ni aumentar falsos confiables.

## Evaluación final

El runner guarda etiquetas YOLO-seg de todas las imágenes evaluadas y genera
automáticamente métricas downstream globales y por fuente, resolución, contacto
con borde y cantidad de hojas. El test interno se ejecuta una sola vez después de
congelar configuración y semillas. El piloto externo queda como auditoría final
cualitativa y nunca como fuente de entrenamiento.
