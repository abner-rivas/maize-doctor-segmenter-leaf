# Preprocesado

Antes de que un modelo vea una sola imagen, esta pasa por una serie de pasos comunes a todos los
pipelines del proyecto: normalizado, división estratificada, balanceo y data augmentation. El
baseline no reinventa nada aquí, hereda tal cual este preprocesado compartido. La documentación
completa vive en [Preprocesado](../preprocessed/index.md).

## Especificidad del baseline

Lo único que cambia respecto al pipeline principal es qué subconjunto de datos llega a ver el
baseline, no cómo se procesa cada imagen.

El baseline consume el split `outputs/splits/seed_42_baseline/`, generado con
`make splits-baseline`. A diferencia del split completo (`seed_42`), aplica un **cap de 1 500
imágenes por clase** sobre las 9 clases: recorta solo las clases mayoritarias y conserva íntegras
las minoritarias (potasio 266, nitrógeno 523, fósforo 612). Todo lo demás (normalización
ImageNet, estratificación por `label + environment`, seed 42, sampler y augmentation) es idéntico
al pipeline principal.
