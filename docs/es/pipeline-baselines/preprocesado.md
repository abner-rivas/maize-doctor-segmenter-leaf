# Preprocesado

El preprocesado (normalizado, división estratificada, balanceo y data augmentation) es
compartido entre todos los pipelines. Ver la documentación completa en
[Preprocesado](../preprocesado/index.md).

## Especificidad del baseline

El baseline consume el split `outputs/splits/seed_42_baseline/`, generado con
`make splits-baseline`. A diferencia del split completo (`seed_42`), aplica un **cap de 1 500
imágenes por clase** sobre las 9 clases: recorta solo las clases mayoritarias y conserva íntegras
las minoritarias (potasio 266, nitrógeno 523, fósforo 612). Todo lo demás —normalización
ImageNet, estratificación por `label + environment`, seed 42, sampler y augmentation— es idéntico
al pipeline principal.
