# Preprocesado

El baseline no reinventa el preprocesado: hereda tal cual el pipeline compartido del proyecto
(normalizado, división estratificada, balanceo y data augmentation). Toda esa lógica está
documentada en [Preprocesado](../preprocessed/index.md) y aquí no se repite.

Lo que cambia en el baseline es **qué datos llega a ver** y **cómo se balancean en la práctica**,
no cómo se procesa cada imagen individual. Esta página cubre solo esas diferencias.

## Qué subconjunto ve el baseline

El baseline consume un split generado por un comando específico del pipeline. Incluye las mismas 9 clases del dataset completo, pero sobre cada una se aplica un **límite de 1,500 imágenes por clase**.

La clave está en _cuándo_ se aplica ese límite: se recorta sobre el conjunto completo de imágenes
válidas **antes de dividir** en train/val/test. Por eso el tope de 1 500 es un total por clase
(sumando los tres cortes), no un tope por corte. El recorte solo afecta a las clases mayoritarias;
las minoritarias, que ya están por debajo del límite, quedan **íntegras** (potasio ≈ 266,
nitrógeno ≈ 523, fósforo ≈ 612).

Cuando una clase se recorta, el muestreo es **proporcional por entorno**: conserva la mezcla
`lab`/`real` original de esa clase en vez de sesgarla hacia el dominio más abundante.

El propósito es puramente pragmático: el baseline existe para **comparar arquitecturas rápido y
barato**, y un tope por clase reduce el coste de cada corrida sin tocar el dataset real. El límite
es un valor por defecto configurable: se puede subir, bajar, o desactivar por completo para
entrenar con el 100 % de las imágenes disponibles.

## Cómo se balancea

El pipeline compartido describe una estrategia de balanceo de dos capas (sampler + pérdida
ponderada). El baseline usa una versión más simple:

- **El propio cap ya actúa como un undersampling suave** de las clases mayoritarias: al topar a
  1 500 imágenes, reduce su dominio frente a las minoritarias antes de que empiece el
  entrenamiento. Es reversible (se regenera el split) y no borra datos del dataset original.
- **`WeightedRandomSampler`** (la primera capa) se aplica igual: repite las muestras minoritarias dentro de cada epoch, y como la augmentation es en caliente cada repetición se ve distinta.
- **La pérdida es una `CrossEntropyLoss` estándar, sin ponderar por clase.** A diferencia de la
  estrategia completa, el baseline se apoya solo en el cap y en el sampler para compensar el
  desbalance, no en pesos en la función de pérdida. Esto mantiene las corridas simples y
  comparables entre modelos.

## Tamaño de imagen por arquitectura

El baseline entrena varias arquitecturas en la misma corrida, y no todas esperan la misma
resolución de entrada. Cada modelo resuelve su propio tamaño de imagen y el tamaño de batch se
**auto-escala** en función de esa resolución (imágenes más grandes ---> batch más pequeño) para
acotar el uso de memoria. El resize de la augmentation se adapta a ese tamaño; el resto del
pipeline de transformaciones (flips, rotación, color, normalización ImageNet) es idéntico.

## Lo que es idéntico al pipeline principal

Todo lo demás se hereda sin cambios: el normalizado en caliente (corrección EXIF, RGB estricto,
estadísticas de ImageNet), la estratificación por `label + environment`, la seed fija 42, y los
pipelines de augmentation (estándar, extendido para minoritarias, y determinista para val/test).
Ver [Preprocesado](../preprocessed/index.md) para el detalle de cada uno.
