# Integración con el clasificador

`export_classifier_dataset` convierte una corrida terminada en un árbol `<clase>/<entorno>/<image_id>.<extensión>`. Por defecto incluye únicamente estados `reliable`, fija el entorno a `lab` para el layout esperado y materializa con hardlinks cuando origen y destino comparten filesystem. El proceso trabaja en un directorio temporal, escribe manifiesto, resumen y README, y sólo al final renombra atómicamente. Si la salida ya existe, se niega a sobrescribirla.

La exportación observada incluyó 28,360 imágenes y excluyó 5,077: 4,020 `failed` y 1,057 `uncertain`. El manifiesto fuente tiene SHA-256 `1b672ca01e4a95f869a94677ddd1e7fe8eeb107e42f7b3f15358049b6ea43c12`. Los hardlinks hacen el dataset autocontenido dentro del filesystem sin duplicar bytes, aunque el ZIP posterior sí ocupa 5.9 GB.

Table: Retención por clase al exportar sólo máscaras confiables.

| Clase | Entrada | Exportada | Excluida | Retención |
|---|---:|---:|---:|---:|
| Roya común | 2,256 | 2,100 | 156 | 93.09% |
| Gusano cogollero | 4,857 | 2,557 | 2,300 | 52.65% |
| Mancha gris | 1,930 | 1,836 | 94 | 95.13% |
| Sana | 8,744 | 8,008 | 736 | 91.58% |
| Necrosis letal | 6,415 | 5,390 | 1,025 | 84.02% |
| Deficiencia de nitrógeno | 846 | 682 | 164 | 80.61% |
| Tizón norteño | 6,830 | 6,525 | 305 | 95.53% |
| Deficiencia de fósforo | 938 | 804 | 134 | 85.71% |
| Deficiencia de potasio | 621 | 458 | 163 | 73.75% |

![El filtro de confiabilidad cambia tanto el tamaño como la composición relativa del corpus que recibe el clasificador.](figures/class_retention.png)

Este filtrado crea un posible sesgo de selección. Cogollero pierde casi la mitad de sus ejemplos, mientras tizón pierde menos de 5%. Si las imágenes rechazadas representan daño severo, fondos complejos u hojas pequeñas, entrenar sólo con las aceptadas puede eliminar precisamente la variabilidad que el diagnóstico necesita. Por eso el manifiesto de exclusiones debe conservarse y el clasificador debe comparar al menos tres protocolos: originales completos, segmentadas confiables y un esquema mixto con fallback explícito.

La exportación no demuestra mejora de clasificación. Su estructura es compatible y su procedencia es trazable, pero faltan experimentos pareados que mantengan splits y modelo del clasificador constantes. También debe verificarse que los cuatro duplicados del corpus se eliminen antes de crear splits clasificatorios; el segmentador los procesó a propósito y no es responsable de esa deduplicación downstream.

## Contrato recomendado

El consumidor debe leer `reliability_status`, `output_mode`, checkpoint y configuración, no inferir éxito a partir de la existencia de un JPG. Las imágenes `uncertain` y `failed` no deben desaparecer de las métricas operativas: aunque se excluyan del entrenamiento segmentado, son parte de la población real y deben medirse como fallback.

