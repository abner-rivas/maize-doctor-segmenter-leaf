# Experimentos

Antes de fijar el baseline canónico de 9 clases, se realizó una **exploración de arquitecturas**
sobre un subconjunto reducido de **4 clases** (`healthy`, `common_rust`, `fall_armyworm`,
`nitrogen_deficiency`), en dos regímenes de datos:

- **Cap de 500 imágenes por clase** - comparación rápida y barata + sonda de eficiencia muestral.
- **Sin límite por clase** - mismas 4 clases con todas las imágenes disponibles.

El objetivo fue comparar el comportamiento de cada arquitectura (convergencia, colapso de clases,
sensibilidad al volumen de datos) a bajo costo, no medir el rendimiento final.

## Arquitecturas exploradas

Se probaron hasta **8 arquitecturas ligeras**:

- `efficientnet_b0`
- `efficientnet_b4`
- `efficientnet_lite0`
- `fastvit_t8`
- `ghostnetv2_100`
- `mobilenet_v3_large`
- `mobilenet_v3_small`
- `shufflenet_v2_x1_0`

De esta exploración se adoptaron como baselines definitivos `efficientnet_b0`,
`shufflenet_v2_x1_0` y `efficientnet_lite0` (ver [Baselines](../baselines/)).

## Resultados completos

Los outputs completos de estas corridas exploratorias (métricas, matrices de confusión, reportes
LIME) se archivan por separado:

<!-- TODO(davidderas50): subir el ZIP y pegar el enlace de Google Drive -->
📦 **Resultados de experimentos (4 clases):** [ZIP en Google Drive](https://drive.google.com/drive/folders/1lHWjZ04V3__OROMNqs38fPD5sd3t8laI?usp=sharing)
