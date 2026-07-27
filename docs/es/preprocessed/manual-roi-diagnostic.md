# Diagnóstico de imagen completa frente a ROI manual

## Objetivo y alcance

`diagnostic_full_vs_manual_roi` mide cómo responden tres checkpoints históricos,
entrenados con imágenes completas, cuando durante inferencia se sustituye su entrada
habitual por una región de interés (ROI) manual de la hoja. Es una prueba diagnóstica
pareada sobre el piloto; **no es entrenamiento, no es un baseline oficial y no evalúa
un clasificador entrenado con ROI**.

La decisión formal derivada de este resultado está en
[ADR: ROI manual no activado](../decisions/adr-manual-roi-diagnostic-result.md).

Los artefactos completos están en
[`outputs/leaf_detection/pilot/diagnostic_experiment/`](../../../outputs/leaf_detection/pilot/diagnostic_experiment/).
No se copiaron resultados a `docs/`.

## Datos y trazabilidad

- Dataset fuente externo: `/home/desarrolloab/Documentos/ML/maize_dataset/data`.
- Dataset validado: 31 622 imágenes soportadas, 28 071 reales, 3 551 de
  laboratorio y 9 clases, sin errores críticos.
- Piloto: 100 imágenes de entorno real, selección balanceada, semilla 42 y sin
  duplicados.
- Manifiesto: [`roi_manifest.csv`](../../../data/leaf_detection/pilot/manifests/roi_manifest.csv),
  SHA-256 `f046e3cab6171fdb9c7152f8ca80b4a95f70cd8162e1c8768c9347180aa3f705`.
- Casos válidos para métricas: 99.
- Exclusión: `image_0021`, marcada `ambiguous`, con área ROI `0.092799`, menor
  que el mínimo `0.15`.
- Fallbacks usados: 0.

La anotación oficial del piloto es el XML nativo de CVAT
[`annotations.xml`](../../../data/leaf_detection/pilot/annotations/cvat/annotations.xml):
100 imágenes y 100 cajas `maize_leaf`. El exportador YOLO produjo sólo 48
etiquetas directas; las otras 52 cajas tenían rotación y se recuperaron desde XML.
La conversión rota calcula las cuatro esquinas alrededor del centro, toma su
envolvente alineada a ejes, redondea hacia afuera, limita la caja a la imagen y
valida la geometría. De las 100 cajas importadas, 48 fueron directas, 52 rotadas
y 36 necesitaron clipping.

## Metodología

Para cada checkpoint se evaluaron las mismas filas y etiquetas con dos rutas:

```text
baseline_full:
imagen completa → resize directo histórico → normalización → clasificador

baseline_roi:
imagen RGB → bbox manual → validación → clipping → margen → recorte
→ letterbox → padding negro → normalización → mismo clasificador histórico
```

El perfil ROI fue local al script
`scripts/experiments/compare_full_vs_manual_roi.py`. La configuración global se
mantuvo en:

```yaml
processing_profile: baseline_full

leaf_detection:
  enabled: false
```

No se activó detector, no se entrenó ningún modelo, no se modificaron checkpoints
y no se utilizó ninguna corrida de `outputs/aborted_runs`.

## Modelos y checkpoints

Los tres checkpoints tienen 30 epochs históricos y conservan `best.pth`,
`last.pth`, historial, predicciones y métricas:

| Modelo | Checkpoint usado |
|---|---|
| EfficientNet-B0 | `outputs/baselines/efficientnet_b0/20260709_040040/best.pth` |
| ShuffleNetV2-x1.0 | `outputs/baselines/shufflenet_v2_x1_0/20260709_042946/best.pth` |
| EfficientNet-Lite0 | `outputs/baselines/efficientnet_lite0/20260709_045817/best.pth` |

Cada directorio diagnóstico conserva `predictions_full.csv`,
`predictions_roi.csv`, `comparison.csv`, `summary.json`, `run_metadata.json`,
`improved_cases/`, `worsened_cases/`, `changed_predictions/` y `previews/`.
Las fuentes principales son:

- [Resumen EfficientNet-B0](../../../outputs/leaf_detection/pilot/diagnostic_experiment/efficientnet_b0/summary.json)
- [Resumen ShuffleNetV2-x1.0](../../../outputs/leaf_detection/pilot/diagnostic_experiment/shufflenet_v2_x1_0/summary.json)
- [Resumen EfficientNet-Lite0](../../../outputs/leaf_detection/pilot/diagnostic_experiment/efficientnet_lite0/summary.json)

Todos los resúmenes registran `official_baseline: false` y
`training_performed: false`.

## Resultados globales

| Modelo | Full Accuracy | ROI Accuracy | Δ Accuracy | Full Macro-F1 | ROI Macro-F1 | Δ Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 0.8889 | 0.8586 | -0.0303 | 0.8827 | 0.8561 | -0.0266 |
| ShuffleNetV2-x1.0 | 0.9091 | 0.7576 | -0.1515 | 0.9064 | 0.7582 | -0.1482 |
| EfficientNet-Lite0 | 0.9091 | 0.6162 | -0.2929 | 0.9052 | 0.6101 | -0.2951 |

| Modelo | Full Loss | ROI Loss | Cambio |
|---|---:|---:|---:|
| EfficientNet-B0 | 0.5153 | 0.5361 | +0.0208 |
| ShuffleNetV2-x1.0 | 0.5435 | 0.8403 | +0.2969 |
| EfficientNet-Lite0 | 0.5836 | 2.9971 | +2.4135 |

| Modelo | Predicciones cambiadas | Errores corregidos | Aciertos perdidos | Cambio medio de confianza |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 7 | 2 | 5 | -0.0306 |
| ShuffleNetV2-x1.0 | 21 | 3 | 18 | -0.0569 |
| EfficientNet-Lite0 | 39 | 2 | 31 | -0.0574 |

## Resultados por modelo

### EfficientNet-B0

Fue el modelo más robusto al cambio: accuracy `0.8889 → 0.8586` y Macro-F1
`0.8827 → 0.8561`. Corrigió `image_0011` e `image_0091`; perdió los aciertos
de `image_0006`, `image_0015`, `image_0024`, `image_0035` e `image_0044`.

ROI mostró mejora o estabilidad en `fall_armyworm`, `healthy`,
`northern_corn_leaf_blight`, `common_rust`, `gray_leaf_spot` y
`lethal_necrosis`. Las clases más afectadas fueron `nitrogen_deficiency`,
`phosphorus_deficiency` y `potassium_deficiency`.

### ShuffleNetV2-x1.0

La accuracy cambió `0.9091 → 0.7576` y el Macro-F1 `0.9064 → 0.7582`.
Corrigió `image_0009`, `image_0045` e `image_0098`, pero perdió 18 aciertos.
La reducción principal se concentró en `nitrogen_deficiency`,
`northern_corn_leaf_blight`, `phosphorus_deficiency` y
`potassium_deficiency`.

### EfficientNet-Lite0

Fue el más sensible: accuracy `0.9091 → 0.6162` y Macro-F1
`0.9052 → 0.6101`. Corrigió `image_0009` e `image_0073`, cambió 39
predicciones y perdió 31 aciertos. Las caídas más severas se observaron en
`common_rust`, `gray_leaf_spot`, `nitrogen_deficiency` y
`potassium_deficiency`.

## Interpretación científica

Los checkpoints históricos fueron entrenados con imágenes completas y no fueron
adaptados a entradas ROI. Aplicar recorte, cambio de escala, letterbox y padding
únicamente durante inferencia produjo un cambio de distribución que redujo el
rendimiento en los tres modelos.

El ROI no aprende: el clasificador es el componente que debe aprender a trabajar
con imágenes ROI. En este diagnóstico cambiaron simultáneamente el encuadre, la
escala de la hoja, la cantidad de fondo, la relación de aspecto, el método de
redimensionamiento, la presencia de padding y el contexto visual.

Por ello, el experimento demuestra que **no es seguro activar ROI sólo durante
inferencia con modelos entrenados usando imágenes completas**. No demuestra que
un modelo entrenado desde el principio con ROI vaya a funcionar peor. Tampoco
permite concluir que el ROI, el detector o el aprendizaje hayan fallado.

## Limitaciones e hipótesis pendientes

El piloto tiene sólo 99 casos evaluables y no aísla cada cambio de
preprocesamiento. Las siguientes explicaciones son hipótesis, no causas
confirmadas:

1. cambio de distribución entre entrenamiento e inferencia;
2. pérdida de contexto útil alrededor de la hoja;
3. margen ROI insuficiente;
4. padding negro no visto durante el entrenamiento;
5. cambio de escala de los síntomas;
6. sensibilidad distinta entre arquitecturas;
7. necesidad de reentrenamiento con ROI;
8. necesidad potencial de conservar más extensión de la hoja para diferenciar
   deficiencias nutricionales.

## Próximos experimentos

La ablación pendiente debe separar los efectos del recorte, letterbox, padding y
escala:

```text
A. Imagen completa + resize histórico
B. Imagen completa + letterbox
C. ROI + resize directo
D. ROI + letterbox con padding negro
E. ROI + letterbox con padding neutro
```

El padding neutro candidato es RGB `(124, 116, 104)`, aproximación a la media de
ImageNet; todavía no se ha probado.

La comparación definitiva y justa será:

```text
baseline_full: entrenado con imágenes completas → evaluado con imágenes completas
baseline_roi:  entrenado con ROI → evaluado con ROI
```

La ruta de bounding boxes preparada originalmente se conserva como alternativa
histórica:

```text
Anotar 300–500 imágenes
        ↓
Entrenar detector de hoja
        ↓
Generar ROI para train, val y test
        ↓
Validar una muestra
        ↓
Congelar manifiestos ROI
        ↓
Entrenar clasificador baseline_roi
        ↓
Comparar contra baseline_full
```

No es necesario anotar manualmente las 10 020 imágenes de los splits.

La primera parte de esa ampliación ya está preparada: se seleccionaron 350
imágenes nuevas de train y 75 de val para anotación multihoja, mientras que el
piloto quedó retenido como test. Todavía no se anotaron esos lotes ni se entrenó
YOLO26n. Consulte
[Preparación del dataset del detector](../leaf-detection/yolo26-detector-dataset.md).

Después se priorizó explorar segmentación y se auditaron dos fuentes externas
YOLO/COCO. La auditoría identificó clases de hoja completa separables de las
lesiones, 11 líneas YOLO recuperables desde COCO, una etiqueta vacía y cero
duplicados contra el piloto. El paso inmediato es filtrar, revisar y consolidar
esas fuentes; no es entrenar todavía. Consulte
[Auditoría de datasets externos de segmentación](../leaf-detection/external-segmentation-datasets-eda.md).

## Conclusión

Los tres modelos históricos presentaron una reducción de rendimiento cuando el
ROI manual se aplicó únicamente durante inferencia. EfficientNet-B0 fue el más
robusto y EfficientNet-Lite0 el más sensible. La caída no basta para descartar
el aislamiento de la hoja, porque los clasificadores nunca fueron entrenados
con este tipo de entrada.

El resultado confirma que debe mantenerse `baseline_full` como configuración
activa y que cualquier clasificador ROI futuro debe entrenarse o ajustarse con
imágenes procesadas mediante el mismo pipeline ROI.
