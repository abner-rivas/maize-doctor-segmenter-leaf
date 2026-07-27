# Preprocesado

## Ubicación de datos

El dataset completo vive fuera del repositorio bajo `DATASET_ROOT`. Los datos derivados
reproducibles del proyecto viven en `data/`: splits, imágenes del piloto, anotaciones y
manifiestos. Los modelos, métricas, validaciones, previews y diagnósticos viven en
`outputs/`. Esta separación evita mezclar entradas con resultados.

## Normalizado

Antes de que una imagen llegue al modelo tiene que pasar por un único punto de entrada que la deje en un formato consistente, sin importar si viene de una cámara de laboratorio o de un smartphone en el campo. Esto evita que se cuele ruido en nuestro dataset y que el modelo aprenda artefactos irrelevantes.

Toda imagen se carga en caliente, mediante dos pasos en orden:

1. **Corrección EXIF**: corrige la orientación física de fotos tomadas con smartphones antes de cualquier transformación.
2. **Conversión a RGB estricta**: elimina el canal alfa (RGBA) y expande imágenes monocromáticas, garantizando 3 canales en todos los tensores.

Ya como parte de los pipelines de transformación, cada tensor se normaliza con las medias y desviaciones estándar de ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Se eligieron porque todos los backbones preentrenados esperan esta normalización.

## División

El siguiente paso es decidir qué imágenes ve el modelo en cada etapa (entrenamiento, validación, evaluación) y garantizar que esa división sea reproducible (que es uno de los puntos más importantes de este proyecto académico).

La división separa el dataset en:
- **70% train**
- **15% val**
- **15% test**

Con una seed fija default de 42 (declarada en `config/dataset.yaml`) para garantizar reproducibilidad exacta entre ejecuciones.

Teniendo en cuenta que en algunas clases se manejan 2 dominios, la estratificación inicialmente se hace por `label + ambiente`, no solo por clase, sin embargo, se prevee que en etapas posteriores se realicen cambios para garantizar mayores pruebas en entornos reales, para validar la generalización del modelo.

La división no crea copias físicas de las imágenes, sino que genera tres CSV (`train.csv`, `val.csv` y `test.csv`) con la ruta relativa de cada imagen y su etiqueta. Esto permite que el dataset sea reproducible y que los CSV puedan ser versionados en Git si se desea.

## Balanceo

El dataset completo que se tiene a la fecha está lejos de ser uniforme, algunas clases tienen miles de imágenes y otras apenas un par de cientos. Sin corrección, un modelo entrenado directamente sobre esa distribución tendería a ignorar las clases minoritarias.

### Diagnóstico del desbalance (para train)

| Clase | N | Ratio vs healthy (sana) |
|---|---|---|
| potassium_deficiency | 186 | 32.9x |
| nitrogen_deficiency | 364 | 16.8x |
| phosphorus_deficiency | 428 | 14.3x |
| gray_leaf_spot | 778 | 7.9x |
| common_rust | 1575 | 3.9x |
| fall_armyworm | 3223 | 1.9x |
| lethal_necrosis | 4491 | 1.4x |
| northern_corn_leaf_blight | 4774 | 1.3x |
| healthy | 6118 | 1.0x |

### Técnicas discutidas

Antes de llegar a la estrategia actual se evaluaron otras alternativas más simples, y se descartaron por razones concretas:

- **Undersampling agresivo:** se descartó porque elimina datos reales y escasos. Bajar la enorme variedad de imágenes de clases como `healthy` o `lethal_necrosis` a 500 o 1 000 imágenes por clase sería un desperdicio de información valiosa. Además, el modelo perdería la oportunidad de aprender patrones de fondo y variaciones de iluminación que solo aparecen en clases mayoritarias.
- **Oversampling físico (crear más copias físicas en disco):** se descartó (de momento) porque con técnicas como `WeightedRandomSampler` se logra el mismo efecto en memoria, es reversible y se combina con la augmentation en caliente.
- **Focal Loss:** se contempla como opción para el pipeline principal si los resultados no mejoran en las clases minoritarias.

### Estrategia planeada: 2 capas complementarias

Para el pipeline principal se plantea una estrategia de balanceo de **dos capas**:

**Capa 1: `WeightedRandomSampler`.** Cada muestra recibe un peso `1 / count_of_its_class`. El sampler repite muestras minoritarias dentro de cada epoch sin inflar su tamaño (`num_samples` = tamaño original). Combinado con augmentation en caliente, cada repetición recibe transformaciones distintas.

**Capa 2: `CrossEntropyLoss` ponderada.** Peso por clase `w_i = total / (num_clases x count_i)`. Reforzaría el gradiente de clases minoritarias incluso cuando aparecen en menor proporción dentro de un batch, complementando al sampler que actúa sobre la frecuencia de aparición.

> El **pipeline de baselines** implementa por ahora solo la Capa 1 (sampler) con una `CrossEntropyLoss` estándar sin ponderar, para mantener las corridas simples y comparables entre arquitecturas.

## Data Augmentation

Además de balancear cuántas veces ve el modelo cada clase, hace falta variar cómo las ve para que no memorice detalles irrelevantes del fondo o del encuadre. La augmentation se aplica en caliente, en cada carga de imagen, y con más intensidad en las clases que más lo necesitan.

Esto se basa en el valor de la configuración:
```
augmentation:
  minority_ratio_threshold: 4.0
```

Lo que se traduce en que las clases cuya frecuencia es superada en **más de 4x** por la clase más numerosa del split reciben un pipeline de augmentación más agresivo, mientras que las que quedan por debajo de ese umbral reciben el pipeline estándar. El umbral se evalúa sobre la distribución real del split de train, no sobre el dataset completo.

Esto se aplica únicamente en train, en cambio val y test usan transformaciones deterministas para garantizar una evaluación justa.

### Pipeline estándar de augmentación

```
Resize(224x224)
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(±15°, BILINEAR)
ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0)
ToTensor()
Normalize(ImageNet)
```

El ColorJitter es conservador (sin saturación ni hue) porque las deficiencias nutricionales se diagnostican por color. Alteraciones agresivas de tono destruirían esa señal.

### Pipeline extendido de augmentación para clases minoritarias

Sobre el dataset completo (train, con `healthy` = 6 118 como techo de referencia), las clases que cruzan el umbral de 4x son cuatro: `potassium_deficiency` (32.9x), `nitrogen_deficiency` (16.8x), `phosphorus_deficiency` (14.3x) y `gray_leaf_spot` (7.9x). `common_rust` (3.9x) queda justo por debajo y recibe el pipeline estándar. A esas cuatro clases se les aplica en caliente el pipeline extendido:

```
RandomResizedCrop(224x224, scale=(0.7, 1.0))   <--- recortes aleatorios
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(±30°, BILINEAR)                 <--- más agresivo que estándar
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
GaussianBlur(kernel=3, sigma=(0.1, 1.5))       <--- ruido gaussiano suave
ToTensor()
Normalize(ImageNet)
```

Se mantiene `hue=0.05` (mínimo) para no destruir la señal de clorosis en deficiencias nutricionales, a pesar de ser clases minoritarias.

### Augmentación mínima para Val / Test

```
Resize(224x224)
ToTensor()
Normalize(ImageNet)
```

Sin ninguna augmentation aleatoria.

## Región de interés de hoja

Una **región de interés** (ROI) es el rectángulo de la fotografía que contiene la hoja que se
quiere analizar. Los componentes de `src/preprocessing/` pueden validar un bounding box,
limitarlo a la imagen, agregar un margen de seguridad y producir una imagen compatible con el
tamaño de entrada del clasificador. Todavía no existe un detector integrado: el bounding box
se proporciona manualmente o podrá venir de otra fuente en fases posteriores.

El margen se calcula por separado sobre el ancho y el alto del bbox. Por ejemplo,
`margin_ratio: 0.08` agrega un 8 % del ancho a cada lado y un 8 % del alto arriba y abajo, sin
salirse de la fotografía. Los márgenes entre 0 y 1 son válidos; un valor mayor que 1 se rechaza
como probable error de configuración.

Después del recorte se aplica **letterbox**: la ROI se redimensiona proporcionalmente, se
centra y se completa con padding RGB hasta el tamaño exacto. Esto evita que un
`Resize((224, 224))` directo convierta una hoja alargada en una forma artificialmente ancha o
alta. La convención del proyecto para `target_size` es `(alto, ancho)`, aunque Pillow reporta
los tamaños de imagen como `(ancho, alto)`.

Cuando el bbox no es válido existen tres fallbacks aislados:

- `original`: usa una copia RGB de la fotografía completa.
- `center_crop`: conserva una proporción configurable de ambos ejes, centrada en la imagen
  (80 % por defecto).
- `reject`: devuelve un rechazo controlado y no produce una región para clasificación.

Validación manual en Linux:

```bash
python3 scripts/checks/validate_leaf_roi.py \
  --image /ruta/imagen.jpg \
  --bbox 100 50 500 350 \
  --margin-ratio 0.08 \
  --target-size 224 224 \
  --padding-value 0 \
  --fallback original \
  --output outputs/leaf_detection/roi_validation
```

Validación manual en PowerShell:

```powershell
python scripts/checks/validate_leaf_roi.py `
  --image "C:\ruta\imagen.jpg" `
  --bbox 100 50 500 350 `
  --margin-ratio 0.08 `
  --target-size 224 224 `
  --padding-value 0 `
  --fallback original `
  --output "outputs\leaf_detection\roi_validation"
```

La salida contiene la imagen original, las dos visualizaciones de bbox, el recorte, el
letterbox y `metadata.json`. La sección `leaf_detection` de `config/dataset.yaml` queda con
`enabled: false`. El perfil global también continúa en `processing_profile: baseline_full`.
La selección automática de detecciones y su caché pertenecen a fases posteriores.

## Piloto manual y diagnóstico ROI

La selección reproducible del piloto, la guía de anotación, los importadores YOLO/CSV, el
manifiesto final, las comprobaciones de fuga y las vistas previas se documentan en
[Piloto manual de regiones de interés](manual-roi-pilot.md).

La ejecución ya completada con los checkpoints históricos, sus métricas y su
interpretación metodológica se documentan en
[Diagnóstico de imagen completa frente a ROI manual](manual-roi-diagnostic.md).

El diagnóstico define dos rutas sin alterar el baseline histórico:

```text
baseline_full: imagen completa → transformación histórica → clasificador
baseline_roi:  RGB → bbox → validación → clipping → margen → recorte
               → letterbox → augmentations → normalización → clasificador
```

Las augmentations sólo son válidas durante entrenamiento. Evaluación, inferencia y
explicabilidad deben usar la misma preparación ROI sin transformaciones aleatorias. En esta
fase `baseline_roi` se usa únicamente en
`scripts/experiments/compare_full_vs_manual_roi.py`; aún no se activa en `CornDataset`,
`predict.py`, LIME ni Grad-CAM.

El diagnóstico obtuvo una reducción de rendimiento en los tres modelos cuando
ROI se aplicó sólo durante inferencia. Este resultado evidencia un cambio de
distribución, no el rendimiento de un clasificador entrenado con ROI. Por eso
`processing_profile: baseline_full` sigue activo y
`leaf_detection.enabled: false`.

## Fuentes externas de segmentación de hoja

Se auditaron dos exportaciones YOLO con sus respaldos COCO como candidatas para
un futuro segmentador binario de hoja. La auditoría recalculó 1,003 y 157
imágenes, validó 14,619 líneas de polígonos, contrastó errores con COCO y
comprobó duplicados contra el piloto retenido.

No se entrenó YOLO, no se repararon fuentes y no se creó un dataset consolidado.
Los hallazgos, decisiones por clase, gráficos y próximos pasos están en
[Auditoría de datasets externos de segmentación](../leaf-detection/external-segmentation-datasets-eda.md).

La secuencia de decisiones —bounding boxes, piloto, diagnóstico, cambio de
distribución y estrategia de máscaras— está consolidada en
[Historia del aislamiento de hojas](../leaf-detection/history.md).

## Auditoría de clases y splits (Fase 3.5)

Antes de restaurar splits o entrenar, la configuración, la documentación y las carpetas de
`clean/` deben coincidir. El diagnóstico, los reportes reproducibles, el respaldo seguro y la
validación de fugas se describen en
[Auditoría de clases y restauración de splits](dataset-class-audit.md).
