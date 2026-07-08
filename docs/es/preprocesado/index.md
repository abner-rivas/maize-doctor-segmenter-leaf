# Preprocesado

## Normalizado

Antes de que una imagen llegue al modelo tiene que pasar por un único punto de entrada que la deje en un formato consistente, sin importar si viene de una cámara de laboratorio o de un smartphone en el campo. Esto evita que diferencias de orientación, canal de color o escala se cuelen como ruido durante el entrenamiento.

Toda imagen se carga en caliente mediante `load_and_normalize_image()` (`src/data/loader.py`), que aplica tres pasos en orden:

1. **Corrección EXIF** (`ImageOps.exif_transpose`): corrige la orientación física de fotos tomadas con smartphones antes de cualquier transformación.
2. **Conversión a RGB estricta**: elimina el canal alfa (RGBA) y expande imágenes monocromáticas, garantizando 3 canales en todos los tensores.
3. **Estadísticas de normalización**: se usan las medias y desviaciones estándar de ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Se eligieron porque todos los backbones preentrenados esperan esta normalización. Cuando se calcule el EDA propio del dataset, podrán sustituirse por estadísticas reales del corpus.

## División

Antes de balancear clases o aplicar augmentation hace falta decidir qué imágenes ve el modelo en cada etapa (entrenamiento, validación, prueba) y garantizar que esa división sea reproducible y representativa del dataset completo.

Implementada en `HierarchicalStratifiedSplitter` (`src/data/splitter.py`), la división separa el dataset en 70% train, 15% val y 15% test, con la seed fija en 42 (declarada en `config/dataset.yaml`) para garantizar reproducibilidad exacta entre ejecuciones.

La estratificación es jerárquica: se hace por `label + environment`, no solo por clase. Esto asegura que la proporción de imágenes de laboratorio frente a las de campo sea consistente en los tres splits; un split aleatorio simple podría concentrar todas las imágenes de laboratorio de una clase en train, sesgando val y test.

Los CSV resultantes son inmutables: `outputs/splits/seed_42/train.csv`, `val.csv` y `test.csv` son la fuente de verdad y no se modifican a mano. Las exclusiones de clases se aplican en tiempo de construcción del dataset (`exclude_classes`), no en el CSV.

Baseline vs. principal: el pipeline principal usa `seed_42/` (9 clases sin cap), mientras que el baseline usa `seed_42_baseline/` (9 clases con cap de 1500 imágenes por clase). Ambos comparten la misma estratificación y la misma seed.

## Balanceo

El dataset está lejos de ser uniforme: algunas clases tienen miles de imágenes y otras apenas un par de cientos. Sin corrección, un modelo entrenado directamente sobre esa distribución tiende a ignorar las clases minoritarias porque optimizar para las mayoritarias ya le da buena precisión global. Por eso el balanceo combina un diagnóstico cuantitativo con varias técnicas complementarias.

### Diagnóstico del desbalance (train)

| Clase | N | Ratio vs healthy |
|---|---|---|
| potassium_deficiency | 186 | 32.9x |
| nitrogen_deficiency | 364 | 16.8x |
| phosphorus_deficiency | 428 | 14.3x |
| gray_leaf_spot | 778 | 7.9x |
| common_rust | 1575 | 3.9x |
| lethal_necrosis | 4491 | 1.4x |
| fall_armyworm | 3223 | 1.9x |
| northern_corn_leaf_blight | 4774 | 1.3x |
| healthy | 6118 | 1.0x |

### Técnicas descartadas

Antes de llegar a la estrategia actual se evaluaron otras alternativas más simples, y se descartaron por razones concretas:

- **Undersampling agresivo:** descartado porque elimina datos reales y escasos. Con 186 imágenes de `potassium_deficiency`, reducir la mayoría a ese nivel destruiría el 97% del corpus.
- **Oversampling físico (copias en disco):** descartado porque `WeightedRandomSampler` logra el mismo efecto en memoria, es reversible y se combina con augmentation en caliente sin duplicar archivos.
- **Focal Loss:** descartada en esta etapa. Es más útil cuando hay muchos ejemplos fáciles que saturan el gradiente (por ejemplo, detección de objetos con fondo masivo). En clasificación de hojas con 9 clases y muestras mayoritariamente difíciles, Weighted Cross Entropy es más interpretable. Se revisará si los resultados experimentales no mejoran.

### Estrategia adoptada: tres capas complementarias

**Capa 1: sin exclusiones de clase.** `aphids_pest` fue considerada inicialmente pero se descartó definitivamente porque no se encontraron suficientes fuentes de imágenes adicionales: con solo ~77 fotos, el data augmentation no produciría variedad visual real sino repetición sintética. En su lugar se incorporó `lethal_necrosis` (~6415 imágenes de campo real), que sí tiene masa crítica para el entrenamiento. El pipeline no excluye ninguna clase actualmente.

**Capa 2: `WeightedRandomSampler`.** Cada muestra recibe un peso `1 / count_of_its_class`. El sampler repite muestras minoritarias dentro de cada epoch sin inflar su tamaño (`num_samples` = tamaño original). Combinado con augmentation en caliente, cada repetición recibe transformaciones distintas.

**Capa 3: `CrossEntropyLoss` ponderada.** Peso por clase `w_i = total / (num_clases × count_i)`. Refuerza el gradiente de clases minoritarias incluso cuando aparecen en menor proporción dentro de un batch, complementando al sampler que actúa sobre frecuencia de aparición.

## Data Augmentation

Además de balancear cuántas veces ve el modelo cada clase, hace falta variar cómo las ve para que no memorice detalles irrelevantes del fondo o del encuadre. La augmentation se aplica en caliente, en cada carga de imagen, y con más intensidad en las clases que más lo necesitan.

Implementada en `src/data/transforms.py`, se aplica únicamente en train; val y test usan transformaciones deterministas para garantizar una evaluación justa.

### Pipeline estándar (`CornTrainingTransforms`): todas las clases de train

```
Resize(224×224)
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(±15°, BILINEAR)
ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0)
ToTensor()
Normalize(ImageNet)
```

El ColorJitter es conservador (sin saturación ni hue) porque las deficiencias nutricionales se diagnostican por color. Alteraciones agresivas de tono destruirían la señal diagnóstica.

### Pipeline extendido (`CornMinorityTransforms`): clases con ratio > 4x

Aplicado en caliente a `potassium_deficiency`, `nitrogen_deficiency`, `phosphorus_deficiency`, `gray_leaf_spot` y `common_rust`. `lethal_necrosis` no entra en este grupo (ratio ~1.4x, bien representada):

```
RandomResizedCrop(224×224, scale=(0.7, 1.0))   ← recortes aleatorios
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(±30°, BILINEAR)                 ← más agresivo que estándar
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
GaussianBlur(kernel=3, sigma=(0.1, 1.5))       ← ruido gaussiano suave
ToTensor()
Normalize(ImageNet)
```

Se mantiene `hue=0.05` (mínimo) para no destruir la señal de clorosis en deficiencias nutricionales, a pesar de ser clases minoritarias.

### Val / Test (`CornValidationTransforms`)

```
Resize(224×224)
ToTensor()
Normalize(ImageNet)
```

Sin ninguna augmentation aleatoria.
