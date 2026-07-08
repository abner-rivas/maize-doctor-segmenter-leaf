# Análisis Exploratorio de Datos

El EDA busca responder tres preguntas antes de diseñar el pipeline de entrenamiento: ¿qué hay en el dataset?, ¿qué problemas tiene?, y ¿qué decisiones impone?

El análisis completo y reproducible, con todo el código, está en la notebook [`notebooks/01_eda.ipynb`](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb). Esta página resume los hallazgos y las decisiones que se derivaron de ellos.

---

## Composición del dataset

El dataset consolidado en `data/clean/` contiene **31 622 imágenes** distribuidas en **9 clases**, procedentes de 6 fuentes públicas. Cada imagen pertenece a un entorno de captura: `lab` (fondo controlado, iluminación artificial) o `real` (campo abierto, iluminación solar).

| Clase | Lab | Real | Total |
|---|---:|---:|---:|
| `healthy` | 0 | 8 744 | **8 744** |
| `northern_corn_leaf_blight` | 888 | 5 942 | **6 830** |
| `lethal_necrosis` | 0 | 6 415 | **6 415** |
| `fall_armyworm` | 0 | 4 857 | **4 857** |
| `common_rust` | 2 150 | 106 | **2 256** |
| `gray_leaf_spot` | 513 | 606 | **1 119** |
| `phosphorus_deficiency` | 0 | 612 | **612** |
| `nitrogen_deficiency` | 0 | 523 | **523** |
| `potassium_deficiency` | 0 | 266 | **266** |

---

## 1. Distribución de clases

El dataset presenta un **desbalance severo**: `healthy` (8 744 imágenes) supera en **32.9×** a `potassium_deficiency` (266). Con este desbalance se corre el riesgo de que el modelo aprenda a predecir siempre la clase mayoritaria, ignorando las clases minoritarias.

![Distribución de imágenes por clase](/eda/eda_01_distribucion_clases.png)

Para mitigar este desbalance, inicialmente se planea usar `WeightedRandomSampler` en el DataLoader para igualar la frecuencia efectiva de cada clase durante el entrenamiento. Para las clases más escasas (`potassium_deficiency`, `nitrogen_deficiency`, `phosphorus_deficiency`) aplicar un pipeline de augmentation mas agresivo (`CornMinorityTransforms`).

- [Ver análisis completo: sección 1.2 de la notebook](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb)

---

## 2. Sesgo de entorno (lab vs real)

La proporción de imágenes de laboratorio vs campo varía drásticamente entre clases, lo que introduce riesgo de *domain shortcut*: una red puede aprender el fondo uniforme de PlantVillage en lugar de los síntomas foliares.

![Proporción lab vs real por clase](/eda/eda_02_lab_vs_real.png)

El caso más crítico es `common_rust`: **95.4 %** de sus 2 256 imágenes proviene de entorno controlado (fondo negro/blanco uniforme). En campo solo hay 106 imágenes.

Esto representa un riesgo significativo de sobreajuste a condiciones de laboratorio, por lo que se deberán aplicar técnicas de augmentation específicas, asi como intentar validar solo sobre imágenes de campo, para comprobar la generalización del modelo.

- [Ver análisis completo: sección 1.3 de la notebook](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb)

---

## 3. Resolución y dimensiones

Las imágenes tienen resoluciones muy dispares. Las de `corn_leaf_roboflow` ya vienen redimensionadas a **640 × 640 px** (Roboflow). El resto conserva su resolución original, que va desde imágenes pequeñas hasta varios megapíxeles.

![Distribución de resoluciones](/eda/eda_03_resoluciones.png)

En el pipeline se deberá aplicar `Resize` a la resolución de entrada del modelo (224 x 224 px para MobileNetV3) como primera transformación, seguido de `CenterCrop` en validación y `RandomResizedCrop` en entrenamiento.

- [Ver análisis completo: sección 1.4 de la notebook](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb)

---

## 4. Calidad de imagen

Se evaluaron tres métricas sobre una muestra estratificada de hasta 400 imágenes por clase: desenfoque (varianza del Laplaciano), subexposición (brillo medio < 40) y sobreexposición (brillo medio > 230).

![Problemas de calidad por clase](/eda/eda_04_calidad.png)

![Distribuciones de calidad por clase](/eda/eda_04b_calidad_boxplots.png)

El porcentaje de imágenes con problemas es bajo y distribuido uniformemente entre clases, sin concentración en ninguna categoría específica. Las imágenes oscuras en clases de laboratorio (fondo negro) son artefactos del umbral global, no defectos reales.

- [Ver análisis completo: sección 1.5 de la notebook](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb)

---

## 5. Duplicados y limpieza

Durante la etapa de construcción del dataset se detectaron duplicados entre fuentes distintas mediante **hash perceptual (pHash)** con `imagededup` (threshold = 0, solo copias exactas a nivel perceptual).

![Duplicados eliminados por clase y fuente](/eda/eda_05_duplicados.png)

Se eliminaron **8 538 imágenes** agrupadas en **8 050 grupos**. Las fuentes con mayor contaminación cruzada fueron `maize_desease` (~6 508 eliminadas) y `multi_desease` (~1 980). Sin esta limpieza, imágenes idénticas habrían aparecido tanto en train como en validación, inflando artificialmente las métricas.

El dataset actual (y publicado en <a href="https://huggingface.co/datasets/daiv05/corn-leaf-diseases-pests-and-deficiencies" target="_blank" rel="noopener noreferrer">Hugging Face</a>) es **post-deduplicación**. Los registros de cada ejecución se almacenan en `src/cleanup/results/` del repositorio oficial.

- [Ver análisis completo: sección 1.6 de la notebook](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb)

---

## 6. Sesgos identificados

A partir de los análisis anteriores se identifican cinco sesgos que afectan directa o indirectamente el entrenamiento:

![Composición lab/real y desbalance relativo](/eda/eda_06_sesgos.png)

| Sesgo | Clases afectadas | Riesgo |
|---|---|---|
| Desbalance de clases (32.9× entre extremos) | Todas | Alto |
| Dominio de imágenes de laboratorio | `common_rust` (95.4 % lab) | Alto |
| Heterogeneidad visual dentro de la clase | `fall_armyworm` (daño vs. daño + gusano) | Medio-alto |
| Fuente única en clases pequeñas | `nitrogen`, `phosphorus`, `potassium` | Medio-alto |
| Concentración geográfica | `northern_corn_leaf_blight` | Medio |

El sesgo de `fall_armyworm` es especialmente relevante, se decidió mezclar las dos fuentes de imágenes: `hoja con daño sin insecto visible` y `hoja con daño y gusano visible`. Esto se hizo porque al final la clase es `fall_armyworm` y la clasificación y recomendación de tratamiento no depende de la presencia del insecto, sino del patrón de daño foliar. Sin embargo, esto introduce un sesgo, por lo que se tendrá especial cuidado en la validación y en la interpretación de métricas,

- [Ver análisis completo: sección 1.7 de la notebook](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb)

---

## Conclusiones

1. **Muestreo ponderado** El ratio 32.9× entre `healthy` y `potassium_deficiency` hace que entrenar sin `WeightedRandomSampler` o `class_weight` produzca un clasificador que ignorará las clases minoritarias.

2. **La validación debe medir generalización de campo.** Incluir imágenes de laboratorio en validación daría una falsa sensación de buen rendimiento para clases con sesgo de dominio fuerte (`common_rust`).

3. **Duplicados entre fuentes** La deduplicación fue crítica para la integridad de los splits: sin ella, data leakage habría inflado las métricas de validación.

4. **Resolución heterogénea:** El dataset mezcla imágenes con resoluciones muy dispares. Las de `corn_leaf_roboflow` ya vienen fijas a 640×640 px. El pipeline de entrenamiento debe aplicar redimensionamiento consistente a la resolución de entrada del modelo.

5. **Fuentes limitadas en clases pequeñas:** `potassium_deficiency`, `nitrogen_deficiency` y `phosphorus_deficiency` tienen pocas fuentes de origen, lo que reduce la diversidad de condiciones de captura y aumenta el riesgo de sobreajuste a patrones específicos.
