# Análisis Exploratorio de Datos

El EDA busca responder tres preguntas antes de diseñar el pipeline de entrenamiento: ¿qué hay en el dataset?, ¿qué problemas tiene?, y ¿qué decisiones impone?

El análisis completo y reproducible, con todo el código, está en la notebook [`notebooks/01_eda.ipynb`](https://github.com/daiv05/maize-doctor-classifier/blob/master/notebooks/01_eda.ipynb). Esta página resume los hallazgos y las decisiones que se derivaron de ellos.

## Composición del dataset

El dataset consolidado en `data/clean/` contiene **31 622 imágenes** distribuidas en **9 clases**, procedentes de 6 fuentes públicas. Cada imagen pertenece a un entorno de captura: `lab` (fondo controlado, iluminación artificial) o `real` (campo abierto, iluminación solar).

#### Enfermedades foliares

| Clase | Lab | Real | Total |
|---|---:|---:|---:|
| `common_rust` | 2 150 | 106 | **2 256** |
| `northern_corn_leaf_blight` | 888 | 5 942 | **6 830** |
| `gray_leaf_spot` | 513 | 606 | **1 119** |
| `lethal_necrosis` | 0 | 6 415 | **6 415** |

#### Plagas

| Clase | Lab | Real | Total |
|---|---:|---:|---:|
| `fall_armyworm` | 0 | 4 857 | **4 857** |

#### Deficiencias nutricionales

| Clase | Lab | Real | Total |
|---|---:|---:|---:|
| `nitrogen_deficiency` | 0 | 523 | **523** |
| `phosphorus_deficiency` | 0 | 612 | **612** |
| `potassium_deficiency` | 0 | 266 | **266** |

#### Control (ausencia de enfermedad)

| Clase | Lab | Real | Total |
|---|---:|---:|---:|
| `healthy` | 0 | 8 744 | **8 744** |

## Muestra visual representativa

Antes de las métricas numéricas, un grid de 4 imágenes aleatorias por clase (seed=42) permite evaluar rápidamente la variabilidad intraclase, la calidad de las etiquetas y las diferencias entre entornos lab y campo.

![Muestra visual representativa por clase](/eda/eda_00_muestra_visual.png)

- [Ver análisis completo: sección 1.1 de la notebook](https://github.com/daiv05/maize-doctor-classifier/blob/master/notebooks/01_eda.ipynb)

## 1. Distribución de clases

El dataset presenta un **desbalance severo**: `healthy` (8 744 imágenes) supera en **32.9x** a `potassium_deficiency` (266). Con este desbalance se corre el riesgo de que el modelo aprenda a predecir siempre la clase mayoritaria, ignorando las clases minoritarias.

![Distribución de imágenes por clase](/eda/eda_01_distribucion_clases.png)

Para mitigar este desbalance, inicialmente se planea usar `WeightedRandomSampler` en el DataLoader para igualar la frecuencia efectiva de cada clase durante el entrenamiento. Para las clases más escasas (`potassium_deficiency`, `nitrogen_deficiency`, `phosphorus_deficiency`) aplicar un pipeline de augmentation mas agresivo (`CornMinorityTransforms`).

- [Ver análisis completo: sección 1.2 de la notebook](https://github.com/daiv05/maize-doctor-classifier/blob/master/notebooks/01_eda.ipynb)

## 2. Sesgo de entorno (lab vs real)

La proporción de imágenes de laboratorio vs campo varía drásticamente entre clases, lo que introduce riesgo de *domain shortcut*: una red puede aprender el fondo uniforme de PlantVillage en lugar de los síntomas foliares.

![Proporción lab vs real por clase](/eda/eda_02_lab_vs_real.png)

El caso más crítico es `common_rust`: **95.4 %** de sus 2 256 imágenes proviene de entorno controlado (fondo negro/blanco uniforme). En campo solo hay 106 imágenes.

Esto representa un riesgo significativo de sobreajuste a condiciones de laboratorio, por lo que se deberán aplicar técnicas de augmentation específicas, asi como intentar validar solo sobre imágenes de campo, para comprobar la generalización del modelo.

### Distribución por fuente de origen

Además de la segmentación lab/real, un heatmap fuente×clase revela concentraciones de dependencia: si una clase proviene exclusivamente de una fuente, el modelo podría sobreajustarse a las condiciones específicas de captura de esa fuente.

![Heatmap de distribución por fuente y clase](/eda/eda_02b_heatmap_fuente_clase.png)

Las tres clases de deficiencias nutricionales provienen de una **fuente única** (`maize_nutrient`), y `lethal_necrosis` depende exclusivamente de `maize_africa`. Esto limita la diversidad de condiciones de captura y aumenta el riesgo de sobreajuste a patrones específicos de esas fuentes.

- [Ver análisis completo: sección 1.3 de la notebook](https://github.com/daiv05/maize-doctor-classifier/blob/master/notebooks/01_eda.ipynb)

## 3. Resolución y dimensiones

Las imágenes tienen resoluciones muy dispares. Los histogramas revelan una distribución **claramente multimodal**:

- **Pico en ~256 px:** imágenes de `cropdg` (PlantVillage), capturas de laboratorio a baja resolución. Incluye la mayor parte de `common_rust` y `gray_leaf_spot` en entorno lab.
- **Pico dominante en ~600-640 px:** asociado a `corn_leaf_roboflow` (pre-redimensionadas a 640×640) y `maize_nutrient`. Aquí caen las tres clases de deficiencias nutricionales y parte de `fall_armyworm`.
- **Cola larga hacia 1000-5000+ px:** imágenes de alta resolución de `maize_africa` y `maize_field`, capturadas con cámaras de campo. Incluye la mayoría de `northern_corn_leaf_blight`, `lethal_necrosis` y `healthy`.

![Distribución de resoluciones](/eda/eda_03_resoluciones.png)

Al redimensionar a 224×224 (target del proyecto), las imágenes del primer grupo apenas pierden información, mientras que las de alta resolución sufren una reducción de ~13x, perdiendo detalles finos de las lesiones.

### Resolución y aspect ratio por clase

Los violin plots confirman que **la resolución no es independiente de la clase**:

![Resolución y aspect ratio por clase](/eda/eda_03b_resolucion_por_clase.png)

- **Deficiencias nutricionales y `fall_armyworm`:** concentradas en ~640 px con aspect ratio ~1.0 (cuadradas). El resize a 224×224 es moderado (~2.9x) y sin distorsión.
- **`common_rust` y `gray_leaf_spot`:** distribuciones bimodales (lab ~256 px, campo ~640 px).
- **`healthy`, `lethal_necrosis` y `northern_corn_leaf_blight`:** colas más largas (hasta 3000-5000+ px), con mayor dispersión de aspect ratio. Estas clases sufren la mayor pérdida de detalle durante el resize, y el resize a cuadrado distorsionará la geometría de las lesiones en imágenes con aspect ratio lejos de 1.0.

- [Ver análisis completo: sección 1.4 de la notebook](https://github.com/daiv05/maize-doctor-classifier/blob/master/notebooks/01_eda.ipynb)

## 4. Calidad de imagen

Se evaluaron tres métricas sobre una muestra estratificada de hasta 400 imágenes por clase (seed=42): desenfoque (varianza del Laplaciano), subexposición (brillo medio < 40) y sobreexposición (brillo medio > 230).

![Problemas de calidad por clase](/eda/eda_04_calidad.png)

![Distribuciones de calidad por clase](/eda/eda_04b_calidad_boxplots.png)

**El desenfoque es el único problema de calidad significativo.** Las categorías de subexposición y sobreexposición son prácticamente inexistentes (~0%). En contraste, el blur afecta de forma desigual: `gray_leaf_spot` es la clase más afectada (~38%), seguida de `nitrogen_deficiency` (~28%) y `potassium_deficiency` (~21%), precisamente clases minoritarias, lo que agrava el problema.

**Hallazgo en los boxplots de brillo:** `common_rust` presenta un brillo mediano notablemente inferior (~95) comparado con el resto (~120-140). Esto es evidencia cuantitativa directa del sesgo de dominio: sus imágenes de laboratorio usan fondos oscuros que bajan el brillo global. Si el modelo aprende a asociar brillo bajo con `common_rust`, fallará en imágenes de campo con fondo de vegetación.

> **Nota metodológica:** el umbral de blur (varianza del Laplaciano < 100) es absoluto y no normaliza por resolución. Imágenes de alta resolución naturalmente tienen mayor varianza que imágenes pequeñas, lo que podría subestimar el blur en imágenes de baja resolución (~256 px) y sobreestimarlo en imágenes grandes (~3000+ px).

### Distribución de color (HSV) por clase

En un problema de enfermedades foliares, el color es una señal diagnóstica primaria: `common_rust` produce pústulas anaranjadas, `nitrogen_deficiency` causa amarillamiento generalizado, y `gray_leaf_spot` genera manchas grisáceas.

![Distribución de canales HSV por clase](/eda/eda_04c_color_hsv.png)

![Hue medio: lab vs real](/eda/eda_04d_hue_lab_vs_real.png)

El análisis del canal Hue por clase y entorno permite verificar si las firmas cromáticas de cada enfermedad son detectables estadísticamente, y si existen diferencias de dominio de color entre imágenes lab y real que podrían actuar como shortcuts para el modelo.

- [Ver análisis completo: sección 1.5 de la notebook](https://github.com/daiv05/maize-doctor-classifier/blob/master/notebooks/01_eda.ipynb)

## 5. Duplicados y limpieza

Durante la etapa de construcción del dataset se detectaron duplicados entre fuentes distintas mediante **hash perceptual (pHash)** con `imagededup` (threshold = 0, solo copias exactas a nivel perceptual).

![Duplicados eliminados por clase y fuente](/eda/eda_05_duplicados.png)

Se eliminaron **8 538 imágenes** agrupadas en **8 050 grupos**. Las fuentes con mayor contaminación cruzada fueron `maize_desease` (~6 508 eliminadas) y `multi_desease` (~1 980). Sin esta limpieza, imágenes idénticas habrían aparecido tanto en train como en validación, inflando artificialmente las métricas.

El dataset actual (y publicado en <a href="https://huggingface.co/datasets/daiv05/corn-leaf-diseases-pests-and-deficiencies" target="_blank" rel="noopener noreferrer">Hugging Face</a>) es **post-deduplicación**. Los registros de cada ejecución se almacenan en `src/cleanup/results/` del repositorio oficial.

- [Ver análisis completo: sección 1.6 de la notebook](https://github.com/daiv05/maize-doctor-classifier/blob/master/notebooks/01_eda.ipynb)

## 6. Sesgos identificados

A partir de los análisis anteriores se identifican cinco sesgos que afectan directa o indirectamente el entrenamiento:

![Composición lab/real y desbalance relativo](/eda/eda_06_sesgos.png)

| Sesgo | Clases afectadas | Riesgo |
|---|---|---|
| Desbalance de clases (32.9x entre extremos) | Todas | Alto |
| Dominio de imágenes de laboratorio | `common_rust` (95.4 % lab) | Alto |
| Heterogeneidad visual dentro de la clase | `fall_armyworm` (daño vs. daño + gusano) | Medio-alto |
| Fuente única en clases pequeñas | `nitrogen`, `phosphorus`, `potassium` | Medio-alto |

El sesgo de `fall_armyworm` es especialmente relevante, se decidió mezclar las dos fuentes de imágenes: `hoja con daño sin insecto visible` y `hoja con daño y gusano visible`. Esto se hizo porque al final la clase es `fall_armyworm` y la clasificación y recomendación de tratamiento no depende de la presencia del insecto, sino del patrón de daño foliar. Sin embargo, esto introduce un sesgo, por lo que se tendrá especial cuidado en la validación y en la interpretación de métricas,

- [Ver análisis completo: sección 1.7 de la notebook](https://github.com/daiv05/maize-doctor-classifier/blob/master/notebooks/01_eda.ipynb)

## Conclusiones

1. **Muestreo ponderado** El ratio 32.9x entre `healthy` y `potassium_deficiency` hace que entrenar sin `WeightedRandomSampler` o `class_weight` produzca un clasificador que ignorará las clases minoritarias.

2. **La validación debe medir generalización de campo.** Incluir imágenes de laboratorio en validación daría una falsa sensación de buen rendimiento para clases con sesgo de dominio fuerte (`common_rust`).

3. **Duplicados entre fuentes** La deduplicación fue crítica para la integridad de los splits: sin ella, data leakage habría inflado las métricas de validación.

4. **Resolución heterogénea y dependiente de la clase:** El dataset mezcla resoluciones multimodales (~256 px lab, ~640 px Roboflow, ~3000+ px campo). Las clases de deficiencias están concentradas en ~640 px mientras que `healthy` y `lethal_necrosis` tienen colas hasta 5000+ px. El pipeline aplica resize uniforme a 224×224.

5. **Fuentes limitadas en clases pequeñas:** `potassium_deficiency`, `nitrogen_deficiency` y `phosphorus_deficiency` tienen una sola fuente de origen (`maize_nutrient`), lo que reduce la diversidad de condiciones de captura y aumenta el riesgo de sobreajuste a patrones específicos.

6. **El brillo bajo de `common_rust` es evidencia del sesgo lab:** su mediana de brillo (~95) es significativamente inferior al resto de clases (~120-140), reflejando los fondos oscuros de laboratorio. El modelo podría usar esta señal como shortcut en lugar de aprender los síntomas foliares.
