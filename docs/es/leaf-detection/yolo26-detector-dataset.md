# Preparación del dataset para el detector YOLO26n

> Estado histórico/alternativo: los lotes de 350 train y 75 val siguen
> pendientes de anotación y no se entrenó YOLO26n. Después del EDA de fuentes
> externas, el paso inmediato es consolidar y validar segmentación. Esta ruta
> de bounding boxes se conserva como alternativa y evidencia, no como pipeline
> activo ya entrenado.

## Alcance

Esta fase prepara imágenes y paquetes de anotación para un futuro detector de
hojas. **No entrena YOLO, no instala Ultralytics y no descarga `yolo26n.pt`.**
Tampoco activa inferencia ni entrenamiento fuera del flujo de segmentación.

El detector tendrá una sola clase, `maize_leaf`. Su función será localizar
hojas; cualquier otra tarea queda fuera del alcance de este repositorio.

## Decisión inicial

Se propone Ultralytics YOLO26n para detección de objetos por ser la variante
nano y priorizar eficiencia y un futuro despliegue móvil. La documentación
oficial presenta `yolo26n.pt` como el modelo de detección nano de la familia
[YOLO26](https://docs.ultralytics.com/models/yolo26). `Ultralytics YOLO
Detection 1.0`, visto en CVAT, era el nombre del formato de exportación y no la
versión de un modelo.

La elección todavía es provisional:

```text
framework: Ultralytics
versión candidata: 8.4.104
modelo futuro: yolo26n.pt
tarea: object detection
clase: maize_leaf
estado: no instalado, no descargado, no entrenado
```

Sólo se comparará YOLO26n con YOLO26s si la variante nano no alcanza el recall
requerido.

## Auditoría de dependencias

La versión candidata está fijada como `ultralytics==8.4.104`; no se debe usar
una versión flotante. Su metadata oficial en
[PyPI](https://pypi.org/project/ultralytics/8.4.104/) declara Python `>=3.8`,
PyTorch `>=1.8.0` y torchvision `>=0.9.0`. Esas restricciones son compatibles
con `pyproject.toml`, que declara Python `>=3.11`, PyTorch `>=2.2,<2.13` y
torchvision `>=0.17,<0.28`.

Auditoría del shell usado para preparar la selección:

| Componente | Estado |
|---|---|
| Python | 3.12.3 |
| Plataforma | Linux 7.0.0-28 generic, x86_64, glibc 2.39 |
| PyTorch | no instalado en este shell |
| torchvision | no instalado en este shell |
| CUDA reportada | no disponible porque PyTorch no está instalado |
| Ultralytics | no instalado |
| Candidata | `ultralytics==8.4.104` |

No existen `requirements*.txt`, `poetry.lock`, `uv.lock` ni otro lockfile de
Python. `package-lock.json` corresponde al sitio Node/VitePress y no fija
dependencias Python. Antes de instalar Ultralytics se debe crear o activar el
entorno del proyecto, resolver conjuntamente el pin candidato y los rangos de
PyTorch/torchvision, revisar el resultado del resolver y ejecutar la suite. Esta
fase no hizo esa instalación.

## Origen y particiones

La fuente activa es el dataset autocontenido
`data/leaf_detection/detector_dataset/`. Sus manifiestos y locks conservan la
procedencia de cada imagen sin depender de archivos de otro proyecto.

| Partición del detector | Imágenes | Estado inicial | Origen |
|---|---:|---|---|
| train | 350 | `pending` | sólo split oficial train |
| val | 75 | `pending` | sólo split oficial val |
| test evaluable | 99 | `annotated` | piloto retenido del test oficial |
| test documentado no evaluable | 1 | `ambiguous` | `image_0021` |

Las 100 imágenes del piloto están documentadas dentro de test, pero sólo 99
tienen etiqueta YOLO y entrarán en las métricas principales. `image_0021`
conserva su estado ambiguo y no tiene etiqueta generada. Ninguna imagen del
piloto fue usada en train o val.

Estas 99 cajas siguen siendo válidas y trazables bajo la regla histórica de
**hoja principal**. Antes de considerar oficial una evaluación multihoja del
detector, el test retenido debe revisarse con la regla nueva para añadir otras
hojas visibles cuando corresponda. Usarlo sin esa revisión podría contar como
falsos positivos detecciones correctas de hojas que el piloto no etiquetó. El
manifiesto marca explícitamente esta revisión pendiente.

## Estrategia reproducible

El selector `scripts/dataset/build_leaf_detector_annotation_set.py` usa semilla
42 y verifica extensión, integridad PIL y SHA-256 antes de copiar. Equilibra las
clases y reserva 80 % para imágenes reales y 20 % para laboratorio. Dentro de
cada clase y entorno intercala señales automáticas verificables:

- orientación vertical, horizontal o aproximadamente cuadrada;
- resolución pequeña, mediana o grande;
- relación de aspecto estrecha, moderada o ancha.

No inventa atributos visuales. Complejidad del fondo, tamaño aparente de la
hoja, presencia de varias hojas y hojas parcialmente cortadas quedan marcados
como revisión manual pendiente en cada manifiesto.

Distribución obtenida:

| Clase | Train | Val |
|---|---:|---:|
| common_rust | 39 | 9 |
| fall_armyworm | 39 | 9 |
| gray_leaf_spot | 39 | 9 |
| healthy | 39 | 8 |
| lethal_necrosis | 39 | 8 |
| nitrogen_deficiency | 39 | 8 |
| northern_corn_leaf_blight | 39 | 8 |
| phosphorus_deficiency | 39 | 8 |
| potassium_deficiency | 38 | 8 |

| Entorno | Train | Val |
|---|---:|---:|
| real | 280 | 60 |
| lab | 70 | 15 |

| Orientación | Train | Val |
|---|---:|---:|
| horizontal | 135 | 33 |
| vertical | 98 | 21 |
| aproximadamente cuadrada | 117 | 21 |

No hubo exclusiones de candidatos de train o val: las imágenes retenidas del
piloto ya pertenecían al split oficial test y los splits oficiales no
presentaron cruces.

## Estructura materializada

```text
data/leaf_detection/detector_dataset/
├── annotation_batches/
│   ├── train/images/              # 350 imágenes, sin labels
│   └── val/images/                # 75 imágenes, sin labels
├── test/
│   ├── images/                    # 100 retenidas y documentadas
│   └── labels/                    # 99 etiquetas válidas
├── manifests/
│   ├── train_selection.csv
│   ├── val_selection.csv
│   ├── test_selection.csv
│   ├── selection_summary.json
│   └── leakage_report.json
├── cvat/
│   ├── train_annotation_batch.zip
│   └── val_annotation_batch.zip
├── dataset.yaml.template
└── README.md
```

No existen `annotation_batches/train/labels`,
`annotation_batches/val/labels` ni la carpeta final `yolo/`. Se crearán sólo
después de recibir y validar anotaciones reales.

## Regla de anotación para el detector

La regla nueva es diferente de la del piloto ROI:

| Piloto anterior | Dataset del detector |
|---|---|
| una caja para la hoja principal | cajas para todas las hojas visibles y claras |
| orientado a un recorte diagnóstico | orientado a aprender detección multihoja |
| varias hojas podían volver ambiguo el caso | varias cajas por fotografía son válidas |

En CVAT se debe:

1. usar únicamente la clase `maize_leaf`;
2. marcar todas las hojas de maíz visibles y suficientemente claras;
3. permitir varias cajas por fotografía;
4. no marcar tallos, suelo, manos, cielo ni lesiones aisladas;
5. evitar cajas degeneradas;
6. marcar imágenes imposibles como `rejected` o `ambiguous`.

Los ZIP de train y val contienen sólo imágenes y `annotation_guide.md`; no
incluyen etiquetas ficticias.

## Trazabilidad del test

`test_selection.csv` conserva, por imagen:

- fuente XML de CVAT;
- identificador original del piloto;
- caja convertida;
- rotación original;
- método `direct_bbox` o `rotated_to_axis_aligned`;
- indicador de clipping;
- área relativa;
- estado y notas.

Los archivos originales
`data/leaf_detection/pilot/manifests/roi_manifest.csv` y
`data/leaf_detection/pilot/annotations/cvat/annotations.xml` no fueron
modificados. Las 99 etiquetas de test son una materialización YOLO para
evaluación y mantienen su trazabilidad mediante el manifiesto.

## Protección contra fugas

[`leakage_report.json`](../../../data/leaf_detection/detector_dataset/manifests/leakage_report.json)
compara ruta original, nombre de archivo y SHA-256:

| Comparación | Rutas | Nombres | SHA-256 |
|---|---:|---:|---:|
| train vs. val | 0 | 0 | 0 |
| train vs. test | 0 | 0 | 0 |
| val vs. test | 0 | 0 | 0 |

También confirma que cada fila procede de su split declarado.

## Reproducción

El dataset congelado se verifica mediante sus manifiestos y fingerprints; no
se reconstruye a partir de insumos de otro proyecto. La salida es protegida y no
sobrescribe una selección previa.

## Plantilla futura

`dataset.yaml.template` queda preparada, pero no debe usarse todavía:

```yaml
path: data/leaf_detection/detector_dataset/yolo
train: images/train
val: images/val
test: images/test

names:
  0: maize_leaf
```

El dataset final sólo se materializará después de anotar y validar train y val.
Hasta entonces, sus manifiestos continúan como evidencia histórica y no como entrada
activa de entrenamiento.
