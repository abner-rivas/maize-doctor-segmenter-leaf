# Datos derivados del proyecto

Esta carpeta contiene datos materializados para flujos reproducibles. No contiene el
dataset completo de 31,622 imágenes, que continúa fuera del repositorio y se resuelve
mediante `DATASET_ROOT`.

```text
data/
├── splits/
│   └── seed_42_baseline/       # train.csv, val.csv, test.csv y auditoría
└── leaf_detection/
    ├── pilot/
    │   ├── images/             # 100 imágenes copiadas del piloto
    │   ├── labels/             # etiquetas YOLO, cuando aplique
    │   ├── manifests/          # selección, importación y roi_manifest.csv
    │   ├── annotations/cvat/   # annotations.xml: fuente oficial, 100 cajas
    │   └── packages/           # ZIP y paquetes portables
    ├── external_sources/
    │   ├── *_yolo26/           # fuentes externas YOLO, inmutables
    │   ├── *_coco_segmentation/# respaldos COCO para contraste
    │   └── packages/           # paquetes originales, no resultados
    └── detector_dataset/
        ├── annotation_batches/ # 350 train + 75 val pendientes de CVAT
        ├── test/               # piloto retenido: 99 válidas + 1 ambigua
        ├── all/                # pool definitivo: 1 155 imágenes + TXT
        ├── images/{train,val,test}/ # 809/173/173 imágenes derivadas
        ├── labels/{train,val,test}/ # TXT correspondientes
        ├── manifests/          # consolidación, splits, locks y fugas
        ├── previews/           # revisión visual previa a crear splits
        ├── cvat/               # paquetes train/val sin etiquetas ficticias
        ├── dataset.yaml        # configuración portable de los tres splits
        └── dataset.yaml.template
```

Separación de responsabilidades:

- `data/`: splits, imágenes piloto, anotaciones y manifiestos.
- `outputs/`: modelos, checkpoints, métricas, predicciones, auditorías, previews,
  validaciones y experimentos diagnósticos.
- `scripts/`: herramientas ejecutables organizadas por propósito.
- `src/`: código reutilizable del paquete.
- `docs/`: documentación del proyecto.
- `public/`: recursos visuales publicados por el sitio de documentación.

`PROJECT_DATA_ROOT` permite cambiar esta raíz en servidores. Si no se define, se usa
`<repositorio>/data`.

El piloto actual tiene 100 imágenes reales, seleccionadas con semilla 42 y sin
duplicados. Su XML nativo de CVAT conserva 100 cajas `maize_leaf`: 48 sin
rotación y 52 convertidas desde cajas rotadas; 36 necesitaron clipping. El
manifiesto ROI contiene 99 filas `annotated` y una fila `ambiguous`
(`image_0021`), sin filas `pending` o `rejected`.

Los previews y la validación están bajo `outputs/leaf_detection/pilot/`. Los
resultados diagnósticos de los tres modelos históricos están en
`outputs/leaf_detection/pilot/diagnostic_experiment/`; no pertenecen a
`data/` porque son resultados, no entradas reproducibles.

Las dos fuentes externas de segmentación permanecen sin modificar bajo
`data/leaf_detection/external_sources/`. Su auditoría reproducible, tablas,
gráficos y previews viven en
`outputs/leaf_detection/external_sources_eda/`; los gráficos publicados se
copian a `public/leaf_detection/external_sources_eda/`. Consulte
[`docs/es/leaf-detection/external-segmentation-datasets-eda.md`](../docs/es/leaf-detection/external-segmentation-datasets-eda.md).

La selección inicial del detector usa exclusivamente 350 imágenes de
`train.csv` y 75 de `val.csv`, con semilla 42, balance de clase y una cuota
80/20 de imágenes reales/laboratorio. Train y val permanecen `pending` y no
tienen carpetas de etiquetas. El piloto se conserva sólo como test; sus 99
casos válidos tienen una materialización YOLO trazable y `image_0021` permanece
ambigua. Consulte
[`docs/es/leaf-detection/yolo26-detector-dataset.md`](../docs/es/leaf-detection/yolo26-detector-dataset.md).

La consolidación externa de segmentación añade un pool definitivo separado
bajo `all/`: 1 155 imágenes, 1 155 etiquetas y 1 224 polígonos de una sola
clase `0 = maize_leaf`. Se aplicaron 35 decisiones humanas únicas: 16
`approved`, 16 `exclude` y 3 `needs_reannotation`. Los tres últimos casos
permanecen fuera del pool y están documentados en `reannotation_queue.csv`.
Ninguna aprobación anuló las validaciones semánticas, sintácticas o
topológicas.

La trazabilidad completa está en
`detector_dataset/manifests/consolidation_manifest.csv`; los resultados de
validación están en
`outputs/leaf_detection/detector_dataset_consolidation/`.

El gate actual está en
`detector_dataset/manifests/dataset_lock.json`. Su estado es
`ready_for_split_generation`: no quedan revisiones pendientes, contradicciones,
duplicados, fugas con el piloto ni errores geométricos. El fingerprint
definitivo es
`c087af60c2bad1c133c4ea8b14cee945405bfe4976aa80c4faf089d7a4b9e38c`.

Sobre ese padre congelado se generaron 1 035 grupos con semilla 42. La
materialización final contiene 809/173/173 imágenes y 858/183/183 máscaras en
train/val/test. La fuente minoritaria `corn` quedó 109/23/23 y la fuente
grande 700/150/150. No hay fugas exactas, grupales, Roboflow, perceptuales
(Hamming menor o igual a 4) ni contra el piloto. El
`detector_dataset/manifests/split_lock.json` quedó
`ready_for_training_preflight`.

## Ciclo de vida de artefactos

Rutas activas:

- `data/splits/seed_42_baseline/`: splits oficiales;
- `data/leaf_detection/pilot/`: piloto retenido y fuente CVAT;
- `data/leaf_detection/external_sources/`: fuentes externas inmutables;
- `data/leaf_detection/detector_dataset/`: lotes de anotación, test retenido y
  pool definitivo con splits de segmentación reproducibles.

Existen tres copias exactas de las 100 imágenes del piloto: la fuente activa,
el paquete desempaquetado y `detector_dataset/test/images/`. La última es una
materialización intencional del test; los paquetes son evidencia protegida.
Ninguna se usa como entrenamiento y ninguna fue eliminada.

Los resultados históricos y la clasificación completa de artefactos se
documentan en
[`docs/es/leaf-detection/history.md`](../docs/es/leaf-detection/history.md) y
`outputs/repository_audit/`.
