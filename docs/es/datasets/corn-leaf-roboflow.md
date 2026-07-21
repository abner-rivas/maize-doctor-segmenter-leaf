# Corn Leaf - Roboflow (corn-leaf-hd9iy)

## Identificador

`corn_leaf_roboflow`

## Descripción

Dataset de detección de enfermedades y deficiencias nutricionales en hojas de maíz, publicado en Roboflow Universe por el workspace `labonis-workspace`. Las imágenes están anotadas en formato YOLO (bounding boxes y segmentación), capturadas en condiciones de campo real.

Las clases incluidas son:

| ID YOLO | Clase | Descripción |
|---|---|---|
| 0 | Fall Armyworm Damage | Daño foliar por gusano cogollero (*Spodoptera frugiperda*) |
| 1 | Healthy Leaf | Hoja sana sin síntomas visibles |
| 2 | Leaf Spot | Mancha foliar (mixta: GLS + otras) |
| 3 | Magnesium Deficiency | Deficiencia de magnesio |
| 4 | Nitrogen Deficiency | Deficiencia de nitrógeno |
| 5 | Northern Leaf Blight | Tizón foliar del norte (NCLB) |
| 6 | Phosphorus Deficiency | Deficiencia de fósforo |
| 7 | Potassium Deficiency | Deficiencia de potasio |

## Estructura

El dataset sigue el formato estándar YOLO con división en splits:

```
corn-leaf-roboflow/
├── train/
│   ├── images/   (2 888 imágenes)
│   └── labels/
├── valid/
│   ├── images/   (702 imágenes)
│   └── labels/
├── test/
│   ├── images/   (353 imágenes)
│   └── labels/
├── data.yaml
├── README.dataset.txt
└── README.roboflow.txt
```

Las anotaciones mezclan formato YOLO bbox (5 campos por línea) y segmentación de polígono (N > 5 campos). Ambos son válidos; el `class_id` es siempre el primer token.

## Aspectos importantes

- La clase dominante en anotaciones es `Leaf Spot` (ID 2), con ~3 946 instancias frente a ~55 de Nitrogen Deficiency y ~46 de Phosphorus Deficiency. El desbalance es significativo.
- Las imágenes fueron preprocesadas: auto-orientación EXIF y redimensionado a **640 x 640 px** (stretch). No se aplicaron técnicas de augmentation.
- Las imágenes son de campo real; no existen contrapartes de laboratorio en este dataset.

::: warning Desbalance severo en deficiencias nutricionales
Las clases `Nitrogen Deficiency` (~55 instancias), `Phosphorus Deficiency` (~46) y `Potassium Deficiency` (~52) tienen muy baja representación. Requieren augmentation o fuentes adicionales antes de usarse en entrenamiento.
:::

## Tamaño y distribución

- **Total de imágenes:** 3 943
- **Resolución:** 640 x 640 px
- **Formato:** JPEG / JPG

Distribución aproximada de instancias por clase (conteo de etiquetas en todos los splits):

| Clase | Instancias (aprox.) |
|---|---|
| Fall Armyworm Damage | ~503 |
| Healthy Leaf | ~619 |
| Leaf Spot | ~3 946 |
| Magnesium Deficiency | ~779 |
| Nitrogen Deficiency | ~55 |
| Northern Leaf Blight | ~292 |
| Phosphorus Deficiency | ~46 |
| Potassium Deficiency | ~52 |

## Formato

- Anotaciones en **YOLOv8** (bbox + segmentación de polígono)
- Imágenes en formato JPEG

## Auditoría visual de anotaciones (Fase 1)

Antes de diseñar o entrenar un detector se deben auditar las anotaciones originales. La
herramienta `scripts/dataset/audit_leaf_annotations.py` relaciona imágenes y archivos `.txt`
por su ruta relativa y nombre base, valida las líneas YOLO, dibuja únicamente los bounding
boxes compatibles y señala polígonos o registros inesperados. El dataset fuente se trata como
solo lectura.

Ejemplo para PowerShell, pasando directamente la carpeta original de Roboflow:

```powershell
python scripts/dataset/audit_leaf_annotations.py `
  --dataset-root "C:\ruta\al\corn-leaf-roboflow" `
  --samples 100 `
  --seed 42 `
  --output "outputs\leaf_detection\annotation_audit" `
  --splits train valid val test
```

Si se omite `--dataset-root`, el script usa la resolución oficial del proyecto y busca
`corn-leaf-roboflow` bajo `DATASET_ROOT/raw/`. Si se omite `--output`, escribe bajo
`OUTPUT_ROOT/leaf_detection/annotation_audit/` (o `outputs/leaf_detection/annotation_audit/`
cuando `OUTPUT_ROOT` no está definido).

Las copias visuales quedan en `images/<split>/`; `summary.json` contiene los totales globales,
la distribución por split y clase, ejemplos de problemas, la semilla y las rutas utilizadas.
Los conteos `valid_annotations`, `invalid_annotations` y `possible_polygons` cubren todos los
archivos de etiqueta encontrados, mientras `audited_images` indica cuántas imágenes componen
la muestra visual. `summary.csv` presenta los mismos conteos en formato tabular.

Después de revisar las imágenes se debe decidir si las anotaciones delimitan hojas completas,
lesiones, zonas sintomáticas u objetos mezclados, y si su consistencia y cobertura justifican
reutilizarlas. Esta fase no presupone que representen la hoja principal ni autoriza todavía el
entrenamiento de un detector.

## Origen

1. [Dataset en Roboflow Universe](https://universe.roboflow.com/labonis-workspace/corn-leaf-hd9iy/dataset/4)

## Citación

@misc{ corn-leaf-hd9iy_dataset,
  title = { Corn Leaf Dataset },
  type = { Open Source Dataset },
  author = { Labonis Workspace },
  howpublished = { \url{ https://universe.roboflow.com/labonis-workspace/corn-leaf-hd9iy } },
  url = { https://universe.roboflow.com/labonis-workspace/corn-leaf-hd9iy },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2026 },
  month = { may },
  note = { visited on 2026-06-11 },
}

## Licencia

- **CC BY 4.0**
