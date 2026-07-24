# Piloto manual y diagnóstico de regiones de interés

Esta fase prepara un conjunto pequeño y trazable para comprobar si aislar la hoja principal
puede mejorar la clasificación. No se anotan las 31 622 imágenes inicialmente porque primero
se necesita medir la calidad y el costo de la regla de anotación, detectar casos ambiguos y
validar el flujo antes de escalarlo. El diagnóstico posterior usa un checkpoint ya existente:
no entrena un detector o clasificador y no modifica imágenes ni checkpoints.

## 1. Seleccionar el piloto

El selector lee un CSV existente con `image_path,label,environment`; no modifica ni regenera
el split. `balanced` recorre las clases equitativamente, prioriza `real`, elimina rutas
duplicadas y redistribuye los cupos que una clase no pueda cubrir. La semilla hace reproducible
la selección. Si el split contiene `source_dataset`, se conserva; de lo contrario se registra
`unknown`.

Linux:

```bash
python3 scripts/dataset/build_leaf_detection_pilot.py \
  --split-csv data/splits/seed_42_baseline/test.csv \
  --samples 100 \
  --seed 42 \
  --environments real \
  --copy-mode copy \
  --selection-strategy balanced \
  --output data/leaf_detection/pilot
```

PowerShell:

```powershell
python scripts/dataset/build_leaf_detection_pilot.py `
  --split-csv "data\splits\seed_42_baseline\test.csv" `
  --samples 100 `
  --seed 42 `
  --environments real `
  --copy-mode copy `
  --selection-strategy balanced `
  --output "data\leaf_detection\pilot"
```

`copy` es el modo portable. `hardlink` y `symlink` también están disponibles, pero fallan con
un mensaje explícito si el sistema de archivos o los permisos no los admiten; nunca cambian de
modo silenciosamente. Un CSV opcional pasado con `--priority-manifest` puede priorizar errores
o baja confianza cuando incluye `image_path` y al menos una de las columnas `correct`,
`pred_label` o `pred_prob`.

La estructura generada es:

```text
pilot/
├── annotations/cvat/
├── images/
├── labels/
├── manifests/
│   ├── pilot_manifest.csv
│   └── pilot_summary.json
├── packages/
├── annotation_guide.md
└── README.md
```

El manifiesto inicial usa exactamente:

```text
pilot_id,pilot_image_path,original_image_path,original_filename,image_sha256,label,split,environment,source_dataset,selected_by,annotation_status,copy_mode
```

## 2. Anotar la hoja principal

La única clase de ROI es `0 = maize_leaf`. Se marca solamente la hoja principal que debería
analizar el clasificador: primero la de mayor área, luego la más centrada y con síntomas
visibles. No se marcan lesiones, hojas del fondo, suelo, manos, tallos ni cielo. La enfermedad
no se vuelve a anotar porque ya está en `label`.

Use `ambiguous` si varias hojas tienen importancia similar o no es posible elegir una sin una
decisión arbitraria. Use `rejected` si no hay una hoja de maíz útil, la imagen es ilegible, la
hoja es extremadamente pequeña o la calidad es insuficiente. No se elige automáticamente una
de varias cajas YOLO: una etiqueta con múltiples líneas queda como `ambiguous`.

Los formatos aceptados son una caja YOLO normalizada por archivo:

```text
0 center_x center_y width height
```

o un CSV en píxeles:

```text
pilot_id,x1,y1,x2,y2,status,notes
```

También se acepta el XML nativo de CVAT con `--format cvat_xml`. En el piloto
actual ésta es la fuente oficial:

```text
data/leaf_detection/pilot/annotations/cvat/annotations.xml
```

Contiene 100 imágenes y 100 cajas de la clase `maize_leaf`, sin imágenes sin
caja ni imágenes con cajas múltiples. El exportador YOLO sólo materializó 48
archivos porque 52 cajas tenían rotación. Esas 52 anotaciones no se perdieron:
el importador XML rota las cuatro esquinas alrededor del centro de la caja,
calcula la envolvente alineada a ejes, redondea hacia afuera, aplica clipping a
la imagen y valida la ROI. El resultado fue 48 cajas directas, 52 convertidas
desde rotación y 36 cajas limitadas por clipping.

## 3. Importar y construir el manifiesto ROI

Linux, usando el XML oficial de CVAT:

```bash
python3 scripts/dataset/import_manual_leaf_annotations.py \
  --pilot-manifest data/leaf_detection/pilot/manifests/pilot_manifest.csv \
  --annotations data/leaf_detection/pilot/annotations/cvat/annotations.xml \
  --format cvat_xml \
  --min-area-ratio 0.15 \
  --output data/leaf_detection/pilot/manifests/imported_annotations.csv

python3 scripts/dataset/build_roi_manifest.py \
  --imported-manifest data/leaf_detection/pilot/manifests/imported_annotations.csv \
  --output data/leaf_detection/pilot/manifests/roi_manifest.csv
```

Alternativamente, usando etiquetas YOLO guardadas en `pilot/labels/`:

```bash
python3 scripts/dataset/import_manual_leaf_annotations.py \
  --pilot-manifest data/leaf_detection/pilot/manifests/pilot_manifest.csv \
  --annotations data/leaf_detection/pilot/labels \
  --format yolo \
  --min-area-ratio 0.15 \
  --output data/leaf_detection/pilot/manifests/imported_annotations.csv

python3 scripts/dataset/build_roi_manifest.py \
  --imported-manifest data/leaf_detection/pilot/manifests/imported_annotations.csv \
  --output data/leaf_detection/pilot/manifests/roi_manifest.csv
```

PowerShell, usando un CSV manual:

```powershell
python scripts/dataset/import_manual_leaf_annotations.py `
  --pilot-manifest "data\leaf_detection\pilot\manifests\pilot_manifest.csv" `
  --annotations "data\leaf_detection\pilot\annotations.csv" `
  --format csv `
  --min-area-ratio 0.15 `
  --output "data\leaf_detection\pilot\manifests\imported_annotations.csv"

python scripts/dataset/build_roi_manifest.py `
  --imported-manifest "data\leaf_detection\pilot\manifests\imported_annotations.csv" `
  --output "data\leaf_detection\pilot\manifests\roi_manifest.csv"
```

El manifiesto final usa exactamente:

```text
pilot_id,image_path,original_image_path,image_sha256,label,split,environment,source_dataset,roi_x1,roi_y1,roi_x2,roi_y2,roi_width,roi_height,roi_area_ratio,roi_confidence,roi_source,annotation_status,notes
```

Una fila `annotated` manual tiene `roi_confidence=1.0` y `roi_source=manual`. Las filas
`ambiguous` y `rejected` conservan el estado y la nota, pero dejan vacíos todos los datos de
la ROI. Una fila sin archivo de anotación queda `pending` y no está lista para el experimento.

El manifiesto actual contiene 100 filas estructuralmente válidas: 99
`annotated`, 1 `ambiguous`, 0 `pending` y 0 `rejected`.

## 4. Validar y revisar visualmente

Linux:

```bash
python3 scripts/checks/validate_roi_manifest.py \
  --roi-manifest data/leaf_detection/pilot/manifests/roi_manifest.csv \
  --output outputs/leaf_detection/pilot/validation \
  --min-area-ratio 0.15 \
  --preview-samples 100 \
  --preview-output outputs/leaf_detection/pilot/previews
```

PowerShell:

```powershell
python scripts/checks/validate_roi_manifest.py `
  --roi-manifest "data\leaf_detection\pilot\manifests\roi_manifest.csv" `
  --output "outputs\leaf_detection\pilot\validation" `
  --min-area-ratio 0.15 `
  --preview-samples 100 `
  --preview-output "outputs\leaf_detection\pilot\previews"
```

La validación produce `roi_validation_summary.json` y `roi_validation_rows.csv`. Comprueba
archivos, hashes, vocabularios, geometría, área mínima, duplicados y posibles fugas entre
splits. Un nombre repetido es sólo una advertencia; una misma imagen por hash en varios splits
es un error. La cobertura se resume por clase, split, entorno, estado, origen y área.

Cada vista previa contiene cinco paneles: original, bbox, margen, recorte y letterbox. Revise
que la caja incluya la hoja diagnóstica sin incorporar demasiado fondo, que el margen no corte
síntomas y que el letterbox no deforme la hoja. El piloto está listo para un experimento
comparativo cuando no tiene filas inválidas ni `pending`, no presenta fuga por ruta o hash, las
clases y entornos previstos tienen cobertura, y una revisión visual confirma una calidad de ROI
aceptable.

La vista histórica de `image_0021` no debe borrarse. Esa imagen conserva
`annotation_status=ambiguous`; su caja reportada cubría `0.092799` de la imagen, por debajo
del mínimo `0.15`.

## 5. Perfiles de procesamiento

Desde el diagnóstico existen dos perfiles explícitos y diferentes:

```text
baseline_full
imagen completa → transformación histórica → clasificador

baseline_roi
imagen RGB → bbox del manifiesto → validación → clipping → margen
→ recorte → letterbox → augmentations → normalización → clasificador
```

`baseline_full` conserva exactamente la referencia histórica. El valor global de
`processing_profile` es `baseline_full` y `leaf_detection.enabled` continúa en `false`.
`baseline_roi` sólo se selecciona de manera local en el experimento; todavía no está conectado
a `CornDataset`, `predict.py`, LIME o Grad-CAM.

`LeafProcessingProfile` y `LeafImageProcessor`, en `src/preprocessing/leaf_processor.py`, son
la entrada reutilizable. La API impide pasar augmentations en `val`, `test`, `inference`,
`lime` o `gradcam`. Cada ejecución ROI conserva perfil, fuente y coordenadas, área y confianza,
margen, tamaño, padding, conservación de aspecto, fallback, ruta y SHA-256 del manifiesto y
versión del procesador.

## 6. Experimento `diagnostic_full_vs_manual_roi`

El experimento ejecuta inferencia pareada sobre las mismas 100 filas del manifiesto con un
checkpoint existente. `image_0021` aparece en los CSV y previews, pero queda fuera de las
métricas principales; cada ejecución conservó 100 filas, 99 incluidas, 1 excluida y 0
fallbacks.

```bash
python scripts/experiments/compare_full_vs_manual_roi.py \
  --checkpoint outputs/baselines/efficientnet_b0/20260709_040040/best.pth \
  --model efficientnet_b0 \
  --roi-manifest data/leaf_detection/pilot/manifests/roi_manifest.csv \
  --config config/dataset.yaml \
  --output outputs/leaf_detection/pilot/diagnostic_experiment/efficientnet_b0 \
  --device auto
```

El script valida existencia, modelo, número de clases y formas del `state_dict` antes de crear
salidas. Construye el modelo con `pretrained=False`, así que no descarga pesos. Calcula
accuracy, precision/recall/F1 macro, loss, matriz de confusión, métricas por clase y cambios
entre perfiles. Produce:

```text
diagnostic_experiment/
├── efficientnet_b0/
├── shufflenet_v2_x1_0/
└── efficientnet_lite0/
```

Esta salida es un diagnóstico sobre un clasificador ya entrenado con imágenes completas. No es
entrenamiento, no constituye un baseline oficial y no demuestra por sí sola el rendimiento de
un futuro clasificador entrenado con ROI.

Los tres diagnósticos ya fueron ejecutados con checkpoints históricos. No deben
volver a ejecutarse para documentar sus resultados. Consulte
[Diagnóstico de imagen completa frente a ROI manual](manual-roi-diagnostic.md)
para ver metodología, tablas globales, análisis por modelo, limitaciones,
hipótesis y próximos experimentos.
