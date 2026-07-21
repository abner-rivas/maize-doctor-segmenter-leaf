# Piloto manual de regiones de interés (Fase 3)

Esta fase prepara un conjunto pequeño y trazable para comprobar si aislar la hoja principal
puede mejorar la clasificación. No se anotan las 31 622 imágenes inicialmente porque primero
se necesita medir la calidad y el costo de la regla de anotación, detectar casos ambiguos y
validar el flujo antes de escalarlo. Esta fase no entrena un detector, no modifica imágenes
originales y no conecta las ROI con el clasificador.

## 1. Seleccionar el piloto

El selector lee un CSV existente con `image_path,label,environment`; no modifica ni regenera
el split. `balanced` recorre las clases equitativamente, prioriza `real`, elimina rutas
duplicadas y redistribuye los cupos que una clase no pueda cubrir. La semilla hace reproducible
la selección. Si el split contiene `source_dataset`, se conserva; de lo contrario se registra
`unknown`.

Linux:

```bash
python3 scripts/dataset/build_leaf_detection_pilot.py \
  --split-csv outputs/splits/seed_42_baseline/test.csv \
  --samples 100 \
  --seed 42 \
  --environments real \
  --copy-mode copy \
  --selection-strategy balanced \
  --output outputs/leaf_detection/pilot
```

PowerShell:

```powershell
python scripts/dataset/build_leaf_detection_pilot.py `
  --split-csv "outputs\splits\seed_42_baseline\test.csv" `
  --samples 100 `
  --seed 42 `
  --environments real `
  --copy-mode copy `
  --selection-strategy balanced `
  --output "outputs\leaf_detection\pilot"
```

`copy` es el modo portable. `hardlink` y `symlink` también están disponibles, pero fallan con
un mensaje explícito si el sistema de archivos o los permisos no los admiten; nunca cambian de
modo silenciosamente. Un CSV opcional pasado con `--priority-manifest` puede priorizar errores
o baja confianza cuando incluye `image_path` y al menos una de las columnas `correct`,
`pred_label` o `pred_prob`.

La estructura generada es:

```text
pilot/
├── images/
├── labels/
├── manifests/
│   ├── pilot_manifest.csv
│   └── pilot_summary.json
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

## 3. Importar y construir el manifiesto ROI

Linux, usando etiquetas YOLO guardadas en `pilot/labels/`:

```bash
python3 scripts/dataset/import_manual_leaf_annotations.py \
  --pilot-manifest outputs/leaf_detection/pilot/manifests/pilot_manifest.csv \
  --annotations outputs/leaf_detection/pilot/labels \
  --format yolo \
  --min-area-ratio 0.15 \
  --output outputs/leaf_detection/pilot/manifests/imported_annotations.csv

python3 scripts/dataset/build_roi_manifest.py \
  --imported-manifest outputs/leaf_detection/pilot/manifests/imported_annotations.csv \
  --output outputs/leaf_detection/pilot/manifests/roi_manifest.csv
```

PowerShell, usando un CSV manual:

```powershell
python scripts/dataset/import_manual_leaf_annotations.py `
  --pilot-manifest "outputs\leaf_detection\pilot\manifests\pilot_manifest.csv" `
  --annotations "outputs\leaf_detection\pilot\annotations.csv" `
  --format csv `
  --min-area-ratio 0.15 `
  --output "outputs\leaf_detection\pilot\manifests\imported_annotations.csv"

python scripts/dataset/build_roi_manifest.py `
  --imported-manifest "outputs\leaf_detection\pilot\manifests\imported_annotations.csv" `
  --output "outputs\leaf_detection\pilot\manifests\roi_manifest.csv"
```

El manifiesto final usa exactamente:

```text
pilot_id,image_path,original_image_path,image_sha256,label,split,environment,source_dataset,roi_x1,roi_y1,roi_x2,roi_y2,roi_width,roi_height,roi_area_ratio,roi_confidence,roi_source,annotation_status,notes
```

Una fila `annotated` manual tiene `roi_confidence=1.0` y `roi_source=manual`. Las filas
`ambiguous` y `rejected` conservan el estado y la nota, pero dejan vacíos todos los datos de
la ROI. Una fila sin archivo de anotación queda `pending` y no está lista para el experimento.

## 4. Validar y revisar visualmente

Linux:

```bash
python3 scripts/checks/validate_roi_manifest.py \
  --roi-manifest outputs/leaf_detection/pilot/manifests/roi_manifest.csv \
  --output outputs/leaf_detection/pilot/validation \
  --preview-samples 25 \
  --preview-output outputs/leaf_detection/pilot/preview
```

PowerShell:

```powershell
python scripts/checks/validate_roi_manifest.py `
  --roi-manifest "outputs\leaf_detection\pilot\manifests\roi_manifest.csv" `
  --output "outputs\leaf_detection\pilot\validation" `
  --preview-samples 25 `
  --preview-output "outputs\leaf_detection\pilot\preview"
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
aceptable. El experimento comparativo y la integración con el clasificador pertenecen a una
fase posterior.
