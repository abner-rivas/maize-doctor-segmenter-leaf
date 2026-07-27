# Auditoría de clases y restauración reproducible de splits (Fase 3.5)

La configuración define qué etiquetas entiende el modelo, pero `clean/` determina qué datos
existen realmente. Entrenar cuando ambas fuentes discrepan puede omitir una carpeta completa,
producir un `class_to_idx` incorrecto o hacer que resultados y documentación representen otra
versión del dataset. Por eso la auditoría es una compuerta previa a cualquier split o
experimento ROI.

## Estado actual

La restauración posterior quedó validada en
`outputs/dataset_audit_final/`: `DATASET_ROOT` apunta a
`/home/desarrolloab/Documentos/ML/maize_dataset/data`, existen las nueve clases,
hay 31 622 imágenes admitidas —3 551 de laboratorio y 28 071 reales—, no hay
discrepancias críticas y `ready_for_splits=true`.

`outputs/dataset_audit_updated/` contiene los mismos tres archivos, idénticos
por SHA-256, y se conserva como candidato de limpieza pendiente de aprobación.
`outputs/dataset_audit/` no es el estado actual: preserva el diagnóstico
histórico descrito a continuación.

## Hallazgo histórico de la copia anterior

La revisión inicial del repositorio y del árbol físico encontró una copia local
anterior del corpus:

- `clean/` conserva 77 imágenes en `aphids_pest` y no contiene `lethal_necrosis`.
- Git registra en el commit `7967572` el cambio de alcance que sustituyó `aphids_pest` por
  `lethal_necrosis` en la configuración y documentación.
- El mismo commit documentó 6 415 imágenes de `lethal_necrosis` y un total provisional de
  31 623. El commit `9a79c45` corrigió `fall_armyworm` de 4 858 a 4 857 y el total a 31 622.
- La notebook EDA guardada fue ejecutada sobre las nueve clases nuevas y registra 31 622
  imágenes, incluidas 6 415 de `lethal_necrosis`.
- La copia local tiene 25 284 archivos con extensiones admitidas por el generador. Además posee
  un TIFF de `fall_armyworm` que el generador oficial no indexa. Los 25 284 admitidos más ese
  TIFF coinciden con los 25 285 archivos de la tabla anterior al cambio de alcance.

En ese momento, la configuración y la documentación describían una revisión
posterior mientras `clean/` local no había sido actualizado. La auditoría no
renombró, eliminó ni sustituyó nada y marcó `ready_for_splits=false`. Esa
evidencia se conserva para explicar la transición; no describe el corpus activo.

## Ejecutar la auditoría

Linux:

```bash
python3 scripts/checks/audit_dataset_classes.py \
  --dataset-root /home/desarrolloab/Documentos/ML/maize_dataset/data \
  --config config/dataset.yaml \
  --documentation docs/es/cleanup-and-ordered/index.md \
  --output outputs/dataset_audit_current \
  --fail-on-mismatch
```

PowerShell:

```powershell
python scripts/checks/audit_dataset_classes.py `
  --dataset-root "C:\ruta\maize_dataset" `
  --config "config\dataset.yaml" `
  --documentation "docs\es\cleanup-and-ordered\index.md" `
  --output "outputs\dataset_audit_current" `
  --fail-on-mismatch
```

Si se omite `--dataset-root`, se usa `get_dataset_root()`. Si se omite `--output`, se usa
`get_output_root()/dataset_audit`. Use una salida nueva para no sobrescribir
evidencia histórica. La ruta recibida siempre representa `DATASET_ROOT`; la
carpeta fuente se resuelve mediante `paths.raw_dir: clean` de la configuración.

Las salidas son:

- `class_counts.csv`: cobertura por clase y entorno, junto con presencia física,
  configuración y documentación.
- `class_mismatches.csv`: discrepancias tipadas y su evidencia.
- `class_audit.json`: diagnóstico agregado y la compuerta `ready_for_splits`.

Una clase configurada ausente o una carpeta adicional bloquea los splits. Una carpeta de
entorno vacía se registra como advertencia porque algunas clases sólo tienen imágenes reales;
una clase completa sin imágenes es un error. Un archivo no admitido se ignora explícitamente,
igual que hace el generador oficial, y nunca se elimina.

## Reconstrucción segura de `clean/`

No use `--force` directamente sobre la copia activa. El procedimiento seguro es:

```text
clean actual
    ↓
inventario, hashes y respaldo
    ↓
descarga en otro DATASET_ROOT/clean
    ↓
auditoría de clean_candidate
    ↓
comparación y aprobación manual
    ↓
sustitución manual recuperable
```

Inventario y respaldo en Linux:

```bash
cd /home/desarrolloab/Documentos/ML/maize_dataset
find clean -type f -print0 | sort -z | xargs -0 sha256sum \
  > /ruta/respaldo/clean_before.sha256
tar -caf /ruta/respaldo/clean_before.tar.gz clean
```

Inventario y respaldo en PowerShell:

```powershell
Get-ChildItem "C:\ruta\maize_dataset\clean" -File -Recurse |
  Sort-Object FullName |
  Get-FileHash -Algorithm SHA256 |
  Export-Csv "C:\ruta\respaldo\clean_before.csv" -NoTypeInformation
tar.exe -caf "C:\ruta\respaldo\clean_before.tar.gz" `
  -C "C:\ruta\maize_dataset" clean
```

Descarga aislada en Linux, usando la fuente configurada en `.env`:

```bash
mkdir -p /ruta/maize_dataset_candidate
DATASET_ROOT=/ruta/maize_dataset_candidate \
  python3 scripts/dataset/download_dataset.py --source hf

python3 scripts/checks/audit_dataset_classes.py \
  --dataset-root /ruta/maize_dataset_candidate \
  --config config/dataset.yaml \
  --output outputs/dataset_audit_candidate \
  --fail-on-mismatch
```

PowerShell:

```powershell
$previousDatasetRoot = $env:DATASET_ROOT
$env:DATASET_ROOT = "C:\ruta\maize_dataset_candidate"
python scripts/dataset/download_dataset.py --source hf
$env:DATASET_ROOT = $previousDatasetRoot

python scripts/checks/audit_dataset_classes.py `
  --dataset-root "C:\ruta\maize_dataset_candidate" `
  --config "config\dataset.yaml" `
  --output "outputs\dataset_audit_candidate" `
  --fail-on-mismatch
```

La descarga depende de que `HF_DATASET_REPO` apunte a la revisión correcta. Antes de aprobar
la sustitución se deben confirmar las nueve clases, 31 622 imágenes admitidas, 3 551 de
laboratorio, 28 071 reales y ausencia de discrepancias críticas. La sustitución de `clean/` no
está automatizada deliberadamente.

## Generación temporal y reproducibilidad

El generador oficial aplica seed 42, deduplicación SHA-256 determinista, estratificación
`label + environment`, proporciones 70/15/15 y un límite baseline de 1 500 imágenes por clase.
No se implementa un segundo algoritmo.

Después de aprobar `clean_candidate`, ejecute dos generaciones aisladas:

```bash
DATASET_ROOT=/ruta/maize_dataset_candidate PROJECT_DATA_ROOT=/tmp/project_data_run_1 \
  python3 scripts/pipeline/create_splits.py --baseline
DATASET_ROOT=/ruta/maize_dataset_candidate PROJECT_DATA_ROOT=/tmp/project_data_run_2 \
  python3 scripts/pipeline/create_splits.py --baseline

python3 scripts/checks/validate_splits.py \
  --dataset-root /ruta/maize_dataset_candidate \
  --splits-dir /tmp/project_data_run_1/splits/seed_42_baseline \
  --compare-dir /tmp/project_data_run_2/splits/seed_42_baseline \
  --output /tmp/project_data_run_1/splits/seed_42_baseline \
  --fail-on-error
```

En PowerShell use dos valores temporales de `$env:PROJECT_DATA_ROOT` y luego:

```powershell
python scripts/checks/validate_splits.py `
  --dataset-root "C:\ruta\maize_dataset_candidate" `
  --splits-dir "C:\temp\project_data_run_1\splits\seed_42_baseline" `
  --compare-dir "C:\temp\project_data_run_2\splits\seed_42_baseline" `
  --output "C:\temp\project_data_run_1\splits\seed_42_baseline" `
  --fail-on-error
```

El validador comprueba columnas, rutas, clases, entornos, rutas duplicadas, hashes repetidos,
fugas entre splits, cobertura y hashes exactos de los CSV comparados. Produce
`split_validation.json` y `split_counts.csv`.

Los splits oficiales sólo pueden generarse cuando ambas auditorías terminan sin
errores, las dos generaciones son idénticas y una persona aprueba el reemplazo
del dataset. Esa compuerta ya fue superada para la revisión activa de 31 622
imágenes y produjo `data/splits/seed_42_baseline/`; debe volver a aplicarse ante
cualquier revisión futura del corpus.
