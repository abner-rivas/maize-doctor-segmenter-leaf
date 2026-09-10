# Reproducibilidad y trazabilidad

La reproducibilidad se apoya en identidades separadas para código, datos, configuración y pesos. El commit de Git describe el software; el commit del dataset identifica la publicación; los locks capturan contenido y membresía de splits; el SHA-256 del YAML fija thresholds; y el hash del checkpoint fija parámetros. Ninguna de estas identidades sustituye a las otras.

Table: Identificadores principales conservados por el proyecto.

| Elemento | Identificador |
|---|---|
| Commit del repositorio en la corrida masiva | `985a920c5f3028ee81a7e0f7cd32507fa9b337c5` |
| Commit del corpus completo | `e515ab2f1e4c5729f8447520f1a630cf14c532dc` |
| Padre segmentado | `7a4a5c083fc64b067df12bcc95ec976d5a7e3b8a585d0a090b6b3940af4d7d5c` |
| Asignación combinada de splits | `96833e43a46c959f0d5c86615b1d1ea6decb139063eea9c877986a61084c0e1` |
| Checkpoint C-01 | `4f66456d05d87f9e7080155eb5cd80c583f34849415ec820c950bd97f9c5ec6f` |
| Checkpoint D-01 | `a2bf4f201ca4f5e32c349cdc66d7ac39a6b012a330b182149401e533b2ecb8ab` |
| Configuración de la corrida masiva | `d537af08613441b5fe90a44bc9008004aca80b77c8d4c293c34efba6b1d5adc3` |
| Manifiesto de inferencia exportado | `1b672ca01e4a95f869a94677ddd1e7fe8eeb107e42f7b3f15358049b6ea43c12` |

## Reanudación y no sobrescritura

Una corrida nueva falla si el directorio de salida existe. Una reanudación exige directorio previo, ausencia de `summary.json`, inventario compatible y mismo dataset root. Tampoco permite `limit` ni modo sólo inventario. Antes de continuar compara procedencia anterior con runtime actual y acumula tiempo de pared. Al terminar escribe `summary.json`; ese archivo convierte el run en cerrado para futuras reanudaciones.

La exportación al clasificador aplica un patrón similar: prepara un staging dentro del directorio padre y lo renombra únicamente después de escribir todos los artefactos. Ante una excepción elimina sólo el staging que acaba de crear. Esta atomicidad evita dejar un árbol aparentemente final con imágenes parciales.

## Comandos de reproducción

La corrida completa registra el comando exacto y la reanudación. En forma abreviada:

```bash
python -m scripts.pipeline.infer_dataset \
  --dataset /ruta/data/clean \
  --metadata data/leaf_detection/external_sources/.../metadata.csv \
  --dataset-commit e515ab2f... \
  --output outputs/full_dataset_inference/2026-08-27_updated_hf_dataset \
  --resume --progress-every 100

python -m scripts.pipeline.export_classifier_dataset \
  outputs/full_dataset_inference/2026-08-27_updated_hf_dataset
```

El informe se regenera con `generate_assets.py` y `build_report.py`. El primer script vuelve a calcular gráficos desde `manifest.csv`; el segundo transforma las secciones Markdown en HTML y LaTeX. `main.pdf` se imprime desde el HTML con Chromium para no depender de una distribución TeX local. `main.tex` queda disponible para compilar con Tectonic o XeLaTeX.

::: evidence Trazabilidad por imagen
Cada fila del manifiesto contiene `image_id`, ruta, clase, SHA-256, estado, razón, propuestas, score, geometría, tiempo, warnings y rutas de artefactos. El JSON individual añade configuración del gate y metadata completa del modelo.
:::

