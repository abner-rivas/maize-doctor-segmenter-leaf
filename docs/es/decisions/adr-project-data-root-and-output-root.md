# ADR: separación entre datos del proyecto y resultados

- Estado: aceptada
- Fecha: 2026-07-27
- Alcance: rutas de datos, resultados y compatibilidad histórica

## Contexto

El proyecto maneja tres clases de artefactos con ciclos de vida diferentes:

1. el corpus completo, demasiado grande para vivir dentro del repositorio;
2. datos derivados reproducibles que sí son entradas de pipelines;
3. resultados de ejecuciones, diagnósticos y modelos.

En una etapa anterior, los splits y algunos materiales del piloto se escribían
bajo `outputs/`. Esto mezclaba entradas reproducibles con resultados y produjo
rutas históricas como `/outputs/splits/seed_42_baseline` en ejecuciones remotas.

## Decisión

Se mantienen tres raíces independientes:

| Raíz | Responsabilidad | Valor local predeterminado |
|---|---|---|
| `DATASET_ROOT` | corpus externo `clean/` y fuentes grandes | variable obligatoria |
| `PROJECT_DATA_ROOT` | splits, imágenes copiadas, anotaciones, manifiestos y paquetes | `<repo>/data` |
| `OUTPUT_ROOT` | checkpoints, métricas, auditorías, previews y diagnósticos | `<repo>/outputs` |

Las rutas activas son:

- `data/splits/seed_42_baseline/`;
- `data/leaf_detection/pilot/`;
- `data/leaf_detection/external_sources/`;
- `data/leaf_detection/detector_dataset/`;
- `outputs/baselines/`;
- `outputs/leaf_detection/pilot/`;
- `outputs/leaf_detection/external_sources_eda/`;
- `outputs/preflight/`.

Los paquetes originales y anotaciones son datos. Los previews, validaciones y
métricas son resultados.

## Migración

| Ruta histórica | Ruta activa |
|---|---|
| `outputs/splits/seed_42_baseline/` | `data/splits/seed_42_baseline/` |
| `outputs/leaf_detection/pilot/images/` | `data/leaf_detection/pilot/images/` |
| `outputs/leaf_detection/pilot/manifests/` | `data/leaf_detection/pilot/manifests/` |
| `outputs/leaf_detection/pilot/annotations.xml` | `data/leaf_detection/pilot/annotations/cvat/annotations.xml` |
| `outputs/leaf_detection/pilot/packages/` | `data/leaf_detection/pilot/packages/` |

## Compatibilidad histórica

Los `summary.json` oficiales no se reescriben para cambiar una ruta. Las
corridas remotas de julio de 2026 conservan
`/outputs/splits/seed_42_baseline`. `src.training.common.load_run_metadata`
intenta esa ubicación y, cuando no existe localmente, utiliza el split
organizado que recibe desde `PROJECT_DATA_ROOT`.

La prueba de ese fallback conserva deliberadamente una ruta `/outputs/splits`;
no es una referencia activa.

## Consecuencias

- Los nuevos comandos y textos de ayuda deben nombrar `PROJECT_DATA_ROOT`.
- Cambiar `OUTPUT_ROOT` no reubica splits.
- Los splits oficiales no se editan manualmente.
- Los datos fuente no se escriben desde auditorías o entrenamientos.
- Las rutas históricas continúan interpretables sin alterar la evidencia.

## Artefactos relacionados

- `src/config.py`;
- `src/training/common.py`;
- `data/README.md`;
- `outputs/repository_audit/path_migration_map.csv`.
