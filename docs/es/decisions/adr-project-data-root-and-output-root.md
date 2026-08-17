# ADR: separación entre datos del proyecto y resultados

- Estado: aceptada
- Fecha: 2026-07-27
- Alcance: rutas de datos, resultados y compatibilidad histórica

## Contexto

El segmentador maneja tres clases de artefactos con ciclos de vida diferentes:

1. el corpus completo, demasiado grande para vivir dentro del repositorio;
2. datos derivados reproducibles que sí son entradas de pipelines;
3. resultados de ejecuciones, diagnósticos y modelos.

En una etapa anterior, algunos materiales del piloto se escribían bajo
`outputs/`. Esto mezclaba entradas reproducibles con resultados.

## Decisión

Se mantienen tres raíces independientes:

| Raíz | Responsabilidad | Valor local predeterminado |
|---|---|---|
| `DATASET_ROOT` | corpus externo y fuentes grandes | variable opcional |
| `PROJECT_DATA_ROOT` | imágenes, anotaciones, manifiestos y paquetes | `<repo>/data` |
| `OUTPUT_ROOT` | checkpoints, métricas, auditorías, previews y diagnósticos | `<repo>/outputs` |

Las rutas activas son:

- `data/leaf_detection/pilot/`;
- `data/leaf_detection/external_sources/`;
- `data/leaf_detection/detector_dataset/`;
- `outputs/leaf_detection/pilot/`, creado sólo cuando se regeneran auditorías
  o previews del piloto y actualmente ausente;
- `outputs/leaf_detection/segmenter/`.

Los directorios `outputs/leaf_detection/external_sources_eda/`,
`detector_dataset_consolidation/` y `detector_dataset_splits/` se crean sólo al
regenerar sus auditorías. Actualmente están ausentes porque sus datos y
manifiestos canónicos permanecen bajo `data/`.

Los paquetes originales y anotaciones son datos. Los previews, validaciones y
métricas son resultados.

## Migración

| Ruta histórica | Ruta activa |
|---|---|
| `outputs/leaf_detection/pilot/images/` | `data/leaf_detection/pilot/images/` |
| `outputs/leaf_detection/pilot/manifests/` | `data/leaf_detection/pilot/manifests/` |
| `outputs/leaf_detection/pilot/annotations.xml` | `data/leaf_detection/pilot/annotations/cvat/annotations.xml` |
| `outputs/leaf_detection/pilot/packages/` | `data/leaf_detection/pilot/packages/` |

## Consecuencias

- Los nuevos comandos y textos de ayuda deben nombrar `PROJECT_DATA_ROOT`.
- Cambiar `OUTPUT_ROOT` no reubica los datos ni los splits del segmentador.
- Los splits congelados del segmentador no se editan manualmente.
- Los datos fuente no se escriben desde auditorías o entrenamientos.
- Los artefactos históricos permanecen fuera del flujo activo y pueden
  eliminarse de `outputs/` si sus datos de entrada reproducibles siguen bajo
  `data/`.

## Artefactos relacionados

- `src/config.py`;
- `data/README.md`;
- `config/segmentation.yaml`.
