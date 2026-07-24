# Datos derivados del proyecto

Esta carpeta contiene datos materializados para flujos reproducibles. No contiene el
dataset completo de 31,622 imágenes, que continúa fuera del repositorio y se resuelve
mediante `DATASET_ROOT`.

```text
data/
├── splits/
│   └── seed_42_baseline/       # train.csv, val.csv, test.csv y auditoría
└── leaf_detection/
    └── pilot/
        ├── images/             # 100 imágenes copiadas del piloto
        ├── labels/             # etiquetas YOLO, cuando aplique
        ├── manifests/          # selección, importación y roi_manifest.csv
        ├── annotations/cvat/   # annotations.xml: fuente oficial, 100 cajas
        └── packages/           # ZIP y paquetes portables
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
