# Datos del segmentador

La raíz activa es `data/leaf_detection/`:

```text
leaf_detection/
├── pilot/                    piloto externo retenido de 100 imágenes
├── external_sources/         fuentes YOLO/COCO inmutables
└── detector_dataset/
    ├── all/                  padre consolidado y congelado
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    ├── manifests/            trazabilidad, decisiones, locks y fingerprints
    ├── previews/             evidencia de revisión
    ├── annotation_batches/   ruta bbox histórica, no activa
    └── dataset.yaml          configuración portable de Ultralytics
```

El padre final contiene 1 155 imágenes y 1 224 máscaras de clase única
`0 = maize_leaf`. Los splits contienen 809/173/173 imágenes y no comparten hashes,
grupos, variantes cercanas ni elementos con el piloto.

`data/` contiene entradas reproducibles; `outputs/leaf_detection/` contiene modelos,
paquetes, predicciones, métricas, previews generados y auditorías. No editar locks o
manifiestos derivados a mano.
