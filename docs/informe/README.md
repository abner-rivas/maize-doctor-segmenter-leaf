# Informe técnico del segmentador de Doctor Maíz

Este directorio contiene un informe reproducible sobre el repositorio exclusivo del
segmentador de hojas. La entrega principal es `main.pdf`; `main.html` ofrece la misma
versión navegable y `main.tex` permite una compilación LaTeX independiente.

## Contenido

- 23 secciones fuente bajo `sections/` (resumen y 22 secciones numeradas);
- 15 figuras del cuerpo, doce analíticas y tres del EDA;
- cuatro activos estáticos copiados para que la entrega sea autocontenida;
- 17 tablas integradas en el documento;
- bibliografía en `referencias.bib`;
- manifiesto generado en `MANIFEST.generated.json`.

## Fuentes de verdad

- Arquitectura: grafo Codebase Memory del proyecto `corn-leaf-desease-project`.
- Datos y protocolo: `docs/es/leaf-detection/` y `docs/es/decisions/`.
- Configuración activa: `config/segmentation.yaml`.
- Plan experimental: `cloud_training/configs/experiments/plan.yaml`.
- Inferencia completa:
  `outputs/full_dataset_inference/2026-08-27_updated_hf_dataset/`.
- Checkpoint operativo:
  `outputs/leaf_detection/models/doctor_maiz_leaf_segmenter_best.pt`.

El informe distingue el checkpoint C-01 usado en la inferencia masiva del candidato
D-01 descrito por la evidencia experimental. También señala que el dataset congelado,
el piloto y los outputs D-01 no están materializados en el checkout inspeccionado.

## Regeneración

Desde este directorio, con el entorno del repositorio:

```bash
MPLCONFIGDIR=/tmp/doctor-maiz-segmenter-report \
  ../../.venv/bin/python generate_assets.py

../../.venv/bin/python build_report.py
```

Para imprimir de nuevo el PDF con Chrome/Chromium:

```bash
../../.venv/bin/python build_report.py --pdf --chrome google-chrome
```

La impresión headless puede requerir permisos del entorno gráfico. Si se dispone de
Tectonic, la fuente LaTeX equivalente se compila con:

```bash
tectonic main.tex --keep-logs --keep-intermediates
```

## Verificación

```bash
../../.venv/bin/python -m py_compile build_report.py generate_assets.py
pdfinfo main.pdf
pdftotext main.pdf -
```

La suite del proyecto se ejecutó sin caché: 349 pruebas aprobaron, cuatro fallaron por
la ausencia local del dataset canónico y tres subtests aprobaron. `ruff check` pasó.
No se abrió test, no se ejecutó el piloto y no se inició entrenamiento para elaborar
este documento.
