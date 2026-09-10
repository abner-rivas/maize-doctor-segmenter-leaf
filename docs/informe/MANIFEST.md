# Manifiesto del informe

## Documento

- `main.pdf`: entrega final paginada en A4.
- `main.html`: versión navegable e imprimible.
- `main.tex`: versión LaTeX generada desde las mismas secciones.
- `report.css`: estilo de portada, tipografía, tablas, figuras y paginación.
- `referencias.bib`: siete fuentes técnicas y de gobernanza.

## Fuentes editables

`sections/` contiene resumen, introducción, objetivos, método, arquitectura, fuentes,
consolidación, splits, preflight cloud, entrenamiento, D-01, inferencia y selección,
quality gate, evaluación downstream, corrida masiva, fallos, integración con el
clasificador, reproducibilidad, calidad de software, ética, limitaciones, plan de
cierre y conclusiones.

## Generación

- `generate_assets.py` valida 33,437 filas y produce gráficos deterministas desde el
  manifiesto de inferencia; también copia el logo y tres figuras EDA para que el
  informe no dependa de rutas externas al directorio.
- `build_report.py` interpreta el subconjunto Markdown usado por las secciones y
  genera `main.html`, `main.tex` y `MANIFEST.generated.json`.
- `main.pdf` se imprime desde `main.html` con Chrome headless porque el entorno de
  elaboración no contiene una distribución TeX.

## Figuras analíticas

- composición de splits y fuentes;
- comparación C-01/D-01 y métricas del checkpoint D-01;
- métricas downstream sobre 150 imágenes;
- status y retención por clase en 33,437 imágenes;
- razones de rechazo, áreas de máscara y latencia;
- flujo de arquitectura y casos cualitativos deterministas.

## Declaraciones que el informe evita

- no presenta la tasa `reliable` como exactitud;
- no atribuye la corrida masiva a D-01;
- no afirma evaluación final de test ni piloto;
- no afirma estabilidad multisemilla;
- no afirma mejora del clasificador;
- no afirma disponibilidad de los datos canónicos ausentes;
- no presenta el sistema como diagnóstico autónomo.

