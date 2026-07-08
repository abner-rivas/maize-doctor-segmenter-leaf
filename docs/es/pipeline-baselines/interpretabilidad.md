# Interpretabilidad

La explicabilidad es **post-hoc** y no está acoplada al entrenamiento: se corre sobre checkpoints
ya entrenados con `best.pth`. Combina LIME (regiones de superpíxeles que sostienen el diagnóstico)
y Grad-CAM (mapa de activación de la clase predicha).

## Scripts

| Script | Salida |
|---|---|
| `explain_lime.py` (`make explain-lime`) | Reporte visual LIME + Grad-CAM por imagen |
| `explain_report.py` (`make explain-report`) | Fidelidad agregada y dispersión por clase |
| `explain_report.py --errors-only` (`make explain-errors`) | LIME dirigido a errores (label ≠ pred) |

## Consistencia de etiquetas

El mapeo clase→índice y el `image_size` de cada reporte se leen del `summary.json` del run
(fuente de verdad del head entrenado), no se reconstruyen desde el YAML. Esto garantiza que la
etiqueta mostrada en el panel coincida con la predicción real del modelo.

## Qué observar

- En las clases fáciles (roya, necrosis letal) los mapas se concentran en el tejido de la hoja.
- En las deficiencias N/P/K y en imágenes de campo con fondo cargado, la atribución puede
  dispersarse hacia el fondo — señal de contexto espurio a vigilar, coherente con la confusión
  entre deficiencias observada en la evaluación.
