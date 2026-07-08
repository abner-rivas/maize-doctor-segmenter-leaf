# Interpretabilidad

Saber cuánto acierta un modelo no basta si no se entiende en qué se fija para decidir, sobre todo
en las clases donde el margen de error ya es alto. Por eso la explicabilidad es **post-hoc** y no
está acoplada al entrenamiento: se corre sobre checkpoints ya entrenados con `best.pth`. Combina
LIME (regiones de superpíxeles que sostienen el diagnóstico) y Grad-CAM (mapa de activación de la
clase predicha).

## Scripts

Tres scripts cubren distintos niveles de detalle, desde inspeccionar una imagen puntual hasta
agregar la fidelidad sobre todo el conjunto de test.

| Script | Salida |
|---|---|
| `explain_lime.py` (`make explain-lime`) | Reporte visual LIME + Grad-CAM por imagen |
| `explain_report.py` (`make explain-report`) | Fidelidad agregada y dispersión por clase |
| `explain_report.py --errors-only` (`make explain-errors`) | LIME dirigido a errores (label ≠ pred) |

## Consistencia de etiquetas

Es un detalle fácil de pasar por alto, pero si se ignora rompe la confianza en los reportes: de
dónde sale la etiqueta que se muestra en cada panel. El mapeo clase→índice y el `image_size` de
cada reporte se leen del `summary.json` del run (fuente de verdad del head entrenado), no se
reconstruyen desde el YAML. Esto garantiza que la etiqueta mostrada en el panel coincida con la
predicción real del modelo.

## Qué observar

Revisar los mapas de atribución con ojo crítico ayuda a distinguir cuándo el modelo realmente
aprende rasgos de la enfermedad y cuándo se apoya en atajos poco confiables. En las clases fáciles
(roya, necrosis letal) los mapas se concentran en el tejido de la hoja. En cambio, en las
deficiencias N/P/K y en imágenes de campo con fondo cargado, la atribución puede dispersarse hacia
el fondo, una señal de contexto espurio a vigilar y coherente con la confusión entre deficiencias
observada en la evaluación.
