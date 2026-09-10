# Inferencia y selección de la hoja objetivo

El modelo puede devolver cero, una o varias propuestas. `instances_from_ultralytics_result` convierte cada máscara a resolución original, asocia confianza, bbox, clase e índice de procedencia, y rechaza desalineaciones entre cantidades. El selector no toma simplemente la primera detección: filtra evidencia inválida y calcula un score trazable.

Una propuesta es elegible si pertenece a la clase cero, tiene confianza finita al menos 0.50, cubre al menos 1% de la imagen, contiene una máscara no vacía y no cubre exactamente el 100% del marco. Las propuestas con confianza entre 0.20 y 0.50 se preservan en la predicción para auditoría, pero no pueden convertirse en hoja objetivo. Esta separación mantiene sensibilidad en la generación de candidatos sin relajar la aceptación.

Para cada instancia elegible se calcula área relativa respecto a la mayor, cercanía del centroide al centro de la imagen y confianza. El score vigente es:

```text
score = 0.45 * area_relativa + 0.35 * cercania_centro + 0.20 * confianza
```

La lista se ordena por score, área, confianza y, finalmente, por el menor índice original como desempate determinista. Cuando hay varias hojas, el resultado conserva todas las trazas y añade una advertencia. El selector no afirma que la hoja mayor sea siempre la más relevante para el diagnóstico; implementa una regla reproducible que debe contrastarse con el uso real.

Table: Parámetros activos de inferencia y selección.

| Parámetro | Valor | Función |
|---|---:|---|
| `image_size` | 640 | Entrada al modelo |
| Confianza de propuesta | 0.20 | Retener candidatos para auditoría |
| Confianza de selección | 0.50 | Aceptar candidato como hoja |
| IoU NMS | 0.70 | Suprimir propuestas solapadas |
| Máximo de detecciones | 20 | Limitar candidatos |
| Área mínima | 0.01 | Rechazar máscaras diminutas |
| Peso de área | 0.45 | Preferir hoja dominante |
| Peso de centro | 0.35 | Preferir objetivo centrado |
| Peso de confianza | 0.20 | Incorporar evidencia del modelo |

## Perfiles de salida

`mask_black` conserva el tamaño original y pone el fondo en negro; es el perfil activo. `bbox_crop` recorta la caja sin máscara; `crop_mask_black` recorta la región ya enmascarada; y `crop_mask_letterbox` ajusta ese recorte a un tamaño objetivo sin deformar proporciones. La corrida completa usa `mask_black`, de modo que las imágenes confiables mantienen resolución y encuadre, mientras el exterior queda neutralizado.

Si no hay selección, el resultado marca `fallback_used=true`, guarda la razón y entrega la imagen original. El fallo no desaparece: queda visible en status, warnings, metadata y rutas de artefactos. Este diseño favorece degradación controlada frente a una salida vacía o engañosa.

