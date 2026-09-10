# Quality gate y calibración

El quality gate transforma el resultado del procesador en tres estados. `reliable` significa que existe una instancia completa y no se activó ninguna regla de ambigüedad o geometría. `uncertain` conserva una máscara disponible, pero evita publicarla como salida final porque su evidencia es ambigua. `failed` indica fallback, evidencia incompleta, ausencia de detección, confianza insuficiente o máscara inválida.

Table: Thresholds activos del quality gate.

| Regla | Valor | Resultado si se activa |
|---|---:|---|
| Rechazar automáticamente varias elegibles | `false` | Se evalúa margen en vez de fallar por conteo |
| Área máxima de máscara | 0.999 | `uncertain` por área excesiva |
| Umbral de máscara grande | 0.25 | Habilita prueba geométrica combinada |
| Ocupación mínima dentro del bbox | 0.80 | Señal de máscara grande dispersa |
| Perímetro normalizado máximo | 8.0 | Señal de geometría compleja |
| Margen mínimo entre scores | 0.33 | `uncertain` si varias hojas compiten |

La regla geométrica combinada no rechaza cualquier hoja grande. Sólo marca como sospechosa una máscara con área al menos 0.25, ocupación dentro de su bbox menor a 0.80 y perímetro normalizado mayor a 8.0. Una máscara que toca bordes puede seguir siendo válida. El máximo 0.999 captura coberturas casi totales que suelen indicar que el modelo absorbió el fondo.

El margen entre instancias se aplica cuando hay más de una elegible. Si los dos mejores scores están separados por menos de 0.33, el sistema no sabe con suficiente claridad cuál hoja publicar. La máscara seleccionada se conserva para auditoría, pero la salida final usa la imagen completa. Esta política evita que una decisión arbitraria quede silenciosa.

## Calibración humana

La calibración evaluó candidatos sobre un manifiesto congelado de 42 imágenes revisadas por personas. El objetivo fue alcanzar al menos 0.95 de precisión entre las salidas confiables y, dentro de los candidatos viables, maximizar máscaras `GOOD` conservadas, cobertura, precisión y cercanía a la política base. El gate elegido mantuvo las 24 máscaras `GOOD` y redujo falsos confiables de uno a cero.

Esa cifra es in-sample: la misma muestra sirvió para escoger el gate y medirlo. No estima por sí sola desempeño futuro. El protocolo exige confirmarla con imágenes nuevas, especialmente fondos complejos, oclusiones, hojas pequeñas, hojas parciales y escenas con varias hojas.

::: evidence Política conservadora
En la corrida completa, toda salida `uncertain` o `failed` terminó en modo `full_image`. El clasificador exportado incluyó únicamente `reliable`; no se filtró sólo por confianza del detector.
:::

