# Análisis de fallos y variación por clase

La cobertura del gate no fue uniforme. Tizón norteño, mancha gris y roya superaron 93% de `reliable`; cogollero quedó en 52.65%, potasio en 73.75% y nitrógeno en 80.61%. Como el modelo no predice la clase de enfermedad, esta diferencia indica que la apariencia, escala, fondo o daño asociados a cada subconjunto afectan la detección de la hoja. No prueba que el segmentador “reconozca” unas enfermedades mejor que otras.

Table: Estado del quality gate por clase en el corpus de 33,437 imágenes.

| Clase | Total | Reliable | Tasa | Uncertain | Failed |
|---|---:|---:|---:|---:|---:|
| Roya común | 2,256 | 2,100 | 93.09% | 58 | 98 |
| Gusano cogollero | 4,857 | 2,557 | 52.65% | 246 | 2,054 |
| Mancha gris | 1,930 | 1,836 | 95.13% | 46 | 48 |
| Sana | 8,744 | 8,008 | 91.58% | 170 | 566 |
| Necrosis letal | 6,415 | 5,390 | 84.02% | 347 | 678 |
| Deficiencia de nitrógeno | 846 | 682 | 80.61% | 47 | 117 |
| Tizón norteño | 6,830 | 6,525 | 95.53% | 107 | 198 |
| Deficiencia de fósforo | 938 | 804 | 85.71% | 21 | 113 |
| Deficiencia de potasio | 621 | 458 | 73.75% | 15 | 148 |

![Cogollero concentra la mayor proporción de fallos, mientras tizón y mancha gris muestran alta cobertura del gate.](figures/status_by_class.png)

Los 5,077 casos no confiables se reparten exactamente entre seis razones. `no_detection` aporta 1,853; confianza inferior a 0.50, 1,385; máscara inválida, 782; margen ambiguo, 373; área excesiva, 373; y geometría grande sospechosa, 311. Las tres primeras producen `failed`; las tres últimas, `uncertain`.

![La ausencia de detección y la confianza baja explican 63.8% de las salidas no confiables.](figures/rejection_reasons.png)

Cogollero concentra 1,333 ausencias de detección y 702 propuestas con confianza baja. Necrosis letal concentra 416 máscaras inválidas y 273 áreas excesivas. En hojas sanas aparecen 341 máscaras inválidas y 89 áreas excesivas. Estas concentraciones sugieren tipos de error diferentes: el primer grupo requiere mejorar recall del modelo; el segundo exige estudiar decodificación o degeneración de máscaras; el tercero puede involucrar encuadres donde la hoja llena casi todo el marco.

El panel siguiente no pretende estimar frecuencia. Muestra el contrato operativo: una salida confiable publica fondo neutral; una geometría sospechosa conserva artefactos de máscara pero devuelve el original; y una ausencia de detección genera máscara vacía y fallback.

![Ejemplos deterministas de una salida reliable, una uncertain por geometría y una failed por ausencia de detección.](figures/qualitative_cases.png)

## Hipótesis de mejora

La siguiente recolección propuesta contiene 300 imágenes reales nuevas con daño severo de cogollero, oclusión, hojas parciales, hojas pequeñas, fondos complejos y varias hojas. El patrón masivo respalda esa prioridad, sobre todo para cogollero. Sin embargo, las nuevas imágenes deben tener máscaras humanas y separación por sesión de captura; usar predicciones del propio modelo como verdad de referencia amplificaría sus fallos.

::: risk Sesgo del gate
Una tasa baja puede reflejar peor segmentación, un gate demasiado estricto o ambos. Sin ground truth en las 33,437 imágenes no es posible separar esos efectos. La revisión debe muestrear tanto fallos como confiables por clase.
:::

