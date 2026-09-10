# Limitaciones

1. **Test interno no consumido.** La evaluación oficial de 173 imágenes permanece bloqueada por la discrepancia 183/182 de instancias. No hay métrica final de test aprobada.
2. **Piloto externo retenido.** Las 100 imágenes del piloto no se evaluaron después del test; por diseño, no hay evidencia cualitativa final independiente.
3. **Una sola semilla para D-01.** La mejora de 0.00593 en el pico de Mask mAP50-95 puede estar dentro de variación aleatoria.
4. **Evaluación downstream parcial.** Los 150 casos con ground truth proceden únicamente de la fuente grande y del split de validación.
5. **Corrida masiva sin ground truth.** Los 33,437 estados miden el gate, no exactitud. No se pueden calcular IoU, Dice, sensibilidad ni precisión reales allí.
6. **Checkpoint distinto.** La corrida masiva utilizó C-01, aunque D-01 es el candidato mejorado. No existe comparación pareada a gran escala.
7. **Datos canónicos ausentes.** El árbol segmentado y el piloto no están en el checkout, lo que impide recalcular locks y explica cuatro fallas de pruebas.
8. **Outputs D-01 ausentes.** No se pudieron inspeccionar directamente checkpoint, curvas, CSV y casos de revisión descritos por la documentación.
9. **Cambios sin seguimiento.** Los módulos de inferencia masiva y exportación no están comprometidos; el notebook EDA aparece eliminado.
10. **Calibración in-sample.** El quality gate se seleccionó y midió sobre las mismas 42 imágenes humanas.
11. **Sesgo de selección downstream.** Exportar sólo `reliable` reduce de forma desigual las clases; cogollero pierde 47.35% y potasio 26.25%.
12. **Heurística de hoja objetivo.** Área y centro no garantizan intención del usuario en escenas multihoja.
13. **Benchmark limitado.** Los tiempos provienen de una CPU concreta y de un pipeline que genera muchos artefactos. No representan latencia Android, iOS o servidor GPU.
14. **Cobertura geográfica desconocida.** No hay evaluación representativa de fincas, variedades, dispositivos o campañas salvadoreñas.
15. **Licencias compuestas.** La licencia del contenedor de datos no sustituye revisar licencias y consentimiento de cada fuente original.
16. **Duplicados en el corpus grande.** La inferencia incluye cuatro duplicados; cualquier split downstream debe eliminarlos o agruparlos antes de evaluar.

::: risk Conclusión restringida
La evidencia permite afirmar que el pipeline funciona, es trazable y obtiene métricas altas en un subconjunto de validación. No permite afirmar generalización en campo salvadoreño, mejora causal del clasificador ni preparación para uso autónomo.
:::

