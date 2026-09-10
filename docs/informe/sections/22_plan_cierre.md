# Plan de cierre y recomendaciones

## 1. Restaurar y verificar la evidencia canónica

Restaurar `data/leaf_detection/detector_dataset`, el piloto y los outputs D-01 desde el almacenamiento autorizado. Verificar hashes antes de ejecutar cualquier experimento. Correr la suite completa y exigir 353 pruebas aprobadas más los tres subtests. Confirmar que el notebook eliminado y los módulos sin seguimiento reflejan decisiones intencionales.

## 2. Resolver el contrato de test

Inspeccionar los dos polígonos con clase y bbox idénticos. La solución debe quedar en una ADR: corregir una anotación si es un duplicado semántico, representar explícitamente dos instancias si son distintas y la herramienta lo permite, o ajustar el contrato de evaluación con justificación. No debe desactivarse el guard únicamente para producir métricas.

## 3. Confirmar D-01 entre semillas

Ejecutar E-01 con semillas 7 y 1337 usando exactamente los hiperparámetros ganadores de D-01. Reportar media, desviación estándar y curvas para recall foliar, Dice y Mask mAP50-95. Mantener test y piloto cerrados. Si la ganancia no supera variación, conservar C-01 o continuar con las ablaciones priorizadas.

## 4. Evaluar una vez y auditar el piloto

Congelar checkpoint, configuración de inferencia y quality gate. Abrir el test interno una sola vez y producir resumen aprobado con fingerprint y hash de pesos. Sólo después ejecutar el piloto externo como auditoría cualitativa, sin usarlo para retocar thresholds. Publicar resultados por fuente, borde, resolución, área y número de hojas.

## 5. Validar el gate fuera de muestra

Muestrear de la corrida masiva casos `reliable`, `uncertain` y `failed` por clase y razón. Obtener máscaras humanas con doble revisión. Medir precisión de `reliable`, cobertura de máscaras buenas y falsos confiables. Incluir suficientes casos de cogollero, necrosis, potasio y fondos complejos.

## 6. Comparar impacto en clasificación

Crear un estudio pareado con idénticos splits, arquitectura, semilla y entrenamiento del clasificador. Comparar originales, segmentadas C-01, segmentadas D-01 y política con fallback. Reportar macro-F1, resultados por clase y entorno, además de cobertura del segmentador. El objetivo no es maximizar una métrica aislada del segmentador, sino mejorar diagnóstico sin excluir casos difíciles.

## 7. Preparar despliegue observable

Exportar el modelo al formato objetivo y medir latencia, memoria y equivalencia numérica en el dispositivo real. La API o app debe registrar versión, status, razón, tiempo y aceptación del usuario sin almacenar fotografías innecesariamente. Establecer umbrales de alerta si baja la tasa `reliable` o aumenta el fallback por clase.

Table: Gates para declarar cierre del segmentador.

| Gate | Criterio mínimo | Estado actual |
|---|---|---|
| Integridad local | Datos restaurados, hashes coincidentes, suite completa verde | Pendiente |
| Estabilidad | D-01 en tres semillas y mejora mayor que variación | Pendiente |
| Test interno | Ejecución única aprobada sobre 173 imágenes | Bloqueado |
| Piloto externo | Auditoría posterior al test | No ejecutado |
| Gate fuera de muestra | Precisión y cobertura en revisión humana nueva | Pendiente |
| Impacto downstream | Comparación controlada del clasificador | Pendiente |
| Dispositivo objetivo | Latencia, memoria y equivalencia verificadas | Pendiente |

