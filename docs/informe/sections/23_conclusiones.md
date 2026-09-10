# Conclusiones

El segmentador de Doctor Maíz presenta una arquitectura coherente para una función acotada: detectar instancias de hoja, escoger una objetivo, neutralizar fondo y rechazar salidas dudosas. Su valor técnico no proviene sólo de YOLO26n-seg, sino de las capas adicionales de trazabilidad: normalización, selección determinista, quality gate, fallback, manifiestos por imagen, hashes de configuración y pesos, y políticas de no sobrescritura.

La preparación del dataset es una fortaleza. Dos fuentes heterogéneas se auditaron a nivel sintáctico, topológico y semántico; se excluyeron 13,392 lesiones; se resolvieron casos humanos; y se congeló un padre de 1,155 imágenes y 1,224 máscaras. Los splits agrupan variantes y vecinos perceptuales, preservan proporciones 70/15/15 y documentan cero fugas conocidas contra el piloto.

D-01 aporta la mejor evidencia experimental disponible. Desactivar mosaic elevó el pico de Mask mAP50-95 de 0.93806 a 0.94399 con la misma semilla. El checkpoint validado reportó 0.94404 y el pipeline real obtuvo IoU 0.98122, Dice 0.99046 y recall foliar 0.99375 en 150 imágenes, sin fallbacks. Son resultados fuertes, pero continúan siendo validación de una sola semilla y una sola fuente.

La corrida masiva demuestra operación y revela el principal reto. C-01 procesó 33,437 imágenes sin errores de ejecución y publicó 28,360 máscaras confiables. Sin embargo, 5,077 casos usaron fallback y la cobertura varió de 52.65% a 95.53% según clase. Cogollero explica una gran parte de ausencias de detección y confianza baja. Esta heterogeneidad exige revisión humana estratificada y evita usar 84.82% como una medida de exactitud.

La integración con el clasificador es reproducible, pero introduce selección desigual. El subconjunto de 28,360 imágenes conserva sólo `reliable`, lo que puede eliminar casos difíciles y modificar la distribución clínica. Antes de adoptarlo como nuevo dataset de entrenamiento debe demostrarse, mediante un experimento pareado, que segmentar mejora el diagnóstico y no sacrifica las clases o condiciones más importantes.

El estado final es “candidato funcional con validación prometedora”, no “modelo final”. Faltan restaurar la evidencia local, resolver el conflicto 183/182, repetir D-01 con dos semillas adicionales, abrir test una vez, auditar el piloto, validar el gate fuera de muestra y medir el dispositivo objetivo. Mantener esos gates es una señal de calidad: el proyecto prefiere una conclusión incompleta pero verificable a una métrica final obtenida rompiendo su propio protocolo.

::: evidence Dictamen
El pipeline está listo para continuar su validación controlada y para producir datasets experimentales con trazabilidad. No está listo para operar como filtro obligatorio ni como sustituto silencioso de la imagen original en un diagnóstico agrícola autónomo.
:::

