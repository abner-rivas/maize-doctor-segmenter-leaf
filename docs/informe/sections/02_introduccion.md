# Introducción

Un clasificador foliar puede aprender tanto la enfermedad como elementos accidentales del fondo, la escala de captura o la forma de encuadrar una hoja. El segmentador aborda una parte concreta de ese riesgo: identificar el contorno de la hoja y neutralizar los píxeles externos antes del diagnóstico. Esta separación de responsabilidades es deliberada. El repositorio del segmentador no contiene entrenamiento, inferencia ni explicabilidad del clasificador; su salida es una imagen preprocesada y un conjunto de metadatos que otro componente puede consumir.

La tarea elegida es segmentación de instancias. A diferencia de una caja, una máscara conserva la silueta irregular de la hoja y permite retirar fondo sin cortar todo el rectángulo circundante. A diferencia de una máscara semántica única, las instancias separadas permiten razonar sobre fotografías con varias hojas y escoger una objetivo. Los modelos `-seg` de Ultralytics producen máscaras, clases y niveles de confianza por instancia [@ultralytics2026segment]. El proyecto reduce esa salida general a una clase `maize_leaf` y añade una política propia de selección y confiabilidad.

Esta decisión introduce una tensión importante. Recortar tejido enfermo destruye información que el clasificador ya no puede recuperar; conservar algo de fondo suele ser menos irreversible. Por esa razón, las métricas downstream priorizan recall de píxel de hoja y subsegmentación antes que precisión de píxel. El fallback también es conservador: cuando la máscara falla o resulta incierta, el sistema entrega la imagen original en vez de publicar una segmentación dudosa como válida.

El trabajo se apoya en dos planos de evidencia. El primero es el experimento controlado: dataset segmentado, splits sin fuga, entrenamiento de C-01 y D-01, y evaluación con ground truth. El segundo es la operación sobre el corpus de clasificación: 33,437 entradas, artefactos por imagen, estados del quality gate y exportación de 28,360 salidas confiables. Confundir ambos planos llevaría a conclusiones incorrectas. El experimento estima exactitud en validación; la operación masiva mide cobertura y comportamiento del gate en un dominio mayor.

El informe adopta una disciplina cercana a una model card [@mitchell2019modelcards] y una datasheet [@gebru2021datasheets]: declara procedencia, configuración, población evaluada, limitaciones y usos no autorizados. Los números se toman de archivos estructurados o documentación respaldada por hashes. Cuando el artefacto primario no está en el checkout actual, se identifica expresamente como evidencia documental no reejecutada.

## Organización del documento

Las primeras secciones definen objetivos, alcance y arquitectura. Luego se describen las fuentes, la consolidación, los splits, el preflight y el entrenamiento cloud. Las secciones centrales estudian D-01, la inferencia, la selección de instancia y el quality gate. Después se analizan la evaluación downstream y la corrida completa. El cierre revisa integración con el clasificador, reproducibilidad, calidad de software, riesgos éticos, limitaciones y gates pendientes.

