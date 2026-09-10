# Ética, impacto social y gobernanza

El propósito social del segmentador es reducir ruido visual antes de un diagnóstico de enfermedades y deficiencias en maíz. Una máscara correcta puede hacer más consistente la entrada del clasificador y facilitar revisión visual. Sin embargo, el beneficio es potencial hasta que un estudio pareado demuestre que el diagnóstico mejora en campo y no sólo en el corpus de desarrollo.

El coste de error es asimétrico. La subsegmentación puede eliminar lesiones, bordes necróticos o patrones de color y conducir a un diagnóstico falso. La sobresegmentación introduce fondo, pero preserva tejido. El diseño refleja esa asimetría al priorizar recall foliar y usar fallback original. Aun así, un fallback no vuelve correcta la clasificación: sólo evita sustituir una imagen válida por una máscara insegura.

La selección de una hoja dominante también incorpora un supuesto de uso. En una fotografía con varias hojas, el área y la cercanía al centro pueden elegir una hoja distinta de la que el agricultor pretendía consultar. La interfaz final debe mostrar la máscara elegida o pedir confirmación, especialmente cuando el margen de score sea pequeño. Ocultar la selección convertiría una heurística técnica en una decisión agronómica no consentida.

## Representatividad

Las fuentes de segmentación son externas y no incluyen una muestra geográfica documentada de El Salvador. El corpus masivo combina clases y estilos de captura, pero su metadata no basta para atribuir diferencias a región, dispositivo o condiciones productivas. La fuerte caída de cobertura en cogollero y potasio muestra que una métrica global puede esconder poblaciones relevantes. Los reportes operativos deben mantener tasas por clase, fuente, entorno, resolución y condiciones difíciles.

Filtrar sólo `reliable` modifica la distribución del dataset del clasificador. Si los rechazos se concentran en daño severo o fondos complejos, excluirlos podría mejorar métricas internas mientras empeora utilidad real. La gobernanza debe conservar los 5,077 fallbacks, muestrearlos para revisión y decidir explícitamente si se usan como originales, se reanotan o se excluyen.

## Licencias, privacidad y comunicación

Las dos fuentes de segmentación registran CC BY 4.0 y requieren atribución. El corpus de Hugging Face consultado declara licencia MIT, pero las procedencias internas de imágenes deben conservarse y revisarse antes de redistribuir un ZIP derivado. Fotografías nuevas de productores necesitan consentimiento, finalidad definida, retención y control de acceso.

La salida del sistema debe comunicarse como apoyo, no diagnóstico definitivo. Un estado `reliable` describe la máscara bajo un gate calibrado, no la certeza de enfermedad. El usuario debe poder ver advertencias y continuar con la imagen original. Los reportes públicos deben incluir versión del modelo, fecha, población de prueba y limitaciones, siguiendo prácticas de model cards [@mitchell2019modelcards].

Table: Salvaguardas recomendadas para despliegue.

| Riesgo | Salvaguarda |
|---|---|
| Tejido enfermo recortado | Priorizar recall, mostrar overlay y permitir original |
| Hoja objetivo equivocada | Exponer selección y pedir confirmación en casos ambiguos |
| Sesgo por clase o captura | Monitorear cobertura por subgrupo y revisar rechazos |
| Falsa seguridad por `reliable` | Separar confianza de máscara y confianza diagnóstica |
| Licencia o consentimiento incompleto | Mantener procedencia, atribución y registro de uso |
| Deriva de dominio | Auditorías periódicas con máscaras humanas nuevas |

