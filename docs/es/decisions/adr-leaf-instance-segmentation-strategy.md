# ADR: segmentación de instancias de hoja

## Estado

Aceptado.

## Decisión

Usar instance segmentation de clase única `maize_leaf` y seleccionar una instancia
objetivo de forma determinista mediante área relativa, cercanía al centro y confidence.

Las salidas soportadas son máscara a resolución original, fondo neutral, recorte de bbox
y recorte en letterbox. Cada ejecución conserva máscara, bbox, score, warnings,
fingerprint del checkpoint y metadata del runtime.

## Consecuencias

- las lesiones y etiquetas ajenas a `maize_leaf` se excluyen al consolidar;
- múltiples hojas permanecen trazables y se someten al margen del quality gate;
- las máscaras vacías, degeneradas, demasiado pequeñas o geométricamente sospechosas no
  pasan silenciosamente como confiables;
- el piloto externo permanece retenido para evaluación cualitativa independiente.
