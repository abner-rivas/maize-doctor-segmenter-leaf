# Auditoría del quality gate de segmentación

El quality gate determina si una máscara puede publicarse como `reliable`, debe marcarse
`uncertain` o representa un fallo. No ejecuta ningún modelo ajeno al segmentador.

## Evidencia

Para cada imagen se conservan:

- confidence de la propuesta y de la selección;
- área de máscara y bbox;
- componentes conexos, perímetro normalizado y contacto con bordes;
- cantidad de instancias elegibles y margen entre sus scores;
- razón de rechazo, thresholds y versión del gate;
- panel original, overlay, máscara y salida con fondo neutral.

Los thresholds activos están en `config/segmentation.yaml`:

```yaml
quality_gate:
  reject_multiple_eligible: false
  max_mask_area_ratio: 0.999
  large_mask_area_ratio: 0.50
  min_large_mask_bbox_ratio: 0.70
  max_large_mask_normalized_perimeter: 8.0
  min_multi_instance_score_margin: 0.33
```

La auditoría compara la política previa con la política propuesta sobre el manifiesto
humano `scripts/experiments/manifests/segmentation_reliability_audit_v1.csv`.

```bash
make leaf-segmentation-reliability-audit
```

El resultado incluye `audit_metrics.csv`, `summary.json`, resultados estructurados y
paneles visuales. No sobrescribe una auditoría existente.
