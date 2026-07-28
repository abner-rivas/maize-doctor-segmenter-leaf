# Checklist de preparación para el entrenamiento cloud

Verificado el 2026-07-28. Marque cada casilla con evidencia, no de memoria.
Ninguna casilla de una etapa posterior debe marcarse antes que las anteriores.

## Antes de salir de la máquina local

- [x] `dataset_lock.status = ready_for_split_generation`
- [x] `split_lock.status = ready_for_training_preflight`
- [x] Fingerprint padre `c087af60…9e38c` recalculado y coincidente
- [x] Los cuatro fingerprints de split coinciden (`make leaf-segmentation-verify-locks`)
- [x] 809/173/173 imágenes y 858/183/183 máscaras verificadas en disco
- [x] Clase única `{0: 1224}`, cero bbox mezclados, cero TXT vacíos
- [x] Cero fugas entre splits y cero fugas contra el piloto
- [x] Piloto intacto: 100 imágenes, sin participación en train/val/test
- [x] `processing_profile=baseline_full` y `leaf_detection.enabled=false`
- [x] `pytest` en verde y `ruff check src/ scripts/ tests/` limpio
- [ ] **Paquete reconstruido con el código corregido** (el paquete del 2026-07-28
      09:19 contiene versiones anteriores de `run_ultralytics.py`, `lib.sh` y
      `Makefile`) → `make leaf-segmentation-cloud-prepare`
- [ ] Nuevo SHA-256 del paquete anotado y comparado tras la subida

## Gate 1 — Cloud ready

- [ ] Paquete extraído **en un volumen persistente**, no en disco efímero
- [ ] `sha256sum -c` del paquete correcto en la máquina remota
- [ ] `nvidia-smi` responde y reporta la GPU esperada
- [ ] `bash cloud_training/bootstrap_cloud.sh` termina sin error
- [ ] El dry-run **no** intentó reemplazar `torch` ni `torchvision`
- [ ] `pip check` sin conflictos
- [ ] `pip_freeze.txt` y `runtime_environment.lock` guardados
- [ ] `torch.cuda.is_available()` sigue en `true` tras instalar Ultralytics
- [ ] `bash cloud_training/preflight_cloud.sh` → `ready_for_smoke_training`
- [ ] `weights_manifest.json` con ruta, tamaño, SHA-256 y versión de Ultralytics
- [ ] **`model_check.segmentation_output = true`** (el forward produjo máscaras)
- [ ] Licencia de Ultralytics resuelta y anotada
- [ ] VRAM total y libre registradas en `gpu.json`
- [ ] Al menos 10 GiB libres para resultados

## Gate 2 — Smoke approved

`CONFIRM_SEGMENTATION_SMOKE_TRAINING=1 make leaf-segmentation-cloud-smoke`

- [ ] Una época completa sin error
- [ ] `smoke_summary.status = passed`
- [ ] Pérdidas finitas (sin `NaN` ni `inf`)
- [ ] GPU realmente utilizada durante el entrenamiento, no sólo disponible
- [ ] `last.pt` escrito y con hash registrado
- [ ] **`selected_batch` es un entero positivo**, no `-1`
- [ ] `train_yolo26n_seg.final.yaml` generado con ese batch
- [ ] VRAM pico registrada, con margen suficiente
- [ ] Memoria estable (sin crecimiento entre iteraciones)
- [ ] Validación ejecutable al final de la época
- [ ] Reanudación comprobada (que `resume` arranca, sin completarla)

Mediciones a anotar para las decisiones posteriores:

- [ ] imágenes/segundo
- [ ] tiempo por batch y por época (train y val por separado)
- [ ] **utilización media de GPU** (si < 80 %, el cuello es el DataLoader)
- [ ] utilización de CPU y RAM
- [ ] espera del DataLoader y lectura de disco

## Decisiones que sólo pueden tomarse tras el smoke

- [ ] Congelar el batch medido como entero en la configuración del baseline
- [ ] Decidir `cache` (false frente a disk) según la utilización de GPU
- [ ] Ajustar `workers` a los vCPU reales de la máquina
- [ ] Rellenar `cloud_cost_estimator.csv` y estimar el coste del baseline
- [ ] Confirmar que el coste estimado es aceptable **antes** de lanzar 150 épocas

## Gate 3 — Baseline completed

`CONFIRM_SEGMENTATION_TRAINING=1 make leaf-segmentation-cloud-train`

- [ ] Entrenamiento finalizado (o detenido por `patience`)
- [ ] `best.pt` y `last.pt` presentes
- [ ] `results.csv`, `args.yaml` y gráficos guardados
- [ ] Métricas de val registradas
- [ ] Configuración congelada y anotada
- [ ] Hashes de checkpoints calculados (`make leaf-segmentation-cloud-checksums`)
- [ ] Fila añadida a `experiment_registry.csv`
- [ ] **`outputs/leaf_detection/` descargado antes de apagar la máquina**

## Gate 4 — Configuration frozen

- [ ] Ablaciones ejecutadas, un factor cada una
- [ ] Configuración elegida **usando sólo val**
- [ ] Confirmación con tres semillas; media y desviación calculadas
- [ ] La diferencia entre configuraciones supera la desviación entre semillas
- [ ] Ni el test interno ni el piloto se usaron para elegir nada

## Gate 5 — Internal test completed

- [ ] Configuración congelada antes de mirar el test
- [ ] `make leaf-segmentation-cloud-test` ejecutado **una sola vez**
- [ ] Métricas y análisis de errores registrados
- [ ] **Ningún hiperparámetro ajustado después**

## Gate 6 — External pilot completed

- [ ] `test_summary.status = passed`
- [ ] `CONFIRM_PILOT_EVALUATION=1 make leaf-segmentation-pilot-evaluate`
- [ ] Revisión visual de las 100 predicciones
- [ ] Tasa de fallback medida
- [ ] Diferencia con el test interno interpretada como generalización de dominio
- [ ] Anotado que las métricas cuantitativas no aplican mientras el piloto
      conserve la regla de hoja principal

## Gate 7 — Downstream classifier experiment ready

- [ ] Máscaras generadas una sola vez y congeladas
- [ ] Manifiesto con fingerprint de las máscaras
- [ ] Variante bbox ROI generada
- [ ] Variante masked ROI generada
- [ ] Tasa de fallback registrada por imagen
- [ ] **`data/splits/seed_42_baseline/` intacto** (los tres SHA-256 sin cambios)
- [ ] `baseline_full` disponible como referencia sin modificar

## Prohibiciones permanentes

- No ejecutar el piloto antes del Gate 5.
- No reejecutar el test interno tras verlo.
- No elegir hiperparámetros con test ni con piloto.
- No modificar splits, decisiones humanas ni el piloto.
- No crear un objetivo `make` que encadene todo el entrenamiento.
- No activar `leaf_detection` ni cambiar `baseline_full` sin evidencia
  downstream.
