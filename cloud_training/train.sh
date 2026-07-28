#!/usr/bin/env bash
set -euo pipefail

[[ "${CONFIRM_SEGMENTATION_TRAINING:-0}" == "1" ]] || {
  echo "Entrenamiento bloqueado: use CONFIRM_SEGMENTATION_TRAINING=1" >&2
  exit 2
}
source "$(dirname "$0")/lib.sh"
OUTPUT_ROOT="${LEAF_SEGMENTATION_OUTPUT:-outputs/leaf_detection}"
CLOUD_DIR="${CLOUD_TRAINING_DIR:-cloud_training}"
CONFIG_PATH="${CONFIG:-${OUTPUT_ROOT}/segmenter/configs/train_yolo26n_seg.final.yaml}"
if [[ "$#" -ne 0 ]]; then
  echo "train.sh no acepta argumentos libres; use CONFIG=<archivo.yaml>" >&2
  exit 2
fi
[[ -f "${CONFIG_PATH}" ]] || {
  echo "Falta configuración congelada: ${CONFIG_PATH}" >&2
  echo "Ejecute primero el smoke y revise train_yolo26n_seg.final.yaml" >&2
  exit 2
}
require_status "${OUTPUT_ROOT}/cloud_preflight/summary.json" ready_for_smoke_training
require_status "${OUTPUT_ROOT}/segmenter/smoke_summary.json" passed
record_invocation "${OUTPUT_ROOT}/segmenter" "$0" "--config" "${CONFIG_PATH}"
"${PYTHON_BIN}" "${CLOUD_DIR}/run_ultralytics.py" train \
  --config "${CONFIG_PATH}"
