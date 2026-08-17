#!/usr/bin/env bash
set -euo pipefail

[[ "${CONFIRM_SEGMENTATION_TRAINING:-0}" == "1" ]] || {
  echo "Reanudación bloqueada: use CONFIRM_SEGMENTATION_TRAINING=1" >&2
  exit 2
}
source "$(dirname "$0")/lib.sh"
OUTPUT_ROOT="${LEAF_SEGMENTATION_OUTPUT:-outputs/leaf_detection}"
CLOUD_DIR="${CLOUD_TRAINING_DIR:-cloud_training}"
ACTIVE_MANIFEST="${ACTIVE_MANIFEST:-${OUTPUT_ROOT}/segmenter/active_run_manifest.json}"
[[ -f "${ACTIVE_MANIFEST}" ]] || {
  echo "Falta identidad del run: ${ACTIVE_MANIFEST}" >&2
  echo "Use leaf-segmentation-cloud-train para iniciar un run nuevo." >&2
  exit 2
}
RUN_STATUS="$(json_value "${ACTIVE_MANIFEST}" status)"
case "${RUN_STATUS}" in
  running|failed) ;;
  completed)
    echo "El run activo ya está completo; no requiere reanudación." >&2
    exit 2
    ;;
  *)
    echo "Estado de run no reanudable: ${RUN_STATUS}" >&2
    exit 2
    ;;
esac
CHECKPOINT="$(json_value "${ACTIVE_MANIFEST}" expected_last_checkpoint)"
[[ -f "${CHECKPOINT}" ]] || { echo "Falta ${CHECKPOINT}" >&2; exit 2; }
require_status "${OUTPUT_ROOT}/cloud_preflight/summary.json" ready_for_smoke_training
record_invocation "${OUTPUT_ROOT}/segmenter" "$0" "$@"
"${PYTHON_BIN}" "${CLOUD_DIR}/run_ultralytics.py" resume \
  --checkpoint "${CHECKPOINT}" --active-manifest "${ACTIVE_MANIFEST}" "$@"
