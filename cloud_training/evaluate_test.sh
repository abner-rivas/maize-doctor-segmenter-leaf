#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib.sh"
OUTPUT_ROOT="${LEAF_SEGMENTATION_OUTPUT:-outputs/leaf_detection}"
CLOUD_DIR="${CLOUD_TRAINING_DIR:-cloud_training}"
BEST="${OUTPUT_ROOT}/segmenter/yolo26n_seg_baseline/weights/best.pt"
TEST_SUMMARY="${OUTPUT_ROOT}/segmenter_evaluation/test_summary.json"

# Gate de un solo uso: el test interno evalua la configuracion congelada una
# unica vez. Repetirlo tras ver el resultado invalida su valor metodologico.
if [[ -f "${TEST_SUMMARY}" && "${FORCE_INTERNAL_TEST_RERUN:-0}" != "1" ]]; then
  echo "Test interno ya ejecutado: ${TEST_SUMMARY}" >&2
  echo "Repetirlo permite ajustar sobre test e invalida la evaluacion." >&2
  echo "Solo con una decision formal registrada: FORCE_INTERNAL_TEST_RERUN=1" >&2
  exit 2
fi
[[ -f "${BEST}" ]] || { echo "Falta ${BEST}" >&2; exit 2; }
record_invocation "${OUTPUT_ROOT}/segmenter_evaluation" "$0" "$@"
"${PYTHON_BIN}" "${CLOUD_DIR}/run_ultralytics.py" evaluate --checkpoint "${BEST}" \
  --config "${CLOUD_DIR}/configs/validate_yolo26n_seg.yaml" --split test "$@"
