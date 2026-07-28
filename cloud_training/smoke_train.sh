#!/usr/bin/env bash
set -euo pipefail

[[ "${CONFIRM_SEGMENTATION_SMOKE_TRAINING:-0}" == "1" ]] || {
  echo "Smoke bloqueado: use CONFIRM_SEGMENTATION_SMOKE_TRAINING=1" >&2
  exit 2
}
source "$(dirname "$0")/lib.sh"
OUTPUT_ROOT="${LEAF_SEGMENTATION_OUTPUT:-outputs/leaf_detection}"
CLOUD_DIR="${CLOUD_TRAINING_DIR:-cloud_training}"
require_status "${OUTPUT_ROOT}/cloud_preflight/summary.json" ready_for_smoke_training
record_invocation "${OUTPUT_ROOT}/segmenter/smoke" "$0" "$@"
"${PYTHON_BIN}" "${CLOUD_DIR}/run_ultralytics.py" smoke \
  --config "${CLOUD_DIR}/configs/smoke_yolo26n_seg.yaml" "$@"
