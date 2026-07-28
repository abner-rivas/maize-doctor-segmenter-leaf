#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib.sh"
OUTPUT_ROOT="${LEAF_SEGMENTATION_OUTPUT:-outputs/leaf_detection}"
CLOUD_DIR="${CLOUD_TRAINING_DIR:-cloud_training}"
BEST="${OUTPUT_ROOT}/segmenter/yolo26n_seg_baseline/weights/best.pt"
[[ -f "${BEST}" ]] || { echo "Falta ${BEST}" >&2; exit 2; }
record_invocation "${OUTPUT_ROOT}/segmenter_evaluation" "$0" "$@"
"${PYTHON_BIN}" "${CLOUD_DIR}/run_ultralytics.py" evaluate --checkpoint "${BEST}" \
  --config "${CLOUD_DIR}/configs/validate_yolo26n_seg.yaml" --split val "$@"
