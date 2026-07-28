#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib.sh"
OUT="${LEAF_SEGMENTATION_OUTPUT:-outputs/leaf_detection}/cloud_preflight"
mkdir -p "${OUT}"
record_invocation "${OUT}" "$0" "$@"
"${PYTHON_BIN}" scripts/pipeline/leaf_segmentation_cloud_preflight.py "$@" \
  2>&1 | tee "${OUT}/preflight.log"
require_status "${OUT}/summary.json" "ready_for_smoke_training"
