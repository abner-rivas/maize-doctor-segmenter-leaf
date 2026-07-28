#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON:-python}"

# Reutiliza el entorno creado por bootstrap_cloud.sh cuando no hay venv activo,
# para que preflight/smoke/train usen el mismo intérprete que instaló Ultralytics.
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv-cloud/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv-cloud/bin/activate
  PYTHON_BIN=python
fi

json_value() {
  local file="$1"
  local expression="$2"
  "${PYTHON_BIN}" - "${file}" "${expression}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for part in sys.argv[2].split("."):
    value = value[part]
print(str(value).lower() if isinstance(value, bool) else value)
PY
}

require_status() {
  local file="$1"
  local expected="$2"
  [[ -f "${file}" ]] || { echo "Falta gate: ${file}" >&2; exit 2; }
  local actual
  actual="$(json_value "${file}" status)"
  [[ "${actual}" == "${expected}" ]] || {
    echo "Gate bloqueado: ${file}: ${actual} != ${expected}" >&2
    exit 2
  }
}

record_invocation() {
  local output_dir="$1"
  shift
  mkdir -p "${output_dir}"
  {
    printf 'utc='; date -u +%Y-%m-%dT%H:%M:%SZ
    printf 'commit='; git rev-parse HEAD 2>/dev/null || printf 'unavailable\n'
    printf 'command='; printf '%q ' "$@"; printf '\n'
    printf 'seed=42\n'
  } >> "${output_dir}/invocations.log"
}
