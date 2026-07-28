#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Todos los pasos posteriores al bootstrap usan exactamente su intérprete,
# aunque el proveedor haya activado otro virtualenv en la sesión.
if [[ -x .venv-cloud/bin/python ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv-cloud/bin/python"
elif [[ -n "${PYTHON:-}" ]] && command -v "${PYTHON}" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "No se encontró un intérprete Python" >&2
  exit 2
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
