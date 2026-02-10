#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found" >&2
  exit 1
fi

if [[ -x "${VENV_DIR}/bin/python" && -x "${VENV_DIR}/bin/pip" ]]; then
  exit 0
fi

if [[ -e "${VENV_DIR}" ]]; then
  # Avoid `rm -rf` (some environments block it). Use Python's shutil instead.
  VENV_DIR="${VENV_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
import shutil

venv_dir = os.environ["VENV_DIR"]
shutil.rmtree(venv_dir, ignore_errors=True)
PY
fi

"${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"

GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"
TMP_DIR="$(mktemp -d)"
GET_PIP="${TMP_DIR}/get-pip.py"

if command -v curl >/dev/null 2>&1; then
  curl -fsSL "${GET_PIP_URL}" -o "${GET_PIP}"
elif command -v wget >/dev/null 2>&1; then
  wget -qO "${GET_PIP}" "${GET_PIP_URL}"
else
  "${PYTHON_BIN}" - <<'PY'
import urllib.request
url = "https://bootstrap.pypa.io/get-pip.py"
data = urllib.request.urlopen(url, timeout=60).read()
open("get-pip.py", "wb").write(data)
PY
  mv get-pip.py "${GET_PIP}"
fi

"${VENV_DIR}/bin/python" "${GET_PIP}" --no-warn-script-location
