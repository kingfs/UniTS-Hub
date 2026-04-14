#!/usr/bin/env sh
set -eu

KRONOS_RUNTIME_REPO="${KRONOS_RUNTIME_REPO:-https://github.com/shiyu-coder/Kronos.git}"
KRONOS_RUNTIME_REF="${KRONOS_RUNTIME_REF:-master}"
KRONOS_RUNTIME_PATH="${KRONOS_RUNTIME_PATH:-/opt/kronos-runtime}"

rm -rf "${KRONOS_RUNTIME_PATH}"
git clone --depth 1 --branch "${KRONOS_RUNTIME_REF}" "${KRONOS_RUNTIME_REPO}" "${KRONOS_RUNTIME_PATH}"

if [ -f "${KRONOS_RUNTIME_PATH}/requirements.txt" ]; then
  uv pip install -r "${KRONOS_RUNTIME_PATH}/requirements.txt"
fi
