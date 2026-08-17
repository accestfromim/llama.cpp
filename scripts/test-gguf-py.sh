#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
        PYTHON="${ROOT_DIR}/.venv/bin/python"
    else
        PYTHON="python3"
    fi
fi

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "missing Python interpreter: ${PYTHON}" >&2
    exit 1
fi

if ! "${PYTHON}" -c 'import pytest' >/dev/null 2>&1; then
    echo "pytest is required; install the test dependencies before running this script" >&2
    exit 1
fi

PYTHONPATH="${ROOT_DIR}/gguf-py:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    exec "${PYTHON}" -m pytest -q "${ROOT_DIR}/gguf-py/tests" "$@"
