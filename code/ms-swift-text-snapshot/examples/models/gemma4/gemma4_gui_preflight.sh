#!/usr/bin/env bash

set -euo pipefail

GEMMA4_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${GEMMA4_SCRIPT_DIR}/gemma4_gui_env.sh"

cd "${SWIFT_ROOT}"
exec "${GEMMA4_CONDA_ENV}/bin/python" \
    "${GEMMA4_SCRIPT_DIR}/gemma4_gui_preflight.py" \
    "$@"
