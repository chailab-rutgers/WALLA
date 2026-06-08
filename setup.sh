#!/usr/bin/env bash
# Create .venv and install Python dependencies for WALLA.
#
# Usage (from the repo root):
#   bash setup.sh
#
# Then activate:
#   source .venv/bin/activate

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: $PYTHON not found; install Python 3.12+ or set PYTHON=..." >&2
    exit 1
fi

PY_MAJOR=$("$PYTHON" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "error: need Python >= 3.10 (found $($PYTHON --version))" >&2
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment at .venv (prompt: WALLA)..."
    "$PYTHON" -m venv --prompt WALLA .venv
else
    echo "Using existing .venv"
fi

echo "Installing dependencies from requirements.txt..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

cat <<'EOF'

Done.

Activate the environment:
  source .venv/bin/activate

Your shell prompt should show (WALLA).

Verify the environment (no GPU or model downloads):
  .venv/bin/python scripts/verify_setup.py

Run scripts with the venv Python (no activation required):
  .venv/bin/python scripts/wagering_pipeline.py examples/configs/wagering_training/walla_v1_2models_mmlu.yaml

Optional API keys (wandb, OpenAI, Hugging Face): copy and edit .api_keys.yaml locally.
See README.md for details.

EOF
