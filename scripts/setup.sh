#!/usr/bin/env bash
# Initialize a virtual environment for MiniChatBot using uv.
#   https://docs.astral.sh/uv/
#
# Creates .venv with `uv venv` and installs the project (editable) plus extras.
# torch is fetched via uv's --torch-backend, which auto-detects your CUDA driver
# by default and falls back to a CPU wheel when no NVIDIA GPU is present.
#
# Override via environment variables:
#   TORCH_BACKEND=cu126   Force a backend: auto, cpu, cu118, cu121, cu124, cu126, cu128.
#   CUDA=cu128            Backward-compatible alias for TORCH_BACKEND.
#   USE_CPU=1             Shortcut for TORCH_BACKEND=cpu.
#   NO_EXTRAS=1           Skip the [dev,tensorboard,data] extras.
#   FORCE=1               Reinstall torch (e.g. switching CPU<->CUDA wheels).
#   PYTHON=3.10           Python version or interpreter for the venv (default: python3).
#
# Examples:
#   ./scripts/setup.sh                 # auto-detect torch backend, full extras
#   TORCH_BACKEND=cu128 ./scripts/setup.sh
#   USE_CPU=1 ./scripts/setup.sh
#   FORCE=1 ./scripts/setup.sh

set -euo pipefail

USE_CPU="${USE_CPU:-0}"
NO_EXTRAS="${NO_EXTRAS:-0}"
FORCE="${FORCE:-0}"
PYTHON="${PYTHON:-python3}"
if [ "$USE_CPU" = "1" ]; then
    BACKEND="cpu"
else
    BACKEND="${TORCH_BACKEND:-${CUDA:-auto}}"
fi

if [ ! -f "pyproject.toml" ]; then
    echo "pyproject.toml not found in $(pwd). Run this from the project root." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it, then re-run:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "  https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

VENV_PY=".venv/bin/python"

echo "Creating .venv (python: $PYTHON) ..."
uv venv --seed --python "$PYTHON" .venv

PIP_FLAGS=()
if [ "$FORCE" = "1" ]; then
    PIP_FLAGS+=("--reinstall-package" "torch")
fi

if [ "$NO_EXTRAS" = "1" ]; then
    TARGET="."
else
    TARGET=".[dev,tensorboard,data]"
fi

echo "Installing project (editable) from $TARGET with torch backend '$BACKEND' ..."
uv pip install --python "$VENV_PY" --torch-backend="$BACKEND" -e "$TARGET" "${PIP_FLAGS[@]}"

echo
echo "--- Verification ---"
"$VENV_PY" -c 'import torch; print("torch:", torch.__version__); c = torch.cuda.is_available(); print("cuda.is_available:", c); print("device:", torch.cuda.get_device_name(0) if c else "cpu")'
