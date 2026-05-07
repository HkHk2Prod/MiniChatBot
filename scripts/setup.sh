#!/usr/bin/env bash
# Initialize a CUDA-enabled venv for MiniChatBot.
#
# CUDA is the default. Override via environment variables:
#   CUDA=cu126        Use a different CUDA version (cu118, cu121, cu124, cu126).
#   USE_CPU=1         Install CPU-only torch instead.
#   NO_EXTRAS=1       Skip [dev,tensorboard] extras.
#   FORCE=1           Force-reinstall torch (e.g., switching CPU<->CUDA wheels).
#   PYTHON=python3    Python interpreter used to create the venv.
#
# Examples:
#   ./scripts/setup.sh
#   CUDA=cu126 ./scripts/setup.sh
#   USE_CPU=1 ./scripts/setup.sh
#   FORCE=1 ./scripts/setup.sh

set -euo pipefail

CUDA="${CUDA:-cu124}"
USE_CPU="${USE_CPU:-0}"
NO_EXTRAS="${NO_EXTRAS:-0}"
FORCE="${FORCE:-0}"
PYTHON="${PYTHON:-python3}"

if [ ! -f "pyproject.toml" ]; then
    echo "pyproject.toml not found in $(pwd). Run this from the project root." >&2
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating .venv with $PYTHON ..."
    "$PYTHON" -m venv .venv
fi

PY=".venv/bin/python"
"$PY" -m pip install --upgrade pip --quiet

PIP_FLAGS=("--quiet")
if [ "$FORCE" = "1" ]; then
    PIP_FLAGS+=("--force-reinstall")
fi

if [ "$USE_CPU" = "1" ]; then
    echo "Installing torch (CPU) ..."
    "$PY" -m pip install torch "${PIP_FLAGS[@]}"
else
    INDEX="https://download.pytorch.org/whl/${CUDA}"
    echo "Installing torch from $INDEX ..."
    "$PY" -m pip install torch --index-url "$INDEX" "${PIP_FLAGS[@]}"
fi

if [ "$NO_EXTRAS" = "1" ]; then
    TARGET="."
else
    TARGET=".[dev,tensorboard,data]"
fi
echo "Installing project (editable) from $TARGET ..."
"$PY" -m pip install -e "$TARGET" --quiet

echo
echo "--- Verification ---"
"$PY" -c "import torch; print(f'torch: {torch.__version__}'); cuda = torch.cuda.is_available(); print(f'cuda.is_available: {cuda}'); print(f'device: {torch.cuda.get_device_name(0) if cuda else chr(34)+chr(99)+chr(112)+chr(117)+chr(34)}')"
