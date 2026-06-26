#!/bin/bash
# ==============================================================================
# Environment Setup — ConcurRL-vLLM Benchmarking
# ==============================================================================
# Sets up a Python virtual environment with pinned dependencies for
# high-concurrency vLLM benchmarking on dual A800 GPUs.
#
# Target hardware:
#   GPU:     2× NVIDIA A800-SXM4-80GB (Ampere, sm_80)
#   CUDA:    12.8
#   Driver:  570.x (R570 branch)
#   Python:  3.11
#
# Usage:
#   bash setup.sh
#   bash setup.sh --fresh   # Delete .venv and start from scratch
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FRESH=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh) FRESH=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# -------------------------------------------------------------------
# Version pins
# -------------------------------------------------------------------
TORCH_VERSION="2.11.0"
TORCH_INDEX="https://mirrors.nju.edu.cn/pytorch/whl/cu128"
VLLM_VERSION="0.21.0"

AIOHTTP_VERSION="3.9.0"
NUMPY_VERSION="1.26.0"
PYDANTIC_VERSION="2.0.0"
PSUTIL_VERSION="5.9.0"
FASTAPI_VERSION="0.104.0"
UVICORN_VERSION="0.24.0"

echo "============================================"
echo " ConcurRL-vLLM — Environment Setup"
echo " Target: NVIDIA A800 x2 | CUDA 12.8 | Python 3.11"
echo "============================================"
echo ""

# -------------------------------------------------------------------
# Python version check
# -------------------------------------------------------------------
PYTHON=$(which python3.11 2>/dev/null || which python3 || which python)
PY_VER=$($PYTHON --version 2>&1)
echo "[setup] Using: $PY_VER"

if ! echo "$PY_VER" | grep -q "3\.11"; then
    echo "[setup] WARNING: Expected Python 3.11, got: $PY_VER"
    echo "[setup] Recommended: apt install python3.11 python3.11-venv"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# -------------------------------------------------------------------
# Virtual environment
# -------------------------------------------------------------------
if $FRESH && [ -d ".venv" ]; then
    echo ""
    echo "[setup] Removing existing .venv (--fresh)..."
    rm -rf .venv
fi

if [ ! -d ".venv" ]; then
    echo ""
    echo "[setup] Creating Python virtual environment..."
    $PYTHON -m venv .venv --clear
fi

echo "[setup] Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null

PIP_PATH=$(which pip)
if ! echo "$PIP_PATH" | grep -q ".venv"; then
    echo "[setup] ERROR: pip is not from .venv: $PIP_PATH"
    exit 1
fi
echo "[setup]   pip: $PIP_PATH"

# -------------------------------------------------------------------
# Upgrade pip
# -------------------------------------------------------------------
echo ""
echo "[setup] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel --quiet

# -------------------------------------------------------------------
# Step 1: PyTorch with CUDA 12.8
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo " Step 1/3: PyTorch $TORCH_VERSION (CUDA 12.8)"
echo "============================================"

echo "[setup] Installing torch==$TORCH_VERSION from cu128 index..."
pip install \
    "torch==$TORCH_VERSION" \
    --index-url "$TORCH_INDEX" \
    # --extra-index-url "https://pypi.org/simple"

echo "[setup] Verifying PyTorch CUDA support..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('  WARNING: CUDA not available!')
"

# -------------------------------------------------------------------
# Step 2: vLLM
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo " Step 2/3: vLLM $VLLM_VERSION"
echo "============================================"

echo "[setup] Installing vllm==$VLLM_VERSION..."
if pip install "vllm==$VLLM_VERSION" 2>&1 | tee /tmp/vllm_install.log; then
    echo "[setup] vLLM installed successfully."
else
    echo "[setup] vLLM $VLLM_VERSION failed — trying latest..."
    pip install vllm || {
        echo "[setup] ERROR: vLLM installation failed."
        exit 1
    }
fi

# Re-install torch cu128 (vLLM may override it)
echo "[setup] Ensuring torch cu128 is preserved..."
pip install "torch==$TORCH_VERSION" \
    --index-url "$TORCH_INDEX" \
    --extra-index-url "https://pypi.org/simple" \
    --force-reinstall --no-deps 2>/dev/null || true

echo "[setup] Verifying vLLM..."
python -c "import vllm; print(f'  vLLM: {vllm.__version__}')" 2>/dev/null \
    || echo "[setup]   WARNING: Could not verify vLLM version"

# -------------------------------------------------------------------
# Step 3: Supporting packages
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo " Step 3/3: Supporting packages"
echo "============================================"

echo "[setup] Installing aiohttp, pydantic, psutil, fastapi, uvicorn, numpy..."
pip install \
    "aiohttp>=$AIOHTTP_VERSION" \
    "pydantic>=$PYDANTIC_VERSION" \
    "psutil>=$PSUTIL_VERSION" \
    "fastapi>=$FASTAPI_VERSION" \
    "uvicorn>=$UVICORN_VERSION" \
    "numpy>=$NUMPY_VERSION"

# -------------------------------------------------------------------
# Final verification
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo " Verifying installation"
echo "============================================"

python -c "
import sys, torch, aiohttp, numpy, pydantic, psutil
print(f'  Python:    {sys.version.split()[0]}')
print(f'  PyTorch:   {torch.__version__}')
print(f'  CUDA:      {torch.cuda.is_available()} ({torch.version.cuda if torch.cuda.is_available() else \"N/A\"})')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}:      {props.name} ({props.total_mem // (1024**3)} GB)')
print(f'  aiohttp:   {aiohttp.__version__}')
print(f'  numpy:     {numpy.__version__}')
print(f'  pydantic:  {pydantic.__version__}')
print(f'  psutil:    {psutil.__version__}')
try:
    import vllm
    print(f'  vLLM:      {vllm.__version__}')
except ImportError:
    print('  vLLM:      NOT FOUND')
try:
    import fastapi
    print(f'  FastAPI:   {fastapi.__version__}')
except ImportError:
    print('  FastAPI:   NOT FOUND (01_compile_check.py will skip)')
"

# -------------------------------------------------------------------
# Done
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo " Environment:"
echo "   Python:    3.11"
echo "   PyTorch:   $TORCH_VERSION (CUDA 12.8 / cu128)"
echo "   vLLM:      $VLLM_VERSION"
echo "   GPU:       2x A800-SXM4-80GB"
echo ""
echo " Next steps:"
echo ""
echo " 1. Download model weights:"
echo "    huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir ./models/Qwen3-30B-A3B"
echo ""
echo " 2. Run compile check (no model needed):"
echo "    bash run_all.sh --phase test"
echo ""
echo " 3. Run full Phase 1 pipeline:"
echo "    bash run_all.sh --phase 1"
echo ""
echo " 4. Run everything:"
echo "    bash run_all.sh --phase all"
echo ""
