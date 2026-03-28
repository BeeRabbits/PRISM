#!/usr/bin/env bash
# PRISM: Full Environment Setup
# Run once on a fresh Ubuntu 24.04 GPU instance (RunPod A100 / RTX 4090)
# Usage: bash setup.sh

set -euo pipefail

echo "════════════════════════════════════════════════════"
echo " PRISM Environment Setup"
echo "════════════════════════════════════════════════════"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "[1/7] Updating system packages..."
apt-get update && apt-get upgrade -y

echo "[2/7] Installing system dependencies..."
apt-get install -y \
    git curl wget build-essential \
    python3.11 python3.11-venv python3-pip \
    nvidia-cuda-toolkit docker.io docker-compose

# ---------------------------------------------------------------------------
# 2. uv (fast Python package manager)
# ---------------------------------------------------------------------------
echo "[3/7] Installing uv..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "  uv version: $(uv --version)"

# ---------------------------------------------------------------------------
# 3. Python virtual environment
# ---------------------------------------------------------------------------
echo "[4/7] Creating Python 3.11 virtual environment..."
uv venv .venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 4. Python dependencies
# ---------------------------------------------------------------------------
echo "[5/7] Installing Python dependencies (PyTorch + CUDA 12.4)..."
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

echo "  Installing ML / training dependencies..."
uv pip install \
    transformers \
    accelerate \
    peft \
    datasets \
    bitsandbytes \
    unsloth \
    trl \
    einops \
    sentencepiece \
    huggingface_hub \
    wandb

echo "  Installing server / data dependencies..."
uv pip install \
    fastapi \
    uvicorn \
    sqlalchemy \
    aiosqlite \
    asyncpg \
    psycopg2-binary \
    pgvector \
    python-dotenv \
    apscheduler \
    anthropic \
    openai \
    rich \
    typer

echo "  Installing test / dev dependencies..."
uv pip install pytest pytest-asyncio httpx

# ---------------------------------------------------------------------------
# 5. Verify CUDA
# ---------------------------------------------------------------------------
echo "[6/7] Verifying CUDA..."
python3 -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU             : {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  VRAM            : {mem:.1f} GB')
"

# ---------------------------------------------------------------------------
# 6. Project directory structure
# ---------------------------------------------------------------------------
echo "[7/7] Creating project directories..."
mkdir -p prism_inside/{model,adapters,data,training,server,memory_adapter,scripts,tests,logs}
mkdir -p model/weights
mkdir -p adapters/{current,titans}

# ---------------------------------------------------------------------------
# 7. Copy .env.example to .env if .env doesn't exist
# ---------------------------------------------------------------------------
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "  Copied .env.example → .env (fill in your tokens)"
fi

echo ""
echo "════════════════════════════════════════════════════"
echo " Setup complete."
echo " Next steps:"
echo "   1. Fill in .env (HF_TOKEN, WANDB_API_KEY)"
echo "   2. source .venv/bin/activate"
echo "   3. python scripts/download_model.py"
echo "   4. python main.py"
echo "════════════════════════════════════════════════════"
