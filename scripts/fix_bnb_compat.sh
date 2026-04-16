#!/usr/bin/env bash
# Fix bitsandbytes + torch 2.6 + triton 3.x incompatibility.
#
# Problem: bitsandbytes 0.45.x imports triton.ops.matmul_perf_model at
# module load, but triton 3.x (shipped with torch 2.6) removed that
# module. Newer bnb (0.46+) either drags torch up past 2.6 (cu13 stack
# needs driver >= 12.8 which many pods don't have) or breaks torch's
# NCCL symbols.
#
# Fix: reinstall the known-good torch 2.6.0+cu124 / bnb 0.45.0 combo,
# then install a pass-through stub for triton.ops.matmul_perf_model.
# These stubs are only exercised by bnb's int8 matmul autotune path,
# which PRISM doesn't hit (4-bit NF4 quantization goes through a
# different code path entirely).
#
# Idempotent — safe to re-run.

set -euo pipefail

if [ ! -d .venv ]; then
    echo "Error: .venv not found. Run from the PRISM project root." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[1/3] Reinstalling torch 2.6.0+cu124 and bitsandbytes 0.45.0..."
uv pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.6.0" \
    "torchvision" \
    "bitsandbytes==0.45.0"

echo "[2/3] Installing triton.ops.matmul_perf_model shim..."
PYTHON_VER=$(.venv/bin/python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
TRITON_OPS_DIR=".venv/lib/${PYTHON_VER}/site-packages/triton/ops"
mkdir -p "$TRITON_OPS_DIR"
: > "$TRITON_OPS_DIR/__init__.py"
cat > "$TRITON_OPS_DIR/matmul_perf_model.py" <<'EOF'
"""Shim for triton.ops.matmul_perf_model removed in triton 3.x.

bitsandbytes 0.45.x imports these symbols unconditionally but only uses
them for int8 matmul autotuning, which PRISM doesn't exercise (it uses
4-bit NF4). Pass-through stubs let the import succeed.
"""


def early_config_prune(configs, named_args=None, **kwargs):
    return configs


def estimate_matmul_time(*args, **kwargs):
    return 0.0
EOF

echo "[3/3] Verifying imports..."
.venv/bin/python - <<'PYEOF'
import torch
import bitsandbytes
import transformers
import peft

print(f"  torch        : {torch.__version__}")
print(f"  cuda avail   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu          : {torch.cuda.get_device_name(0)}")
print(f"  bitsandbytes : {bitsandbytes.__version__}")
print(f"  transformers : {transformers.__version__}")
print(f"  peft         : {peft.__version__}")
PYEOF

echo ""
echo "Done. If cuda avail is True, run: python main.py"
