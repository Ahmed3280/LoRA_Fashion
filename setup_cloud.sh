#!/bin/bash
# =============================================================================
# setup_cloud.sh — One-command environment setup for Vast.ai / RunPod
# =============================================================================
# Run this ONCE after spinning up your cloud GPU instance.
#
# Tested on: PyTorch 2.4.0 + CUDA 12.1 base image (use this template on Vast.ai)
#
# Usage:
#   bash setup_cloud.sh
# =============================================================================

set -e  # stop on any error

echo "=============================================="
echo "  CatVTON Cloud Setup"
echo "  $(date)"
echo "=============================================="

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "[1/5] Installing system packages ..."
apt-get update -qq
apt-get install -y -q zip unzip git wget curl

# ── 2. Python packages ────────────────────────────────────────────────────────
echo ""
echo "[2/5] Installing Python packages ..."
# NOTE: torch/torchvision are intentionally excluded — the RunPod base image
# ships with CUDA-enabled torch. Installing from PyPI would replace it with
# a CPU-only build and break training.
pip install --quiet -r requirements_train.txt

# ── 3. Detectron2 (required by DensePose) ────────────────────────────────────
echo ""
echo "[3/5] Installing detectron2 ..."
# Pre-built wheel for torch 2.4 + CUDA 12.1
pip install --quiet detectron2 \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.4/index.html \
    || echo "[WARN] detectron2 pre-built wheel failed, trying from source ..."  \
    && pip install --quiet "git+https://github.com/facebookresearch/detectron2.git"

# ── 4. Verify GPU ─────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Verifying GPU ..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  GPU   : {torch.cuda.get_device_name(0)}')
print(f'  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'  CUDA  : {torch.version.cuda}')
print(f'  Torch : {torch.__version__}')
"

# ── 5. Quick import check ─────────────────────────────────────────────────────
echo ""
echo "[5/5] Checking key imports ..."
python -c "
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
print('  diffusers   OK')
print('  peft        OK')
print('  accelerate  OK')
"

echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Upload your dataset zip:"
echo "       scp dataset.zip root@<vast-ip>:<port>:/workspace/"
echo "       unzip dataset.zip -d /workspace/data/mena"
echo ""
echo "  2. Upload your code zip:"
echo "       scp catvton.zip root@<vast-ip>:<port>:/workspace/"
echo "       unzip catvton.zip -d /workspace/catvton"
echo "       cd /workspace/catvton"
echo ""
echo "  3. Generate masks (if not done locally):"
echo "       python prepare_dataset.py \\"
echo "           --raw_data_dir /workspace/data/mena/raw \\"
echo "           --output_dir   /workspace/data/mena \\"
echo "           --cloth_type   overall"
echo ""
echo "  4. Run training:"
echo "       python train_lora.py \\"
echo "           --data_root       /workspace/CatVTON \\"
echo "           --output_dir      /workspace/output/lora_mena \\"
echo "           --cloth_type      overall \\"
echo "           --height          1024 \\"
echo "           --width           768 \\"
echo "           --batch_size      1 \\"
echo "           --num_epochs      20 \\"
echo "           --mixed_precision bf16"
echo "=============================================="
