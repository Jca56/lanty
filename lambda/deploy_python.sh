#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Lanty - Lambda Cloud Python Training Pipeline
# ============================================================
# Usage: ./deploy_python.sh <INSTANCE_IP>
#
# Uploads project, installs Python ML stack, runs LoRA fine-tuning
# of Qwen2.5-7B-Instruct on Lanty data, builds llama.cpp, converts
# to GGUF, downloads everything back.
# ============================================================

LAMBDA_USER="${LAMBDA_USER:-ubuntu}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/lanty}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
QUANT="${QUANT:-Q4_K_M}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <INSTANCE_IP>"
    echo ""
    echo "Environment variables:"
    echo "  BASE_MODEL    HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)"
    echo "  EPOCHS        Training epochs (default: 5)"
    echo "  BATCH_SIZE    Per-device batch size (default: 2)"
    echo "  GRAD_ACCUM    Gradient accumulation steps (default: 8)"
    echo "  LR            Learning rate (default: 2e-4)"
    echo "  QUANT         GGUF quantization type (default: Q4_K_M)"
    exit 1
fi

INSTANCE_IP="$1"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $SSH_KEY"
RSYNC="rsync -avz -e \"$SSH\""
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

START_TIME=$(date +%s)

echo "============================================"
echo "  Lanty Python Training Deployment"
echo "============================================"
echo "Instance:    ${LAMBDA_USER}@${INSTANCE_IP}"
echo "Base model:  ${BASE_MODEL}"
echo "Epochs:      ${EPOCHS}"
echo "Quant:       ${QUANT}"
echo ""

# --- Step 1: Wait for SSH ---
echo "[1/8] Waiting for SSH..."
for i in $(seq 1 30); do
    if $SSH ${LAMBDA_USER}@${INSTANCE_IP} "echo ok" &>/dev/null; then
        echo "  Connected!"
        break
    fi
    [ "$i" -eq 30 ] && { echo "ERROR: Could not connect"; exit 1; }
    sleep 8
done

# --- Step 2: Check GPU ---
echo ""
echo "[2/8] Checking GPU..."
$SSH ${LAMBDA_USER}@${INSTANCE_IP} "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"

# --- Step 3: Upload project ---
echo ""
echo "[3/8] Uploading project..."
$SSH ${LAMBDA_USER}@${INSTANCE_IP} "mkdir -p ${REMOTE_DIR}"
eval $RSYNC \
    --exclude '__pycache__/' \
    --exclude '.git/' \
    --exclude 'lambda/' \
    --exclude 'data/lore/' \
    --exclude 'data/lanty_voice/' \
    --exclude 'data/training/batch_*.txt' \
    --exclude 'scripts/.venv/' \
    --exclude 'models/*.gguf' \
    --exclude 'models/lanty-qwen-merged/' \
    "${PROJECT_ROOT}/" "${LAMBDA_USER}@${INSTANCE_IP}:${REMOTE_DIR}/"
echo "  Upload complete!"

# --- Step 4: Install Python dependencies ---
echo ""
echo "[4/8] Installing Python ML stack..."
$SSH ${LAMBDA_USER}@${INSTANCE_IP} bash -s <<'EOF'
set -e
cd /home/ubuntu/lanty
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip
# Install PyTorch with CUDA 12.8 wheel (matches Lambda H100 driver 570/CUDA 12.8).
# The default torch wheel on PyPI ships with cu130 which needs driver 580+.
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128
pip install --quiet \
    transformers \
    peft \
    trl \
    datasets \
    accelerate \
    bitsandbytes \
    sentencepiece \
    protobuf
echo "  Python packages installed."
python -c "import torch; print(f'  torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}')"
EOF

# --- Step 5: Prepare chat dataset (if not already done) ---
echo ""
echo "[5/8] Building chat-format dataset..."
$SSH ${LAMBDA_USER}@${INSTANCE_IP} bash -s <<EOF
set -e
cd ${REMOTE_DIR}
source .venv/bin/activate
python scripts/prepare_chat_dataset.py
EOF

# --- Step 6: Train LoRA adapter ---
echo ""
echo "[6/8] Training LoRA adapter..."
TRAIN_START=$(date +%s)
$SSH ${LAMBDA_USER}@${INSTANCE_IP} bash -s <<EOF
set -e
cd ${REMOTE_DIR}
source .venv/bin/activate
# --no-quantize: use full bf16 LoRA. QLoRA (4-bit base) has known NaN issues
# with newer torch/bnb versions. We have plenty of VRAM on Lambda anyway.
python scripts/train_lanty.py \
    --no-quantize \
    --base-model ${BASE_MODEL} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --lr ${LR}
EOF
TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$((TRAIN_END - TRAIN_START))
echo "  Training finished in ${TRAIN_ELAPSED}s"

# --- Step 7: Merge LoRA + convert to GGUF ---
echo ""
echo "[7/8] Merging LoRA + building llama.cpp + converting to GGUF..."
$SSH ${LAMBDA_USER}@${INSTANCE_IP} bash -s <<EOF
set -e
cd ${REMOTE_DIR}
source .venv/bin/activate

# Merge LoRA into base model
python scripts/merge_and_export.py

# Build llama.cpp if not present
if [ ! -d /home/ubuntu/llama.cpp ]; then
    cd /home/ubuntu
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    cmake -B build
    cmake --build build --config Release -j\$(nproc)
fi

# Use a SEPARATE venv for llama.cpp's Python deps so its requirements.txt
# doesn't downgrade our cu128 torch in the training venv (which causes
# CUDA to disappear and training to fail).
if [ ! -d /home/ubuntu/llama.cpp/.venv ]; then
    cd /home/ubuntu/llama.cpp
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    deactivate
fi

# Convert merged model to GGUF f16 using llama.cpp's isolated venv
source /home/ubuntu/llama.cpp/.venv/bin/activate
cd /home/ubuntu/llama.cpp
python convert_hf_to_gguf.py ${REMOTE_DIR}/models/lanty-qwen-merged \
    --outfile ${REMOTE_DIR}/models/lanty-qwen-f16.gguf \
    --outtype f16
deactivate

# Quantize to Q4_K_M
./build/bin/llama-quantize \
    ${REMOTE_DIR}/models/lanty-qwen-f16.gguf \
    ${REMOTE_DIR}/models/lanty-qwen-${QUANT}.gguf \
    ${QUANT}

# Clean up the f16 — we only need the quantized version
rm -f ${REMOTE_DIR}/models/lanty-qwen-f16.gguf

ls -lh ${REMOTE_DIR}/models/
EOF

# --- Step 8: Download trained model ---
echo ""
echo "[8/8] Downloading trained model..."
mkdir -p "${PROJECT_ROOT}/models"
eval $RSYNC \
    "${LAMBDA_USER}@${INSTANCE_IP}:${REMOTE_DIR}/models/lanty-qwen-${QUANT}.gguf" \
    "${PROJECT_ROOT}/models/"
eval $RSYNC \
    "${LAMBDA_USER}@${INSTANCE_IP}:${REMOTE_DIR}/models/lanty-qwen-lora/" \
    "${PROJECT_ROOT}/models/lanty-qwen-lora/"

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================"
echo "  Training Complete!"
echo "============================================"
echo "  Total time:    ${TOTAL_ELAPSED}s"
echo "  Training time: ${TRAIN_ELAPSED}s"
echo ""
echo "  Files downloaded:"
echo "    models/lanty-qwen-${QUANT}.gguf  (for llama.cpp inference)"
echo "    models/lanty-qwen-lora/          (LoRA adapter)"
echo ""
echo "  Test locally:"
echo "    python inference/chat.py"
echo ""
echo "  Remember to terminate your Lambda instance!"
echo "============================================"
