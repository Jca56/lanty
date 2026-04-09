#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Lanty - Lambda Cloud Training Deployment Script
# ============================================================
# Usage: ./deploy.sh <INSTANCE_IP> [OPTIONS]
#
# Provisions a Lambda Cloud GPU instance with Rust + CUDA,
# uploads lanty source + training data, runs training,
# and downloads the trained model back.
# ============================================================

# --- Configuration (override via environment variables) ---
LAMBDA_USER="${LAMBDA_USER:-ubuntu}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
MODEL_SIZE="${MODEL_SIZE:-small}"
EPOCHS="${EPOCHS:-20}"
SEQ_LEN="${SEQ_LEN:-128}"
VOCAB_SIZE="${VOCAB_SIZE:-8192}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/lanty}"
HOURLY_RATE="${HOURLY_RATE:-1.10}"  # Default A100 rate

# --- Parse arguments ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <INSTANCE_IP> [--rate HOURLY_RATE]"
    echo ""
    echo "Environment variables:"
    echo "  LAMBDA_USER   SSH user (default: ubuntu)"
    echo "  SSH_KEY        SSH key path (default: ~/.ssh/id_ed25519)"
    echo "  MODEL_SIZE     tiny or small (default: small)"
    echo "  EPOCHS         Training epochs (default: 20)"
    echo "  SEQ_LEN        Sequence length (default: 128)"
    echo "  VOCAB_SIZE     Tokenizer vocab size (default: 8192)"
    echo "  HOURLY_RATE    Instance cost/hr for estimate (default: 1.10)"
    exit 1
fi

INSTANCE_IP="$1"
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rate) HOURLY_RATE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $SSH_KEY"
SSH_CMD="ssh $SSH_OPTS ${LAMBDA_USER}@${INSTANCE_IP}"
SCP_CMD="scp $SSH_OPTS"
RSYNC_CMD="rsync -avz --progress -e \"ssh $SSH_OPTS\""

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
START_TIME=$(date +%s)

echo "============================================"
echo "  Lanty - Lambda Cloud Training Deployment"
echo "============================================"
echo ""
echo "Instance:    ${LAMBDA_USER}@${INSTANCE_IP}"
echo "Model size:  ${MODEL_SIZE}"
echo "Epochs:      ${EPOCHS}"
echo "Seq length:  ${SEQ_LEN}"
echo "Vocab size:  ${VOCAB_SIZE}"
echo "Rate:        \$${HOURLY_RATE}/hr"
echo ""

# --- Step 1: Wait for instance to be reachable ---
echo "[1/7] Waiting for instance to be reachable..."
for i in $(seq 1 30); do
    if $SSH_CMD "echo ok" &>/dev/null; then
        echo "  Connected!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ERROR: Could not connect after 30 attempts"
        exit 1
    fi
    echo "  Attempt $i/30 - retrying in 10s..."
    sleep 10
done

# --- Step 2: Check GPU and CUDA ---
echo ""
echo "[2/7] Checking GPU and CUDA on instance..."
$SSH_CMD "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
CUDA_PATH=$($SSH_CMD "ls -d /usr/local/cuda* 2>/dev/null | tail -1 || echo /usr/local/cuda")
echo "  CUDA path: $CUDA_PATH"
$SSH_CMD "ls ${CUDA_PATH}/bin/nvcc" &>/dev/null || {
    echo "  ERROR: nvcc not found at ${CUDA_PATH}/bin/nvcc"
    echo "  Lambda instances should have CUDA pre-installed."
    exit 1
}
$SSH_CMD "${CUDA_PATH}/bin/nvcc --version" | tail -1
echo "  GPU and CUDA OK!"

# --- Step 3: Install Rust ---
echo ""
echo "[3/7] Setting up Rust toolchain..."
$SSH_CMD "command -v cargo &>/dev/null" &>/dev/null && {
    echo "  Rust already installed"
    $SSH_CMD "rustc --version"
} || {
    echo "  Installing Rust via rustup..."
    $SSH_CMD "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    echo "  Rust installed!"
}

# --- Step 4: Upload source code and data ---
echo ""
echo "[4/7] Uploading project source and training data..."
$SSH_CMD "mkdir -p ${REMOTE_DIR}"

# Upload source (excluding target dir and models)
eval $RSYNC_CMD \
    --exclude 'target/' \
    --exclude 'models/' \
    --exclude '.git/' \
    --exclude 'lambda/' \
    "${PROJECT_ROOT}/" "${LAMBDA_USER}@${INSTANCE_IP}:${REMOTE_DIR}/"

echo "  Upload complete!"

# --- Step 5: Build ---
echo ""
echo "[5/7] Building lanty (release mode with CUDA)..."
$SSH_CMD "source ~/.cargo/env && \
    export PATH=${CUDA_PATH}/bin:\$PATH && \
    export CUDA_PATH=${CUDA_PATH} && \
    cd ${REMOTE_DIR} && \
    cargo build --release 2>&1 | tail -3"
echo "  Build complete!"

# --- Step 6: Train ---
echo ""
echo "[6/7] Training..."
TRAIN_START=$(date +%s)

# Train tokenizer if needed
$SSH_CMD "test -f ${REMOTE_DIR}/models/tokenizer.json" &>/dev/null || {
    echo "  Training tokenizer (vocab_size=${VOCAB_SIZE})..."
    $SSH_CMD "source ~/.cargo/env && \
        export PATH=${CUDA_PATH}/bin:\$PATH && \
        export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH && \
        cd ${REMOTE_DIR} && \
        cargo run --release --bin lanty-train-tokenizer -- data ${VOCAB_SIZE} 2>&1 | tail -5"
    echo "  Tokenizer trained!"
}

# Train model
echo "  Training model (${MODEL_SIZE}, ${EPOCHS} epochs, seq_len=${SEQ_LEN})..."
$SSH_CMD "source ~/.cargo/env && \
    export PATH=${CUDA_PATH}/bin:\$PATH && \
    export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH && \
    cd ${REMOTE_DIR} && \
    cargo run --release --bin lanty-train -- \
        --${MODEL_SIZE} \
        --epochs ${EPOCHS} \
        --seq-len ${SEQ_LEN} 2>&1" | while IFS= read -r line; do
    echo "  $line"
done

TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$((TRAIN_END - TRAIN_START))
echo ""
echo "  Training finished in $(date -u -d @${TRAIN_ELAPSED} +%H:%M:%S 2>/dev/null || echo "${TRAIN_ELAPSED}s")"

# --- Step 7: Download trained model ---
echo ""
echo "[7/7] Downloading trained model..."
mkdir -p "${PROJECT_ROOT}/models"
eval $RSYNC_CMD \
    "${LAMBDA_USER}@${INSTANCE_IP}:${REMOTE_DIR}/models/" \
    "${PROJECT_ROOT}/models/"
echo "  Models downloaded to ${PROJECT_ROOT}/models/"

# --- Summary ---
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
HOURS=$(echo "scale=2; ${TOTAL_ELAPSED} / 3600" | bc)
COST=$(echo "scale=2; ${HOURS} * ${HOURLY_RATE}" | bc)

echo ""
echo "============================================"
echo "  Training Complete!"
echo "============================================"
echo "  Total time:     $(date -u -d @${TOTAL_ELAPSED} +%H:%M:%S 2>/dev/null || echo "${TOTAL_ELAPSED}s")"
echo "  Training time:  $(date -u -d @${TRAIN_ELAPSED} +%H:%M:%S 2>/dev/null || echo "${TRAIN_ELAPSED}s")"
echo "  Est. cost:      \$${COST} (${HOURS} hrs @ \$${HOURLY_RATE}/hr)"
echo ""
echo "  Run 'lanty' to chat with your trained model!"
echo "  Remember to terminate your Lambda instance!"
echo "============================================"
