#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# deploy.sh — Hugging Face Spaces Deployment Script
# ════════════════════════════════════════════════════════════════════
# Usage:
#   HF_TOKEN=hf_... ./deploy.sh [USERNAME/SPACE_NAME]
#
# Examples:
#   HF_TOKEN=hf_... ./deploy.sh
#   HF_TOKEN=hf_... ./deploy.sh my-org/violence-detection
# ════════════════════════════════════════════════════════════════════

set -e

SPACE_NAME="${1:-violence-detection}"

echo "================================================================="
echo "  Deploying Violence Detection System to Hugging Face Spaces"
echo "================================================================="

# 1. Resolve Python binary (support Windows Git Bash/WSL and native Linux/macOS)
PYTHON_BIN=""
for cmd in python.exe python3 python; do
    if command -v "$cmd" &>/dev/null; then
        if "$cmd" -c "import sys" &>/dev/null; then
            PYTHON_BIN="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python 3.10+ is required but not found on PATH."
    exit 1
fi

# 2. Check and install huggingface_hub if missing
if ! $PYTHON_BIN -c "import huggingface_hub" &>/dev/null; then
    echo "[INFO] Installing huggingface_hub..."
    $PYTHON_BIN -m pip install --quiet huggingface_hub
fi

# 3. Verify the deployment engine and Docker configuration are present
if [ ! -f "deploy.py" ] || [ ! -f "Dockerfile" ]; then
    echo "[ERROR] Run this script from the project root; deploy.py and Dockerfile are required."
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[ERROR] Set HF_TOKEN in the environment before deployment."
    exit 1
fi

if [ ! -s "backend/violence_lstm.onnx" ]; then
    echo "[ERROR] backend/violence_lstm.onnx is missing. Train and export the model first."
    exit 1
fi

# 4. Create/update the Docker Space and upload the filtered project directory.
echo "[DEPLOY] Initializing Space and uploading repository..."
export HF_TOKEN
"$PYTHON_BIN" deploy.py --space-name "$SPACE_NAME"
