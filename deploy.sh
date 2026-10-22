#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# deploy.sh — Hugging Face Spaces Deployment Script
# ════════════════════════════════════════════════════════════════════
# Usage:
#   ./deploy.sh [SPACE_NAME] [HF_TOKEN]
#
# Examples:
#   ./deploy.sh
#   ./deploy.sh violence-detection
#   ./deploy.sh my-org/violence-detection hf_xxxxxxxxxxxx
# ════════════════════════════════════════════════════════════════════

set -e

SPACE_NAME="${1:-violence-detection}"
HF_TOKEN="${2:-$HF_TOKEN}"

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

# 4. Invoke automated deployment engine. It reads HF_TOKEN from the environment.
echo "[DEPLOY] Initializing Space and uploading repository..."
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
fi

"$PYTHON_BIN" deploy.py --space-name "$SPACE_NAME"
