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

# 1. Check Python installation
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "[ERROR] Python is not found on PATH. Please install Python 3.10+."
    exit 1
fi

PYTHON_BIN="python3"
if ! command -v python3 &>/dev/null; then
    PYTHON_BIN="python"
fi

# 2. Check and install huggingface_hub if missing
if ! $PYTHON_BIN -c "import huggingface_hub" &>/dev/null; then
    echo "[INFO] Installing huggingface_hub..."
    $PYTHON_BIN -m pip install --quiet huggingface_hub
fi

# 3. Verify git status and stage all required files
echo "[GIT] Ensuring repository is clean and committed..."
if [ ! -d ".git" ]; then
    git init
fi

git add .
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    git commit -m "deploy: update production Dockerfile and requirements for Hugging Face Spaces" || true
fi

# 4. Invoke automated deployment engine
echo "[DEPLOY] Initializing Space and uploading repository..."
if [ -n "$HF_TOKEN" ]; then
    $PYTHON_BIN deploy.py --space-name "$SPACE_NAME" --token "$HF_TOKEN"
else
    $PYTHON_BIN deploy.py --space-name "$SPACE_NAME"
fi
