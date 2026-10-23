"""
deploy.py — Hugging Face Spaces Automated Deployment Engine
============================================================

Utilizes huggingface_hub to:
1. Authenticate using the active token or environment variable.
2. Automatically create a Docker Space on Hugging Face (port 7860).
3. Push all repository assets (code, Dockerfile, static frontend build) directly to the Space.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

try:
    from huggingface_hub import HfApi, whoami
except ImportError:
    print("[ERROR] huggingface_hub is not installed.")
    print("Run: pip install huggingface_hub")
    sys.exit(1)


def deploy(space_name: str = "violence-detection", token: str = None, private: bool = False):
    print("=" * 65)
    print("  Hugging Face Spaces Automated Deployment")
    print("=" * 65)

    # 1. Resolve authentication token
    resolved_token = token or os.getenv("HF_TOKEN")
    api = HfApi(token=resolved_token)

    try:
        user_info = api.whoami(token=resolved_token)
        username = user_info.get("name")
        print(f"[AUTH] Authenticated as Hugging Face user: @{username}")
    except Exception as e:
        print("\n[AUTH ERROR] Not logged in to Hugging Face or invalid token.")
        print("Please authenticate using either:")
        print("  1) Run: huggingface-cli login")
        print("  2) Or set: export HF_TOKEN=\"hf_your_token_here\" (Linux/Mac)")
        print("  3) Or set: $env:HF_TOKEN=\"hf_your_token_here\" (PowerShell)")
        print(f"\nDetails: {e}")
        sys.exit(1)

    # 2. Determine full repository ID (username/space_name)
    if "/" in space_name:
        repo_id = space_name
    else:
        repo_id = f"{username}/{space_name}"

    print(f"[SPACE] Target Space: https://huggingface.co/spaces/{repo_id}")
    print("[SPACE] SDK: Docker (port 7860)")

    # 3. Create or verify the Space on Hugging Face
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=private,
            exist_ok=True,
            token=resolved_token,
        )
        print(f"[SPACE] Initialized Space repository: {repo_id}")
    except Exception as e:
        print(f"[SPACE ERROR] Failed to create or access Space: {e}")
        sys.exit(1)

    # 4. Upload repository files
    root_dir = Path(__file__).parent.resolve()
    print(f"\n[UPLOAD] Uploading project directory ({root_dir}) to Hugging Face...")

    ignore_patterns = [
        "node_modules/*",
        "frontend/node_modules/*",
        ".next/*",
        "frontend/.next/*",
        ".git/*",
        ".venv/*",
        "venv/*",
        "__pycache__/*",
        "backend/__pycache__/*",
        "*.pyc",
        "*.db",
        "*.db-wal",
        "*.db-shm",
        "*.jsonl",
        "data/raw/**",
        "data/processed/**",
        "**/data/raw/**",
        "**/data/processed/**",
        "*.npy",
        "*.pt",
        "kaggle.json",
        "**/kaggle.json",
        ".env",
        ".env.*",
        "**/.env",
        "**/.env.*",
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",
        "**/secrets/**",
    ]

    try:
        commit_info = api.upload_folder(
            folder_path=str(root_dir),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            commit_message="feat: automated production deployment to Hugging Face Spaces",
            token=resolved_token,
        )
        print("\n" + "=" * 65)
        print("  DEPLOYMENT SUCCESSFUL!")
        print("=" * 65)
        print(f"Space URL:    https://huggingface.co/spaces/{repo_id}")
        print(f"Commit OID:   {commit_info}")
        print("Your Docker container is now building and will launch on port 7860.")
        print("=" * 65)
    except Exception as e:
        print(f"\n[UPLOAD ERROR] upload_folder failed: {e}")
        print("Attempting Git remote push fallback...")

        # Git fallback
        remote_url = f"https://huggingface.co/spaces/{repo_id}"
        try:
            remotes = subprocess.run(
                ["git", "remote"], check=True, capture_output=True, text=True
            ).stdout.splitlines()
            if "space" in remotes:
                subprocess.run(
                    ["git", "remote", "set-url", "space", remote_url], check=True
                )
            else:
                subprocess.run(["git", "remote", "add", "space", remote_url], check=True)
            subprocess.run(["git", "push", "space", "HEAD:main"], check=True)
            print(f"[GIT PUSH] Pushed successfully to {remote_url}")
        except Exception as git_err:
            print(f"[GIT ERROR] Git fallback failed: {git_err}")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Violence Detection System to Hugging Face Spaces")
    parser.add_argument("--space-name", default="violence-detection", help="Name of the Space (default: violence-detection)")
    parser.add_argument("--token", default=None, help="Hugging Face User Access Token (optional if logged in)")
    parser.add_argument("--private", action="store_true", help="Create space as private")
    args = parser.parse_args()

    deploy(space_name=args.space_name, token=args.token, private=args.private)
