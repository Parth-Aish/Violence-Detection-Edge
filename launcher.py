"""Interactive Unified Launcher for Edge Vision & AI Violence Detection Platform.

Provides a 1-click interactive terminal menu to run:
1. Web Dashboard & Batch REST API (Next.js + FastAPI)
2. Live Edge Webcam Inference (50+ FPS HUD)
3. Offline Altercation Video Test (V_1.mp4)
4. Automated Concat Benchmark Suite
5. Full Dataset Performance Evaluation
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_banner() -> None:
    print(r"""
 ==================================================================================
  * EDGE VISION -- SOTA SPATIAL-TEMPORAL VIOLENCE DETECTION PLATFORM
  ST-GCN * Relational Strike Gating * Scene Context * Occlusion Kalman Tracking
 ==================================================================================
""")


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    try:
        subprocess.run(cmd, cwd=cwd or Path.cwd(), check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as exc:
        print(f"\n[ERROR] Command failed: {exc}")


def option_cloud_dashboard() -> None:
    print("\n[LAUNCHER] Starting Cloud Web Stack (FastAPI Backend + Next.js UI)...")
    print("  • Backend:  http://127.0.0.1:8000")
    print("  • Frontend: http://localhost:3000\n")

    backend_cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
    frontend_cmd = ["npm", "run", "dev"]

    frontend_dir = Path("frontend").resolve()

    p_backend = subprocess.Popen(backend_cmd)
    time.sleep(1.5)
    p_frontend = subprocess.Popen(frontend_cmd, cwd=frontend_dir, shell=True)

    print("\n[ACTIVE] Services running! Press Ctrl+C in this terminal to terminate both.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Terminating servers...")
        p_backend.terminate()
        p_frontend.terminate()
        p_backend.wait()
        p_frontend.wait()
        print("[SHUTDOWN] Cleaned up successfully.")


def main() -> None:
    clear_screen()
    print_banner()

    options = [
        ("1", "Launch Web Dashboard & Cloud API (Next.js 16 + FastAPI)"),
        ("2", "Run Live Edge Surveillance (Local USB Webcam @ 50+ FPS)"),
        ("3", "Run Live Edge on Altercation Video Sample (data/raw/Violence/V_1.mp4)"),
        ("4", "Run Automated Benchmark Concat Test (Verification Suite)"),
        ("5", "Run Full Dataset Evaluation Suite (1,951 Videos Report)"),
        ("6", "Run Quick Balanced Benchmark (10-Clip Sanity Check)"),
        ("0", "Exit"),
    ]

    for key, label in options:
        print(f"  [{key}] {label}")
    print()

    choice = input("Select an option [0-6]: ").strip()

    if choice == "1":
        option_cloud_dashboard()
    elif choice == "2":
        print("\n[LAUNCHER] Starting live webcam edge surveillance...")
        run_command([sys.executable, "run_live.py", "--source", "0"])
    elif choice == "3":
        sample = Path("data/raw/Violence/V_1.mp4")
        if not sample.is_file():
            sample = Path("frontend/public/samples/violence_sample.mp4")
        print(f"\n[LAUNCHER] Starting edge playback on sample: {sample}...")
        run_command([sys.executable, "run_live.py", "--source", str(sample)])
    elif choice == "4":
        print("\n[LAUNCHER] Running automated concat benchmark test...")
        run_command([sys.executable, "backend/benchmark_tests.py"])
    elif choice == "5":
        print("\n[LAUNCHER] Running dataset-wide evaluation...")
        run_command([sys.executable, "backend/evaluate_all.py"])
    elif choice == "6":
        print("\n[LAUNCHER] Running quick 10-clip evaluation...")
        run_command([sys.executable, "backend/evaluate_all.py", "--limit", "5"])
    elif choice == "0":
        print("\nExiting. Have a great day!")
        sys.exit(0)
    else:
        print(f"\n[ERROR] Invalid option '{choice}'. Please select 0-6.")


if __name__ == "__main__":
    main()
