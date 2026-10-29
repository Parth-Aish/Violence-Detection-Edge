"""Convenience launcher for local real-time edge violence detection."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script_path = Path(__file__).resolve().parent / "backend" / "run_live.py"
    cmd = [sys.executable, str(script_path)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
