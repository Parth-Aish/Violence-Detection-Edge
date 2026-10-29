"""Process raw (2000, 30, 34) pose data into (2000, 30, 82) kinematic feature tensor."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from kinematics_extractor import extract_kinematic_batch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"


def main() -> int:
    input_path = DATA_DIR / "X_train.npy"
    output_path = DATA_DIR / "X_train_kinematic.npy"

    if not input_path.exists():
        raise FileNotFoundError(f"Source features not found at {input_path}")

    print(f"Loading raw poses from: {input_path}")
    raw_x = np.load(input_path, allow_pickle=False).astype(np.float32)
    print(f"Loaded raw X shape: {raw_x.shape}")

    t0 = time.perf_counter()
    print("Extracting 82-dimensional biomechanical features across all samples...")
    kinematic_x = extract_kinematic_batch(raw_x)
    elapsed = time.perf_counter() - t0

    print(f"Processed shape: {kinematic_x.shape} in {elapsed:.2f}s")
    assert kinematic_x.shape == (2000, 30, 82), f"Unexpected shape {kinematic_x.shape}"
    assert np.isfinite(kinematic_x).all(), "Extracted features contain NaN or Inf values!"

    np.save(output_path, kinematic_x)
    print(f"Saved {output_path.stat().st_size / (1024 * 1024):.2f} MB to: {output_path}")
    print("Biomechanical feature preparation complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
