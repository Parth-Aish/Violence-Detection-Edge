"""Automated Benchmark Suite & Concat Test for Edge Video Kinematics.

Proves system accuracy and real-time inference throughput (> 30 FPS)
on concatenated non-violence and violence footage.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure backend directory is in sys.path
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from core.inference import PoseSTGCNDetector, PoseLSTMDetector


def locate_sample(name: str, fallback_candidates: list[Path]) -> Path:
    """Find a video sample file or raise FileNotFoundError."""
    for candidate in fallback_candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not locate required sample video '{name}'")


def build_concatenated_video(
    nv_video_path: Path,
    v_video_path: Path,
    output_path: Path,
    target_fps: float = 30.0,
) -> tuple[int, int]:
    """Concatenate NV and V clips into a single video file."""
    nv_cap = cv2.VideoCapture(str(nv_video_path))
    if not nv_cap.isOpened():
        raise RuntimeError(f"Failed to open non-violence clip: {nv_video_path}")
    nv_frames: list[np.ndarray] = []
    while True:
        ret, frame = nv_cap.read()
        if not ret or frame is None:
            break
        nv_frames.append(frame)
    nv_cap.release()

    v_cap = cv2.VideoCapture(str(v_video_path))
    if not v_cap.isOpened():
        raise RuntimeError(f"Failed to open violence clip: {v_video_path}")
    v_frames: list[np.ndarray] = []
    height, width = nv_frames[0].shape[:2]
    while True:
        ret, frame = v_cap.read()
        if not ret or frame is None:
            break
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        v_frames.append(frame)
    v_cap.release()

    total_nv = len(nv_frames)
    total_v = len(v_frames)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))
    for frame in nv_frames + v_frames:
        writer.write(frame)
    writer.release()

    return total_nv, total_v


def run_concat_benchmark() -> None:
    print("=" * 72)
    print("  EDGE VIDEO KINEMATICS - AUTOMATED BENCHMARK & CONCAT TEST")
    print("=" * 72)

    project_root = Path(__file__).resolve().parent.parent
    nv_path = locate_sample(
        "NV_1.mp4",
        [
            project_root / "data" / "raw" / "NonViolence" / "NV_1.mp4",
            project_root / "frontend" / "public" / "samples" / "safe_sample.mp4",
            project_root / "data" / "raw" / "NV_1.mp4",
        ],
    )
    v_path = locate_sample(
        "V_1.mp4",
        [
            project_root / "data" / "raw" / "Violence" / "V_1.mp4",
            project_root / "frontend" / "public" / "samples" / "violence_sample.mp4",
            project_root / "data" / "raw" / "V_1.mp4",
        ],
    )

    print(f"Loaded Non-Violence Sample: {nv_path}")
    print(f"Loaded Violence Sample:     {v_path}")

    temp_concat_video = Path(tempfile.gettempdir()) / "concat_benchmark_test.mp4"
    print("\nBuilding concatenated validation video...")
    total_nv, total_v = build_concatenated_video(nv_path, v_path, temp_concat_video)
    print(
        f"Concatenated Video Created: {temp_concat_video} "
        f"({total_nv} NV frames + {total_v} V frames = {total_nv + total_v} total frames)"
    )

    # Initialize detector unlocked with optimal threads for local hardware
    num_threads = min(4, os.cpu_count() or 4)
    print(f"\nInitializing PoseSTGCNDetector (threads={num_threads}, pose_stride=2)...")
    detector = PoseSTGCNDetector(intra_op_threads=num_threads)
    detector.pose_stride = 2

    cap = cv2.VideoCapture(str(temp_concat_video))
    if not cap.isOpened():
        raise RuntimeError("Failed to open concatenated video for evaluation.")

    fps = 30.0
    dt = 1.0 / fps
    detector.reset()

    predictions: list[bool] = []
    probabilities: list[float] = []
    latencies_ms: list[float] = []

    print("Running sequential frame-by-frame evaluation...")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            # Time strictly inference & kinematic extraction (excluding video decode)
            t_start = time.perf_counter()
            pred = detector.process_frame(frame, dt=dt)
            inference_dur_ms = (time.perf_counter() - t_start) * 1000.0

            latencies_ms.append(inference_dur_ms)
            predictions.append(pred.is_fighting)
            probabilities.append(pred.probability)
            frame_idx += 1
    finally:
        cap.release()
        temp_concat_video.unlink(missing_ok=True)

    # Segmentation Analysis
    nv_preds = predictions[:total_nv]
    v_preds = predictions[total_nv:]

    nv_false_alarms = sum(nv_preds)
    v_detections = sum(v_preds)

    tp = v_detections
    fp = nv_false_alarms
    fn = total_v - v_detections
    tn = total_nv - nv_false_alarms

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_inference_ms = float(np.mean(latencies_ms))
    p95_inference_ms = float(np.percentile(latencies_ms, 95))
    throughput_fps = 1000.0 / avg_inference_ms if avg_inference_ms > 0 else 0.0

    print("\n" + "-" * 72)
    print("  VALIDATION BENCHMARK RESULTS")
    print("-" * 72)
    print(f"Non-Violence Segment: {total_nv} frames")
    print(f"  False Alarms:       {nv_false_alarms} / {total_nv} ({nv_false_alarms/total_nv*100:.1f}%)")
    print(f"Violence Segment:     {total_v} frames")
    print(f"  Valid Detections:   {v_detections} / {total_v} ({v_detections/total_v*100:.1f}%)")
    print(f"Classification Metrics:")
    print(f"  Precision:          {precision:.4f} ({precision*100:.1f}%)")
    print(f"  Recall:             {recall:.4f} ({recall*100:.1f}%)")
    print(f"  F1 Score:           {f1:.4f}")
    print(f"Inference Latency (excluding video decode):")
    print(f"  Mean Latency:       {avg_inference_ms:.2f} ms per frame")
    print(f"  95th Percentile:    {p95_inference_ms:.2f} ms")
    print(f"  Throughput:         {throughput_fps:.1f} FPS (Target > 30.0 FPS)")
    print("-" * 72)

    # ── Assertion 1: Segment flip check & F1 validity ───────────────────────
    assert nv_false_alarms == 0, (
        f"FAILED Assertion 1: Expected 0 false alarms on non-violent segment, "
        f"but got {nv_false_alarms} positive triggers."
    )
    assert v_detections > 0, (
        "FAILED Assertion 1: Expected system to flip to True during violent segment, "
        "but 0 violence triggers were recorded."
    )
    print("[PASS] Assertion 1 PASSED: Model output is False during NV segment and flips to True during V segment.")

    # -- Assertion 2: Real-time throughput assertion (> 30 FPS) -------------
    if avg_inference_ms >= 33.0:
        raise AssertionError(
            f"FAILED Assertion 2: Target throughput failed ({avg_inference_ms:.2f} ms >= 33.0 ms, "
            f"yielding {throughput_fps:.1f} FPS < 30 FPS).\n"
            "Optimization instructions:\n"
            "  1. Ensure pose_stride >= 2 in PoseLSTMDetector.\n"
            "  2. Vectorize NMS / bounding box conversions in detect_entities.\n"
            "  3. Check ORT intra_op_threads setting (optimal is 4 on desktop CPUs).\n"
            "  4. Verify ONNX graph optimization is enabled (GraphOptimizationLevel.ORT_ENABLE_ALL)."
        )

    print(f"[PASS] Assertion 2 PASSED: Inference throughput is {throughput_fps:.1f} FPS (> 30 FPS capability guaranteed).")
    print("\nALL BENCHMARKS PASSED SUCCESSFULLY.\n")


if __name__ == "__main__":
    run_concat_benchmark()
