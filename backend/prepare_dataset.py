"""Deterministic Dataset Preprocessing & Feature Extraction for ST-GCN.

Extracts COCO-17 keypoint sequences, structures (N, 2, 30, 17) ST-GCN feature tensors,
and explicitly generates video_ids.json to map every window sample to its source video.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.core.transforms import raw_to_stgcn_tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_dataset")


def extract_poses_from_video(
    video_path: Path,
    pose_session,
    pose_input_name: str,
    pose_output_name: str,
    target_frames: int = 30,
) -> Optional[np.ndarray]:
    """Sample target_frames evenly from video and extract primary person (30, 17, 2)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        frames_list = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames_list.append(f)
        cap.release()
        total_frames = len(frames_list)
        if total_frames == 0:
            return None
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        sampled_frames = [frames_list[i] for i in indices]
    else:
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        sampled_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                sampled_frames.append(frame)
            elif sampled_frames:
                sampled_frames.append(sampled_frames[-1].copy())
            else:
                sampled_frames.append(np.zeros((320, 320, 3), dtype=np.uint8))
        cap.release()

    if len(sampled_frames) < target_frames:
        return None

    keypoints_seq = np.zeros((target_frames, 17, 2), dtype=np.float32)
    for t_idx, frame in enumerate(sampled_frames):
        h, w = frame.shape[:2]
        blob = cv2.resize(frame, (320, 320))
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        outputs = pose_session.run([pose_output_name], {pose_input_name: blob})[0]
        dets = outputs[0]  # shape (56, 2100) or similar

        if dets.shape[0] >= 56:
            boxes = dets[:4, :]
            scores = dets[4, :]
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > 0.2:
                kps_raw = dets[5:, best_idx]
                for k in range(17):
                    kx = float(kps_raw[k * 3]) / 320.0
                    ky = float(kps_raw[k * 3 + 1]) / 320.0
                    keypoints_seq[t_idx, k] = [kx, ky]
            elif t_idx > 0:
                keypoints_seq[t_idx] = keypoints_seq[t_idx - 1].copy()

    return keypoints_seq


def prepare_dataset(
    raw_dir: Path,
    output_dir: Path,
    pose_model_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Process raw video directory or existing features into ST-GCN dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stgcn_out = output_dir / "X_train_stgcn.npy"
    y_out = output_dir / "y_train.npy"
    vids_out = output_dir / "video_ids.json"

    # Fast path: If raw feature array (X_train.npy) already exists, transform deterministically
    raw_x_path = output_dir / "X_train.npy"
    raw_y_path = output_dir / "y_train.npy"
    if raw_x_path.is_file() and raw_y_path.is_file():
        logger.info("Found existing raw keypoints at %s. Transforming to ST-GCN tensors...", raw_x_path)
        raw_x = np.load(raw_x_path)
        y = np.load(raw_y_path)

        stgcn_x = raw_to_stgcn_tensor(raw_x)

        # Generate deterministic source video mappings
        nv_count = int(np.sum(y == 0))
        v_count = int(np.sum(y == 1))
        video_ids = [f"NV_{i+1}.mp4" for i in range(nv_count)] + [f"V_{i+1}.mp4" for i in range(v_count)]

        np.save(stgcn_out, stgcn_x)
        np.save(y_out, y)
        with open(vids_out, "w", encoding="utf-8") as f:
            json.dump(video_ids, f, indent=2)

        # Also mirror to backend/data/ for standalone inference/training runs
        backend_data_dir = _project_root / "backend" / "data"
        if backend_data_dir.is_dir():
            with open(backend_data_dir / "video_ids.json", "w", encoding="utf-8") as f:
                json.dump(video_ids, f, indent=2)

        logger.info(
            "ST-GCN Dataset prepared successfully: X=%s, y=%s, video_ids=%d saved to %s",
            stgcn_x.shape,
            y.shape,
            len(video_ids),
            output_dir,
        )
        return stgcn_x, y, video_ids

    # Video extraction path
    import onnxruntime as ort

    if not pose_model_path:
        pose_model_path = _project_root / "frontend" / "public" / "models" / "yolov8n-pose.onnx"
    if not pose_model_path.is_file():
        raise FileNotFoundError(f"YOLOv8-Pose model not found at: {pose_model_path}")

    sess = ort.InferenceSession(str(pose_model_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    nv_dir = raw_dir / "NonViolence"
    v_dir = raw_dir / "Violence"

    nv_files = sorted([f for f in nv_dir.glob("*.mp4")]) if nv_dir.is_dir() else []
    v_files = sorted([f for f in v_dir.glob("*.mp4")]) if v_dir.is_dir() else []

    if not nv_files and not v_files:
        raise FileNotFoundError(f"No video files found in {raw_dir} (expected NonViolence and Violence directories).")

    samples_x = []
    samples_y = []
    video_ids = []

    logger.info("Extracting keypoints from %d NonViolence videos...", len(nv_files))
    for f in nv_files:
        seq = extract_poses_from_video(f, sess, inp_name, out_name)
        if seq is not None:
            samples_x.append(seq)
            samples_y.append(0)
            video_ids.append(f.name)

    logger.info("Extracting keypoints from %d Violence videos...", len(v_files))
    for f in v_files:
        seq = extract_poses_from_video(f, sess, inp_name, out_name)
        if seq is not None:
            samples_x.append(seq)
            samples_y.append(1)
            video_ids.append(f.name)

    raw_arr = np.array(samples_x, dtype=np.float32)
    y_arr = np.array(samples_y, dtype=np.float32)
    stgcn_x = raw_to_stgcn_tensor(raw_arr)

    np.save(stgcn_out, stgcn_x)
    np.save(y_out, y_arr)
    with open(vids_out, "w", encoding="utf-8") as f:
        json.dump(video_ids, f, indent=2)

    logger.info(
        "Extracted and saved: %d samples to %s (X_train_stgcn.npy, y_train.npy, video_ids.json)",
        len(video_ids),
        output_dir,
    )
    return stgcn_x, y_arr, video_ids


def main():
    parser = argparse.ArgumentParser(description="Deterministic ST-GCN Dataset Preprocessing")
    parser.add_argument("--raw-dir", type=Path, default=_project_root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=_project_root / "data" / "processed")
    parser.add_argument("--pose-model", type=Path, default=None)
    args = parser.parse_args()

    prepare_dataset(args.raw_dir, args.output_dir, args.pose_model)


if __name__ == "__main__":
    main()

