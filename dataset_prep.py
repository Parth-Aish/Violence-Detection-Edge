"""Extract fixed-length YOLOv8-Pose sequences for the violence LSTM."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Add root to sys.path to allow imports from backend.core
_root_dir = str(Path(__file__).resolve().parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from backend.core.transforms import raw_to_stgcn_tensor


LOGGER = logging.getLogger("dataset_prep")
VIDEO_SUFFIXES = {".avi", ".mp4", ".mkv", ".mov", ".webm"}
SEQUENCE_LENGTH = 30
FEATURES = 34
POSE_IMAGE_SIZE = 320
MODEL_LOCAL = threading.local()


def label_from_path(path: Path) -> Optional[int]:
    """Return 1 for violence and 0 for non-violence, based on class folders."""
    positive = {"violence", "violent", "fight", "fighting"}
    negative = {"nonviolence", "nonviolent", "nonviolentvideos", "normal", "safe"}
    labels = set()
    for part in path.parts[:-1]:
        normalized = re.sub(r"[^a-z]", "", part.lower())
        if normalized in positive:
            labels.add(1)
        elif normalized in negative:
            labels.add(0)
    if len(labels) > 1:
        raise ValueError(f"Conflicting class directories in path: {path}")
    return next(iter(labels), None)


def _pose_model(model_path: str) -> YOLO:
    model = getattr(MODEL_LOCAL, "pose_model", None)
    if model is None:
        model = YOLO(model_path)
        MODEL_LOCAL.pose_model = model
    return model


def _frame_indices(frame_count: int) -> np.ndarray:
    if frame_count < 1:
        raise ValueError("Video contains no readable frames.")
    return np.linspace(
        0, frame_count - 1, num=SEQUENCE_LENGTH, dtype=np.int64
    )


def _extract_pose_features(frame: np.ndarray, result) -> np.ndarray:
    height, width = frame.shape[:2]
    best_keypoints: Optional[np.ndarray] = None
    best_confidence = -1.0

    if result.keypoints is not None and result.boxes is not None:
        coordinates = result.keypoints.xy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for person_kps, confidence in zip(coordinates, confidences):
            if confidence > best_confidence:
                best_keypoints = person_kps
                best_confidence = float(confidence)

    if best_keypoints is None:
        return np.zeros(FEATURES, dtype=np.float32)

    normalized = best_keypoints.astype(np.float32, copy=True)
    normalized[:, 0] = np.clip(normalized[:, 0] / max(width, 1), 0.0, 1.0)
    normalized[:, 1] = np.clip(normalized[:, 1] / max(height, 1), 0.0, 1.0)
    return normalized.reshape(FEATURES)


def extract_video(path: Path, label: int, model_path: str) -> tuple[np.ndarray, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            frame_count = 0
            while capture.read()[0]:
                frame_count += 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        indices = _frame_indices(frame_count)
        selected_indices = {int(index) for index in indices}
        selected_frames = {}
        frame_index = 0
        last_index = int(indices[-1])
        while frame_index <= last_index:
            success, frame = capture.read()
            if not success or frame is None:
                break
            if frame_index in selected_indices:
                selected_frames[frame_index] = frame
            frame_index += 1
        if len(selected_frames) != len(selected_indices):
            raise ValueError(f"Could not decode all sampled frames from {path}")
        frames = [selected_frames[int(index)] for index in indices]

        results = _pose_model(model_path).predict(
            frames,
            verbose=False,
            conf=0.25,
            imgsz=POSE_IMAGE_SIZE,
            batch=SEQUENCE_LENGTH,
        )
        sequence = np.stack(
            [
                _extract_pose_features(frame, result)
                for frame, result in zip(frames, results)
            ]
        ).astype(np.float32, copy=False)
        if sequence.shape != (SEQUENCE_LENGTH, FEATURES):
            raise RuntimeError(
                f"Pose model returned {len(sequence)} frames for {path}; "
                f"expected {SEQUENCE_LENGTH}."
            )
        return sequence, label
    finally:
        capture.release()


def prepare_dataset(
    dataset_dir: Path,
    output_dir: Path,
    model_path: str,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    videos = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            label = label_from_path(path.relative_to(dataset_dir))
            if label is not None:
                videos.append((path, label))

    if not videos:
        raise FileNotFoundError(
            "No labeled video files found. Expected class folders such as "
            "Violence/ and NonViolence/."
        )

    extracted: dict[int, tuple[np.ndarray, int]] = {}
    failures: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pose") as pool:
        futures = {
            pool.submit(extract_video, path, label, model_path): (index, path)
            for index, (path, label) in enumerate(videos)
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            index, path = futures[future]
            try:
                sequence, label = future.result()
            except Exception as exc:
                failures.append((path, str(exc)))
                LOGGER.exception("Skipping failed video %s", path)
            else:
                extracted[index] = (sequence, label)
            if completed % 100 == 0 or completed == len(videos):
                LOGGER.info(
                    "Processed %d/%d videos; %d successful, %d failed",
                    completed,
                    len(videos),
                    len(extracted),
                    len(failures),
                )

    if not extracted:
        raise RuntimeError(f"Pose extraction failed for all {len(videos)} videos.")

    ordered = [extracted[index] for index in sorted(extracted)]
    features = np.stack([sequence for sequence, _ in ordered]).astype(
        np.float32, copy=False
    )
    targets = np.asarray([label for _, label in ordered], dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)

    stgcn_features = raw_to_stgcn_tensor(features)
    for filename, array in (
        ("X_train.npy", features),
        ("X_train_stgcn.npy", stgcn_features),
        ("y_train.npy", targets),
    ):
        temporary = output_dir / f".{filename}.tmp"
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary, output_dir / filename)

    LOGGER.info(
        "Saved %d sequences (%d violence, %d non-violence), skipped %d; X_raw=%s, X_stgcn=%s, y=%s",
        len(targets),
        int(np.sum(targets == 1)),
        int(np.sum(targets == 0)),
        len(failures),
        features.shape,
        stgcn_features.shape,
        targets.shape,
    )
    return features, targets


def convert_existing_to_stgcn(processed_dir: Path) -> Path:
    """Convert an existing X_train.npy into X_train_stgcn.npy."""
    raw_path = processed_dir / "X_train.npy"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing {raw_path}")
    raw_features = np.load(raw_path)
    LOGGER.info("Loading existing %s with shape %s", raw_path, raw_features.shape)
    stgcn_features = raw_to_stgcn_tensor(raw_features)
    stgcn_path = processed_dir / "X_train_stgcn.npy"
    np.save(stgcn_path, stgcn_features)
    LOGGER.info("Saved ST-GCN tensor to %s with shape %s", stgcn_path, stgcn_features.shape)
    return stgcn_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--convert-only", action="store_true", help="Convert existing X_train.npy to ST-GCN tensor")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.convert_only:
        convert_existing_to_stgcn(args.output_dir.resolve())
        return 0

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    try:
        prepare_dataset(
            args.dataset_dir.resolve(),
            args.output_dir.resolve(),
            args.model,
            args.workers,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
