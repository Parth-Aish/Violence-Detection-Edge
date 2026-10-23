"""YOLOv8 pose extraction followed by a 30-frame ONNX LSTM classifier."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("inference")

SEQUENCE_LENGTH = 30
KEYPOINT_COUNT = 17
FEATURE_COUNT = KEYPOINT_COUNT * 2
VIOLENCE_THRESHOLD = 0.85
POSE_IMAGE_SIZE = 320


class Severity(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class ViolencePrediction:
    is_fighting: bool
    probability: float
    severity: Severity
    confidence: float
    bbox: Optional[tuple[float, float, float, float]]
    timestamp: float
    inference_ms: float
    fps: float


class PoseLSTMDetector:
    """Extract a pose each frame and classify its rolling coordinate sequence."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        *,
        threshold: float = VIOLENCE_THRESHOLD,
        intra_op_threads: Optional[int] = None,
    ) -> None:
        self.model_path = Path(
            model_path or os.environ.get(
                "ONNX_MODEL_PATH", Path(__file__).with_name("violence_lstm.onnx")
            )
        )
        self.pose_model_path = pose_model_path or os.environ.get(
            "POSE_MODEL_PATH", "yolov8n-pose.pt"
        )
        self.threshold = threshold
        if not 0.0 < threshold < 1.0:
            raise ValueError("Violence threshold must be between zero and one.")
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"ONNX violence model not found at {self.model_path}. "
                "Run train_model.py after preparing the Kaggle dataset."
            )

        try:
            import onnxruntime as ort
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime and Ultralytics are required. Install requirements.txt."
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads or int(
            os.environ.get("ONNX_INTRA_OP_THREADS", "1")
        )
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one ONNX model input and one output.")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        self.pose_model = YOLO(self.pose_model_path)
        self._poses: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        self._last_frame_time: Optional[float] = None
        self._fps = 0.0
        self._lock = threading.Lock()
        logger.info(
            "Loaded pose/LSTM inference (pose=%s, onnx=%s, threshold=%.2f)",
            self.pose_model_path,
            self.model_path,
            self.threshold,
        )

    def _extract_largest_person(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, Optional[tuple[float, float, float, float]]]:
        height, width = frame.shape[:2]
        results = self.pose_model(
            frame, verbose=False, conf=0.25, imgsz=POSE_IMAGE_SIZE
        )
        best_score = -1.0
        best_pose: Optional[np.ndarray] = None
        best_bbox: Optional[tuple[float, float, float, float]] = None
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            coordinates = result.keypoints.xy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            for index, confidence in enumerate(confidences):
                if confidence > best_score:
                    best_score = float(confidence)
                    best_pose = coordinates[index]
                    best_bbox = tuple(float(value) for value in boxes[index])

        if best_pose is None:
            return np.zeros(FEATURE_COUNT, dtype=np.float32), 0.0, None

        normalized = best_pose.astype(np.float32, copy=True)
        normalized[:, 0] = np.clip(normalized[:, 0] / max(width, 1), 0.0, 1.0)
        normalized[:, 1] = np.clip(normalized[:, 1] / max(height, 1), 0.0, 1.0)
        return normalized.reshape(FEATURE_COUNT), max(best_score, 0.0), best_bbox

    def process_frame(self, frame: np.ndarray) -> ViolencePrediction:
        """Return a timestamped score and decision for one BGR frame."""
        started = time.perf_counter()
        now = time.time()
        pose, confidence, bbox = self._extract_largest_person(frame)
        with self._lock:
            self._poses.append(pose)
            if self._last_frame_time is not None:
                elapsed = now - self._last_frame_time
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    self._fps = instant_fps if self._fps == 0 else (
                        0.8 * self._fps + 0.2 * instant_fps
                    )
            self._last_frame_time = now

            probability = 0.0
            if len(self._poses) == SEQUENCE_LENGTH:
                sequence = np.stack(self._poses).astype(np.float32, copy=False)
                output = self.session.run(
                    [self.output_name],
                    {self.input_name: sequence[np.newaxis, :, :]},
                )[0]
                probability = float(np.asarray(output).reshape(-1)[0])
                if not np.isfinite(probability):
                    raise RuntimeError("ONNX model returned a non-finite probability.")
                probability = float(np.clip(probability, 0.0, 1.0))

        is_fighting = probability > self.threshold
        if not is_fighting:
            severity = Severity.NONE
        elif probability >= 0.98:
            severity = Severity.CRITICAL
        elif probability >= 0.94:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ViolencePrediction(
            is_fighting=is_fighting,
            probability=probability,
            severity=severity,
            confidence=confidence,
            bbox=bbox,
            timestamp=now,
            inference_ms=elapsed_ms,
            fps=self._fps,
        )

    def reset(self) -> None:
        """Clear temporal features when the input source changes."""
        with self._lock:
            self._poses.clear()
            self._last_frame_time = None
            self._fps = 0.0


# Retain the old class import name for downstream integrations.
TemporalVelocityClassifier = PoseLSTMDetector
