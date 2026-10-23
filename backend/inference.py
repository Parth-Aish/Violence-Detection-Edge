"""CPU-only ONNX pose extraction followed by a rolling ONNX LSTM classifier."""

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

import cv2
import numpy as np

cv2.setNumThreads(1)

logger = logging.getLogger("inference")

SEQUENCE_LENGTH = 30
KEYPOINT_COUNT = 17
KEYPOINT_VALUES = 3
FEATURE_COUNT = KEYPOINT_COUNT * 2
POSE_IMAGE_SIZE = 320
POSE_CONFIDENCE = 0.25
POSE_NMS_THRESHOLD = 0.45
VIOLENCE_THRESHOLD = 0.75
CONSECUTIVE_TRIGGERS_REQUIRED = 3
POSE_STRIDE = max(1, int(os.environ.get("POSE_STRIDE", "2")))


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
    """Run exported YOLOv8-pose and LSTM models without PyTorch or Ultralytics."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        *,
        threshold: Optional[float] = None,
        intra_op_threads: Optional[int] = None,
    ) -> None:
        import onnxruntime as ort

        if threshold is None:
            threshold = float(
                os.environ.get("VIOLENCE_THRESHOLD", str(VIOLENCE_THRESHOLD))
            )
        self.model_path = Path(
            model_path
            or os.environ.get(
                "ONNX_MODEL_PATH", Path(__file__).with_name("violence_lstm.onnx")
            )
        )
        self.pose_model_path = Path(
            pose_model_path
            or os.environ.get(
                "POSE_MODEL_PATH", Path(__file__).with_name("yolov8n-pose.onnx")
            )
        )
        self.threshold = threshold
        if not 0.0 < threshold < 1.0:
            raise ValueError("Violence threshold must be between zero and one.")
        cv2.setNumThreads(1)
        for model_path in (self.pose_model_path, self.model_path):
            if not model_path.is_file():
                raise FileNotFoundError(f"Required ONNX model not found: {model_path}")

        threads = (
            intra_op_threads
            if intra_op_threads is not None
            else int(
                os.environ.get(
                    "ORT_THREADS", os.environ.get("ONNX_INTRA_OP_THREADS", "1")
                )
            )
        )
        if threads < 1:
            raise ValueError("ONNX intra-op thread count must be at least one.")
        self.intra_op_threads = threads
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = True

        self.pose_session = ort.InferenceSession(
            str(self.pose_model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        pose_inputs = self.pose_session.get_inputs()
        pose_outputs = self.pose_session.get_outputs()
        classifier_inputs = self.session.get_inputs()
        classifier_outputs = self.session.get_outputs()
        if len(pose_inputs) != 1 or len(pose_outputs) != 1:
            raise ValueError("Pose ONNX model must have one input and one output.")
        if len(classifier_inputs) != 1 or len(classifier_outputs) != 1:
            raise ValueError("LSTM ONNX model must have one input and one output.")
        self.pose_input_name = pose_inputs[0].name
        self.pose_output_name = pose_outputs[0].name
        self.input_name = classifier_inputs[0].name
        self.output_name = classifier_outputs[0].name
        self._poses: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        self.consecutive_triggers = 0
        self.pose_stride = POSE_STRIDE
        self._frame_index = 0
        self._frames_since_detection = 0
        self._cached_pose = np.zeros(FEATURE_COUNT, dtype=np.float32)
        self._last_detected_pose: Optional[np.ndarray] = None
        self._pose_velocity = np.zeros(FEATURE_COUNT, dtype=np.float32)
        self._cached_confidence = 0.0
        self._cached_bbox: Optional[tuple[float, float, float, float]] = None
        self._last_frame_time: Optional[float] = None
        self._fps = 0.0
        self._lock = threading.Lock()

        logger.info(
            "Loaded ONNX pose/LSTM models (pose=%s, classifier=%s, threads=%d)",
            self.pose_model_path,
            self.model_path,
            threads,
        )

    @staticmethod
    def _letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        scale = min(POSE_IMAGE_SIZE / width, POSE_IMAGE_SIZE / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        resized = cv2.resize(
            frame, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
        )
        pad_x = (POSE_IMAGE_SIZE - resized_width) // 2
        pad_y = (POSE_IMAGE_SIZE - resized_height) // 2
        boxed = cv2.copyMakeBorder(
            resized,
            pad_y,
            POSE_IMAGE_SIZE - resized_height - pad_y,
            pad_x,
            POSE_IMAGE_SIZE - resized_width - pad_x,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return boxed, scale, pad_x, pad_y

    def _extract_largest_person(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, Optional[tuple[float, float, float, float]]]:
        height, width = frame.shape[:2]
        boxed, scale, pad_x, pad_y = self._letterbox(frame)
        image = boxed[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(image[np.newaxis], dtype=np.float32) / 255.0
        output = self.pose_session.run(
            [self.pose_output_name], {self.pose_input_name: tensor}
        )[0]
        predictions = np.asarray(output)
        if predictions.ndim != 3 or predictions.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected pose ONNX output shape: {predictions.shape}"
            )
        rows = predictions[0].T
        expected_columns = 5 + KEYPOINT_COUNT * KEYPOINT_VALUES
        if rows.ndim != 2 or rows.shape[1] != expected_columns:
            raise RuntimeError(
                f"Expected {expected_columns} pose outputs per detection, "
                f"received shape {rows.shape}."
            )

        scores = rows[:, 4]
        candidates = np.flatnonzero(scores >= POSE_CONFIDENCE)
        if candidates.size == 0:
            return np.zeros(FEATURE_COUNT, dtype=np.float32), 0.0, None

        boxes: list[list[float]] = []
        candidate_scores: list[float] = []
        for index in candidates:
            center_x, center_y, box_width, box_height = rows[index, :4]
            boxes.append(
                [
                    float(center_x - box_width / 2),
                    float(center_y - box_height / 2),
                    float(box_width),
                    float(box_height),
                ]
            )
            candidate_scores.append(float(scores[index]))

        kept = cv2.dnn.NMSBoxes(
            boxes, candidate_scores, POSE_CONFIDENCE, POSE_NMS_THRESHOLD
        )
        kept_indices = np.asarray(kept).reshape(-1)
        if kept_indices.size == 0:
            return np.zeros(FEATURE_COUNT, dtype=np.float32), 0.0, None
        selected = int(kept_indices[np.argmax(scores[candidates[kept_indices]])])
        row = rows[candidates[selected]]

        keypoints = row[5:].reshape(KEYPOINT_COUNT, KEYPOINT_VALUES)
        points = keypoints[:, :2].astype(np.float32, copy=True)
        points[:, 0] = (points[:, 0] - pad_x) / scale
        points[:, 1] = (points[:, 1] - pad_y) / scale
        points[:, 0] = np.clip(points[:, 0] / max(width, 1), 0.0, 1.0)
        points[:, 1] = np.clip(points[:, 1] / max(height, 1), 0.0, 1.0)

        center_x, center_y, box_width, box_height = row[:4]
        x1 = np.clip((center_x - box_width / 2 - pad_x) / scale, 0, width)
        y1 = np.clip((center_y - box_height / 2 - pad_y) / scale, 0, height)
        x2 = np.clip((center_x + box_width / 2 - pad_x) / scale, 0, width)
        y2 = np.clip((center_y + box_height / 2 - pad_y) / scale, 0, height)
        bbox = (float(x1), float(y1), float(x2), float(y2))
        return (
            points.reshape(FEATURE_COUNT),
            float(scores[candidates[selected]]),
            bbox,
        )

    def process_frame(self, frame: np.ndarray) -> ViolencePrediction:
        """Return a timestamped violence score for one BGR frame."""
        with self._lock:
            return self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> ViolencePrediction:
        started = time.perf_counter()
        now = time.time()
        self._frames_since_detection += 1
        if self._frame_index % self.pose_stride == 0:
            pose, confidence, bbox = self._extract_largest_person(frame)
            if self._last_detected_pose is not None and self._frames_since_detection:
                self._pose_velocity = (
                    pose - self._last_detected_pose
                ) / self._frames_since_detection
            self._last_detected_pose = pose.copy()
            self._cached_pose = pose.copy()
            self._cached_confidence = confidence
            self._cached_bbox = bbox
            self._frames_since_detection = 0
        else:
            self._cached_pose = self._cached_pose + 0.5 * self._pose_velocity
            self._cached_pose = np.clip(self._cached_pose, 0.0, 1.0)
            pose = self._cached_pose.copy()
            confidence = self._cached_confidence
            bbox = self._cached_bbox
        self._frame_index += 1
        self._poses.append(pose)
        if self._last_frame_time is not None:
            elapsed = now - self._last_frame_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                self._fps = (
                    instant_fps
                    if self._fps == 0
                    else 0.8 * self._fps + 0.2 * instant_fps
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
                raise RuntimeError("ONNX classifier returned a non-finite probability.")
            probability = float(np.clip(probability, 0.0, 1.0))

        if probability > self.threshold:
            self.consecutive_triggers += 1
        else:
            self.consecutive_triggers = 0
        is_fighting = self.consecutive_triggers >= CONSECUTIVE_TRIGGERS_REQUIRED
        if not is_fighting:
            severity = Severity.NONE
        elif probability >= 0.98:
            severity = Severity.CRITICAL
        elif probability >= 0.94:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM
        return ViolencePrediction(
            is_fighting=is_fighting,
            probability=probability,
            severity=severity,
            confidence=confidence,
            bbox=bbox,
            timestamp=now,
            inference_ms=(time.perf_counter() - started) * 1000.0,
            fps=self._fps,
        )

    def reset(self) -> None:
        """Clear temporal features after the source changes."""
        with self._lock:
            self._poses.clear()
            self.consecutive_triggers = 0
            self._frame_index = 0
            self._frames_since_detection = 0
            self._cached_pose.fill(0.0)
            self._last_detected_pose = None
            self._pose_velocity.fill(0.0)
            self._cached_confidence = 0.0
            self._cached_bbox = None
            self._last_frame_time = None
            self._fps = 0.0


TemporalVelocityClassifier = PoseLSTMDetector
