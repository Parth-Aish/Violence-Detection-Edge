"""CPU-only ONNX pose extraction followed by a rolling ONNX LSTM classifier."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
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
PRIMARY_PERSON_CONFIDENCE = 0.30
SECONDARY_PROPOSAL_CONFIDENCE = 0.12
POSE_NMS_THRESHOLD = 0.45
VIOLENCE_THRESHOLD = 0.75
ALERT_HOLD_FRAMES = 4
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
    keypoints: Optional[np.ndarray] = None
    person_count: int = 0
    entity_count: int = 0
    secondary_entity_count: int = 0
    interaction_mode: str = "Solo / Safe"
    person_boxes: tuple[tuple[float, float, float, float], ...] = ()
    person_keypoints: tuple[np.ndarray, ...] = ()
    bio_score: float = 0.0


@dataclass
class TrackedPerson:
    track_id: int
    bbox: tuple[float, float, float, float]
    keypoints: np.ndarray
    model_pose: np.ndarray
    confidence: float
    history: deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=SEQUENCE_LENGTH)
    )
    last_detection_keypoints: np.ndarray = field(
        default_factory=lambda: np.zeros((KEYPOINT_COUNT, 2), dtype=np.float32)
    )
    last_detection_time: float = 0.0
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros((KEYPOINT_COUNT, 2), dtype=np.float32)
    )
    acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros((KEYPOINT_COUNT, 2), dtype=np.float32)
    )
    center_y: float = 0.0
    aspect_ratio: float = 0.0
    center_drop: float = 0.0
    aspect_collapse: float = 0.0
    last_seen: float = 0.0
    activity: float = 0.0


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
        self.pose_stride = POSE_STRIDE
        self._frame_index = 0
        self._tracks: dict[int, TrackedPerson] = {}
        self._next_track_id = 1
        self._last_gray: Optional[np.ndarray] = None
        self._last_frame_size: Optional[tuple[int, int]] = None
        self._secondary_boxes: list[tuple[float, float, float, float]] = []
        self._secondary_count = 0
        self._alert_hold_frames = 0
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

    @staticmethod
    def _box_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = max(0.0, first[2] - first[0]) * max(
            0.0, first[3] - first[1]
        )
        second_area = max(0.0, second[2] - second[0]) * max(
            0.0, second[3] - second[1]
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    def _decode_pose_rows(
        self, rows: np.ndarray, width: int, height: int, scale: float, pad_x: int, pad_y: int
    ) -> list[tuple[np.ndarray, float, tuple[float, float, float, float]]]:
        detections = []
        for row in rows:
            center_x, center_y, box_width, box_height = row[:4]
            keypoints = row[5:].reshape(KEYPOINT_COUNT, KEYPOINT_VALUES)
            points = keypoints[:, :2].astype(np.float32, copy=True)
            points[:, 0] = (points[:, 0] - pad_x) / scale
            points[:, 1] = (points[:, 1] - pad_y) / scale
            points[:, 0] = np.clip(points[:, 0] / max(width, 1), 0.0, 1.0)
            points[:, 1] = np.clip(points[:, 1] / max(height, 1), 0.0, 1.0)
            x1 = np.clip((center_x - box_width / 2 - pad_x) / scale, 0, width)
            y1 = np.clip((center_y - box_height / 2 - pad_y) / scale, 0, height)
            x2 = np.clip((center_x + box_width / 2 - pad_x) / scale, 0, width)
            y2 = np.clip((center_y + box_height / 2 - pad_y) / scale, 0, height)
            detections.append(
                (
                    points,
                    float(row[4]),
                    (float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return detections

    def detect_entities(
        self, frame: np.ndarray
    ) -> tuple[
        list[tuple[np.ndarray, float, tuple[float, float, float, float]]],
        int,
        np.ndarray,
        list[tuple[float, float, float, float]],
    ]:
        """Detect pose-backed people, weak proposals, and nearby foreground motion."""
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
        primary_indices = np.flatnonzero(scores >= PRIMARY_PERSON_CONFIDENCE)
        primary_rows = rows[primary_indices]
        primary_boxes = [
            [
                float(row[0] - row[2] / 2),
                float(row[1] - row[3] / 2),
                float(row[2]),
                float(row[3]),
            ]
            for row in primary_rows
        ]
        primary_scores = [float(row[4]) for row in primary_rows]
        primary_kept = (
            np.asarray(
                cv2.dnn.NMSBoxes(
                    primary_boxes,
                    primary_scores,
                    PRIMARY_PERSON_CONFIDENCE,
                    POSE_NMS_THRESHOLD,
                )
            ).reshape(-1)
            if primary_boxes
            else np.array([], dtype=np.int32)
        )
        if primary_kept.size:
            primary_kept = primary_kept[
                np.argsort(np.asarray(primary_scores)[primary_kept])[::-1][:4]
            ]
        detections = self._decode_pose_rows(
            primary_rows[primary_kept], width, height, scale, pad_x, pad_y
        )

        weak_indices = np.flatnonzero(
            (scores >= SECONDARY_PROPOSAL_CONFIDENCE)
            & (scores < PRIMARY_PERSON_CONFIDENCE)
        )
        weak_rows = rows[weak_indices]
        weak_boxes = [
            [
                float(row[0] - row[2] / 2),
                float(row[1] - row[3] / 2),
                float(row[2]),
                float(row[3]),
            ]
            for row in weak_rows
        ]
        weak_scores = [float(row[4]) for row in weak_rows]
        weak_kept = (
            np.asarray(
                cv2.dnn.NMSBoxes(
                    weak_boxes,
                    weak_scores,
                    SECONDARY_PROPOSAL_CONFIDENCE,
                    POSE_NMS_THRESHOLD,
                )
            ).reshape(-1)
            if weak_boxes
            else np.array([], dtype=np.int32)
        )
        secondary_boxes = []
        for index in weak_kept:
            row = weak_rows[index]
            x1 = np.clip((row[0] - row[2] / 2 - pad_x) / scale, 0, width)
            y1 = np.clip((row[1] - row[3] / 2 - pad_y) / scale, 0, height)
            x2 = np.clip((row[0] + row[2] / 2 - pad_x) / scale, 0, width)
            y2 = np.clip((row[1] + row[3] / 2 - pad_y) / scale, 0, height)
            candidate_box = (float(x1), float(y1), float(x2), float(y2))
            if all(self._box_iou(candidate_box, item[2]) < 0.3 for item in detections):
                secondary_boxes.append(candidate_box)

        secondary_count = len(secondary_boxes)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (max(1, width // 4), max(1, height // 4)))
        motion_found = False
        if self._last_gray is not None and self._last_gray.shape == small.shape and detections:
            difference = cv2.absdiff(self._last_gray, small)
            foreground_mask = np.full(small.shape, 255, dtype=np.uint8)
            for _, _, box in detections:
                x1 = max(0, int((box[0] - (box[2] - box[0]) * 0.1) / 4))
                y1 = max(0, int((box[1] - (box[3] - box[1]) * 0.1) / 4))
                x2 = min(
                    small.shape[1],
                    int((box[2] + (box[2] - box[0]) * 0.1) / 4),
                )
                y2 = min(
                    small.shape[0],
                    int((box[3] + (box[3] - box[1]) * 0.1) / 4),
                )
                foreground_mask[y1:y2, x1:x2] = 0
            difference = cv2.bitwise_and(difference, difference, mask=foreground_mask)
            _, motion = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
            motion = cv2.morphologyEx(
                motion, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
            )
            contours, _ = cv2.findContours(
                motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            small_area = small.shape[0] * small.shape[1]
            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= small_area * 0.03:
                    continue
                moment = cv2.moments(contour)
                if moment["m00"] == 0:
                    continue
                cx = moment["m10"] / moment["m00"] * 4
                cy = moment["m01"] / moment["m00"] * 4
                if any(
                    min(
                        np.hypot(cx - box[0], cy - box[1]),
                        np.hypot(cx - box[2], cy - box[3]),
                    )
                    <= max(width, height) * 0.35
                    for _, _, box in detections
                ):
                    motion_found = True
                    break

        if motion_found:
            secondary_count = max(secondary_count, 1)
        self._last_gray = small
        self._last_frame_size = (width, height)
        self._secondary_boxes = secondary_boxes
        self._secondary_count = secondary_count
        return detections, secondary_count, small, secondary_boxes

    @staticmethod
    def _center(box: tuple[float, float, float, float]) -> np.ndarray:
        return np.array(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

    @staticmethod
    def _box_aspect(box: tuple[float, float, float, float]) -> float:
        return (box[3] - box[1]) / max(box[2] - box[0], 1.0)

    def _update_tracks(
        self,
        detections: list[tuple[np.ndarray, float, tuple[float, float, float, float]]],
        *,
        now: float,
        dt: float,
        frame_width: int,
        frame_height: int,
    ) -> list[TrackedPerson]:
        available_tracks = list(self._tracks.values())
        candidates = []
        max_distance = max(frame_width, frame_height) * 0.35
        for track_index, track in enumerate(available_tracks):
            old_center = self._center(track.bbox)
            for detection_index, (_, _, box) in enumerate(detections):
                distance = float(np.linalg.norm(old_center - self._center(box)))
                if distance <= max_distance:
                    candidates.append((distance, track_index, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignments: dict[int, int] = {}
        for _, track_index, detection_index in sorted(candidates):
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
            assignments[detection_index] = track_index

        time_scale = (1.0 / 15.0) / max(dt, 0.02)
        for detection_index, (points, confidence, box) in enumerate(detections):
            if detection_index in assignments:
                track = available_tracks[assignments[detection_index]]
                previous_velocity = track.velocity.copy()
                correction = points - track.keypoints
                detection_delta = points - track.last_detection_keypoints
                detection_dt = max(now - track.last_detection_time, 0.02)
                measured_velocity = detection_delta / detection_dt
                track.acceleration = (measured_velocity - previous_velocity) / max(
                    detection_dt, 0.02
                )
                track.velocity = measured_velocity
                track.model_pose = np.clip(
                    track.model_pose + correction * time_scale, 0.0, 1.0
                )
                previous_center_y = track.center_y
                previous_aspect = track.aspect_ratio
                track.center_y = float(self._center(box)[1] / max(frame_height, 1))
                track.aspect_ratio = self._box_aspect(box)
                center_drop = max(0.0, track.center_y - previous_center_y)
                aspect_collapse = max(0.0, previous_aspect - track.aspect_ratio)
                track.activity = max(
                    track.activity * 0.65,
                    float(np.linalg.norm(track.velocity[[9, 10, 15, 16]], axis=1).max()),
                    center_drop / max(dt, 0.02),
                    aspect_collapse / max(dt, 0.02),
                )
                track.keypoints = points
                track.last_detection_keypoints = points.copy()
                track.last_detection_time = now
                track.bbox = box
                track.confidence = confidence
                track.last_seen = now
                track.history.append(track.model_pose.reshape(FEATURE_COUNT).copy())
                track.center_drop = center_drop
                track.aspect_collapse = aspect_collapse
            else:
                track = TrackedPerson(
                    track_id=self._next_track_id,
                    bbox=box,
                    keypoints=points,
                    model_pose=points.copy(),
                    last_detection_keypoints=points.copy(),
                    last_detection_time=now,
                    confidence=confidence,
                    center_y=float(self._center(box)[1] / max(frame_height, 1)),
                    aspect_ratio=self._box_aspect(box),
                    last_seen=now,
                )
                track.history.append(track.model_pose.reshape(FEATURE_COUNT).copy())
                track.center_drop = 0.0
                track.aspect_collapse = 0.0
                self._tracks[track.track_id] = track
                self._next_track_id += 1

        for track_index, track in enumerate(available_tracks):
            if track_index in matched_tracks:
                continue
            age = now - track.last_seen
            if age > 1.0:
                self._tracks.pop(track.track_id, None)
                continue
            delta = track.velocity * max(dt, 0.02)
            track.keypoints = np.clip(track.keypoints + delta, 0.0, 1.0)
            track.model_pose = np.clip(
                track.model_pose + delta * time_scale, 0.0, 1.0
            )
            track.history.append(track.model_pose.reshape(FEATURE_COUNT).copy())
            track.activity *= 0.85

        return [
            track
            for track in self._tracks.values()
            if now - track.last_seen <= 1.0
        ]

    def _track_probability(self, track: TrackedPerson) -> float:
        if len(track.history) < 6:
            return 0.0
        sequence = list(track.history)
        if len(sequence) < SEQUENCE_LENGTH:
            sequence = [sequence[0]] * (SEQUENCE_LENGTH - len(sequence)) + sequence
        model_input = np.asarray(sequence[-SEQUENCE_LENGTH:], dtype=np.float32)
        output = self.session.run(
            [self.output_name],
            {self.input_name: model_input[np.newaxis, :, :]},
        )[0]
        probability = float(np.asarray(output).reshape(-1)[0])
        if not np.isfinite(probability):
            raise RuntimeError("ONNX classifier returned a non-finite probability.")
        return float(np.clip(probability, 0.0, 1.0))

    @staticmethod
    def _wrist_points(track: TrackedPerson) -> np.ndarray:
        return track.keypoints[[9, 10]]

    def _biomechanical_score(
        self,
        active_tracks: list[TrackedPerson],
        secondary_boxes: list[tuple[float, float, float, float]],
        *,
        person_count: int,
        secondary_count: int,
        frame_width: int,
        frame_height: int,
    ) -> float:
        if not active_tracks:
            return 0.0
        pair_scores = []
        for track in active_tracks[:2]:
            strike_speed = float(
                np.linalg.norm(track.velocity[[9, 10, 15, 16]], axis=1).max()
            )
            strike_acceleration = float(
                np.linalg.norm(track.acceleration[[9, 10, 15, 16]], axis=1).max()
            )
            strike = min(strike_speed / 2.0, 1.0)
            burst = min(strike_acceleration / 12.0, 1.0)
            knockdown = max(
                min(track.center_drop / 0.12, 1.0),
                min(track.aspect_collapse / 0.7, 1.0),
            )
            pair_scores.append((strike, burst, knockdown))

        proximity = 0.0
        if person_count >= 2 and len(active_tracks) >= 2:
            first, second = active_tracks[:2]
            proximity = 1.0 if self._box_iou(first.bbox, second.bbox) > 0.10 else 0.0
            second_center = self._center(second.bbox) / np.array(
                [max(frame_width, 1), max(frame_height, 1)]
            )
            first_wrists = self._wrist_points(first)
            wrist_distance = float(
                np.linalg.norm(first_wrists - second_center, axis=1).min()
            )
            proximity = max(
                proximity,
                float(wrist_distance < 0.25 * frame_width / max(frame_width, 1)),
            )
        elif person_count == 1 and secondary_count:
            track = active_tracks[0]
            if secondary_boxes:
                for box in secondary_boxes:
                    if self._box_iou(track.bbox, box) > 0.10:
                        proximity = 1.0
                        break
                    target = self._center(box) / np.array(
                        [max(frame_width, 1), max(frame_height, 1)]
                    )
                    if np.linalg.norm(self._wrist_points(track) - target, axis=1).min() < 0.25:
                        proximity = 1.0
                        break
            if proximity == 0.0:
                proximity = 0.5

        strike = max(item[0] for item in pair_scores)
        burst = max(item[1] for item in pair_scores)
        knockdown = max(item[2] for item in pair_scores)
        return float(
            np.clip(
                0.35 * strike + 0.20 * burst + 0.30 * proximity + 0.15 * knockdown,
                0.0,
                1.0,
            )
        )

    def process_frame(self, frame: np.ndarray) -> ViolencePrediction:
        """Return a timestamped violence score for one BGR frame."""
        with self._lock:
            return self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> ViolencePrediction:
        started = time.perf_counter()
        now = time.time()
        elapsed = (
            now - self._last_frame_time if self._last_frame_time is not None else 1 / 15
        )
        dt = max(elapsed, 0.02)
        frame_height, frame_width = frame.shape[:2]
        if self._frame_index % self.pose_stride == 0:
            detections, secondary_count, _, secondary_boxes = self.detect_entities(frame)
            self._secondary_boxes = secondary_boxes
            self._secondary_count = secondary_count
        else:
            detections = []
            secondary_count = self._secondary_count
            secondary_boxes = self._secondary_boxes
        tracks = self._update_tracks(
            detections,
            now=now,
            dt=dt,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        self._frame_index += 1
        if self._last_frame_time is not None:
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                self._fps = (
                    instant_fps
                    if self._fps == 0
                    else 0.8 * self._fps + 0.2 * instant_fps
                )
        self._last_frame_time = now

        person_count = len(tracks)
        active_tracks = sorted(tracks, key=lambda item: item.activity, reverse=True)[:2]
        onnx_probability = max(
            (self._track_probability(track) for track in active_tracks),
            default=0.0,
        )
        total_entities = person_count + secondary_count
        interaction_active = person_count >= 2 or (
            person_count == 1 and secondary_count >= 1
        )
        if person_count >= 2:
            interaction_mode = "2+ Persons"
        elif person_count == 1 and secondary_count >= 1:
            interaction_mode = "Person + Entity/Animal"
        else:
            interaction_mode = "Solo / Safe"

        bio_score = self._biomechanical_score(
            active_tracks,
            secondary_boxes,
            person_count=person_count,
            secondary_count=secondary_count,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if interaction_active:
            probability = max(
                onnx_probability,
                0.65 * bio_score + 0.35 * onnx_probability,
            )
            if probability >= 0.58:
                self._alert_hold_frames = ALERT_HOLD_FRAMES
            elif self._alert_hold_frames:
                self._alert_hold_frames -= 1
            is_fighting = self._alert_hold_frames > 0
        else:
            probability = min(onnx_probability, 0.249)
            self._alert_hold_frames = 0
            is_fighting = False

        if not is_fighting:
            severity = Severity.NONE
        elif probability >= 0.98:
            severity = Severity.CRITICAL
        elif probability >= 0.94:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM
        primary = active_tracks[0] if active_tracks else None
        return ViolencePrediction(
            is_fighting=is_fighting,
            probability=probability,
            severity=severity,
            confidence=primary.confidence if primary else 0.0,
            bbox=primary.bbox if primary else None,
            timestamp=now,
            inference_ms=(time.perf_counter() - started) * 1000.0,
            fps=self._fps,
            keypoints=primary.keypoints.copy() if primary else None,
            person_count=person_count,
            entity_count=total_entities,
            interaction_mode=interaction_mode,
            person_boxes=tuple(track.bbox for track in tracks),
            person_keypoints=tuple(track.keypoints.copy() for track in tracks),
            bio_score=bio_score,
            secondary_entity_count=secondary_count,
        )

    def reset(self) -> None:
        """Clear temporal features after the source changes."""
        with self._lock:
            self._poses.clear()
            self._frame_index = 0
            self._tracks.clear()
            self._next_track_id = 1
            self._last_gray = None
            self._last_frame_size = None
            self._secondary_boxes.clear()
            self._secondary_count = 0
            self._alert_hold_frames = 0
            self._last_frame_time = None
            self._fps = 0.0


TemporalVelocityClassifier = PoseLSTMDetector
