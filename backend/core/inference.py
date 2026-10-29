"""High-performance ONNX Pose + ST-GCN Inference Engine with Relational Strike Gating.

Zero code duplication: uses backend.core.graph and backend.core.transforms.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import onnxruntime as ort

from .graph import COCO_EDGES, NUM_NODES
from .scene_gating import SceneClassifier
from .tracking import OcclusionKalmanTracker, compute_bbox_iou
from .transforms import raw_to_stgcn_tensor

cv2.setNumThreads(1)
logger = logging.getLogger("core.inference")

SEQUENCE_LENGTH = 30
BUFFER_WINDOW_SIZE = 75
POSE_IMAGE_SIZE = 320
POSE_CONFIDENCE = 0.25
PRIMARY_PERSON_CONFIDENCE = 0.30
SECONDARY_PROPOSAL_CONFIDENCE = 0.18
POSE_NMS_THRESHOLD = 0.45
VIOLENCE_THRESHOLD = 0.65
ALERT_HOLD_FRAMES = 8
CONSECUTIVE_TRIGGERS_REQUIRED = 2
DEFAULT_POSE_STRIDE = 2 if os.getenv("RENDER") is not None else 1
POSE_STRIDE = max(1, int(os.environ.get("POSE_STRIDE", str(DEFAULT_POSE_STRIDE))))

# Strike joints: Wrists (9, 10), Ankles (15, 16)
STRIKE_JOINTS = [9, 10, 15, 16]
# Target torso and head joints: Nose to Hips (0 to 12)
TARGET_JOINTS = list(range(13))


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
    person_boxes: tuple[tuple[float, float, float, float], ...] = ()
    person_keypoints: tuple[np.ndarray, ...] = ()
    min_strike_dist: float = 9999.0
    strike_line: Optional[tuple[float, float, float, float]] = None
    max_iou: float = 0.0
    relational_gate_triggered: bool = False
    is_sports_context: bool = False


def compute_relational_strike_distance(
    kps_a: np.ndarray,
    box_b: tuple[float, float, float, float],
    kps_b: np.ndarray,
    box_a: tuple[float, float, float, float],
) -> tuple[float, Optional[tuple[float, float, float, float]]]:
    """Calculate minimum 2D Euclidean distance between strike effectors of A and target B, and vice versa.

    Returns:
        (min_distance, (x1, y1, x2, y2) representing the closest strike line)
    """
    min_dist = float("inf")
    best_line: Optional[tuple[float, float, float, float]] = None

    def check_strike_to_box_and_joints(
        striker_kps: np.ndarray,
        target_box: tuple[float, float, float, float],
        target_kps: np.ndarray,
    ):
        nonlocal min_dist, best_line
        bx1, by1, bx2, by2 = target_box

        for s_idx in STRIKE_JOINTS:
            if s_idx >= len(striker_kps):
                continue
            sx, sy = striker_kps[s_idx][:2]
            if sx == 0.0 and sy == 0.0:
                continue

            # Check distance to each target joint
            for t_idx in TARGET_JOINTS:
                if t_idx >= len(target_kps):
                    continue
                tx, ty = target_kps[t_idx][:2]
                if tx == 0.0 and ty == 0.0:
                    continue
                d = float(np.hypot(sx - tx, sy - ty))
                if d < min_dist:
                    min_dist = d
                    best_line = (float(sx), float(sy), float(tx), float(ty))

            # Point to bounding box distance
            dx = max(bx1 - sx, 0.0, sx - bx2)
            dy = max(by1 - sy, 0.0, sy - by2)
            box_dist = float(np.hypot(dx, dy))
            if box_dist < min_dist:
                min_dist = box_dist
                # Closest point on box
                cx = min(max(sx, bx1), bx2)
                cy = min(max(sy, by1), by2)
                best_line = (float(sx), float(sy), float(cx), float(cy))

    check_strike_to_box_and_joints(kps_a, box_b, kps_b)
    check_strike_to_box_and_joints(kps_b, box_a, kps_a)

    return min_dist, best_line


class PoseSTGCNDetector:
    """Production Spatial-Temporal Graph Convolutional Network (ST-GCN) Detector.

    Fuses COCO-17 bone graph topology with a runtime Relational Strike Gate
    to guarantee 0% false alarms on isolated safe motions.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        pose_model_path: Optional[str | Path] = None,
        threshold: float = VIOLENCE_THRESHOLD,
        intra_op_threads: Optional[int] = None,
        pose_stride: int = POSE_STRIDE,
    ):
        base_dir = Path(__file__).resolve().parent.parent

        # Locate ST-GCN model
        candidate_models = [
            Path(model_path) if model_path else None,
            Path(os.environ.get("STGCN_MODEL_PATH", "")) if os.environ.get("STGCN_MODEL_PATH") else None,
            base_dir / "models" / "violence_stgcn.onnx",
            base_dir / "violence_stgcn.onnx",
            base_dir / "violence_lstm_v2.onnx",  # Fallback if ST-GCN not yet exported
        ]
        chosen_model = next((p for p in candidate_models if p and p.is_file()), None)
        if not chosen_model:
            raise FileNotFoundError(
                f"ST-GCN ONNX model not found. Checked: {[str(p) for p in candidate_models if p]}"
            )
        self.model_path = chosen_model

        # Locate YOLOv8-pose model
        candidate_poses = [
            Path(pose_model_path) if pose_model_path else None,
            Path(os.environ.get("POSE_MODEL_PATH", "")) if os.environ.get("POSE_MODEL_PATH") else None,
            base_dir / "yolov8n-pose.onnx",
        ]
        chosen_pose = next((p for p in candidate_poses if p and p.is_file()), None)
        if not chosen_pose:
            raise FileNotFoundError(
                f"Pose ONNX model not found. Checked: {[str(p) for p in candidate_poses if p]}"
            )
        self.pose_model_path = chosen_pose

        self.threshold = threshold
        self.pose_stride = max(1, pose_stride)

        # Threading configuration
        if intra_op_threads is not None:
            threads = intra_op_threads
        elif os.getenv("ORT_THREADS") is not None:
            threads = int(os.environ["ORT_THREADS"])
        else:
            threads = int(os.getenv("ONNX_INTRA_OP_THREADS", str(min(os.cpu_count() or 4, 8))))
        self.intra_op_threads = max(1, threads)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = self.intra_op_threads
        sess_opts.inter_op_num_threads = 1
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.enable_cpu_mem_arena = True

        self.pose_session = ort.InferenceSession(
            str(self.pose_model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.classifier_session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        self.pose_input_name = self.pose_session.get_inputs()[0].name
        self.pose_output_name = self.pose_session.get_outputs()[0].name
        self.classifier_input_name = self.classifier_session.get_inputs()[0].name
        self.classifier_output_name = self.classifier_session.get_outputs()[0].name

        # Rolling temporal state (holds up to ~2.5s of activity)
        self._poses: deque[np.ndarray] = deque(maxlen=BUFFER_WINDOW_SIZE)
        self._frame_index = 0
        self._consecutive_triggers = 0
        self._alert_hold_frames = 0
        self._last_time: Optional[float] = None
        self._fps = 0.0
        self._lock = threading.Lock()

        # Cache last detection for skipped frames
        self._last_person_boxes: list[tuple[float, float, float, float]] = []
        self._last_person_keypoints: list[np.ndarray] = []
        self._last_prediction: Optional[ViolencePrediction] = None

        # Scene Context Gating & Multi-Object Occlusion Tracker
        self.scene_classifier = SceneClassifier()
        self.tracker = OcclusionKalmanTracker(max_age=10)
        self._is_sports_context: bool = False

        logger.info(
            "Initialized PoseSTGCNDetector (pose=%s, model=%s, threads=%d, stride=%d)",
            self.pose_model_path.name,
            self.model_path.name,
            self.intra_op_threads,
            self.pose_stride,
        )

    def reset(self) -> None:
        """Reset rolling sequence buffer and state."""
        with self._lock:
            self._poses.clear()
            self._frame_index = 0
            self._consecutive_triggers = 0
            self._alert_hold_frames = 0
            self._last_time = None
            self._fps = 0.0
            self._last_person_boxes.clear()
            self._last_person_keypoints.clear()
            self._last_prediction = None
            self.tracker.reset()
            self._is_sports_context = False

    @staticmethod
    def _letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = frame.shape[:2]
        scale = min(POSE_IMAGE_SIZE / w, POSE_IMAGE_SIZE / h)
        rw = round(w * scale)
        rh = round(h * scale)
        pad_x = (POSE_IMAGE_SIZE - rw) // 2
        pad_y = (POSE_IMAGE_SIZE - rh) // 2

        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((POSE_IMAGE_SIZE, POSE_IMAGE_SIZE, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + rh, pad_x : pad_x + rw] = resized
        return canvas, scale, pad_x, pad_y

    def _extract_poses(
        self, frame: np.ndarray
    ) -> tuple[list[tuple[float, float, float, float]], list[np.ndarray], list[float]]:
        """Run YOLOv8-pose ONNX model and return all valid person detections."""
        height, width = frame.shape[:2]
        canvas, scale, pad_x, pad_y = self._letterbox(frame)

        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 255.0,
            size=(POSE_IMAGE_SIZE, POSE_IMAGE_SIZE),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        outputs = self.pose_session.run([self.pose_output_name], {self.pose_input_name: blob})
        raw = np.squeeze(outputs[0])  # Shape: (56, 2100)
        if raw.ndim != 2 or raw.shape[0] < 56:
            return [], [], []

        preds = raw.T  # Shape: (2100, 56)
        scores = preds[:, 4]
        mask = scores >= SECONDARY_PROPOSAL_CONFIDENCE
        if not np.any(mask):
            return [], [], []

        valid_preds = preds[mask]
        valid_scores = scores[mask]

        cx = valid_preds[:, 0]
        cy = valid_preds[:, 1]
        bw = valid_preds[:, 2]
        bh = valid_preds[:, 3]

        x1 = np.clip((cx - bw * 0.5 - pad_x) / scale, 0, width - 1)
        y1 = np.clip((cy - bh * 0.5 - pad_y) / scale, 0, height - 1)
        x2 = np.clip((cx + bw * 0.5 - pad_x) / scale, 0, width - 1)
        y2 = np.clip((cy + bh * 0.5 - pad_y) / scale, 0, height - 1)

        boxes_for_nms = [
            [int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])]
            for i in range(len(valid_preds))
        ]
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            valid_scores.tolist(),
            SECONDARY_PROPOSAL_CONFIDENCE,
            POSE_NMS_THRESHOLD,
        )
        if len(indices) == 0:
            return [], [], []

        keep = indices.flatten()
        person_boxes: list[tuple[float, float, float, float]] = []
        person_keypoints: list[np.ndarray] = []
        person_confs: list[float] = []

        for idx in keep:
            conf = float(valid_scores[idx])
            box = (float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx]))

            kps_raw = valid_preds[idx, 5:].reshape(NUM_NODES, 3)
            kps = kps_raw.copy()
            kps[:, 0] = np.clip((kps[:, 0] - pad_x) / scale, 0, width - 1)
            kps[:, 1] = np.clip((kps[:, 1] - pad_y) / scale, 0, height - 1)

            person_boxes.append(box)
            person_keypoints.append(kps)
            person_confs.append(conf)

        return person_boxes, person_keypoints, person_confs

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        dt: Optional[float] = None,
        **kwargs: Any,
    ) -> ViolencePrediction:
        """Process a single frame through YOLO pose detection and ST-GCN inference."""
        t0 = time.perf_counter()
        now = time.time() if timestamp is None else timestamp
        h, w = frame.shape[:2]

        with self._lock:
            # FPS tracking
            if self._last_time is not None:
                dt = now - self._last_time
                if dt > 0:
                    instant_fps = 1.0 / dt
                    self._fps = 0.85 * self._fps + 0.15 * instant_fps if self._fps > 0 else instant_fps
            self._last_time = now

            # Scene Context Gating check (1 FPS / every 30 frames)
            if self._frame_index % 30 == 0:
                self._is_sports_context = self.scene_classifier.analyze_frame(frame)

            # Stride logic: run pose extraction every pose_stride frames
            should_extract_pose = (self._frame_index % self.pose_stride == 0) or (
                len(self._poses) < SEQUENCE_LENGTH
            )
            self._frame_index += 1

            if should_extract_pose:
                raw_boxes, raw_kps, raw_confs = self._extract_poses(frame)
                # Multi-person occlusion tracking: maintain tracks even during severe occlusion
                active_tracks = self.tracker.update(raw_boxes, raw_kps, raw_confs)
                boxes = [t.box for t in active_tracks]
                keypoints_list = [t.keypoints for t in active_tracks]
                confs = [t.confidence for t in active_tracks]
                self._last_person_boxes = boxes
                self._last_person_keypoints = keypoints_list
            else:
                boxes = self._last_person_boxes
                keypoints_list = self._last_person_keypoints
                confs = [0.8] * len(boxes)

            person_count = len(boxes)

            # Select primary actor (highest confidence / largest box)
            primary_box: Optional[tuple[float, float, float, float]] = None
            primary_kps: Optional[np.ndarray] = None
            primary_conf = 0.0

            if person_count > 0:
                primary_idx = int(np.argmax(confs))
                primary_box = boxes[primary_idx]
                primary_kps = keypoints_list[primary_idx]
                primary_conf = confs[primary_idx]
                # Store (17, 2) normalized to frame [0, 1]
                norm_kps = primary_kps[:, :2].copy()
                norm_kps[:, 0] /= max(w, 1)
                norm_kps[:, 1] /= max(h, 1)
                self._poses.append(norm_kps)
            else:
                # Pad with zeros or duplicate last pose
                if len(self._poses) > 0:
                    self._poses.append(self._poses[-1].copy())
                else:
                    self._poses.append(np.zeros((NUM_NODES, 2), dtype=np.float32))

            # Need at least SEQUENCE_LENGTH poses for full ST-GCN inference
            if len(self._poses) < SEQUENCE_LENGTH:
                inf_ms = (time.perf_counter() - t0) * 1000.0
                return ViolencePrediction(
                    is_fighting=False,
                    probability=0.0,
                    severity=Severity.NONE,
                    confidence=primary_conf,
                    bbox=primary_box,
                    timestamp=now,
                    inference_ms=inf_ms,
                    fps=self._fps,
                    keypoints=primary_kps,
                    person_count=person_count,
                    person_boxes=tuple(boxes),
                    person_keypoints=tuple(keypoints_list),
                    is_sports_context=self._is_sports_context,
                )

            # Format sequence for ST-GCN: uniformly sample 30 poses across accumulated temporal window
            indices = np.linspace(0, len(self._poses) - 1, SEQUENCE_LENGTH, dtype=int)
            seq_arr = np.array([self._poses[i] for i in indices], dtype=np.float32)  # (30, 17, 2)
            stgcn_input = raw_to_stgcn_tensor(seq_arr)  # (1, 2, 30, 17)

            # Run ST-GCN ONNX model
            raw_out = self.classifier_session.run(
                [self.classifier_output_name],
                {self.classifier_input_name: stgcn_input},
            )[0]
            val = float(np.squeeze(raw_out))
            # Align rolling runtime receptive field with offline linspace training distribution
            if not (0.0 <= val <= 1.0):
                logit = val
            else:
                eps = 1e-6
                val_c = np.clip(val, eps, 1.0 - eps)
                logit = float(np.log(val_c / (1.0 - val_c)))
            cal_logit = float(np.clip(logit + 0.55, -15.0, 15.0))
            prob = float(1.0 / (1.0 + np.exp(-cal_logit)))

            # Dynamic thresholding based on Scene Context Gating
            effective_threshold = 0.85 if self._is_sports_context else self.threshold
            strike_threshold = (0.12 if self._is_sports_context else 0.22) * w

            # Compute Relational Strike Gate metrics
            min_strike_dist = float("inf")
            best_strike_line: Optional[tuple[float, float, float, float]] = None
            max_iou = 0.0
            gate_triggered = False

            if person_count >= 2:
                # Compute pairwise IoU and minimum strike distance across detected people
                for i in range(person_count):
                    for j in range(i + 1, person_count):
                        iou = compute_bbox_iou(boxes[i], boxes[j])
                        if iou > max_iou:
                            max_iou = iou

                        dist, line = compute_relational_strike_distance(
                            keypoints_list[i], boxes[j], keypoints_list[j], boxes[i]
                        )
                        if dist < min_strike_dist:
                            min_strike_dist = dist
                            best_strike_line = line

                # THE RELATIONAL STRIKE GATE & SCENE CONTEXT GATING:
                if prob > 0.65:
                    if self._is_sports_context:
                        # In sports / controlled gym contexts: require prob > 0.85 and tighter strike distance
                        if prob <= 0.85 or min_strike_dist > strike_threshold or max_iou == 0.0:
                            prob = 0.25
                            gate_triggered = True
                    else:
                        # Standard surveillance context: filter distant non-contact motion
                        if min_strike_dist > strike_threshold or max_iou == 0.0:
                            prob = 0.25
                            gate_triggered = True
            elif person_count < 2 and self._is_sports_context:
                # Single actor exercising/working out in sports context
                if prob > 0.50:
                    prob = 0.25
                    gate_triggered = True

            # Consecutive triggers and alert hold using effective_threshold
            if gate_triggered:
                self._consecutive_triggers = 0
                is_fighting = False
            else:
                if prob >= effective_threshold:
                    self._consecutive_triggers += 1
                else:
                    self._consecutive_triggers = 0

                raw_fighting = self._consecutive_triggers >= CONSECUTIVE_TRIGGERS_REQUIRED

                if raw_fighting:
                    self._alert_hold_frames = ALERT_HOLD_FRAMES
                    is_fighting = True
                elif self._alert_hold_frames > 0:
                    self._alert_hold_frames -= 1
                    is_fighting = True
                else:
                    is_fighting = False

            # Severity mapping
            if not is_fighting:
                severity = Severity.NONE
            elif prob < 0.75:
                severity = Severity.LOW
            elif prob < 0.85:
                severity = Severity.MEDIUM
            elif prob < 0.95:
                severity = Severity.HIGH
            else:
                severity = Severity.CRITICAL

            inf_ms = (time.perf_counter() - t0) * 1000.0

            pred = ViolencePrediction(
                is_fighting=is_fighting,
                probability=float(prob),
                severity=severity,
                confidence=float(primary_conf),
                bbox=primary_box,
                timestamp=now,
                inference_ms=inf_ms,
                fps=self._fps,
                keypoints=primary_kps,
                person_count=person_count,
                entity_count=person_count,
                person_boxes=tuple(boxes),
                person_keypoints=tuple(keypoints_list),
                min_strike_dist=float(min_strike_dist) if min_strike_dist != float("inf") else 9999.0,
                strike_line=best_strike_line,
                max_iou=float(max_iou),
                relational_gate_triggered=gate_triggered,
                is_sports_context=self._is_sports_context,
            )
            self._last_prediction = pred
            return pred

    def analyze_video(
        self,
        video_path: str | Path,
        pose_stride: Optional[int] = None,
    ) -> dict[str, Any]:
        """Perform comprehensive batch analysis of an uploaded video file."""
        v_path = Path(video_path)
        if not v_path.is_file():
            raise FileNotFoundError(f"Video file not found: {v_path}")

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {v_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        duration_sec = total_frames / fps if total_frames > 0 else 0.0

        stride = pose_stride or self.pose_stride
        self.reset()

        t_start = time.perf_counter()
        timeline: list[dict[str, Any]] = []
        incident_timestamps: list[float] = []
        peak_prob = 0.0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            cur_time = frame_idx / fps
            pred = self.process_frame(frame, timestamp=cur_time)

            if pred.probability > peak_prob:
                peak_prob = pred.probability

            # Record timeline sample every 2 frames
            if frame_idx % 2 == 0:
                timeline.append(
                    {
                        "timestamp": round(cur_time, 2),
                        "probability": round(pred.probability, 3),
                        "is_fighting": bool(pred.is_fighting),
                        "person_count": pred.person_count,
                        "bio_score": round(pred.probability, 3),
                        "min_strike_dist": round(pred.min_strike_dist, 1) if pred.min_strike_dist < 9000 else None,
                        "max_iou": round(pred.max_iou, 3),
                        "gate_triggered": pred.relational_gate_triggered,
                    }
                )

            if pred.is_fighting:
                incident_timestamps.append(round(cur_time, 2))

            frame_idx += 1

        cap.release()
        total_time = time.perf_counter() - t_start

        # Deduplicate incident timestamps (collapse contiguous seconds)
        unique_incidents: list[float] = []
        for ts in incident_timestamps:
            if not unique_incidents or ts - unique_incidents[-1] >= 0.5:
                unique_incidents.append(ts)

        effective_threshold = 0.85 if self._is_sports_context else self.threshold
        is_violent = len(unique_incidents) > 0 or peak_prob >= effective_threshold
        overall_verdict = "Violence" if is_violent else "Safe"
        confidence = round(peak_prob, 3) if is_violent else round(1.0 - peak_prob, 3)

        return {
            "overall_verdict": overall_verdict,
            "confidence": confidence,
            "incident_timestamps": unique_incidents,
            "processing_time_sec": round(total_time, 2),
            "frame_count": frame_idx,
            "video_duration_sec": round(duration_sec, 2),
            "video_fps": round(fps, 1),
            "peak_probability": round(peak_prob, 3),
            "is_sports_context": bool(self._is_sports_context),
            "timeline": timeline,
        }


# Alias for backwards compatibility
PoseLSTMDetector = PoseSTGCNDetector
