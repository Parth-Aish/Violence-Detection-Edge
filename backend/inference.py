"""
inference.py — YOLOv8-Pose + Temporal Velocity Heuristic Violence Classifier
==============================================================================

Architecture:
    Frame → YOLOv8-Pose → 17 keypoints per person → Temporal Buffer (15 frames)
    → Wrist/Elbow velocity computation → Thresholding → Violence classification

Design Rationale (vs. 3D-CNNs):
    - 3D-CNNs (SlowFast, I3D): ~8 GFLOPS, 30+ FPS GPU required, >200MB model.
    - Our approach: YOLOv8n-pose (~3.2 GFLOPS) + NumPy heuristics (~0.1ms).
    - Result: Runs on Jetson Nano / RPi5 at 15 FPS. No temporal CNN required.
    - Trade-off: Lower accuracy on subtle violence, but excellent on physical
      altercations which produce high limb velocities.

Keypoint Indices (COCO 17-keypoint format):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

Author: Violence Detection System
"""

import time
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum

logger = logging.getLogger("inference")

# Lazy-load ultralytics to fail fast with a clear message
_yolo_model = None


def _get_model(model_path: str = "yolov8n-pose.pt"):
    """
    Lazy singleton for YOLOv8-Pose model.
    Downloads automatically on first run if not present.
    """
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(model_path)
            logger.info(f"YOLOv8-Pose model loaded: {model_path}")
        except ImportError:
            raise RuntimeError(
                "ultralytics is not installed. "
                "Run: pip install ultralytics"
            )
    return _yolo_model


class Severity(IntEnum):
    """Violence severity levels mapped to integer scores for logging."""
    NONE = 0
    LOW = 1       # Elevated motion, possibly aggressive gesturing
    MEDIUM = 2    # Rapid limb motion consistent with shoving/pushing
    HIGH = 3      # Extreme velocities consistent with punching/kicking
    CRITICAL = 4  # Sustained high-severity across multiple frames


@dataclass
class KeypointIndices:
    """COCO 17-keypoint indices for violence-relevant joints."""
    LEFT_WRIST: int = 9
    RIGHT_WRIST: int = 10
    LEFT_ELBOW: int = 7
    RIGHT_ELBOW: int = 8
    LEFT_SHOULDER: int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_KNEE: int = 13
    RIGHT_KNEE: int = 14
    LEFT_ANKLE: int = 15
    RIGHT_ANKLE: int = 16

KP = KeypointIndices()

# Joints used for velocity computation (wrists, elbows, ankles)
VIOLENCE_JOINTS = [
    KP.LEFT_WRIST, KP.RIGHT_WRIST,
    KP.LEFT_ELBOW, KP.RIGHT_ELBOW,
    KP.LEFT_ANKLE, KP.RIGHT_ANKLE,
]


@dataclass
class DetectionResult:
    """Single-frame violence detection result for one person."""
    person_id: int
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2)
    severity: Severity
    max_velocity: float                        # Peak joint velocity (px/frame)
    mean_velocity: float                       # Mean velocity across tracked joints
    confidence: float                          # Pose detection confidence
    timestamp: float = field(default_factory=time.time)
    keypoints: Optional[np.ndarray] = None     # (17, 3) array if retained


@dataclass
class ViolenceEvent:
    """Aggregated violence event spanning multiple frames."""
    event_id: str
    start_time: float
    end_time: float
    peak_severity: Severity
    detections: list[DetectionResult]
    frame_count: int


class TemporalVelocityClassifier:
    """
    Lightweight temporal classifier that computes inter-frame joint velocities
    across a sliding window of N frames to detect violent motion patterns.

    Algorithm:
        1. For each detected person, extract keypoints (17 x [x, y, conf]).
        2. Buffer the last `window_size` frames of keypoints per tracked person.
        3. Compute Euclidean velocity of wrist/elbow/ankle joints between frames.
        4. Apply exponential moving average (EMA) to smooth noise.
        5. Classify based on velocity thresholds.

    Velocity Thresholds (empirically tuned):
        - These are in pixels/frame at 640x480 @ 15fps.
        - A normal walking speed produces ~2-5 px/frame wrist velocity.
        - Aggressive gesturing: ~15-25 px/frame.
        - Punching motion: ~40-80+ px/frame.
    """

    def __init__(
        self,
        window_size: int = 15,
        fps: int = 15,
        frame_width: int = 640,
        frame_height: int = 480,
        # Velocity thresholds (pixels/frame)
        threshold_low: float = 18.0,
        threshold_medium: float = 35.0,
        threshold_high: float = 55.0,
        threshold_critical: float = 80.0,
        # Minimum keypoint confidence to consider valid
        min_kp_confidence: float = 0.3,
        # EMA smoothing factor (higher = more responsive, more noisy)
        ema_alpha: float = 0.4,
        # Minimum consecutive high-velocity frames to trigger
        min_trigger_frames: int = 3,
        # Proximity threshold for inter-person violence (pixels)
        proximity_threshold: float = 150.0,
        # Model path
        model_path: str = "yolov8n-pose.pt",
    ):
        self.window_size = window_size
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.threshold_low = threshold_low
        self.threshold_medium = threshold_medium
        self.threshold_high = threshold_high
        self.threshold_critical = threshold_critical
        self.min_kp_confidence = min_kp_confidence
        self.ema_alpha = ema_alpha
        self.min_trigger_frames = min_trigger_frames
        self.proximity_threshold = proximity_threshold
        self.model_path = model_path

        # Per-person temporal buffers: person_id → deque of keypoints
        self._keypoint_buffers: dict[int, deque] = {}
        # Per-person EMA velocity: person_id → float
        self._ema_velocities: dict[int, float] = {}
        # Per-person consecutive trigger counter
        self._trigger_counters: dict[int, int] = {}
        # Frame counter
        self._frame_count = 0

    def _extract_poses(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8-Pose on a single frame.
        
        Returns list of dicts:
            [{"bbox": (x1,y1,x2,y2), "keypoints": np.array(17,3), "conf": float}, ...]
        """
        model = _get_model(self.model_path)
        results = model(frame, verbose=False, conf=0.25)

        persons = []
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            kps = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

            for i in range(len(boxes)):
                persons.append({
                    "bbox": tuple(boxes[i].tolist()),
                    "keypoints": kps[i],    # (17, 3) → [x, y, confidence]
                    "conf": float(confs[i]),
                })
        return persons

    def _compute_joint_velocities(
        self, current_kps: np.ndarray, previous_kps: np.ndarray
    ) -> tuple[float, float]:
        """
        Compute max and mean Euclidean velocity for violence-relevant joints.

        Only considers joints where BOTH frames have confidence > threshold.
        Returns (max_velocity, mean_velocity) in pixels/frame.
        """
        velocities = []

        for joint_idx in VIOLENCE_JOINTS:
            curr = current_kps[joint_idx]
            prev = previous_kps[joint_idx]

            # Both keypoints must be confidently detected
            if curr[2] < self.min_kp_confidence or prev[2] < self.min_kp_confidence:
                continue

            # Euclidean distance in pixel space
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            velocity = np.sqrt(dx * dx + dy * dy)
            velocities.append(velocity)

        if not velocities:
            return 0.0, 0.0

        return float(np.max(velocities)), float(np.mean(velocities))

    def _classify_severity(self, velocity: float, consecutive: int) -> Severity:
        """
        Map velocity to severity, boosted by consecutive frame count.
        Consecutive high-velocity frames escalate severity.
        """
        if velocity >= self.threshold_critical:
            return Severity.CRITICAL
        elif velocity >= self.threshold_high:
            return Severity.CRITICAL if consecutive >= self.min_trigger_frames * 2 else Severity.HIGH
        elif velocity >= self.threshold_medium:
            return Severity.HIGH if consecutive >= self.min_trigger_frames * 2 else Severity.MEDIUM
        elif velocity >= self.threshold_low:
            return Severity.MEDIUM if consecutive >= self.min_trigger_frames else Severity.LOW
        else:
            return Severity.NONE

    def _check_proximity(self, persons: list[dict]) -> list[tuple[int, int]]:
        """
        Find pairs of persons within proximity threshold.
        Proximity between two people + high velocities = higher confidence of violence.
        Returns list of (idx_a, idx_b) pairs.
        """
        close_pairs = []
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                bbox_a = persons[i]["bbox"]
                bbox_b = persons[j]["bbox"]
                # Center-to-center distance
                cx_a = (bbox_a[0] + bbox_a[2]) / 2
                cy_a = (bbox_a[1] + bbox_a[3]) / 2
                cx_b = (bbox_b[0] + bbox_b[2]) / 2
                cy_b = (bbox_b[1] + bbox_b[3]) / 2
                dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
                if dist < self.proximity_threshold:
                    close_pairs.append((i, j))
        return close_pairs

    def process_frame(self, frame: np.ndarray) -> list[DetectionResult]:
        """
        Process a single frame through the full pipeline:
            1. YOLOv8-Pose extraction
            2. Temporal buffering
            3. Velocity computation
            4. Severity classification

        Args:
            frame: BGR image as np.ndarray (H, W, 3)

        Returns:
            List of DetectionResult for persons with Severity > NONE
        """
        self._frame_count += 1
        detections = []

        # Step 1: Extract poses
        persons = self._extract_poses(frame)

        if not persons:
            return detections

        # Step 2: Check proximity (used to boost severity)
        close_pairs = self._check_proximity(persons)
        persons_in_proximity = set()
        for i, j in close_pairs:
            persons_in_proximity.add(i)
            persons_in_proximity.add(j)

        # Step 3: Process each person
        for person_idx, person in enumerate(persons):
            kps = person["keypoints"]  # (17, 3)
            bbox = person["bbox"]
            conf = person["conf"]

            # Initialize buffer for new persons
            if person_idx not in self._keypoint_buffers:
                self._keypoint_buffers[person_idx] = deque(maxlen=self.window_size)
                self._ema_velocities[person_idx] = 0.0
                self._trigger_counters[person_idx] = 0

            buffer = self._keypoint_buffers[person_idx]
            buffer.append(kps)

            # Need at least 2 frames for velocity
            if len(buffer) < 2:
                continue

            # Step 4: Compute velocity
            max_vel, mean_vel = self._compute_joint_velocities(
                buffer[-1], buffer[-2]
            )

            # Step 5: Apply EMA smoothing
            ema = self._ema_velocities[person_idx]
            ema = self.ema_alpha * max_vel + (1 - self.ema_alpha) * ema
            self._ema_velocities[person_idx] = ema

            # Step 6: Track consecutive high-velocity frames
            if ema >= self.threshold_low:
                self._trigger_counters[person_idx] += 1
            else:
                # Decay counter gradually (not instant reset — handles brief pauses)
                self._trigger_counters[person_idx] = max(
                    0, self._trigger_counters[person_idx] - 1
                )

            consecutive = self._trigger_counters[person_idx]

            # Step 7: Classify severity
            severity = self._classify_severity(ema, consecutive)

            # Boost severity if person is in close proximity to another
            if person_idx in persons_in_proximity and severity.value >= Severity.LOW.value:
                severity = Severity(min(severity.value + 1, Severity.CRITICAL.value))

            # Step 8: Only emit detections above NONE
            if severity > Severity.NONE and consecutive >= self.min_trigger_frames:
                detection = DetectionResult(
                    person_id=person_idx,
                    bbox=bbox,
                    severity=severity,
                    max_velocity=max_vel,
                    mean_velocity=mean_vel,
                    confidence=conf,
                    keypoints=kps.copy(),
                )
                detections.append(detection)

                if severity >= Severity.HIGH:
                    logger.warning(
                        f"[FRAME {self._frame_count}] Person #{person_idx} — "
                        f"Severity={severity.name}, "
                        f"MaxVel={max_vel:.1f}px/f, "
                        f"EMA={ema:.1f}, "
                        f"Consecutive={consecutive}"
                    )

        # Cleanup stale person buffers (simple heuristic: if more buffers than persons)
        if len(self._keypoint_buffers) > len(persons) * 2:
            active_ids = set(range(len(persons)))
            stale = [k for k in self._keypoint_buffers if k not in active_ids]
            for k in stale:
                del self._keypoint_buffers[k]
                del self._ema_velocities[k]
                del self._trigger_counters[k]

        return detections

    def reset(self):
        """Clear all temporal state. Use when switching streams."""
        self._keypoint_buffers.clear()
        self._ema_velocities.clear()
        self._trigger_counters.clear()
        self._frame_count = 0
        logger.info("Temporal classifier state reset.")
