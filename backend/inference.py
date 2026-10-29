"""Compatibility layer re-exporting core ST-GCN inference engine with Relational Strike Gating.

Zero code duplication: all logic resides in backend.core.inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add backend directory to sys.path
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from core.inference import (
    ALERT_HOLD_FRAMES,
    CONSECUTIVE_TRIGGERS_REQUIRED,
    DEFAULT_POSE_STRIDE,
    POSE_IMAGE_SIZE,
    POSE_STRIDE,
    PRIMARY_PERSON_CONFIDENCE,
    SECONDARY_PROPOSAL_CONFIDENCE,
    SEQUENCE_LENGTH,
    VIOLENCE_THRESHOLD,
    PoseLSTMDetector,
    PoseSTGCNDetector,
    Severity,
    ViolencePrediction,
    compute_bbox_iou,
    compute_relational_strike_distance,
)

__all__ = [
    "PoseSTGCNDetector",
    "PoseLSTMDetector",
    "ViolencePrediction",
    "Severity",
    "compute_bbox_iou",
    "compute_relational_strike_distance",
    "VIOLENCE_THRESHOLD",
    "ALERT_HOLD_FRAMES",
    "CONSECUTIVE_TRIGGERS_REQUIRED",
    "DEFAULT_POSE_STRIDE",
    "POSE_STRIDE",
    "POSE_IMAGE_SIZE",
    "SEQUENCE_LENGTH",
    "PRIMARY_PERSON_CONFIDENCE",
    "SECONDARY_PROPOSAL_CONFIDENCE",
]
