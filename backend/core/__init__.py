"""Core package for ST-GCN graph definition, mathematical transforms, and inference."""

from .graph import COCO_EDGES, COCO_KEYPOINTS, COCOGraph
from .inference import (
    ALERT_HOLD_FRAMES,
    CONSECUTIVE_TRIGGERS_REQUIRED,
    DEFAULT_POSE_STRIDE,
    POSE_STRIDE,
    VIOLENCE_THRESHOLD,
    PoseLSTMDetector,
    PoseSTGCNDetector,
    Severity,
    ViolencePrediction,
    compute_bbox_iou,
    compute_relational_strike_distance,
)
from .model import STGCN, STGCNBlock, SpatialGraphConv
from .transforms import normalize_stgcn_sequence, raw_to_stgcn_tensor

__all__ = [
    "COCOGraph",
    "COCO_EDGES",
    "COCO_KEYPOINTS",
    "STGCN",
    "STGCNBlock",
    "SpatialGraphConv",
    "raw_to_stgcn_tensor",
    "normalize_stgcn_sequence",
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
]
