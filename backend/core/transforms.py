"""Authoritative mathematical transformations and normalizations for ST-GCN."""

from __future__ import annotations

import numpy as np

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
NUM_NODES = 17
CHANNELS = 2


def normalize_pose_skeleton(coords_17x2: np.ndarray) -> np.ndarray:
    """Normalize a single frame's 17 keypoints (17, 2) using mid-hip centering and torso scaling.

    Missing joints (represented as [0.0, 0.0]) are masked so they remain explicitly [0.0, 0.0]
    and are not shifted by the mid-hip offset into false valid coordinates.

    Args:
        coords_17x2: (17, 2) array containing [x, y] coordinates.

    Returns:
        (17, 2) array centered at mid-hip and normalized by torso length for valid joints.
    """
    pts = coords_17x2.astype(np.float32, copy=True)
    if pts.shape != (NUM_NODES, CHANNELS):
        raise ValueError(f"Expected shape (17, 2), got {pts.shape}")

    # Keypoint validity mask: True for observed joints, False for missing (0, 0)
    valid_mask = np.any(pts != 0.0, axis=1)
    if not np.any(valid_mask):
        return np.zeros_like(pts)

    valid_pts = pts[valid_mask]

    # Calculate mid-hip reference using only valid hip joints
    has_left_hip = bool(valid_mask[LEFT_HIP])
    has_right_hip = bool(valid_mask[RIGHT_HIP])
    if has_left_hip and has_right_hip:
        mid_hip = (pts[LEFT_HIP] + pts[RIGHT_HIP]) * 0.5
    elif has_left_hip:
        mid_hip = pts[LEFT_HIP].copy()
    elif has_right_hip:
        mid_hip = pts[RIGHT_HIP].copy()
    else:
        # Fallback to mean of whatever joints are valid
        mid_hip = np.mean(valid_pts, axis=0)

    # Calculate torso scale using valid shoulder joints
    has_left_shoulder = bool(valid_mask[LEFT_SHOULDER])
    has_right_shoulder = bool(valid_mask[RIGHT_SHOULDER])
    if has_left_shoulder and has_right_shoulder:
        mid_shoulder = (pts[LEFT_SHOULDER] + pts[RIGHT_SHOULDER]) * 0.5
        torso_len = float(np.linalg.norm(mid_shoulder - mid_hip))
    elif has_left_shoulder:
        torso_len = float(np.linalg.norm(pts[LEFT_SHOULDER] - mid_hip))
    elif has_right_shoulder:
        torso_len = float(np.linalg.norm(pts[RIGHT_SHOULDER] - mid_hip))
    else:
        torso_len = 0.0

    if torso_len > 0.01:
        scale = max(torso_len, 0.05)
    else:
        # Fallback scale based on vertical span of valid joints
        valid_centered = valid_pts - mid_hip
        box_h = float(np.max(valid_centered[:, 1]) - np.min(valid_centered[:, 1])) if len(valid_centered) > 0 else 1.0
        scale = max(box_h * 0.5, 0.1)

    # Initialize normalized output as zeros: missing joints remain strictly [0.0, 0.0]
    out = np.zeros_like(pts)
    out[valid_mask] = (pts[valid_mask] - mid_hip) / scale
    return out


def normalize_stgcn_sequence(seq_30x17x2: np.ndarray) -> np.ndarray:
    """Normalize a (30, 17, 2) sequence frame-by-frame."""
    out = np.zeros_like(seq_30x17x2, dtype=np.float32)
    for t in range(seq_30x17x2.shape[0]):
        out[t] = normalize_pose_skeleton(seq_30x17x2[t])
    return out


def raw_to_stgcn_tensor(
    raw_keypoints: np.ndarray,
    keep_batch_dim: bool = True,
) -> np.ndarray:
    """Transform raw keypoint array into ST-GCN input tensor of shape (N, 2, 30, 17).

    Supported input shapes:
      - (N, 30, 34) or (30, 34)
      - (N, 30, 17, 2) or (30, 17, 2)

    Returns:
      np.ndarray of shape (N, 2, 30, 17) if keep_batch_dim=True,
      or (2, 30, 17) if keep_batch_dim=False on single input.
    """
    arr = np.asarray(raw_keypoints, dtype=np.float32)
    is_single = False

    if arr.ndim == 2 and arr.shape == (30, 34):
        is_single = True
        arr = arr.reshape(1, 30, 17, 2)
    elif arr.ndim == 3 and arr.shape == (30, 17, 2):
        is_single = True
        arr = arr.reshape(1, 30, 17, 2)
    elif arr.ndim == 3 and arr.shape[1:] == (30, 34):
        arr = arr.reshape(-1, 30, 17, 2)
    elif arr.ndim == 4 and arr.shape[1:] == (30, 17, 2):
        pass
    else:
        raise ValueError(
            f"Unsupported input shape for ST-GCN transformation: {arr.shape}. "
            "Expected (N, 30, 34), (30, 34), (N, 30, 17, 2), or (30, 17, 2)."
        )

    num_samples = arr.shape[0]
    normalized_batch = np.zeros((num_samples, 30, 17, 2), dtype=np.float32)

    for i in range(num_samples):
        normalized_batch[i] = normalize_stgcn_sequence(arr[i])

    # Transpose from (N, T=30, V=17, C=2) to (N, C=2, T=30, V=17)
    tensor = np.transpose(normalized_batch, (0, 3, 1, 2))

    if is_single and not keep_batch_dim:
        return tensor[0]
    return tensor
