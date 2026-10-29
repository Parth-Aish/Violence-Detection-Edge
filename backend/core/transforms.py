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

    Args:
        coords_17x2: (17, 2) array containing [x, y] coordinates.

    Returns:
        (17, 2) array centered at mid-hip and normalized by torso length.
    """
    pts = coords_17x2.astype(np.float32, copy=True)
    if pts.shape != (NUM_NODES, CHANNELS):
        raise ValueError(f"Expected shape (17, 2), got {pts.shape}")

    # Check if frame is empty/zeros
    if not np.any(pts):
        return pts

    left_hip = pts[LEFT_HIP]
    right_hip = pts[RIGHT_HIP]
    mid_hip = (left_hip + right_hip) * 0.5

    # If hip coordinates are degenerate (0,0), fallback to mean of non-zero points
    if np.all(mid_hip == 0):
        valid = pts[np.any(pts != 0, axis=1)]
        if len(valid) > 0:
            mid_hip = np.mean(valid, axis=0)

    # Root-centering
    pts_centered = pts - mid_hip

    # Torso scale calculation
    left_shoulder = pts[LEFT_SHOULDER]
    right_shoulder = pts[RIGHT_SHOULDER]
    mid_shoulder = (left_shoulder + right_shoulder) * 0.5

    torso_len = float(np.linalg.norm(mid_shoulder - mid_hip))
    # Safe scale factor: clamp between 0.05 and 2.0 to prevent explosive coordinates
    if torso_len > 0.01:
        scale = max(torso_len, 0.05)
    else:
        # Fallback scale based on bounding box extent of valid joints
        valid_centered = pts_centered[np.any(pts != 0, axis=1)]
        if len(valid_centered) > 0:
            box_h = float(np.max(valid_centered[:, 1]) - np.min(valid_centered[:, 1]))
            scale = max(box_h * 0.5, 0.1)
        else:
            scale = 1.0

    return pts_centered / scale


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
