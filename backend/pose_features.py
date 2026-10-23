"""Shared translation- and scale-invariant pose feature normalization."""

from __future__ import annotations

import numpy as np

SEQUENCE_LENGTH = 30
KEYPOINT_COUNT = 17
FEATURE_COUNT = KEYPOINT_COUNT * 2


def normalize_pose_sequence(sequence: np.ndarray) -> np.ndarray:
    """Convert normalized screen coordinates to root-centered pose/velocity features."""
    poses = np.asarray(sequence, dtype=np.float32)
    if poses.shape != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(
            f"Expected pose sequence shape {(SEQUENCE_LENGTH, FEATURE_COUNT)}, "
            f"got {poses.shape}."
        )
    if not np.isfinite(poses).all():
        raise ValueError("Pose sequence contains NaN or infinite values.")

    points = poses.reshape(SEQUENCE_LENGTH, KEYPOINT_COUNT, 2).copy()
    frame_valid = np.any(points != 0.0, axis=(1, 2))
    valid_indices = np.flatnonzero(frame_valid)
    if valid_indices.size == 0:
        return np.zeros((SEQUENCE_LENGTH, FEATURE_COUNT), dtype=np.float32)

    all_indices = np.arange(SEQUENCE_LENGTH)
    for keypoint in range(KEYPOINT_COUNT):
        for coordinate in range(2):
            values = points[:, keypoint, coordinate]
            joint_valid = frame_valid & np.isfinite(values)
            if not joint_valid.any():
                continue
            known = np.flatnonzero(joint_valid)
            values[:] = np.interp(all_indices, known, values[known])

    for frame_index in np.flatnonzero(~frame_valid):
        nearest = valid_indices[np.argmin(np.abs(valid_indices - frame_index))]
        points[frame_index] = points[nearest]

    hips_valid = np.all(points[:, [11, 12], :] != 0.0, axis=(1, 2))
    shoulders_valid = np.all(points[:, [5, 6], :] != 0.0, axis=(1, 2))
    roots = np.empty((SEQUENCE_LENGTH, 2), dtype=np.float32)
    for frame_index in range(SEQUENCE_LENGTH):
        if hips_valid[frame_index]:
            roots[frame_index] = points[frame_index, [11, 12]].mean(axis=0)
        elif shoulders_valid[frame_index]:
            roots[frame_index] = points[frame_index, [5, 6]].mean(axis=0)
        else:
            roots[frame_index] = np.median(points[frame_index], axis=0)

    centered = points - roots[:, np.newaxis, :]
    torso_scales = np.linalg.norm(
        points[:, [5, 6]].mean(axis=1) - roots,
        axis=1,
    )
    fallback_scales = np.percentile(
        np.linalg.norm(centered, axis=2),
        90,
        axis=1,
    )
    scales = np.where(torso_scales > 1e-4, torso_scales, fallback_scales)
    scales = np.maximum(scales, 1e-3)
    centered /= scales[:, np.newaxis, np.newaxis]
    velocity = np.zeros_like(centered)
    velocity[1:] = centered[1:] - centered[:-1]
    features = 0.7 * velocity + 0.3 * centered
    return np.clip(features, -2.0, 2.0).reshape(SEQUENCE_LENGTH, FEATURE_COUNT)


def normalize_pose_batch(sequences: np.ndarray) -> np.ndarray:
    """Normalize a batch of 30-frame, 34-coordinate pose sequences."""
    poses = np.asarray(sequences, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(
            f"Expected pose batch shape (N, {SEQUENCE_LENGTH}, {FEATURE_COUNT}), "
            f"got {poses.shape}."
        )
    return np.stack([normalize_pose_sequence(sequence) for sequence in poses])
