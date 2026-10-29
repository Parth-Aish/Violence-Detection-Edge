"""Shared biomechanical feature engineering: transforms raw (30, 34) pose sequence to (30, 82) kinematic tensor."""

from __future__ import annotations

import numpy as np

SEQUENCE_LENGTH = 30
KEYPOINT_COUNT = 17
RAW_FEATURE_COUNT = KEYPOINT_COUNT * 2  # 34
KINEMATIC_FEATURE_COUNT = 82

# COCO Keypoint mapping
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

STRIKE_JOINTS = [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE]


def extract_kinematic_tensor(sequence: np.ndarray) -> np.ndarray:
    """Transform a (30, 34) raw pose sequence into a physics-aware (30, 82) kinematic tensor.
    
    Breakdown of 82 features:
      0..33 (34): Root-centered, torso-scaled normalized joint coordinates (17 * 2)
      34..67 (34): 1st-order frame-to-frame normalized velocities (17 * 2)
      68..75  (8): 2nd-order acceleration magnitudes (4) + velocity magnitudes (4) for wrists & ankles
      76..81  (6): Cosine angles of elbows (2) & knees (2), torso aspect ratio (1), torso vertical tilt (1)
    """
    raw = np.asarray(sequence, dtype=np.float32)
    if raw.shape != (SEQUENCE_LENGTH, RAW_FEATURE_COUNT):
        raise ValueError(
            f"Expected pose sequence shape ({SEQUENCE_LENGTH}, {RAW_FEATURE_COUNT}), got {raw.shape}."
        )

    pts = raw.reshape(SEQUENCE_LENGTH, KEYPOINT_COUNT, 2).copy()

    # Handle zero/missing frames via linear interpolation
    frame_valid = np.any(pts != 0.0, axis=(1, 2))
    valid_idx = np.flatnonzero(frame_valid)
    if valid_idx.size == 0:
        return np.zeros((SEQUENCE_LENGTH, KINEMATIC_FEATURE_COUNT), dtype=np.float32)

    all_idx = np.arange(SEQUENCE_LENGTH)
    for k in range(KEYPOINT_COUNT):
        for c in range(2):
            v = pts[:, k, c]
            jv = frame_valid & np.isfinite(v)
            if jv.any():
                kn = np.flatnonzero(jv)
                v[:] = np.interp(all_idx, kn, v[kn])

    for fi in np.flatnonzero(~frame_valid):
        nearest = valid_idx[np.argmin(np.abs(valid_idx - fi))]
        pts[fi] = pts[nearest]

    # 1. Mid-hip root-centering & Torso scale normalization (34 features)
    hips_valid = (pts[:, L_HIP] != 0.0).any(axis=1) & (pts[:, R_HIP] != 0.0).any(axis=1)
    shoulders_valid = (pts[:, L_SHOULDER] != 0.0).any(axis=1) & (pts[:, R_SHOULDER] != 0.0).any(axis=1)

    roots = np.empty((SEQUENCE_LENGTH, 2), dtype=np.float32)
    for t in range(SEQUENCE_LENGTH):
        if hips_valid[t]:
            roots[t] = (pts[t, L_HIP] + pts[t, R_HIP]) / 2.0
        elif shoulders_valid[t]:
            roots[t] = (pts[t, L_SHOULDER] + pts[t, R_SHOULDER]) / 2.0
        else:
            roots[t] = np.median(pts[t], axis=0)

    centered = pts - roots[:, None, :]
    mid_shoulder = (pts[:, L_SHOULDER] + pts[:, R_SHOULDER]) / 2.0
    torso_len = np.linalg.norm(mid_shoulder - roots, axis=1)
    fallback = np.percentile(np.linalg.norm(centered, axis=2), 90, axis=1)
    scale = np.where(torso_len > 1e-4, torso_len, fallback)
    scale = np.maximum(scale, 1e-3)
    norm_joints = np.clip(centered / scale[:, None, None], -2.5, 2.5)  # (30, 17, 2)

    # 2. 1st-Order Velocity (34 features)
    velocity = np.zeros_like(norm_joints)
    velocity[1:] = norm_joints[1:] - norm_joints[:-1]
    velocity = np.clip(velocity, -2.5, 2.5)

    # 3. 2nd-Order Acceleration (8 features: 4 acc mags + 4 vel mags for wrists & ankles)
    accel = np.zeros_like(velocity)
    accel[1:] = velocity[1:] - velocity[:-1]
    accel = np.clip(accel, -5.0, 5.0)

    acc_mags = np.linalg.norm(accel[:, STRIKE_JOINTS], axis=-1)   # (30, 4)
    vel_mags = np.linalg.norm(velocity[:, STRIKE_JOINTS], axis=-1) # (30, 4)
    dyn_8 = np.clip(np.hstack([acc_mags, vel_mags]), 0.0, 5.0)     # (30, 8)

    # 4. Angles & Posture (6 features)
    def cos_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        v1 = p1 - p2
        v2 = p3 - p2
        dot = np.sum(v1 * v2, axis=-1)
        norm1 = np.linalg.norm(v1, axis=-1)
        norm2 = np.linalg.norm(v2, axis=-1)
        return np.clip(dot / (norm1 * norm2 + 1e-6), -1.0, 1.0)

    cos_le = cos_angle(pts[:, L_SHOULDER], pts[:, L_ELBOW], pts[:, L_WRIST])[:, None]
    cos_re = cos_angle(pts[:, R_SHOULDER], pts[:, R_ELBOW], pts[:, R_WRIST])[:, None]
    cos_lk = cos_angle(pts[:, L_HIP], pts[:, L_KNEE], pts[:, L_ANKLE])[:, None]
    cos_rk = cos_angle(pts[:, R_HIP], pts[:, R_KNEE], pts[:, R_ANKLE])[:, None]

    # Torso aspect ratio: vertical torso height / shoulder width
    sh_w = np.linalg.norm(pts[:, L_SHOULDER] - pts[:, R_SHOULDER], axis=-1)
    torso_h = np.abs(mid_shoulder[:, 1] - roots[:, 1])
    aspect = np.clip(torso_h / (sh_w + 1e-4), 0.0, 5.0)[:, None]

    # Torso vertical tilt cosine: angle with upright vertical vector (0, -1)
    torso_vec = mid_shoulder - roots
    cos_tilt = np.clip(
        -torso_vec[:, 1] / (np.linalg.norm(torso_vec, axis=-1) + 1e-6),
        -1.0,
        1.0,
    )[:, None]

    angles_6 = np.hstack([cos_le, cos_re, cos_lk, cos_rk, aspect, cos_tilt])  # (30, 6)

    # Assemble complete (30, 82) feature matrix
    features_82 = np.hstack([
        norm_joints.reshape(SEQUENCE_LENGTH, RAW_FEATURE_COUNT),
        velocity.reshape(SEQUENCE_LENGTH, RAW_FEATURE_COUNT),
        dyn_8,
        angles_6,
    ])
    return features_82.astype(np.float32)


def extract_kinematic_batch(sequences: np.ndarray) -> np.ndarray:
    """Transform a batch of shape (N, 30, 34) into (N, 30, 82)."""
    seqs = np.asarray(sequences, dtype=np.float32)
    if seqs.ndim != 3 or seqs.shape[1:] != (SEQUENCE_LENGTH, RAW_FEATURE_COUNT):
        raise ValueError(
            f"Expected batch shape (N, {SEQUENCE_LENGTH}, {RAW_FEATURE_COUNT}), got {seqs.shape}."
        )
    return np.stack([extract_kinematic_tensor(seq) for seq in seqs])
