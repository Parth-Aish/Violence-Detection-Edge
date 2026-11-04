"""Multi-Person Occlusion Tracker with True Kalman State Estimation & Hungarian Assignment.

Maintains track continuity across severe physical occlusions (e.g. ground wrestling,
tackles, and close grappling) by estimating velocity vectors with a Kalman filter
and associating detections via Hungarian algorithm (IoU cost matrix).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def compute_bbox_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union between two bounding boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_box_center_dist(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute Euclidean distance between box centroids."""
    cax = (box_a[0] + box_a[2]) * 0.5
    cay = (box_a[1] + box_a[3]) * 0.5
    cbx = (box_b[0] + box_b[2]) * 0.5
    cby = (box_b[1] + box_b[3]) * 0.5
    return float(np.hypot(cax - cbx, cay - cby))


class KalmanBoxTracker:
    """Kalman Filter State Estimator for 2D Bounding Boxes & Kinematic Velocities.

    State Vector x (8-dim):
        [cx, cy, w, h, vx, vy, vw, vh]
        where (cx, cy) is centroid, (w, h) is box dimensions,
        and (vx, vy, vw, vh) are respective velocity derivatives (dx, dy, dw, dh).
    """

    def __init__(self, bbox: tuple[float, float, float, float]):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        # State vector: [cx, cy, w, h, vx, vy, vw, vh]
        self.state = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # State covariance P (8x8): initial uncertainty
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # High initial uncertainty in velocities

        # Process noise Q (8x8): motion model uncertainty
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.01

        # Measurement noise R (4x4): detector sensor noise
        self.R = np.eye(4, dtype=np.float32)
        self.R[:2, :2] *= 1.0
        self.R[2:, 2:] *= 10.0

    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def y(self) -> float:
        return float(self.state[1])

    @property
    def dx(self) -> float:
        return float(self.state[4])

    @property
    def dy(self) -> float:
        return float(self.state[5])

    def predict(self) -> tuple[float, float, float, float]:
        """Advance state by velocity motion model: x' = F @ x, P' = F @ P @ F^T + Q."""
        # 1. State prediction: cx += vx, cy += vy, w += vw, h += vh
        self.state[:4] += self.state[4:]

        # 2. Fast analytic covariance prediction for F = [[I, I], [0, I]]
        P11 = self.P[:4, :4]
        P12 = self.P[:4, 4:]
        P21 = self.P[4:, :4]
        P22 = self.P[4:, 4:]
        self.P[:4, :4] = P11 + P12 + P21 + P22 + self.Q[:4, :4]
        self.P[:4, 4:] = P12 + P22
        self.P[4:, :4] = P21 + P22
        self.P[4:, 4:] = P22 + self.Q[4:, 4:]

        cx, cy, w, h = self.state[:4]
        w = max(1.0, w)
        h = max(1.0, h)
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        return (float(x1), float(y1), float(x2), float(y2))

    def update(self, bbox: tuple[float, float, float, float]) -> None:
        """Correct Kalman state using YOLOv8 bounding box measurement."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        z = np.array([cx, cy, w, h], dtype=np.float32)

        # Innovation residual: y = z - H @ x (since H = [I_4, 0])
        y = z - self.state[:4]

        # Innovation covariance: S = H @ P @ H^T + R = P[:4, :4] + R
        S = self.P[:4, :4] + self.R

        # Kalman Gain: K = P @ H^T @ S^-1 = P[:, :4] @ S^-1
        # Solve linear system directly (faster and more stable than explicit matrix inversion)
        K = np.linalg.solve(S, self.P[:4, :]).T  # shape (8, 4)

        # Posterior state: x = x + K @ y
        self.state += K @ y

        # Posterior covariance: P = (I - K @ H) @ P = P - K @ P[:4, :]
        self.P -= K @ self.P[:4, :]


@dataclass
class Track:
    track_id: int
    box: tuple[float, float, float, float]
    keypoints: np.ndarray
    confidence: float
    time_since_update: int = 0
    hits: int = 1
    age: int = 1
    kalman: KalmanBoxTracker = field(init=False)

    def __post_init__(self) -> None:
        self.kalman = KalmanBoxTracker(self.box)

    @property
    def velocity(self) -> np.ndarray:
        return self.kalman.state[4:8]

    def predict(self) -> tuple[float, float, float, float]:
        """Advance Kalman state and predict next bounding box location."""
        self.age += 1
        self.time_since_update += 1
        old_cx = (self.box[0] + self.box[2]) * 0.5
        old_cy = (self.box[1] + self.box[3]) * 0.5

        predicted_box = self.kalman.predict()
        self.box = predicted_box

        new_cx = (self.box[0] + self.box[2]) * 0.5
        new_cy = (self.box[1] + self.box[3]) * 0.5
        shift_x = new_cx - old_cx
        shift_y = new_cy - old_cy

        # Extrapolate keypoints along estimated velocity
        if self.keypoints is not None and len(self.keypoints) > 0:
            kps = self.keypoints.copy()
            kps[:, 0] += shift_x
            kps[:, 1] += shift_y
            self.keypoints = kps

        return self.box

    def update(
        self,
        new_box: tuple[float, float, float, float],
        new_kps: np.ndarray,
        conf: float,
    ) -> None:
        """Update track with observed detection using Kalman measurement correction."""
        self.kalman.update(new_box)

        # Synchronize box with corrected Kalman state
        cx, cy, w, h = self.kalman.state[:4]
        w = max(1.0, w)
        h = max(1.0, h)
        self.box = (
            float(cx - w * 0.5),
            float(cy - h * 0.5),
            float(cx + w * 0.5),
            float(cy + h * 0.5),
        )
        self.keypoints = new_kps.copy()
        self.confidence = conf
        self.time_since_update = 0
        self.hits += 1


def associate_detections_to_tracks(
    track_boxes: list[tuple[float, float, float, float]],
    detected_boxes: list[tuple[float, float, float, float]],
    min_iou: float = 0.20,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Associate detections to predicted tracks using Hungarian algorithm / IoU cost matrix."""
    num_tracks = len(track_boxes)
    num_dets = len(detected_boxes)

    if num_tracks == 0:
        return [], list(range(num_tracks)), list(range(num_dets))
    if num_dets == 0:
        return [], list(range(num_tracks)), list(range(num_dets))

    # Compute IoU cost matrix: cost = 1.0 - IoU
    cost_matrix = np.ones((num_tracks, num_dets), dtype=np.float32)
    for i, trk_box in enumerate(track_boxes):
        for j, det_box in enumerate(detected_boxes):
            iou = compute_bbox_iou(trk_box, det_box)
            if iou > 0.0:
                cost_matrix[i, j] = 1.0 - iou
            else:
                dist = compute_box_center_dist(trk_box, det_box)
                diag = np.hypot(det_box[2] - det_box[0], det_box[3] - det_box[1])
                norm_dist = dist / max(diag, 1.0)
                if norm_dist < 1.0:
                    cost_matrix[i, j] = 1.0 - max(0.0, 0.35 - 0.35 * norm_dist)
                else:
                    cost_matrix[i, j] = 1.0

    # Hungarian algorithm assignment
    if num_tracks <= 2 or num_dets <= 2:
        # Fast bipartite matching for small surveillance groups (<= 2 actors)
        matched_rows, matched_cols = [], []
        temp_cost = cost_matrix.copy()
        for _ in range(min(num_tracks, num_dets)):
            min_val = float(np.min(temp_cost))
            if min_val > (1.0 - min_iou):
                break
            r, c = np.unravel_index(np.argmin(temp_cost), temp_cost.shape)
            matched_rows.append(int(r))
            matched_cols.append(int(c))
            temp_cost[r, :] = 1e6
            temp_cost[:, c] = 1e6
        row_ind = np.array(matched_rows, dtype=int)
        col_ind = np.array(matched_cols, dtype=int)
    else:
        try:
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ImportError:
            matched_rows, matched_cols = [], []
            temp_cost = cost_matrix.copy()
            for _ in range(min(num_tracks, num_dets)):
                min_val = float(np.min(temp_cost))
                if min_val > (1.0 - min_iou):
                    break
                r, c = np.unravel_index(np.argmin(temp_cost), temp_cost.shape)
                matched_rows.append(int(r))
                matched_cols.append(int(c))
                temp_cost[r, :] = 1e6
                temp_cost[:, c] = 1e6
            row_ind = np.array(matched_rows, dtype=int)
            col_ind = np.array(matched_cols, dtype=int)

    matches: list[tuple[int, int]] = []
    unmatched_tracks: list[int] = list(range(num_tracks))
    unmatched_detections: list[int] = list(range(num_dets))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= (1.0 - min_iou):
            matches.append((int(r), int(c)))
            if int(r) in unmatched_tracks:
                unmatched_tracks.remove(int(r))
            if int(c) in unmatched_detections:
                unmatched_detections.remove(int(c))

    return matches, unmatched_tracks, unmatched_detections


class OcclusionKalmanTracker:
    """Kalman Multi-Object Tracker with Hungarian Association & Occlusion Memory.

    Keeps occluded tracks alive for max_age frames to maintain 2-actor relational
    interaction state even when one actor is temporarily masked or occluded.
    """

    def __init__(self, max_age: int = 10, min_iou: float = 0.20):
        self.max_age = max(1, max_age)
        self.min_iou = min_iou
        self.tracks: list[Track] = []
        self._next_id: int = 1

    def reset(self) -> None:
        """Reset all active tracks."""
        self.tracks.clear()
        self._next_id = 1

    def update(
        self,
        detected_boxes: list[tuple[float, float, float, float]],
        detected_kps: list[np.ndarray],
        confidences: list[float],
    ) -> list[Track]:
        """Perform predict, associate, update, and track lifecycle management."""
        # 1. Kalman prediction step for existing tracks
        for t in self.tracks:
            t.predict()

        predicted_boxes = [t.box for t in self.tracks]

        # 2. Associate detections to predicted tracks using Hungarian algorithm
        matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
            predicted_boxes,
            detected_boxes,
            min_iou=self.min_iou,
        )

        # 3. Correct matched tracks with YOLOv8 measurement
        for trk_idx, det_idx in matches:
            self.tracks[trk_idx].update(
                detected_boxes[det_idx],
                detected_kps[det_idx],
                confidences[det_idx],
            )

        # 4. Spawn new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(
                track_id=self._next_id,
                box=detected_boxes[det_idx],
                keypoints=detected_kps[det_idx],
                confidence=confidences[det_idx],
            )
            self._next_id += 1
            self.tracks.append(new_track)

        # 5. Prune tracks that exceeded max_age
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return active tracks sorted with most recent/confident first
        return sorted(self.tracks, key=lambda t: (t.time_since_update, -t.confidence))
