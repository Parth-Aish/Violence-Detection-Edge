"""Multi-Person Occlusion Tracker with Bounding Box IoU & Momentum Extrapolation.

Maintains track continuity across severe physical occlusions (e.g. ground wrestling,
tackles, and close grappling) by predicting trajectory momentum for up to max_age=10 frames.
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


@dataclass
class Track:
    track_id: int
    box: tuple[float, float, float, float]
    keypoints: np.ndarray
    confidence: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    time_since_update: int = 0
    hits: int = 1
    age: int = 1

    def predict(self) -> None:
        """Advance state by linear momentum extrapolation with damping."""
        self.age += 1
        self.time_since_update += 1

        # Apply velocity with damping to avoid runaway drift during long occlusion
        damping = 0.92
        self.velocity *= damping
        cur_box = np.array(self.box, dtype=np.float32)
        new_box = cur_box + self.velocity

        # Ensure valid width and height
        if new_box[2] > new_box[0] and new_box[3] > new_box[1]:
            old_cx = (self.box[0] + self.box[2]) * 0.5
            old_cy = (self.box[1] + self.box[3]) * 0.5
            new_cx = (new_box[0] + new_box[2]) * 0.5
            new_cy = (new_box[1] + new_box[3]) * 0.5
            shift_x = new_cx - old_cx
            shift_y = new_cy - old_cy

            self.box = (float(new_box[0]), float(new_box[1]), float(new_box[2]), float(new_box[3]))
            # Extrapolate keypoint positions with centroid translation
            if self.keypoints is not None and len(self.keypoints) > 0:
                kps = self.keypoints.copy()
                kps[:, 0] += shift_x
                kps[:, 1] += shift_y
                self.keypoints = kps

    def update(
        self,
        new_box: tuple[float, float, float, float],
        new_kps: np.ndarray,
        conf: float,
    ) -> None:
        """Update track state with a newly observed detection."""
        inst_vel = np.array(new_box, dtype=np.float32) - np.array(self.box, dtype=np.float32)
        # Momentum smoothing: 60% new velocity, 40% historical velocity
        self.velocity = 0.60 * inst_vel + 0.40 * self.velocity
        self.box = new_box
        self.keypoints = new_kps.copy()
        self.confidence = conf
        self.time_since_update = 0
        self.hits += 1


class OcclusionKalmanTracker:
    """Kalman/Momentum Multi-Object Tracker with Occlusion Memory.

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
        # 1. Prediction step for existing tracks
        for t in self.tracks:
            t.predict()

        num_dets = len(detected_boxes)
        num_tracks = len(self.tracks)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if num_tracks > 0 and num_dets > 0:
            # Build similarity matrix based on IoU and centroid proximity
            similarities = np.zeros((num_tracks, num_dets), dtype=np.float32)
            for i, trk in enumerate(self.tracks):
                for j, det_box in enumerate(detected_boxes):
                    iou = compute_bbox_iou(trk.box, det_box)
                    if iou > 0.0:
                        similarities[i, j] = iou
                    else:
                        # Fallback: proximity metric when IoU is zero due to fast motion
                        dist = compute_box_center_dist(trk.box, det_box)
                        diag = np.hypot(det_box[2] - det_box[0], det_box[3] - det_box[1])
                        norm_dist = dist / max(diag, 1.0)
                        if norm_dist < 1.0:
                            similarities[i, j] = max(0.0, 0.35 - 0.35 * norm_dist)

            # Greedy bipartite matching
            while True:
                best_val = -1.0
                best_i = -1
                best_j = -1
                for i in range(num_tracks):
                    if i in matched_tracks:
                        continue
                    for j in range(num_dets):
                        if j in matched_dets:
                            continue
                        if similarities[i, j] > best_val:
                            best_val = similarities[i, j]
                            best_i = i
                            best_j = j

                if best_val >= self.min_iou and best_i != -1 and best_j != -1:
                    self.tracks[best_i].update(
                        detected_boxes[best_j],
                        detected_kps[best_j],
                        confidences[best_j],
                    )
                    matched_tracks.add(best_i)
                    matched_dets.add(best_j)
                else:
                    break

        # 2. Spawn new tracks for unmatched detections
        for j in range(num_dets):
            if j not in matched_dets:
                new_track = Track(
                    track_id=self._next_id,
                    box=detected_boxes[j],
                    keypoints=detected_kps[j],
                    confidence=confidences[j],
                )
                self._next_id += 1
                self.tracks.append(new_track)

        # 3. Prune tracks that exceeded max_age
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return active tracks sorted with most recent/confident first
        return sorted(self.tracks, key=lambda t: (t.time_since_update, -t.confidence))
