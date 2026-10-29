"""Authoritative 17-node COCO skeletal graph and normalized adjacency math."""

from __future__ import annotations

import numpy as np

NUM_NODES = 17

COCO_KEYPOINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# Anatomical bone connections in COCO 17-keypoint topology
COCO_EDGES = [
    # Head & facial structure
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    # Torso
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    # Upper extremities (Left arm)
    (5, 7),
    (7, 9),
    # Upper extremities (Right arm)
    (6, 8),
    (8, 10),
    # Lower extremities (Left leg)
    (11, 13),
    (13, 15),
    # Lower extremities (Right leg)
    (12, 14),
    (14, 16),
]


class COCOGraph:
    """COCO 17-joint skeletal graph with normalized Laplacian generator."""

    def __init__(self, num_nodes: int = NUM_NODES, edges: list[tuple[int, int]] = COCO_EDGES):
        self.num_nodes = num_nodes
        self.edges = edges
        self.adjacency = self._build_adjacency(self_loops=True)
        self.normalized_adjacency = self._normalize_adjacency(self.adjacency)

    def _build_adjacency(self, self_loops: bool = True) -> np.ndarray:
        """Construct symmetrical adjacency matrix A with shape (17, 17)."""
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i, j in self.edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        if self_loops:
            np.fill_diagonal(adj, 1.0)

        return adj

    @staticmethod
    def _normalize_adjacency(adj: np.ndarray) -> np.ndarray:
        """Calculate symmetrically normalized graph adjacency: D^(-1/2) * A * D^(-1/2)."""
        row_sum = np.sum(adj, axis=1)
        d_inv_sqrt = np.zeros_like(row_sum)
        positive_mask = row_sum > 0
        d_inv_sqrt[positive_mask] = np.power(row_sum[positive_mask], -0.5)
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        return norm_adj.astype(np.float32)

    def get_raw_adjacency(self) -> np.ndarray:
        """Return the unnormalized (17, 17) adjacency matrix."""
        return self.adjacency.copy()

    def get_normalized_adjacency(self) -> np.ndarray:
        """Return symmetrically normalized (17, 17) matrix D^(-1/2) * A * D^(-1/2)."""
        return self.normalized_adjacency.copy()


def get_coco_normalized_adjacency() -> np.ndarray:
    """Helper returning standard COCO-17 normalized adjacency matrix (17, 17)."""
    return COCOGraph().get_normalized_adjacency()
