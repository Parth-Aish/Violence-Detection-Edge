"""Spatial-Temporal Graph Convolutional Network (ST-GCN) Architecture.

Implements spatial graph convolutions over the 17-node COCO skeletal topology
combined with 1D temporal convolutions across 30-frame sequences.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import NUM_NODES, get_coco_normalized_adjacency


class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolution: X_out = Conv2d(X * A_norm, kernel_size=(1, 1)).

    Input:  (B, C_in, T, V)
    A_norm: (V, V)
    Output: (B, C_out, T, V)
    """

    def __init__(self, in_channels: int, out_channels: int, a_norm: torch.Tensor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer("a_norm", a_norm)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, C, T, V). MatMul across the node dimension V:
        # (B, C, T, V) @ (V, V) -> (B, C, T, V)
        x_g = torch.matmul(x, self.a_norm)
        return self.conv(x_g)


class STGCNBlock(nn.Module):
    """Single ST-GCN block fusing spatial graph convolution and temporal 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        a_norm: torch.Tensor,
        temporal_kernel_size: int = 9,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (temporal_kernel_size // 2, 0)

        # 1. Spatial Graph Convolution
        self.sgc = SpatialGraphConv(in_channels, out_channels, a_norm)
        self.bn_s = nn.BatchNorm2d(out_channels)

        # 2. Temporal Convolution (Kernel (9, 1), Padding (4, 0) keeps T=30)
        self.tc = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel_size, 1),
            padding=padding,
            bias=False,
        )
        self.bn_t = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # 3. Residual connection
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)

        # Spatial step
        x = self.sgc(x)
        x = self.bn_s(x)
        x = self.relu(x)

        # Temporal step
        x = self.tc(x)
        x = self.bn_t(x)
        x = self.relu(x + res)
        return self.dropout(x)


class STGCN(nn.Module):
    """Complete Spatial-Temporal Graph Convolutional Network for Violence Detection.

    Input shape:  (Batch, 2, 30, 17)
    Output shape: (Batch, 1) logits (or sigmoid probability when export_sigmoid=True)
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 1,
        num_nodes: int = NUM_NODES,
        export_sigmoid: bool = False,
    ):
        super().__init__()
        self.export_sigmoid = export_sigmoid
        a_norm = torch.from_numpy(get_coco_normalized_adjacency()).float()

        # Batch normalization on raw input keypoints
        self.data_bn = nn.BatchNorm2d(in_channels)

        # 3 ST-GCN Blocks: Channels 2 -> 16 -> 32 -> 64
        self.block1 = STGCNBlock(in_channels, 16, a_norm)
        self.block2 = STGCNBlock(16, 32, a_norm)
        self.block3 = STGCNBlock(32, 64, a_norm)

        # Fully connected head
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 2, 30, 17)
        x = self.data_bn(x)

        x = self.block1(x)  # -> (B, 16, 30, 17)
        x = self.block2(x)  # -> (B, 32, 30, 17)
        x = self.block3(x)  # -> (B, 64, 30, 17)

        # Global Average Pooling across Time (dim 2) and Nodes (dim 3)
        x = x.mean(dim=[2, 3])  # -> (B, 64)

        # Classifier
        logits = self.fc(x)  # -> (B, 1)

        if self.export_sigmoid:
            return torch.sigmoid(logits)
        return logits
