"""Scene Context Gating using Lightweight MobileNetV3-Small.

Suppresses false positive alerts from controlled sports environments
(e.g., boxing rings, wrestling mats, basketball courts, soccer pitches, gyms)
by classifying scene context at 1 FPS with minimal CPU overhead (~14ms).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    torch = None  # type: ignore
    nn = None  # type: ignore
    MobileNet_V3_Small_Weights = None  # type: ignore
    mobilenet_v3_small = None  # type: ignore
    HAS_TORCH = False

logger = logging.getLogger("core.scene_gating")

# ImageNet class indices associated with athletics, contact sports, and controlled arenas
SPORTS_AND_CONTROLLED_CLASSES: set[int] = {
    416,  # balance beam
    422,  # barbell
    429,  # baseball
    430,  # basketball
    433,  # bathing cap
    518,  # crash helmet
    522,  # croquet ball
    543,  # dumbbell
    560,  # football helmet
    574,  # golf ball
    602,  # horizontal bar
    610,  # jersey, T-shirt
    615,  # knee pad
    702,  # parallel bars
    722,  # ping-pong ball
    746,  # puck, hockey puck
    747,  # punching bag, punch bag, punching ball
    752,  # racket, racquet
    768,  # rugby ball
    795,  # ski
    796,  # ski mask
    805,  # soccer ball
    819,  # stage
    841,  # sweatshirt
    842,  # swimming trunks, bathing trunks
    852,  # tennis ball
    884,  # vault
    890,  # volleyball
    902,  # whistle
    981,  # ballplayer, baseball player
}


class SceneClassifier:
    """Lightweight CPU-optimized scene context classifier."""

    def __init__(self, device: str = "cpu"):
        if not HAS_TORCH:
            logger.info("PyTorch not installed. Scene gating gracefully disabled.")
            self.device = None
            self.model: Optional[Any] = None
            self.mean: Optional[np.ndarray] = None
            self.std: Optional[np.ndarray] = None
            return

        self.device = torch.device(device)
        self.model: Optional[Any] = None
        self._load_model()

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _load_model(self) -> None:
        """Initialize MobileNetV3-Small in evaluation mode."""
        if not HAS_TORCH:
            self.model = None
            return

        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            model = mobilenet_v3_small(weights=weights)
            model.eval()
            model.to(self.device)
            # Freeze parameters for inference speed
            for param in model.parameters():
                param.requires_grad = False
            self.model = model
            logger.info("MobileNetV3-Small scene classifier loaded successfully.")
        except Exception as exc:
            logger.warning("Failed to initialize MobileNetV3-Small (%s). Scene gating will default to False.", exc)
            self.model = None

    def analyze_frame(self, frame: np.ndarray) -> bool:
        """Classify a single BGR frame.

        Returns True if the scene indicates a sport or controlled athletic environment,
        otherwise False.
        """
        if not HAS_TORCH or self.model is None or frame is None or frame.size == 0:
            return False

        try:
            # Resize directly to 224x224 and convert BGR -> RGB
            resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Normalize using ImageNet statistics
            normalized = (rgb - self.mean) / self.std

            # Format to (1, 3, 224, 224)
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)

            # Evaluate top-5 predicted classes
            top_vals, top_indices = torch.topk(probs, 5)
            top_indices_set = set(top_indices.cpu().numpy().tolist())

            # Check if top-1 is in sports classes OR top-5 contains sports classes with significant score
            has_sports_match = bool(top_indices_set.intersection(SPORTS_AND_CONTROLLED_CLASSES))
            if has_sports_match:
                # Require cumulative sports probability > 0.08 to avoid spurious edge noise
                sports_prob_sum = sum(
                    float(probs[idx].item())
                    for idx in SPORTS_AND_CONTROLLED_CLASSES
                    if idx in top_indices_set
                )
                if sports_prob_sum >= 0.08:
                    return True

            return False

        except Exception as exc:
            logger.debug("Error in scene context classification: %s", exc)
            return False
