"""Train and export the physics-aware biomechanical violence detection model."""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
SEQUENCE_LENGTH = 30
FEATURES = 82
OPERATIONAL_THRESHOLD = 0.65


class TemporalAttention(nn.Module):
    """Temporal attention pooling: focuses on the exact 3-5 frames of impact/grappling."""

    def __init__(self, feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        scores = self.attn(x)  # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)  # [B, D]
        return context


class BiomechanicalViolenceModel(nn.Module):
    """Input LayerNorm -> Bidirectional GRU/LSTM -> Temporal Attention Pooling -> Classifier."""

    def __init__(
        self,
        input_dim: int = FEATURES,
        hidden_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        rnn_out_dim = hidden_dim * 2  # 128
        self.attention = TemporalAttention(feature_dim=rnn_out_dim, hidden_dim=32)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_out_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: [B, 30, 82]
        normed = self.input_norm(sequence)
        rnn_out, _ = self.rnn(normed)
        context = self.attention(rnn_out)
        logits = self.classifier(context).squeeze(-1)
        return logits


class ProbabilityOutput(nn.Module):
    """Export adapter exposing fighting probability rather than logits."""

    def __init__(self, model: BiomechanicalViolenceModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(sequence))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def export_onnx(model: BiomechanicalViolenceModel, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.onnx")
    adapter = ProbabilityOutput(model).eval()
    sample = torch.zeros(1, SEQUENCE_LENGTH, FEATURES, dtype=torch.float32)
    try:
        torch.onnx.export(
            adapter,
            sample,
            str(temporary),
            input_names=["pose_sequence"],
            output_names=["fighting_probability"],
            opset_version=17,
            dynamo=False,
        )
        import onnx

        onnx.checker.check_model(onnx.load(str(temporary)))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def train(args: argparse.Namespace) -> dict[str, float]:
    seed_everything(args.seed)
    print(f"Loading kinematic features from: {args.features}")
    x = np.load(args.features, allow_pickle=False).astype(np.float32, copy=False)
    y = np.load(args.labels, allow_pickle=False).astype(np.float32, copy=False).reshape(-1)

    if x.ndim != 3 or x.shape[1:] != (SEQUENCE_LENGTH, FEATURES):
        raise ValueError(
            f"Expected X shape (N, {SEQUENCE_LENGTH}, {FEATURES}), got {x.shape}."
        )
    if y.shape[0] != x.shape[0] or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Labels must have one binary 0/1 value per input sequence.")
    if not np.isfinite(x).all():
        raise ValueError("Feature array contains NaN or infinite values.")

    indices = np.arange(len(y))
    test_count = max(2, math.ceil(len(y) * 0.2))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_count,
        random_state=args.seed,
        stratify=y.astype(int),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Training on device: {device}")

    model = BiomechanicalViolenceModel().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Label smoothing to prevent overconfident false positives
    label_smoothing = 0.05
    y_train_raw = y[train_indices]
    y_train_smooth = y_train_raw * (1.0 - 2.0 * label_smoothing) + label_smoothing

    train_dataset = TensorDataset(
        torch.from_numpy(x[train_indices]),
        torch.from_numpy(y_train_smooth.astype(np.float32)),
    )

    for epoch in range(args.epochs):
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        model.train()
        total_loss = 0.0
        examples = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            examples += len(batch_y)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch + 1:03d}/{args.epochs}: loss={total_loss / examples:.5f}")

    # Evaluation on test set
    model.eval()
    x_test_tensor = torch.from_numpy(x[test_indices]).to(device)
    with torch.no_grad():
        logits = model(x_test_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    y_test = y[test_indices].astype(int)
    predictions_operational = (probabilities >= OPERATIONAL_THRESHOLD).astype(int)
    predictions_50 = (probabilities >= 0.50).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, predictions_operational, zero_division=0)),
        "recall": float(recall_score(y_test, predictions_operational, zero_division=0)),
        "f1": float(f1_score(y_test, predictions_operational, zero_division=0)),
        "test_sequences": float(len(test_indices)),
        "operating_threshold": OPERATIONAL_THRESHOLD,
    }
    print(f"\n--- Validation Metrics (Threshold = {OPERATIONAL_THRESHOLD:.2f}) ---")
    for name, value in metrics.items():
        print(f"test_{name}: {value:.4f}" if name != "test_sequences" else f"{name}: {int(value)}")
    print(f"test_f1_at_0.50: {f1_score(y_test, predictions_50, zero_division=0):.4f}")

    # Export to backend/violence_lstm_v2.onnx
    destination = Path(args.onnx_out)
    export_onnx(model.cpu(), destination)
    print(f"Successfully exported ONNX model to: {destination}")

    # Copy to frontend public models directory
    frontend_models_dir = ROOT / "frontend" / "public" / "models"
    if frontend_models_dir.exists():
        fe_dest = frontend_models_dir / destination.name
        shutil.copy2(destination, fe_dest)
        print(f"Synced model to frontend: {fe_dest}")

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data" / "processed" / "X_train_kinematic.npy",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data" / "processed" / "y_train.npy",
    )
    parser.add_argument(
        "--onnx-out",
        type=Path,
        default=ROOT / "backend" / "violence_lstm_v2.onnx",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU.")
    args = parser.parse_args()

    try:
        train(args)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
