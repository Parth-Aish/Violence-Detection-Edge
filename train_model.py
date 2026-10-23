"""Train and export the pose-sequence violence LSTM."""

from __future__ import annotations

import argparse
import math
import os
import random
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
FEATURES = 34
OPERATIONAL_THRESHOLD = 0.75


class ViolenceLSTM(nn.Module):
    """Small binary classifier for a 30-frame, 34-coordinate pose sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEATURES,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(sequence)
        return self.classifier(output[:, -1, :]).squeeze(-1)


class ProbabilityOutput(nn.Module):
    """Export adapter exposing a fighting probability rather than logits."""

    def __init__(self, model: ViolenceLSTM) -> None:
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


def export_onnx(model: ViolenceLSTM, destination: Path) -> None:
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
    x = np.load(args.features, allow_pickle=False).astype(np.float32, copy=False)
    y = np.load(args.labels, allow_pickle=False).astype(np.int64, copy=False).reshape(-1)

    if x.ndim != 3 or x.shape[1:] != (SEQUENCE_LENGTH, FEATURES):
        raise ValueError(
            f"Expected X shape (N, {SEQUENCE_LENGTH}, {FEATURES}), got {x.shape}."
        )
    if y.shape[0] != x.shape[0] or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must have one binary 0/1 value per input sequence.")
    class_counts = np.bincount(y, minlength=2)
    if np.any(class_counts < 2):
        raise ValueError(
            "At least two samples from each class are required for a stratified split."
        )
    if not np.isfinite(x).all():
        raise ValueError("Feature array contains NaN or infinite values.")

    indices = np.arange(len(y))
    test_count = max(2, math.ceil(len(y) * 0.2))
    if test_count >= len(y):
        raise ValueError("Not enough sequences for an 80/20 train/test split.")
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_count,
        random_state=args.seed,
        stratify=y,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y.astype(np.float32, copy=False))
    train_dataset = TensorDataset(x_tensor[train_indices], y_tensor[train_indices])
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = ViolenceLSTM().to(device)
    positive_count = int(np.sum(y[train_indices] == 1))
    negative_count = int(np.sum(y[train_indices] == 0))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        examples = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            examples += len(batch_y)
        print(f"epoch {epoch + 1:03d}/{args.epochs}: loss={total_loss / examples:.5f}")

    model.eval()
    with torch.no_grad():
        logits = model(x_tensor[test_indices].to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    predictions = (probabilities > OPERATIONAL_THRESHOLD).astype(np.int64)
    y_test = y[test_indices]
    metrics = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_sequences": float(len(test_indices)),
        "operating_threshold": OPERATIONAL_THRESHOLD,
    }
    for name, value in metrics.items():
        print(f"test_{name}: {value:.4f}" if name != "test_sequences" else f"{name}: {int(value)}")
    baseline_predictions = (probabilities >= 0.5).astype(np.int64)
    print(
        "test_precision_at_0.50: "
        f"{precision_score(y_test, baseline_predictions, zero_division=0):.4f}"
    )
    print(
        "test_recall_at_0.50: "
        f"{recall_score(y_test, baseline_predictions, zero_division=0):.4f}"
    )
    print(
        "test_f1_at_0.50: "
        f"{f1_score(y_test, baseline_predictions, zero_division=0):.4f}"
    )

    export_onnx(model.cpu(), args.onnx_out)
    print(f"Exported ONNX model: {args.onnx_out}")
    print(f"ONNX decision threshold used by backend: > {OPERATIONAL_THRESHOLD:.2f}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "processed" / "X_train.npy")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "processed" / "y_train.npy")
    parser.add_argument("--onnx-out", type=Path, default=ROOT / "backend" / "violence_lstm.onnx")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    try:
        train(args)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
