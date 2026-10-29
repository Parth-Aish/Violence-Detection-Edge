"""Train Spatial-Temporal Graph Convolutional Network (ST-GCN) for violence detection."""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

# Fix Windows console UTF-8 encoding for PyTorch ONNX export logs
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Add project root to sys.path
_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from backend.core.model import STGCN

LOGGER = logging.getLogger("train_stgcn")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class STGCNDataset(Dataset):
    """PyTorch Dataset for ST-GCN tensors with biomechanical data augmentations."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float().unsqueeze(1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[index].clone()  # Shape: (2, 30, 17)
        y = self.labels[index]

        if self.augment:
            # 1. Random horizontal flip: negate X coords and swap symmetrical joint pairs
            if random.random() > 0.5:
                x[0, :, :] = -x[0, :, :]
                pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
                for l_idx, r_idx in pairs:
                    tmp = x[:, :, l_idx].clone()
                    x[:, :, l_idx] = x[:, :, r_idx]
                    x[:, :, r_idx] = tmp

            # 2. Add subtle Gaussian noise to keypoints
            if random.random() > 0.5:
                noise = torch.randn_like(x) * 0.015
                x = x + noise

            # 3. Random scale jitter (0.95 to 1.05)
            if random.random() > 0.5:
                scale_factor = random.uniform(0.95, 1.05)
                x = x * scale_factor

        return x, y


def apply_label_smoothing(targets: torch.Tensor, smoothing: float = 0.05) -> torch.Tensor:
    """Smooth binary labels towards 0.5 to prevent overconfidence."""
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_pred_prob >= threshold).astype(int)
    y_true = y_true.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def export_stgcn_to_onnx(
    checkpoint_path: Path,
    output_onnx_path: Path,
) -> None:
    """Export a trained PyTorch ST-GCN checkpoint to ONNX with Sigmoid output."""
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    export_model = STGCN(in_channels=2, num_classes=1, export_sigmoid=True)
    export_model.load_state_dict(state)
    export_model.eval()

    dummy_input = torch.randn(1, 2, 30, 17, dtype=torch.float32)

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    LOGGER.info("Successfully exported ONNX ST-GCN model to: %s", output_onnx_path)

    fallback_path = output_onnx_path.parent.parent / "violence_stgcn.onnx"
    try:
        shutil.copyfile(output_onnx_path, fallback_path)
        LOGGER.info("Copied ONNX model to fallback path: %s", fallback_path)
    except Exception as e:
        LOGGER.warning("Could not copy to fallback path: %s", e)


def train_stgcn(
    data_dir: Path,
    output_onnx_path: Path,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    label_smoothing: float = 0.05,
    device: str = "cpu",
) -> dict[str, float]:
    seed_everything(42)

    features_path = data_dir / "X_train_stgcn.npy"
    labels_path = data_dir / "y_train.npy"

    if not features_path.is_file():
        raise FileNotFoundError(f"Missing ST-GCN dataset at {features_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels at {labels_path}")

    X = np.load(features_path)
    y = np.load(labels_path)
    LOGGER.info("Loaded dataset: X=%s, y=%s (Violence: %d, Safe: %d)", X.shape, y.shape, int(np.sum(y == 1)), int(np.sum(y == 0)))

    # Stratified 80/20 train/validation split
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    n_pos_train = int(len(pos_idx) * 0.8)
    n_neg_train = int(len(neg_idx) * 0.8)

    train_idx = np.concatenate([pos_idx[:n_pos_train], neg_idx[:n_neg_train]])
    val_idx = np.concatenate([pos_idx[n_pos_train:], neg_idx[n_neg_train:]])
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    train_dataset = STGCNDataset(X[train_idx], y[train_idx], augment=True)
    val_dataset = STGCNDataset(X[val_idx], y[val_idx], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    torch_device = torch.device(device)
    model = STGCN(in_channels=2, num_classes=1).to(torch_device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.3], device=torch_device))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_metrics = {}
    best_state = None

    LOGGER.info("Starting ST-GCN training for %d epochs on %s...", epochs, device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(torch_device)
            batch_y = batch_y.to(torch_device)

            targets_smooth = apply_label_smoothing(batch_y, smoothing=label_smoothing)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, targets_smooth)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_y)

        scheduler.step()
        train_loss = total_loss / len(train_dataset)

        # Validation step
        model.eval()
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(torch_device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_probs.extend(probs)
                val_targets.extend(batch_y.numpy().flatten())

        val_probs_arr = np.array(val_probs)
        val_targets_arr = np.array(val_targets)
        metrics = compute_metrics(val_targets_arr, val_probs_arr, threshold=0.5)

        LOGGER.info(
            "Epoch %2d/%2d | Loss: %.4f | Val Acc: %.3f | P: %.3f | R: %.3f | F1: %.3f",
            epoch,
            epochs,
            train_loss,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            best_metrics = metrics.copy()
            best_state = model.state_dict().copy()

    LOGGER.info(
        "Training complete. Best Validation Metrics: Acc=%.3f, Precision=%.3f, Recall=%.3f, F1=%.3f",
        best_metrics.get("accuracy", 0.0),
        best_metrics.get("precision", 0.0),
        best_metrics.get("recall", 0.0),
        best_metrics.get("f1", 0.0),
    )

    # Save PyTorch weights checkpoint
    pt_checkpoint = output_onnx_path.with_suffix(".pt")
    torch.save(best_state, pt_checkpoint)
    LOGGER.info("Saved PyTorch checkpoint to: %s", pt_checkpoint)

    # Export to ONNX
    export_stgcn_to_onnx(pt_checkpoint, output_onnx_path)
    return best_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-onnx",
        type=Path,
        default=Path("backend/models/violence_stgcn.onnx"),
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--export-only", action="store_true", help="Export existing .pt checkpoint to ONNX")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    try:
        if args.export_only:
            pt_path = args.output_onnx.with_suffix(".pt")
            if not pt_path.is_file():
                LOGGER.error("Cannot export: checkpoint %s not found.", pt_path)
                return 1
            export_stgcn_to_onnx(pt_path, args.output_onnx.resolve())
            return 0

        train_stgcn(
            data_dir=args.data_dir.resolve(),
            output_onnx_path=args.output_onnx.resolve(),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    except Exception as exc:
        LOGGER.error("Training/Export failed: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
